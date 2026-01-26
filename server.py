#!/usr/bin/env python
"""
Flask server to connect HTML demo to Python AADC backend.

Usage:
    python server.py

Then open http://localhost:8888/ in browser.
The demo will use AADC gradient optimization instead of JS heuristics.
"""

import sys
import time
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.trade_types import generate_market_environment, generate_trades_by_type
from model.simm_allocation_optimizer import reallocate_trades_optimal

app = Flask(__name__, static_folder='.')


@app.route('/')
def index():
    """Serve the main demo page."""
    return send_from_directory('.', 'optimization_demo.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)."""
    return send_from_directory('.', filename)


@app.route('/run_optimization')
def run_optimization():
    """
    Run AADC-based trade allocation optimization.

    Query parameters:
        trades: Number of trades (default: 50)
        portfolios: Number of portfolios (default: 5)
        types: Comma-separated trade types (default: ir_swap)
        avgMaturity: Average trade maturity in years (default: 5)
        maturitySpread: Maturity spread in years (default: 1)

    Returns JSON matching the format expected by optimization_demo.html
    """
    start_time = time.time()

    # Parse parameters
    num_trades = int(request.args.get('trades', 50))
    num_portfolios = int(request.args.get('portfolios', 5))
    trade_types_str = request.args.get('types', 'ir_swap')
    avg_maturity = float(request.args.get('avgMaturity', 5))
    maturity_spread = float(request.args.get('maturitySpread', 1))

    # Parse trade types
    trade_type_map = {
        'ir_swap': 'ir_swap',
        'equity_option': 'equity_option',
        'fx_option': 'fx_option',
        'inflation_swap': 'inflation_swap',
        'xccy_swap': 'xccy_swap',
    }
    trade_types = [trade_type_map.get(t.strip(), 'ir_swap')
                   for t in trade_types_str.split(',') if t.strip()]
    if not trade_types:
        trade_types = ['ir_swap']

    # Generate market environment
    currencies = ['USD', 'EUR', 'GBP', 'JPY']
    market = generate_market_environment(currencies, seed=42)

    # Generate trades
    trades = []
    trades_per_type = max(1, num_trades // len(trade_types))
    for tt in trade_types:
        try:
            tt_trades = generate_trades_by_type(
                tt, trades_per_type, currencies,
                seed=42,
                avg_maturity=avg_maturity,
                maturity_spread=maturity_spread
            )
            trades.extend(tt_trades)
        except Exception as e:
            print(f"Warning: Could not generate {tt} trades: {e}")

    # Trim or pad to exact count
    if len(trades) > num_trades:
        trades = trades[:num_trades]

    T = len(trades)
    if T == 0:
        return jsonify({'error': 'No trades generated'}), 400

    # Initial random allocation
    np.random.seed(42)
    group_ids = np.random.randint(0, num_portfolios, T)
    initial_allocation = np.zeros((T, num_portfolios))
    for t, g in enumerate(group_ids):
        initial_allocation[t, g] = 1.0

    # Run AADC optimization
    try:
        result = reallocate_trades_optimal(
            trades, market, num_portfolios,
            initial_allocation=initial_allocation,
            num_threads=4,
            allow_partial=False,
            method='gradient_descent',
            max_iters=100,
            verbose=False
        )
    except Exception as e:
        return jsonify({'error': f'Optimization failed: {str(e)}'}), 500

    # Build response in expected format

    # Convert trades to JSON-serializable format
    trades_json = []
    for i, trade in enumerate(trades):
        trade_dict = {
            'id': trade.trade_id,
            'type': type(trade).__name__.replace('Trade', ''),
            'notional': getattr(trade, 'notional', 1e6),
            'currency': getattr(trade, 'currency', 'USD'),
            'maturity': getattr(trade, 'maturity', 5.0),
            'sensitivity': result.get('trade_sensitivities', {}).get(i, 0),
            'standalone_im': result.get('trade_standalone_im', {}).get(i, 0),
            'initial_portfolio': int(group_ids[i]),
            'num_sensitivities': 12,  # Approximate
        }
        trades_json.append(trade_dict)

    # Get final allocation
    final_allocation = result['final_allocation']
    final_assignments = [int(np.argmax(final_allocation[t, :])) for t in range(T)]
    initial_assignments = [int(g) for g in group_ids]

    # Calculate portfolio IMs
    def calc_portfolio_ims(assignments, total_im_by_portfolio):
        portfolio_ims = []
        for p in range(num_portfolios):
            trade_count = sum(1 for a in assignments if a == p)
            im = total_im_by_portfolio.get(p, 0)
            portfolio_ims.append({
                'portfolio': p,
                'trades': trade_count,
                'im': float(im)
            })
        return portfolio_ims

    initial_portfolio_ims = calc_portfolio_ims(
        initial_assignments,
        result.get('initial_portfolio_ims', {})
    )
    final_portfolio_ims = calc_portfolio_ims(
        final_assignments,
        result.get('final_portfolio_ims', {})
    )

    # Build movements list
    movements = []
    for t in range(T):
        if initial_assignments[t] != final_assignments[t]:
            movements.append({
                'trade_id': trades[t].trade_id,
                'from_portfolio': initial_assignments[t],
                'to_portfolio': final_assignments[t],
                'standalone_im': trades_json[t]['standalone_im'],
            })

    # Build optimization steps for chart
    im_history = result.get('im_history', [])
    steps = []
    for i, im_val in enumerate(im_history):
        steps.append({
            'iteration': i,
            'total_im': float(im_val),
            'assignments': final_assignments if i == len(im_history) - 1 else initial_assignments,
        })

    # If no history, create simple 2-point history
    if not steps:
        steps = [
            {'iteration': 0, 'total_im': float(result['initial_im']), 'assignments': initial_assignments},
            {'iteration': 1, 'total_im': float(result['final_im']), 'assignments': final_assignments},
        ]

    # Calculate reduction percentage
    initial_im = result['initial_im']
    final_im = result['final_im']
    reduction_pct = (1 - final_im / initial_im) * 100 if initial_im > 0 else 0

    # Calculate maturity stats
    maturities = [getattr(t, 'maturity', 5.0) for t in trades]
    actual_avg_maturity = sum(maturities) / len(maturities) if maturities else 5.0
    min_maturity = min(maturities) if maturities else 5.0
    max_maturity = max(maturities) if maturities else 5.0

    elapsed_time = time.time() - start_time

    response = {
        'config': {
            'num_trades': T,
            'num_portfolios': num_portfolios,
            'trade_types': trade_types_str,
            'num_risk_factors': result.get('num_risk_factors', 100),
            'avg_maturity': round(actual_avg_maturity, 2),
            'maturity_range': [round(min_maturity, 2), round(max_maturity, 2)],
        },
        'trades': trades_json,
        'initial_state': {
            'assignments': initial_assignments,
            'portfolio_ims': initial_portfolio_ims,
            'total_im': float(result['initial_im']),
        },
        'final_state': {
            'assignments': final_assignments,
            'portfolio_ims': final_portfolio_ims,
            'total_im': float(result['final_im']),
        },
        'optimization': {
            'num_iterations': result.get('num_iterations', len(steps)),
            'steps': steps,
            'movements': movements,
            'im_reduction_pct': float(reduction_pct),
            'elapsed_time': elapsed_time,
            'converged': result.get('converged', True),
            'trades_moved': result.get('trades_moved', len(movements)),
        },
    }

    return jsonify(response)


if __name__ == '__main__':
    print("=" * 60)
    print("SIMM Trade Allocation Optimizer - AADC Backend")
    print("=" * 60)
    print()
    print("Starting server on http://0.0.0.0:8888")
    print("Open http://localhost:8888 in your browser")
    print()
    print("The demo will use Python AADC optimization instead of JS heuristics.")
    print("Press Ctrl+C to stop.")
    print()

    app.run(host='0.0.0.0', port=8888, debug=False, threaded=True)
