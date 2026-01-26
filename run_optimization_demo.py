#!/usr/bin/env python3
"""
Run optimization and generate JSON data for HTML visualization.

This script runs the trade allocation optimization and outputs detailed
step-by-step data that can be visualized in the HTML demo page.

Usage:
    python run_optimization_demo.py --trades 10 --portfolios 3 --output demo_data.json
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_optimization_for_demo(num_trades: int, num_portfolios: int, trade_types: str = "ir_swap",
                              avg_maturity: float = None, maturity_spread: float = None):
    """
    Run optimization and capture all intermediate states for visualization.

    Args:
        num_trades: Number of trades to generate
        num_portfolios: Number of portfolios
        trade_types: Comma-separated trade types
        avg_maturity: Average maturity in years (None = random)
        maturity_spread: Spread around average (None = default 1.0)

    Returns a dict with all data needed for the HTML visualization.
    """
    from model.trade_types import generate_market_environment, generate_trades_by_type
    from common.portfolio import allocate_portfolios, run_simm
    from model.simm_portfolio_aadc import precompute_all_trade_crifs

    # Generate market and trades
    currencies = ['USD', 'EUR'][:min(2, num_portfolios)]
    market = generate_market_environment(currencies, seed=42)

    all_trades = []
    for tt in trade_types.split(','):
        tt = tt.strip()
        trades = generate_trades_by_type(tt, num_trades, currencies, seed=42,
                                        avg_maturity=avg_maturity, maturity_spread=maturity_spread)
        all_trades.extend(trades)

    trades = all_trades[:num_trades]  # Limit to requested number

    # Initial allocation
    group_ids = allocate_portfolios(len(trades), num_portfolios)

    # Precompute trade CRIFs
    trade_crifs = precompute_all_trade_crifs(trades, market, num_threads=4)

    # Build trade info for visualization
    trade_info = []
    for i, trade in enumerate(trades):
        if trade.trade_id not in trade_crifs:
            continue
        crif = trade_crifs[trade.trade_id]
        # Compute standalone IM for this trade
        _, standalone_im, _ = run_simm(crif)

        # Get trade details
        info = {
            'id': trade.trade_id,
            'type': type(trade).__name__.replace('Trade', ''),
            'notional': getattr(trade, 'notional', getattr(trade, 'dom_notional', 1e6)),
            'currency': getattr(trade, 'currency', getattr(trade, 'domestic_ccy', 'USD')),
            'maturity': getattr(trade, 'maturity', 1.0),
            'standalone_im': standalone_im,
            'initial_portfolio': int(group_ids[i]),
            'num_sensitivities': len(crif),
        }
        trade_info.append(info)

    # Compute initial portfolio IMs
    initial_portfolio_ims = []
    for p in range(num_portfolios):
        p_trade_ids = [t['id'] for i, t in enumerate(trade_info) if group_ids[i] == p]
        if p_trade_ids:
            p_crifs = [trade_crifs[tid] for tid in p_trade_ids if tid in trade_crifs]
            if p_crifs:
                import pandas as pd
                combined = pd.concat(p_crifs, ignore_index=True)
                _, im, _ = run_simm(combined)
                initial_portfolio_ims.append({'portfolio': p, 'im': im, 'trades': len(p_trade_ids)})
            else:
                initial_portfolio_ims.append({'portfolio': p, 'im': 0, 'trades': 0})
        else:
            initial_portfolio_ims.append({'portfolio': p, 'im': 0, 'trades': 0})

    initial_total_im = sum(p['im'] for p in initial_portfolio_ims)

    # Run optimization with step tracking
    from model.simm_allocation_optimizer import (
        _get_unique_risk_factors,
        _build_sensitivity_matrix,
        record_allocation_im_kernel,
        compute_allocation_gradient,
        project_to_simplex,
    )

    # Get trade IDs that have CRIFs
    valid_trade_ids = [t['id'] for t in trade_info]
    T = len(valid_trade_ids)
    P = num_portfolios

    # Build sensitivity matrix
    filtered_crifs = {tid: trade_crifs[tid] for tid in valid_trade_ids}
    risk_factors = _get_unique_risk_factors(filtered_crifs)
    S = _build_sensitivity_matrix(filtered_crifs, valid_trade_ids, risk_factors)

    # Record kernel
    funcs, x_handles, im_output, _ = record_allocation_im_kernel(S, risk_factors, P)

    # Initial allocation matrix
    x = np.zeros((T, P))
    for i, tid in enumerate(valid_trade_ids):
        orig_idx = next(j for j, t in enumerate(trade_info) if t['id'] == tid)
        x[i, group_ids[orig_idx]] = 1.0

    # Track optimization steps
    optimization_steps = []

    # Run gradient descent with step recording
    max_iters = 50

    # First iteration to compute adaptive learning rate
    gradient, total_im = compute_allocation_gradient(
        funcs, x_handles, None, im_output, S, x, num_threads=4
    )

    # Compute adaptive learning rate based on gradient scale
    grad_max = np.abs(gradient).max()
    if grad_max > 1e-10:
        lr = 0.05 / grad_max  # Target ~5% allocation change per iteration
    else:
        lr = 1e-12

    for iteration in range(max_iters):
        if iteration > 0:
            gradient, total_im = compute_allocation_gradient(
                funcs, x_handles, None, im_output, S, x, num_threads=4
            )

        # Record current state
        current_assignments = np.argmax(x, axis=1).tolist()
        step_data = {
            'iteration': iteration,
            'total_im': total_im,
            'assignments': current_assignments,
            'allocation_matrix': x.tolist(),
        }
        optimization_steps.append(step_data)

        # Check convergence
        if iteration > 0:
            prev_im = optimization_steps[-2]['total_im']
            if abs(total_im - prev_im) / max(abs(prev_im), 1e-10) < 1e-6:
                break

        # Gradient step
        x_new = x - lr * gradient
        x_new = project_to_simplex(x_new)
        x = x_new

    # Final rounded allocation
    final_assignments = np.argmax(x, axis=1).tolist()

    # Compute final portfolio IMs
    final_portfolio_ims = []
    for p in range(num_portfolios):
        p_indices = [i for i in range(T) if final_assignments[i] == p]
        p_trade_ids = [valid_trade_ids[i] for i in p_indices]
        if p_trade_ids:
            import pandas as pd
            p_crifs = [trade_crifs[tid] for tid in p_trade_ids]
            combined = pd.concat(p_crifs, ignore_index=True)
            _, im, _ = run_simm(combined)
            final_portfolio_ims.append({'portfolio': p, 'im': im, 'trades': len(p_trade_ids)})
        else:
            final_portfolio_ims.append({'portfolio': p, 'im': 0, 'trades': 0})

    final_total_im = sum(p['im'] for p in final_portfolio_ims)

    # Identify trade movements
    movements = []
    initial_assignments = [t['initial_portfolio'] for t in trade_info]
    for i, tid in enumerate(valid_trade_ids):
        orig_idx = next(j for j, t in enumerate(trade_info) if t['id'] == tid)
        from_p = initial_assignments[orig_idx]
        to_p = final_assignments[i]
        if from_p != to_p:
            movements.append({
                'trade_id': tid,
                'from_portfolio': from_p,
                'to_portfolio': to_p,
                'standalone_im': trade_info[orig_idx]['standalone_im'],
            })

    # Compute actual maturity stats from generated trades
    maturities = [t['maturity'] for t in trade_info]
    actual_avg_maturity = sum(maturities) / len(maturities) if maturities else 0
    actual_min_maturity = min(maturities) if maturities else 0
    actual_max_maturity = max(maturities) if maturities else 0

    # Build final result
    result = {
        'config': {
            'num_trades': len(trade_info),
            'num_portfolios': num_portfolios,
            'trade_types': trade_types,
            'num_risk_factors': len(risk_factors),
            'avg_maturity': round(actual_avg_maturity, 2),
            'maturity_range': [round(actual_min_maturity, 2), round(actual_max_maturity, 2)],
        },
        'trades': trade_info,
        'initial_state': {
            'assignments': initial_assignments,
            'portfolio_ims': initial_portfolio_ims,
            'total_im': initial_total_im,
        },
        'final_state': {
            'assignments': final_assignments,
            'portfolio_ims': final_portfolio_ims,
            'total_im': final_total_im,
        },
        'optimization': {
            'num_iterations': len(optimization_steps),
            'steps': optimization_steps,
            'movements': movements,
            'im_reduction_pct': (1 - final_total_im / initial_total_im) * 100 if initial_total_im > 0 else 0,
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate optimization demo data")
    parser.add_argument("--trades", type=int, default=10, help="Number of trades (max 15)")
    parser.add_argument("--portfolios", type=int, default=3, help="Number of portfolios")
    parser.add_argument("--trade-types", type=str, default="ir_swap", help="Trade types")
    parser.add_argument("--output", type=str, default="demo_data.json", help="Output JSON file")
    args = parser.parse_args()

    num_trades = args.trades  # No cap - UI handles large datasets differently

    print(f"Running optimization demo: {num_trades} trades, {args.portfolios} portfolios...")

    try:
        result = run_optimization_for_demo(num_trades, args.portfolios, args.trade_types)

        output_path = PROJECT_ROOT / args.output
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Demo data saved to {output_path}")
        print(f"Initial IM: ${result['initial_state']['total_im']:,.2f}")
        print(f"Final IM:   ${result['final_state']['total_im']:,.2f}")
        print(f"Reduction:  {result['optimization']['im_reduction_pct']:.1f}%")
        print(f"Trades moved: {len(result['optimization']['movements'])}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
