"""
JSON Export Layer - Export animation-ready JSON for the visualization dashboard.

Wraps existing analytics modules (optimization, what-if, pre-trade) and
serializes their results to JSON files in data/animation/.

Usage:
    from model.json_export import export_all
    export_all(num_trades=50, num_portfolios=5)
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "animation"


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _json_default(obj):
    """Custom JSON serializer for types json doesn't handle natively."""
    import numpy as np
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def _write_json(filename: str, data: dict):
    _ensure_output_dir()
    path = OUTPUT_DIR / filename
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)
    return path


def _load_latest_execution_log() -> Optional[dict]:
    """Load the most recent ALL row from execution_log_portfolio.csv if available."""
    log_path = PROJECT_ROOT / "data" / "execution_log_portfolio.csv"
    if not log_path.exists():
        return None
    try:
        import csv
        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Find the most recent row (last one, or last 'ALL' row)
        all_rows = [r for r in rows if r.get('trade_group', '') == 'ALL']
        if not all_rows:
            all_rows = rows
        if not all_rows:
            return None
        latest = all_rows[-1]
        result = {}
        for key in ['crif_time', 'simm_time', 'im_sens_time', 'optimize_time',
                     'total_time', 'num_trades', 'num_portfolios', 'initial_im',
                     'final_im', 'im_reduction_pct', 'trades_moved']:
            if key in latest and latest[key]:
                try:
                    result[key] = float(latest[key])
                except (ValueError, TypeError):
                    result[key] = latest[key]
        return result if result else None
    except Exception:
        return None


def export_optimization_data(
    num_trades: int = 10,
    num_portfolios: int = 3,
    trade_types: str = "ir_swap",
    avg_maturity: float = None,
    maturity_spread: float = None,
) -> dict:
    """
    Export optimization data by delegating to run_optimization_demo.

    Returns the result dict and writes it to data/animation/optimization.json.
    """
    from scripts.run_optimization_demo import run_optimization_for_demo

    result = run_optimization_for_demo(
        num_trades, num_portfolios, trade_types,
        avg_maturity=avg_maturity, maturity_spread=maturity_spread,
    )

    result["tab"] = "optimization"
    result["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Enrich performance data with execution log if available
    exec_log = _load_latest_execution_log()
    if exec_log:
        if "performance" not in result:
            result["performance"] = {}
        result["performance"]["execution_log"] = exec_log

    _write_json("optimization.json", result)
    return result


def export_whatif_data(
    num_trades: int = 50,
    seed: int = 42,
) -> dict:
    """
    Export what-if analytics data.

    Generates trades, computes CRIF, runs margin attribution and scenarios,
    then serializes everything to data/animation/whatif.json.
    """
    import numpy as np
    import pandas as pd
    from model.trade_types import generate_market_environment, generate_trades_by_type
    from model.whatif_analytics import (
        compute_margin_attribution_aadc,
        compute_margin_attribution_naive,
        whatif_unwind_top_contributors,
        whatif_add_hedge,
        whatif_stress_scenario,
    )

    currencies = ['USD', 'EUR', 'GBP']
    market = generate_market_environment(currencies, seed=seed)
    trades = generate_trades_by_type('ir_swap', num_trades, currencies, seed=seed)

    # Compute CRIFs
    try:
        from model.simm_portfolio_aadc import compute_crif_aadc
        trade_sensitivities = {}
        all_crif_rows = []
        for trade in trades:
            trade_crif, _, _ = compute_crif_aadc([trade], market, num_threads=4)
            trade_sensitivities[trade.trade_id] = trade_crif
            all_crif_rows.append(trade_crif)

        portfolio_crif = pd.concat(all_crif_rows, ignore_index=True)
        portfolio_crif = portfolio_crif.groupby(
            ['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'AmountCurrency'],
            as_index=False
        ).agg({'TradeID': lambda x: ','.join(x.unique()), 'Amount': 'sum', 'AmountUSD': 'sum'})

        use_aadc = True
    except Exception:
        # Fallback: use bump-and-revalue CRIF
        from model.trade_types import compute_crif_for_trades
        trade_sensitivities = {}
        all_crif_rows = []
        for trade in trades:
            trade_crif = compute_crif_for_trades([trade], market)
            trade_sensitivities[trade.trade_id] = trade_crif
            all_crif_rows.append(trade_crif)

        portfolio_crif = pd.concat(all_crif_rows, ignore_index=True)
        group_cols = ['RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'AmountCurrency']
        if 'ProductClass' in portfolio_crif.columns:
            group_cols = ['ProductClass'] + group_cols
        portfolio_crif = portfolio_crif.groupby(
            group_cols, as_index=False
        ).agg({'TradeID': lambda x: ','.join(x.unique()), 'Amount': 'sum', 'AmountUSD': 'sum'})

        use_aadc = False

    # Attribution
    if use_aadc:
        attribution = compute_margin_attribution_aadc(portfolio_crif, trade_sensitivities, num_threads=4)
    else:
        attribution = compute_margin_attribution_naive(portfolio_crif, trade_sensitivities)

    # Serialize trade contributions
    trade_contributions = []
    for c in attribution.trade_contributions:
        trade_contributions.append({
            "trade_id": c.trade_id,
            "marginal_contribution": c.marginal_contribution,
            "contribution_pct": c.contribution_pct,
            "net_sensitivity": c.net_sensitivity,
            "is_margin_additive": bool(c.is_margin_additive),
        })

    # Scenarios
    # 1. Unwind top contributors
    unwind = whatif_unwind_top_contributors(
        portfolio_crif, trade_sensitivities, attribution, num_to_unwind=5
    )

    # 2. Add hedge (offset largest contributor)
    hedge_result = None
    if attribution.top_margin_consumers:
        top_trade_id = attribution.top_margin_consumers[0]
        hedge_crif = trade_sensitivities[top_trade_id].copy()
        hedge_crif['Amount'] *= -1
        hedge_crif['AmountUSD'] *= -1
        hedge_crif['TradeID'] = 'HEDGE_' + top_trade_id
        hedge_result = whatif_add_hedge(portfolio_crif, hedge_crif, f"Offset {top_trade_id}")

    # 3. Stress scenarios
    stress_scenarios_config = [
        ("IR +50%", {"Risk_IRCurve": 1.5}),
        ("FX +100%", {"Risk_FX": 2.0}),
        ("IR -30%", {"Risk_IRCurve": 0.7}),
        ("Combined +30%", {"Risk_IRCurve": 1.3, "Risk_FX": 1.3}),
    ]
    stress_results = []
    for name, shocks in stress_scenarios_config:
        sr = whatif_stress_scenario(portfolio_crif, shocks, name)
        stress_results.append({
            "scenario_name": sr.scenario_name,
            "shock_factors": shocks,
            "current_im": sr.current_im,
            "scenario_im": sr.scenario_im,
            "im_change": sr.im_change,
            "im_change_pct": sr.im_change_pct,
        })

    result = {
        "tab": "whatif",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "portfolio_summary": {
            "total_im": attribution.total_im,
            "num_trades": attribution.num_trades,
            "num_risk_factors": attribution.num_risk_factors,
        },
        "attribution": {
            "computation_method": attribution.computation_method,
            "computation_time_ms": attribution.computation_time_ms,
            "naive_time_estimate_ms": attribution.naive_time_estimate_ms,
            "speedup": round(attribution.naive_time_estimate_ms / max(attribution.computation_time_ms, 1), 1),
            "trade_contributions": trade_contributions,
            "top_margin_consumers": attribution.top_margin_consumers,
            "top_margin_reducers": attribution.top_margin_reducers,
        },
        "scenarios": {
            "unwind": {
                "scenario_name": unwind.scenario_name,
                "description": unwind.description,
                "current_im": unwind.current_im,
                "scenario_im": unwind.scenario_im,
                "im_change": unwind.im_change,
                "im_change_pct": unwind.im_change_pct,
                "trades_affected": unwind.trades_affected,
            },
            "hedge": {
                "scenario_name": hedge_result.scenario_name if hedge_result else "N/A",
                "description": hedge_result.description if hedge_result else "",
                "current_im": hedge_result.current_im if hedge_result else 0,
                "scenario_im": hedge_result.scenario_im if hedge_result else 0,
                "im_change": hedge_result.im_change if hedge_result else 0,
                "im_change_pct": hedge_result.im_change_pct if hedge_result else 0,
                "trades_affected": hedge_result.trades_affected if hedge_result else [],
            },
            "stress": stress_results,
        },
    }

    _write_json("whatif.json", result)
    return result


def export_pretrade_data(
    num_portfolios: int = 5,
    trades_per_counterparty: int = 30,
    seed: int = 42,
) -> dict:
    """
    Export pre-trade analytics data.

    Generates counterparty portfolios, proposes a new trade, runs routing
    analysis and bilateral vs cleared comparison, then serializes to
    data/animation/pretrade.json.
    """
    import numpy as np
    import pandas as pd
    from model.trade_types import (
        generate_market_environment, generate_trades_by_type, IRSwapTrade,
    )
    from model.pretrade_analytics import (
        analyze_trade_routing, compare_bilateral_vs_cleared, ClearingVenue,
        calculate_simm_margin,
    )

    currencies = ['USD', 'EUR', 'GBP']
    market = generate_market_environment(currencies, seed=seed)

    # Generate counterparty portfolios with different seeds for variety
    counterparty_names = [
        "Goldman Sachs", "JPMorgan", "Citibank", "Barclays", "Deutsche Bank"
    ][:num_portfolios]

    counterparty_portfolios = {}

    try:
        from model.simm_portfolio_aadc import compute_crif_aadc
        use_aadc = True
    except Exception:
        from model.trade_types import compute_crif_for_trades
        use_aadc = False

    for i, cp_name in enumerate(counterparty_names):
        cp_trades = generate_trades_by_type(
            'ir_swap', trades_per_counterparty, currencies, seed=seed + i + 10
        )
        if use_aadc:
            cp_crif, _, _ = compute_crif_aadc(cp_trades, market, num_threads=4)
        else:
            cp_crif = compute_crif_for_trades(cp_trades, market)
        counterparty_portfolios[cp_name] = cp_crif

    # Proposed new trade
    new_trade = IRSwapTrade(
        trade_id="NEW_10Y_USD",
        notional=100_000_000,
        currency="USD",
        maturity=10.0,
        fixed_rate=0.035,
        frequency=2,
        payer=False,
    )

    if use_aadc:
        new_trade_crif, _, _ = compute_crif_aadc([new_trade], market, num_threads=4)
    else:
        new_trade_crif = compute_crif_for_trades([new_trade], market)

    standalone_im = calculate_simm_margin(new_trade_crif)

    # Trade routing analysis
    routing = analyze_trade_routing(
        new_trade_crif,
        counterparty_portfolios,
        num_threads=4,
        use_gradient=False,  # Full recalc for accuracy
    )

    # Bilateral vs cleared
    # Use the recommended counterparty's portfolio for bilateral
    best_cp = routing.recommended_counterparty
    bilateral_vs_cleared = compare_bilateral_vs_cleared(
        new_trade_crif,
        bilateral_portfolio=counterparty_portfolios[best_cp],
        bilateral_counterparty=best_cp,
        cleared_portfolio=None,
        clearing_venue=ClearingVenue.LCH,
    )

    # Serialize
    counterparty_results = []
    for r in routing.counterparty_results:
        counterparty_results.append({
            "counterparty": r.counterparty,
            "current_im": r.current_im,
            "marginal_im": r.marginal_im,
            "new_im": r.new_im,
            "netting_benefit_pct": r.netting_benefit_pct,
        })

    result = {
        "tab": "pretrade",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "new_trade": {
            "trade_id": new_trade.trade_id,
            "type": "IR Swap",
            "notional": new_trade.notional,
            "currency": new_trade.currency,
            "maturity": new_trade.maturity,
            "standalone_im": standalone_im,
        },
        "routing": {
            "recommended_counterparty": routing.recommended_counterparty,
            "margin_savings": routing.margin_savings,
            "counterparty_results": counterparty_results,
        },
        "bilateral_vs_cleared": {
            "bilateral_counterparty": bilateral_vs_cleared.bilateral_counterparty,
            "bilateral_marginal_im": bilateral_vs_cleared.bilateral_marginal_im,
            "clearing_venue": bilateral_vs_cleared.clearing_venue,
            "cleared_marginal_im": bilateral_vs_cleared.cleared_marginal_im,
            "recommendation": bilateral_vs_cleared.recommendation,
            "rationale": bilateral_vs_cleared.rationale,
        },
    }

    _write_json("pretrade.json", result)
    return result


def export_all(
    num_trades: int = 50,
    num_portfolios: int = 5,
    trade_types: str = "ir_swap",
    only: Optional[List[str]] = None,
) -> dict:
    """
    Export all animation datasets and write a manifest.

    Args:
        num_trades: Number of trades for optimization and what-if
        num_portfolios: Number of portfolios/counterparties
        trade_types: Comma-separated trade types
        only: If specified, only export these datasets (e.g. ["optimization", "whatif"])

    Returns:
        Manifest dict listing available datasets.
    """
    _ensure_output_dir()

    datasets = []
    should_run = lambda name: only is None or name in only

    if should_run("optimization"):
        print("Exporting optimization data...")
        try:
            opt = export_optimization_data(
                num_trades=min(num_trades, 15),
                num_portfolios=min(num_portfolios, 5),
                trade_types=trade_types,
            )
            datasets.append({
                "name": "optimization",
                "file": "optimization.json",
                "generated_at": opt["generated_at"],
                "status": "ok",
            })
            print(f"  Optimization: {opt['config']['num_trades']} trades, "
                  f"IM reduction {opt['optimization']['im_reduction_pct']:.1f}%")
        except Exception as e:
            print(f"  Optimization export failed: {e}")
            datasets.append({"name": "optimization", "file": "optimization.json", "status": "error", "error": str(e)})

    if should_run("whatif"):
        print("Exporting what-if analytics data...")
        try:
            wi = export_whatif_data(num_trades=num_trades)
            datasets.append({
                "name": "whatif",
                "file": "whatif.json",
                "generated_at": wi["generated_at"],
                "status": "ok",
            })
            print(f"  What-If: {wi['portfolio_summary']['num_trades']} trades, "
                  f"total IM ${wi['portfolio_summary']['total_im']:,.0f}")
        except Exception as e:
            print(f"  What-If export failed: {e}")
            datasets.append({"name": "whatif", "file": "whatif.json", "status": "error", "error": str(e)})

    if should_run("pretrade"):
        print("Exporting pre-trade analytics data...")
        try:
            pt = export_pretrade_data(num_portfolios=num_portfolios)
            datasets.append({
                "name": "pretrade",
                "file": "pretrade.json",
                "generated_at": pt["generated_at"],
                "status": "ok",
            })
            print(f"  Pre-Trade: recommended {pt['routing']['recommended_counterparty']}, "
                  f"savings ${pt['routing']['margin_savings']:,.0f}")
        except Exception as e:
            print(f"  Pre-Trade export failed: {e}")
            datasets.append({"name": "pretrade", "file": "pretrade.json", "status": "error", "error": str(e)})

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
    }

    _write_json("manifest.json", manifest)
    print(f"\nManifest written to {OUTPUT_DIR / 'manifest.json'}")
    return manifest
