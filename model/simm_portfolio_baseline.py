"""
SIMM Portfolio Baseline - Base SIMM implementation with portfolio allocation.

Generates a portfolio of trades (multiple types), allocates them to groups,
computes CRIF sensitivities via bump-and-revalue, runs SIMM per group,
computes dIM/dsensitivity via bump-and-revalue, and reports per-trade
SIMM contributions via Euler decomposition.

Version: 2.0.0

Usage:
    python -m model.simm_portfolio_baseline \
        --trades 10 --simm-buckets 2 --portfolios 5 --threads 8 \
        --trade-types ir_swap,equity_option,fx_option,inflation_swap,xccy_swap

Supported trade types: ir_swap, equity_option, fx_option, inflation_swap, xccy_swap
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

# Version
MODEL_VERSION = "2.0.0"
MODEL_NAME = "simm_portfolio_baseline_py"

# Bump size for IM sensitivity gradient
IM_SENS_BUMP = 1.0

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.trade_types import (
    MarketEnvironment, compute_crif_for_trades,
)
from src.agg_margins import SIMM
from common.portfolio import (
    parse_common_args, generate_portfolio, run_simm,
    make_empty_group_result,
    save_crif_log, write_log,
    print_results_table, build_log_rows,
    compute_trade_contributions, print_trade_contributions,
    save_trade_contributions_log, LOG_FILE,
)


def compute_crif(trades: list, market: MarketEnvironment) -> tuple:
    """
    Compute CRIF sensitivities via bump-and-revalue.

    Returns:
        (crif_df, crif_time_sec)
    """
    start = time.perf_counter()
    crif = compute_crif_for_trades(trades, market)
    crif_time = time.perf_counter() - start
    return crif, crif_time


def compute_im_gradient(group_crif: pd.DataFrame, base_im: float, bump: float = IM_SENS_BUMP) -> tuple:
    """
    Compute dIM/dsensitivity for each CRIF row via bump-and-revalue.

    Returns:
        (gradient_array, elapsed_time_sec)
    """
    n = len(group_crif)
    gradient = np.zeros(n)
    start = time.perf_counter()

    for i in range(n):
        bumped_crif = group_crif.copy()
        bumped_crif.iloc[i, bumped_crif.columns.get_loc("Amount")] += bump
        bumped_crif.iloc[i, bumped_crif.columns.get_loc("AmountUSD")] += bump
        simm_input = bumped_crif.drop(columns=["TradeID"])
        portfolio = SIMM(simm_input, "USD", 1)
        gradient[i] = (portfolio.simm - base_im) / bump

    elapsed = time.perf_counter() - start
    return gradient, elapsed


def main():
    args = parse_common_args("SIMM Portfolio Baseline Benchmark")

    trade_types = [t.strip() for t in args.trade_types.split(",")]
    num_trades = args.trades
    num_simm_buckets = args.simm_buckets
    num_portfolios = args.portfolios
    num_threads = args.threads
    trade_types_str = ",".join(trade_types)

    print("=" * 80)
    print("           SIMM Portfolio Baseline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model:           {MODEL_NAME} v{MODEL_VERSION}")
    print(f"  Trade Types:     {trade_types_str}")
    print(f"  Trades:          {num_trades}")
    print(f"  SIMM Buckets:    {num_simm_buckets}")
    print(f"  Portfolios:      {num_portfolios}")
    print(f"  Threads:         {num_threads} (baseline: single-threaded)")
    print()

    # Generate portfolio
    market, trades, group_ids, currencies = generate_portfolio(
        trade_types, num_trades, num_simm_buckets, num_portfolios
    )
    num_trades_actual = len(trades)

    # Process each group
    print(f"Processing groups (CRIF -> SIMM -> dIM/ds -> trade contributions)...")
    print()

    group_results = []
    all_crifs = []
    all_contrib_rows = []

    for group in range(num_portfolios):
        group_trade_indices = [i for i in range(len(trades)) if group_ids[i] == group]
        group_trades = [trades[i] for i in group_trade_indices]
        num_group_trades = len(group_trades)

        if num_group_trades == 0:
            group_results.append(make_empty_group_result(group))
            continue

        # 1. Compute CRIF via bump-and-revalue
        group_crif, crif_time = compute_crif(group_trades, market)

        if group_crif.empty:
            group_results.append(make_empty_group_result(group, num_group_trades, crif_time))
            continue

        # Collect CRIF for logging
        crif_with_group = group_crif.copy()
        crif_with_group["GroupID"] = group
        all_crifs.append(crif_with_group)

        # 2. Run SIMM
        portfolio, base_im, simm_time = run_simm(group_crif)

        # 3. Compute IM gradient and per-trade contributions
        if base_im > 0.0:
            gradient, grad_time = compute_im_gradient(group_crif, base_im)
            num_sens = len(gradient)

            # Euler decomposition: per-trade contribution
            contributions = compute_trade_contributions(group_crif, gradient)
            print_trade_contributions(contributions, group, base_im)

            # Collect for CSV output
            for trade_id, contrib in contributions.items():
                all_contrib_rows.append({
                    "model_name": MODEL_NAME,
                    "group_id": group,
                    "trade_id": trade_id,
                    "contribution": contrib,
                    "im_total": base_im,
                    "pct_of_im": (contrib / base_im * 100) if base_im > 0 else 0.0,
                })
        else:
            grad_time = 0.0
            num_sens = 0

        group_results.append({
            "group_id": group,
            "num_group_trades": num_group_trades,
            "im_result": base_im,
            "crif_time_sec": crif_time,
            "simm_time_sec": simm_time,
            "im_sens_time_sec": grad_time,
            "num_im_sensitivities": num_sens,
        })

    # Save logs
    save_crif_log(all_crifs)
    save_trade_contributions_log(all_contrib_rows)
    print()

    # Print results table
    totals = print_results_table(group_results, num_trades_actual)

    # Timing summary
    total_time = totals["total_crif_time"] + totals["total_simm_time"] + totals["total_grad_time"]
    print(f"Timing Summary:")
    print(f"  CRIF computation:  {totals['total_crif_time']:.3f} s ({num_portfolios} groups)")
    print(f"  SIMM aggregation:  {totals['total_simm_time']:.6f} s ({num_portfolios} groups)")
    print(f"  IM gradient:       {totals['total_grad_time']:.3f} s ({totals['total_num_sens']} sensitivities bumped)")
    print(f"  Total:             {total_time:.3f} s")
    print()

    # Write execution log
    log_rows = build_log_rows(
        group_results, totals,
        MODEL_NAME, MODEL_VERSION, trade_types_str,
        num_trades_actual, num_simm_buckets, num_portfolios,
        num_threads,
    )
    write_log(log_rows)
    print(f"Execution logged to {LOG_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
