#!/usr/bin/env python3
"""
ISDA-SIMM Benchmark Runner

Runs both baseline (bump-and-revalue) and AADC implementations with identical
parameters, then displays a performance comparison.

Usage:
    python benchmark_simm.py \
        --trades 10 --simm-buckets 2 --portfolios 5 --threads 4 \
        --trade-types ir_swap,equity_option,fx_option,inflation_swap,xccy_swap
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from common.portfolio import LOG_FILE, LOG_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="ISDA-SIMM Benchmark: Baseline vs AADC")
    parser.add_argument("--trade-types", type=str, default="ir_swap",
                        help="Comma-separated trade types (ir_swap,equity_option,fx_option,inflation_swap,xccy_swap)")
    parser.add_argument("--trades", type=int, default=10,
                        help="Number of trades per type")
    parser.add_argument("--simm-buckets", type=int, default=2,
                        help="Number of currencies (SIMM IR buckets)")
    parser.add_argument("--portfolios", type=int, default=5,
                        help="Number of portfolio groups")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads (AADC workers)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline run (use existing log)")
    parser.add_argument("--skip-aadc", action="store_true",
                        help="Skip AADC run (use existing log)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run full allocation optimization (AADC only)")
    parser.add_argument("--method", type=str, default="gradient_descent",
                        choices=["gradient_descent", "greedy"],
                        help="Optimization method (default: gradient_descent)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Allow partial trade allocation across portfolios")
    parser.add_argument("--max-iters", type=int, default=100,
                        help="Maximum optimization iterations (default: 100)")
    return parser.parse_args()


def run_model(model_module, args_list):
    """Run a model's main() with the given CLI args."""
    saved_argv = sys.argv
    sys.argv = ["benchmark"] + args_list
    try:
        model_module.main()
    finally:
        sys.argv = saved_argv


def print_comparison(baseline_row, aadc_row):
    """Print side-by-side performance comparison."""
    print()
    print("=" * 80)
    print("                    ISDA-SIMM Benchmark Comparison")
    print("=" * 80)

    # Configuration
    row = aadc_row if aadc_row is not None else baseline_row
    print(f"Configuration:")
    print(f"  Trade Types:      {row['trade_types']}")
    print(f"  Trades:           {int(row['num_trades'])}")
    print(f"  SIMM Buckets:     {int(row['num_simm_buckets'])}")
    print(f"  Portfolios:       {int(row['num_portfolios'])}")
    print(f"  Threads:          {int(row['num_threads'])}")
    print()

    # Performance table
    print(f"Performance Comparison:")
    print("-" * 80)
    print(f"  {'Implementation':<20} {'CRIF':>10} {'SIMM':>10} {'IM Grad':>10} {'Total':>10} {'Speedup':>10}")
    print("-" * 80)

    if baseline_row is not None:
        b_crif = baseline_row['crif_time_sec']
        b_simm = baseline_row['simm_time_sec']
        b_grad = baseline_row['im_sens_time_sec']
        b_total = b_crif + b_simm + b_grad
        print(f"  {'Baseline (B&R)':<20} {b_crif:>9.3f}s {b_simm:>9.3f}s {b_grad:>9.3f}s {b_total:>9.3f}s {'1.0x':>10}")
    else:
        b_total = None

    if aadc_row is not None:
        a_crif = aadc_row['crif_time_sec']
        a_simm = aadc_row['simm_time_sec']
        a_grad = aadc_row['im_sens_time_sec']
        a_total = a_crif + a_simm + a_grad
        speedup = f"{b_total / a_total:.1f}x" if b_total and a_total > 0 else "-"
        print(f"  {'AADC (AAD)':<20} {a_crif:>9.3f}s {a_simm:>9.3f}s {a_grad:>9.3f}s {a_total:>9.3f}s {speedup:>10}")

    print("-" * 80)
    print()

    # Results comparison
    print(f"SIMM Results:")
    if baseline_row is not None:
        print(f"  Baseline IM:      ${baseline_row['im_result']:>20,.2f}")
    if aadc_row is not None:
        print(f"  AADC IM:          ${aadc_row['im_result']:>20,.2f}")
    if baseline_row is not None and aadc_row is not None:
        diff = abs(aadc_row['im_result'] - baseline_row['im_result'])
        rel = diff / max(abs(baseline_row['im_result']), 1e-10) * 100
        print(f"  Difference:       ${diff:>20,.2f} ({rel:.4f}%)")
    print()

    # Per-component speedups
    if baseline_row is not None and aadc_row is not None:
        print(f"Per-Component Speedup:")
        if a_crif > 0:
            print(f"  CRIF:             {b_crif / a_crif:.1f}x  (baseline {b_crif:.3f}s vs AADC {a_crif:.3f}s)")
        if a_simm > 0:
            print(f"  SIMM:             {b_simm / a_simm:.1f}x  (baseline {b_simm:.3f}s vs AADC {a_simm:.3f}s)")
        if a_grad > 0:
            print(f"  IM Gradient:      {b_grad / a_grad:.1f}x  (baseline {b_grad:.3f}s vs AADC {a_grad:.3f}s)")
        print()

    print("=" * 80)


def main():
    args = parse_args()

    # Build common CLI args list
    args_list = [
        "--trade-types", args.trade_types,
        "--trades", str(args.trades),
        "--simm-buckets", str(args.simm_buckets),
        "--portfolios", str(args.portfolios),
        "--threads", str(args.threads),
    ]

    # Clear execution log for clean comparison
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    # Run baseline
    if not args.skip_baseline:
        print()
        print("#" * 80)
        print("#  RUNNING BASELINE (bump-and-revalue)")
        print("#" * 80)
        print()
        from model.simm_portfolio_baseline import main as baseline_main
        import model.simm_portfolio_baseline as baseline_mod
        run_model(baseline_mod, args_list)
    else:
        print("[Skipping baseline]")

    # Run AADC
    if not args.skip_aadc:
        print()
        print("#" * 80)
        print("#  RUNNING AADC (automatic adjoint differentiation)")
        print("#" * 80)
        print()
        aadc_args = args_list.copy()
        if args.optimize:
            aadc_args.extend(["--optimize", "--method", args.method])
            if args.allow_partial:
                aadc_args.append("--allow-partial")
            aadc_args.extend(["--max-iters", str(args.max_iters)])
        import model.simm_portfolio_aadc as aadc_mod
        run_model(aadc_mod, aadc_args)
    else:
        print("[Skipping AADC]")

    # Read back results and compare
    if not LOG_FILE.exists():
        print("Error: no execution log found")
        sys.exit(1)

    df = pd.read_csv(LOG_FILE)

    # Get the aggregate (ALL) rows for each model
    baseline_rows = df[(df['model_name'] == 'simm_portfolio_baseline_py') & (df['group_id'] == 'ALL')]
    aadc_rows = df[(df['model_name'] == 'simm_portfolio_aadc_py') & (df['group_id'] == 'ALL')]

    baseline_row = baseline_rows.iloc[-1] if len(baseline_rows) > 0 else None
    aadc_row = aadc_rows.iloc[-1] if len(aadc_rows) > 0 else None

    if baseline_row is None and aadc_row is None:
        print("Error: no results found in execution log")
        sys.exit(1)

    print_comparison(baseline_row, aadc_row)


if __name__ == "__main__":
    main()
