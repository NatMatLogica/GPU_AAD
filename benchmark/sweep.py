#!/usr/bin/env python
"""
Automated parameter sweeps for ISDA-SIMM benchmark.

Collects results into a single CSV for publication analysis.

Usage:
    # Trade scaling
    python -m benchmark.sweep \
        --sweep-trades 50,100,500,1000,5000,10000 \
        --portfolios 5 --threads 8 --min-runs 30

    # Thread scaling (CPU)
    python -m benchmark.sweep \
        --trades 1000 --portfolios 5 \
        --sweep-threads 1,4,8,16,32,64,96 --min-runs 30

    # GPU scaling
    python -m benchmark.sweep \
        --trades 1000 --portfolios 5 \
        --sweep-gpus 1,2,4,8 --min-runs 30
"""

import argparse
import csv
import json
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data_gen import generate_benchmark_data
from benchmark.benchmark_fair import (
    get_available_backends, benchmark_backend, validate_backends,
)
from benchmark.environment import capture_environment
from benchmark.cost import CostTracker

RESULTS_DIR = Path(__file__).parent / "results"

CSV_COLUMNS = [
    "backend", "trades", "portfolios", "threads", "gpus",
    "median_ms", "mean_ms", "p5_ms", "p95_ms", "p99_ms",
    "ci95_lower_ms", "ci95_upper_ms", "cv",
    "speedup_vs_numpy", "cost_usd", "total_im",
    "num_runs", "outliers_excluded",
]


def _parse_int_list(s: str) -> list:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_sweep(args):
    """Execute a parameter sweep and write results to CSV."""
    env = capture_environment(seed=42)
    cost_tracker = CostTracker(
        cost_per_hour=args.cost_per_hour, platform=args.platform
    )
    trade_types = [t.strip() for t in args.trade_types.split(",")]

    # Determine sweep type
    if args.sweep_trades:
        sweep_param = "trades"
        sweep_values = _parse_int_list(args.sweep_trades)
    elif args.sweep_threads:
        sweep_param = "threads"
        sweep_values = _parse_int_list(args.sweep_threads)
    elif args.sweep_gpus:
        sweep_param = "gpus"
        sweep_values = _parse_int_list(args.sweep_gpus)
    else:
        print("Error: specify --sweep-trades, --sweep-threads, or --sweep-gpus")
        sys.exit(1)

    print("=" * 70)
    print(f"SIMM Benchmark Sweep: {sweep_param}")
    print(f"Values: {sweep_values}")
    print("=" * 70)

    # Output file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"sweep_{sweep_param}_{timestamp}.csv"
    json_path = RESULTS_DIR / f"sweep_{sweep_param}_{timestamp}.json"

    all_rows = []
    all_json_results = []

    for val in sweep_values:
        # Set parameters for this sweep step
        trades = val if sweep_param == "trades" else args.trades
        threads = val if sweep_param == "threads" else args.threads
        num_gpus = val if sweep_param == "gpus" else args.num_gpus

        print(f"\n--- {sweep_param}={val} (trades={trades}, threads={threads}, gpus={num_gpus}) ---")

        # Generate data
        try:
            data = generate_benchmark_data(
                num_trades=trades,
                num_portfolios=args.portfolios,
                trade_types=trade_types,
                num_simm_buckets=args.simm_buckets,
                num_threads=threads,
            )
        except Exception as e:
            print(f"  Data generation failed: {e}")
            continue

        # Set up backends
        backend_pairs = get_available_backends(
            num_threads=threads, num_gpus=num_gpus,
        )
        active_backends = []
        for name, backend in backend_pairs:
            try:
                backend.setup(data.factor_meta)
                active_backends.append((name, backend))
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

        if not active_backends:
            print("  No backends available, skipping.")
            continue

        agg_S = (data.S.T @ data.initial_allocation).T  # (P, K)

        # Benchmark each backend
        numpy_median = None
        step_results = {}

        for name, backend in active_backends:
            print(f"  Benchmarking {name}...", end=" ", flush=True)
            try:
                timing = benchmark_backend(
                    backend, agg_S, num_warmup=3, num_runs=args.min_runs
                )
            except Exception as e:
                print(f"FAILED ({e})")
                continue

            if name == "numpy":
                numpy_median = timing["median_ms"]

            speedup = numpy_median / timing["median_ms"] if numpy_median and timing["median_ms"] > 0 else 0
            cost_usd = cost_tracker.compute(timing["median_ms"] / 1000.0)

            row = {
                "backend": name,
                "trades": trades,
                "portfolios": args.portfolios,
                "threads": threads,
                "gpus": num_gpus if "cuda" in name else 0,
                "median_ms": timing["median_ms"],
                "mean_ms": timing["mean_ms"],
                "p5_ms": timing["p5_ms"],
                "p95_ms": timing["p95_ms"],
                "p99_ms": timing["p99_ms"],
                "ci95_lower_ms": timing["ci95_lower_ms"],
                "ci95_upper_ms": timing["ci95_upper_ms"],
                "cv": timing["cv"],
                "speedup_vs_numpy": speedup,
                "cost_usd": cost_usd,
                "total_im": timing["total_im"],
                "num_runs": timing["num_runs_total"],
                "outliers_excluded": timing["outliers_excluded"],
            }
            all_rows.append(row)
            step_results[name] = timing

            print(f"{timing['median_ms']:.3f}ms (speedup: {speedup:.1f}x)")

        all_json_results.append({
            "sweep_param": sweep_param,
            "sweep_value": val,
            "trades": trades,
            "threads": threads,
            "gpus": num_gpus,
            "results": step_results,
        })

    # Write CSV
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
        print(f"\nCSV saved to {csv_path}")

    # Write JSON with environment
    output = {
        "timestamp": datetime.now().isoformat(),
        "environment": env.to_dict(),
        "sweep_param": sweep_param,
        "sweep_values": sweep_values,
        "cost": cost_tracker.to_dict(),
        "results": all_json_results,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"JSON saved to {json_path}")

    print("\n" + "=" * 70)
    print("Sweep complete.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Automated SIMM benchmark parameter sweeps"
    )
    # Sweep parameters (exactly one required)
    parser.add_argument("--sweep-trades", type=str, default=None,
                        help="Comma-separated trade counts to sweep")
    parser.add_argument("--sweep-threads", type=str, default=None,
                        help="Comma-separated thread counts to sweep")
    parser.add_argument("--sweep-gpus", type=str, default=None,
                        help="Comma-separated GPU counts to sweep")
    # Fixed parameters
    parser.add_argument("--trades", type=int, default=1000,
                        help="Trade count (fixed, when not sweeping trades)")
    parser.add_argument("--portfolios", type=int, default=5,
                        help="Number of portfolios")
    parser.add_argument("--threads", type=int, default=8,
                        help="AADC threads (fixed, when not sweeping threads)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs (fixed, when not sweeping GPUs)")
    parser.add_argument("--trade-types", type=str, default="ir_swap,equity_option",
                        help="Comma-separated trade types")
    parser.add_argument("--simm-buckets", type=int, default=3,
                        help="Number of currencies")
    parser.add_argument("--min-runs", type=int, default=30,
                        help="Minimum timing runs per configuration")
    parser.add_argument("--cost-per-hour", type=float, default=0.0,
                        help="Cost per hour in USD")
    parser.add_argument("--platform", type=str, default="",
                        help="Platform name")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
