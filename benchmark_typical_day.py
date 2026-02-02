#!/usr/bin/env python3
"""
Typical Trading Day: SIMM Cloud Cost Benchmark

Measures raw SIMM evaluation throughput per backend, applies configurable
daily activity volumes, and outputs cloud cost comparison.

Usage:
    python benchmark_typical_day.py --trades 1000 --portfolios 5 --threads 8
    python benchmark_typical_day.py --pre-trade-checks 1000 --whatif-scenarios 100
"""
# Version: 1.0.0
MODEL_VERSION = "1.0.0"

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_trading_workflow import (
    setup_portfolio_and_kernel,
    _eval_aadc,
    _eval_gpu,
    _run_cpp_mode,
    _export_shared_data,
    AADC_AVAILABLE,
    CUDA_AVAILABLE,
    CPP_AVAILABLE,
    MODEL_VERSION as WORKFLOW_VERSION,
)
from model.simm_portfolio_cuda import (
    compute_simm_im_only_cuda,
)
from benchmark_aadc_vs_gpu import _build_benchmark_log_row
from common.portfolio import write_log

TYPICAL_DAY_CSV = Path(__file__).parent / "data" / "typical_day.csv"

TYPICAL_DAY_COLUMNS = [
    "timestamp", "num_trades", "num_portfolios", "num_risk_factors", "num_threads",
    "pre_trade_checks", "whatif_scenarios", "optimize_iters", "eod_computations",
    "total_evals",
    "backend", "kernel_recording_sec", "evals_per_sec", "median_eval_sec",
    "daily_time_sec", "cost_per_hour", "daily_cost_usd", "annual_cost_usd",
]

TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# Throughput measurement
# =============================================================================

def measure_throughput(eval_fn, ctx, warmup=5, n_evals=50):
    """
    Measure SIMM evaluation throughput for a gradient-capable backend.

    eval_fn: callable(ctx, agg_S_T) -> (im_values, gradients)
    Returns dict with median_eval_sec, evals_per_sec, im_value.
    """
    S = ctx["S"]
    allocation = ctx["initial_allocation"]
    agg_S_T = np.dot(S.T, allocation).T  # (P, K)

    # Warmup
    for _ in range(warmup):
        eval_fn(ctx, agg_S_T)

    # Timed evaluations with small noise
    rng = np.random.default_rng(42)
    times = []
    im_val = None
    for _ in range(n_evals):
        noise = 1.0 + rng.uniform(-0.001, 0.001, agg_S_T.shape)
        agg_noisy = agg_S_T * noise
        t0 = time.perf_counter()
        ims, _ = eval_fn(ctx, agg_noisy)
        times.append(time.perf_counter() - t0)
        if im_val is None:
            im_val = float(np.sum(ims))

    times = np.array(times)
    median_t = float(np.median(times))
    return {
        "median_eval_sec": median_t,
        "evals_per_sec": 1.0 / median_t if median_t > 0 else 0,
        "im_value": im_val,
    }


def measure_bf_throughput(ctx, warmup=5, n_evals=50):
    """Measure BF GPU throughput (forward-only IM, no gradients)."""
    S = ctx["S"]
    allocation = ctx["initial_allocation"]
    agg_S_T = np.dot(S.T, allocation).T

    rw = ctx["risk_weights"]
    cf = ctx["concentration_factors"]
    bid = ctx["bucket_id"]
    rmi = ctx["risk_measure_idx"]
    brc = ctx["bucket_rc"]
    brm = ctx["bucket_rm"]
    icf = ctx["intra_corr_flat"]
    bgf = ctx["bucket_gamma_flat"]
    nb = ctx["num_buckets"]
    gpu_c = ctx["gpu_constants"]

    def _eval(agg):
        return compute_simm_im_only_cuda(
            agg, rw, cf, bid, rmi, brc, brm, icf, bgf, nb, gpu_arrays=gpu_c,
        )

    for _ in range(warmup):
        _eval(agg_S_T)

    rng = np.random.default_rng(42)
    times = []
    im_val = None
    for _ in range(n_evals):
        noise = 1.0 + rng.uniform(-0.001, 0.001, agg_S_T.shape)
        t0 = time.perf_counter()
        ims = _eval(agg_S_T * noise)
        times.append(time.perf_counter() - t0)
        if im_val is None:
            im_val = float(np.sum(ims))

    times = np.array(times)
    median_t = float(np.median(times))
    return {
        "median_eval_sec": median_t,
        "evals_per_sec": 1.0 / median_t if median_t > 0 else 0,
        "im_value": im_val,
    }


def measure_cpp_throughput(ctx, num_trades, num_portfolios, num_threads,
                           seed=42, timing_iters=50):
    """
    Measure C++ AADC throughput via dedicated throughput mode.
    Runs timing_iters kernel evaluations (with warmup) and reports median time.
    """
    input_dir = os.path.join(os.path.dirname(__file__), "data", "typical_day_cpp_input")
    _export_shared_data(ctx, input_dir)

    parsed = _run_cpp_mode(
        "throughput", num_trades, num_portfolios, num_threads, seed,
        max_iters=timing_iters, input_dir=input_dir,
    )
    if parsed is None:
        return None

    rec_ms = parsed.get("recording_time_ms", parsed.get("kernel_recording_ms", 0))
    median_ms = parsed.get("median_eval_ms", 0)
    evals_per_sec = parsed.get("evals_per_sec", 0)

    if evals_per_sec <= 0:
        return None

    return {
        "recording_time_sec": rec_ms / 1000.0,
        "median_eval_sec": median_ms / 1000.0,
        "evals_per_sec": evals_per_sec,
        "im_value": 0,
    }


# =============================================================================
# Cost computation
# =============================================================================

def compute_daily_cost(evals_per_sec, total_evals, cost_per_hour,
                       kernel_rec_sec=0.0):
    """Compute cloud cost for running total_evals at measured throughput."""
    if evals_per_sec <= 0:
        return {"daily_time_sec": float("inf"), "daily_cost_usd": float("inf")}

    eval_time_sec = total_evals / evals_per_sec
    daily_time_sec = kernel_rec_sec + eval_time_sec
    daily_cost_usd = (daily_time_sec / 3600.0) * cost_per_hour
    return {"daily_time_sec": daily_time_sec, "daily_cost_usd": daily_cost_usd}


# =============================================================================
# Output formatting
# =============================================================================

def _fmt_time(sec):
    if sec < 0.001:
        return f"{sec*1e6:.0f} us"
    if sec < 1.0:
        return f"{sec*1000:.1f} ms"
    return f"{sec:.2f} s"


def _fmt_cost(usd):
    if usd < 0.0001:
        return f"${usd:.6f}"
    if usd < 0.01:
        return f"${usd:.4f}"
    if usd < 1.0:
        return f"${usd:.2f}"
    return f"${usd:,.2f}"


def print_results(results, total_evals, volumes, K, num_trades, num_portfolios,
                  cpu_rate, gpu_rate):
    """Print formatted comparison table."""
    pre, wif, opt, eod = volumes

    print()
    print("=" * 74)
    print("  Typical Trading Day: SIMM Cloud Cost Comparison")
    print("=" * 74)
    print(f"  Portfolio: {num_trades} trades, {num_portfolios} counterparties, "
          f"{K} risk factors")
    print()
    print("  Activity Volumes:")
    print(f"    Pre-trade margin checks:  {pre:>6,}")
    print(f"    What-if scenarios:        {wif:>6,}")
    print(f"    Optimization iterations:  {opt:>6,}")
    print(f"    EOD official margin:      {eod:>6,}")
    print(f"    Total SIMM evaluations:   {total_evals:>6,}")
    print()

    # Backend throughput table
    print("  Backend Throughput:")
    print("  " + "-" * 70)
    print(f"  {'Backend':<20} {'Kern Rec':>10} {'Eval/sec':>10} "
          f"{'Daily Evals':>12} {'Daily Time':>12} {'Daily Cost':>12}")
    print("  " + "-" * 70)

    for name, r in results.items():
        rec = _fmt_time(r["kernel_rec_sec"]) if r["kernel_rec_sec"] > 0.001 else "-"
        eps = f"{r['evals_per_sec']:,.0f}"
        dt = _fmt_time(r["daily_time_sec"])
        dc = _fmt_cost(r["daily_cost_usd"])
        print(f"  {name:<20} {rec:>10} {eps:>10} {total_evals:>12,} {dt:>12} {dc:>12}")

    print("  " + "-" * 70)

    # Annual cost table (sorted by cost)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["daily_cost_usd"])
    min_cost = sorted_results[0][1]["daily_cost_usd"] if sorted_results else 1e-10

    print()
    print(f"  Annual Cost Projection ({TRADING_DAYS_PER_YEAR} trading days):")
    print("  " + "-" * 70)
    print(f"  {'Backend':<20} {'$/hr':>8} {'Daily Cost':>12} "
          f"{'Annual Cost':>12} {'vs Cheapest':>12}")
    print("  " + "-" * 70)

    for name, r in sorted_results:
        annual = r["daily_cost_usd"] * TRADING_DAYS_PER_YEAR
        ratio = r["daily_cost_usd"] / min_cost if min_cost > 0 else 0
        rate = f"${r['cost_per_hour']:.2f}"
        print(f"  {name:<20} {rate:>8} {_fmt_cost(r['daily_cost_usd']):>12} "
              f"{_fmt_cost(annual):>12} {ratio:>11.1f}x")

    print("  " + "-" * 70)
    print()
    print(f"  Cloud rates: CPU ${cpu_rate:.2f}/hr | GPU ${gpu_rate:.2f}/hr")
    print("=" * 74)


# =============================================================================
# CSV output
# =============================================================================

def write_typical_day_csv(results, timestamp, num_trades, num_portfolios, K,
                          num_threads, volumes, total_evals):
    """Write/append to data/typical_day.csv."""
    pre, wif, opt, eod = volumes
    rows = []
    for backend, r in results.items():
        rows.append({
            "timestamp": timestamp,
            "num_trades": num_trades,
            "num_portfolios": num_portfolios,
            "num_risk_factors": K,
            "num_threads": num_threads,
            "pre_trade_checks": pre,
            "whatif_scenarios": wif,
            "optimize_iters": opt,
            "eod_computations": eod,
            "total_evals": total_evals,
            "backend": backend,
            "kernel_recording_sec": r["kernel_rec_sec"],
            "evals_per_sec": r["evals_per_sec"],
            "median_eval_sec": r["median_eval_sec"],
            "daily_time_sec": r["daily_time_sec"],
            "cost_per_hour": r["cost_per_hour"],
            "daily_cost_usd": r["daily_cost_usd"],
            "annual_cost_usd": r["daily_cost_usd"] * TRADING_DAYS_PER_YEAR,
        })

    df = pd.DataFrame(rows, columns=TYPICAL_DAY_COLUMNS)
    TYPICAL_DAY_CSV.parent.mkdir(parents=True, exist_ok=True)

    if TYPICAL_DAY_CSV.exists():
        existing = pd.read_csv(TYPICAL_DAY_CSV, nrows=0)
        if set(existing.columns) != set(TYPICAL_DAY_COLUMNS):
            old = pd.read_csv(TYPICAL_DAY_CSV)
            combined = pd.concat([old, df], ignore_index=True)
            combined = combined.reindex(columns=TYPICAL_DAY_COLUMNS)
            combined.to_csv(TYPICAL_DAY_CSV, mode="w", header=True, index=False)
        else:
            df.to_csv(TYPICAL_DAY_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(TYPICAL_DAY_CSV, mode="w", header=True, index=False)

    print(f"  Wrote {len(rows)} rows to {TYPICAL_DAY_CSV}")


def log_to_execution_log(results, timestamp, num_trades, num_simm_buckets,
                         num_portfolios, K, num_threads, total_evals,
                         trade_types_str):
    """Append summary rows to data/execution_log_portfolio.csv."""
    backend_model_map = {
        "AADC Python": "typical_day_aadc_py",
        "GPU (CUDA)": "typical_day_gpu",
        "BF GPU (no grad)": "typical_day_bf_gpu",
        "C++ AADC": "typical_day_cpp_aadc",
    }
    log_rows = []
    for name, r in results.items():
        model_name = backend_model_map.get(name, name)
        threads = num_threads if "AADC" in name or "C++" in name else 1
        row = _build_benchmark_log_row(
            timestamp=timestamp,
            model_name=model_name,
            model_version=MODEL_VERSION,
            trade_types_str=trade_types_str,
            num_trades=num_trades,
            num_simm_buckets=num_simm_buckets,
            num_portfolios=num_portfolios,
            num_threads=threads,
            im_result=r.get("im_value", 0),
            num_risk_factors=K,
            eval_time_sec=r["daily_time_sec"],
            kernel_recording_sec=r["kernel_rec_sec"] if r["kernel_rec_sec"] > 0 else None,
            num_simm_evals=total_evals,
        )
        log_rows.append(row)

    if log_rows:
        write_log(log_rows)
        print(f"  Logged {len(log_rows)} rows to data/execution_log_portfolio.csv")


# =============================================================================
# Main
# =============================================================================

def run_typical_day(args):
    """Main benchmark orchestrator."""
    trade_types = args.trade_types.split(",")
    num_trades = args.trades
    num_portfolios = args.portfolios
    num_threads = args.threads
    num_simm_buckets = args.simm_buckets

    pre = args.pre_trade_checks
    wif = args.whatif_scenarios
    opt = args.optimize_iters
    eod = args.eod_computations
    total_evals = pre + wif + opt + eod
    volumes = (pre, wif, opt, eod)

    cpu_rate = args.cpu_cost_per_hour
    gpu_rate = args.gpu_cost_per_hour
    warmup = args.warmup
    timing_evals = args.timing_evals

    print("=" * 74)
    print("  Typical Trading Day Benchmark")
    print("=" * 74)
    print(f"  Trades: {num_trades}  Portfolios: {num_portfolios}  "
          f"Threads: {num_threads}  Types: {args.trade_types}")
    print(f"  AADC: {'Yes' if AADC_AVAILABLE else 'No'}  "
          f"CUDA: {'Yes' if CUDA_AVAILABLE else 'No'}  "
          f"C++: {'Yes' if CPP_AVAILABLE else 'No'}")
    print(f"  Warmup: {warmup}  Timing evals: {timing_evals}")
    print()

    # Setup shared context
    print("  Setting up portfolio, correlations, kernel...")
    ctx = setup_portfolio_and_kernel(
        num_trades, num_portfolios, trade_types, num_simm_buckets, num_threads
    )
    K = ctx["K"]
    print(f"  Portfolio ready: T={num_trades}, K={K}, P={num_portfolios}")
    print()

    results = {}  # ordered dict by insertion

    # AADC Python
    if AADC_AVAILABLE:
        print(f"  Measuring AADC Python throughput ({timing_evals} evals)...", end="", flush=True)
        tp = measure_throughput(_eval_aadc, ctx, warmup, timing_evals)
        rec_sec = ctx.get("rec_time", 0) or 0
        cost = compute_daily_cost(tp["evals_per_sec"], total_evals, cpu_rate, rec_sec)
        results["AADC Python"] = {
            **tp, "kernel_rec_sec": rec_sec, "cost_per_hour": cpu_rate, **cost,
        }
        print(f" {tp['evals_per_sec']:,.0f} evals/sec")

    # GPU (CUDA) with gradients
    if CUDA_AVAILABLE:
        print(f"  Measuring GPU (CUDA) throughput ({timing_evals} evals)...", end="", flush=True)
        tp = measure_throughput(_eval_gpu, ctx, warmup, timing_evals)
        cost = compute_daily_cost(tp["evals_per_sec"], total_evals, gpu_rate)
        results["GPU (CUDA)"] = {
            **tp, "kernel_rec_sec": 0.0, "cost_per_hour": gpu_rate, **cost,
        }
        print(f" {tp['evals_per_sec']:,.0f} evals/sec")

    # BF GPU (no gradients)
    if CUDA_AVAILABLE:
        print(f"  Measuring BF GPU throughput ({timing_evals} evals)...", end="", flush=True)
        tp = measure_bf_throughput(ctx, warmup, timing_evals)
        cost = compute_daily_cost(tp["evals_per_sec"], total_evals, gpu_rate)
        results["BF GPU (no grad)"] = {
            **tp, "kernel_rec_sec": 0.0, "cost_per_hour": gpu_rate, **cost,
        }
        print(f" {tp['evals_per_sec']:,.0f} evals/sec")

    # C++ AADC
    if CPP_AVAILABLE:
        print(f"  Measuring C++ AADC throughput ({timing_evals} iters)...", end="", flush=True)
        tp = measure_cpp_throughput(
            ctx, num_trades, num_portfolios, num_threads,
            timing_iters=timing_evals,
        )
        if tp is not None:
            rec_sec = tp.get("recording_time_sec", 0)
            cost = compute_daily_cost(tp["evals_per_sec"], total_evals, cpu_rate, rec_sec)
            results["C++ AADC"] = {
                **tp, "kernel_rec_sec": rec_sec, "cost_per_hour": cpu_rate, **cost,
            }
            print(f" {tp['evals_per_sec']:,.0f} evals/sec")
        else:
            print(" FAILED")

    if not results:
        print("\nERROR: No backends available.")
        return None

    # Print results table
    print_results(results, total_evals, volumes, K, num_trades, num_portfolios,
                  cpu_rate, gpu_rate)

    # Write CSVs
    timestamp = datetime.now().isoformat()
    write_typical_day_csv(results, timestamp, num_trades, num_portfolios, K,
                          num_threads, volumes, total_evals)
    log_to_execution_log(results, timestamp, num_trades, num_simm_buckets,
                         num_portfolios, K, num_threads, total_evals,
                         args.trade_types)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Typical Trading Day: SIMM Cloud Cost Benchmark"
    )
    # Portfolio
    parser.add_argument("--trades", "-t", type=int, default=1000)
    parser.add_argument("--portfolios", "-p", type=int, default=5)
    parser.add_argument("--trade-types", type=str, default="ir_swap")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--simm-buckets", type=int, default=3)

    # Activity volumes
    parser.add_argument("--pre-trade-checks", type=int, default=500,
                        help="Pre-trade margin checks per day (default: 500)")
    parser.add_argument("--whatif-scenarios", type=int, default=50,
                        help="What-if stress scenarios per day (default: 50)")
    parser.add_argument("--optimize-iters", type=int, default=100,
                        help="Optimization iterations per day (default: 100)")
    parser.add_argument("--eod-computations", type=int, default=1,
                        help="EOD official margin computations (default: 1)")

    # Cloud cost rates
    parser.add_argument("--gpu-cost-per-hour", type=float, default=32.77,
                        help="GPU instance $/hr (default: 32.77, p4d.24xlarge)")
    parser.add_argument("--cpu-cost-per-hour", type=float, default=3.06,
                        help="CPU instance $/hr (default: 3.06, c5.18xlarge)")

    # Throughput measurement
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup evaluations before timing (default: 5)")
    parser.add_argument("--timing-evals", type=int, default=50,
                        help="Evaluations for throughput measurement (default: 50)")

    args = parser.parse_args()
    run_typical_day(args)


if __name__ == "__main__":
    main()
