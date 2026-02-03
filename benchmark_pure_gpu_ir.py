#!/usr/bin/env python
"""
Benchmark: Pure GPU vs AADC (Python & C++) - IR Swaps Only

Fair apples-to-apples comparison where all implementations:
1. Generate CRIF sensitivities (GPU: on device, AADC: via AAD)
2. Compute SIMM aggregation
3. Compute gradients for optimization

This benchmark uses ONLY IR swaps to ensure fair comparison,
as PureGPU only supports IR swaps currently.

Usage:
    python benchmark_pure_gpu_ir.py --trades 1000 --portfolios 5 --threads 8

Version: 1.0.0
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent))

MODEL_VERSION = "1.0.0"

# Check backends
try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False

CUDA_SIMULATOR = os.environ.get('NUMBA_ENABLE_CUDASIM', '0') == '1'
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available() or CUDA_SIMULATOR
except ImportError:
    CUDA_AVAILABLE = False

CPP_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "build", "simm_optimizer")
CPP_AVAILABLE = os.path.isfile(CPP_BINARY) and os.access(CPP_BINARY, os.X_OK)

# Project imports
from model.trade_types import (
    generate_trades_by_type,
    generate_market_environment,
    compute_crif_for_trades,
    IRSwapTrade,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    backend: str
    num_trades: int
    num_portfolios: int
    num_threads: int
    crif_time_sec: float
    simm_time_sec: float
    total_time_sec: float
    total_im: float
    num_evals: int = 1
    details: dict = field(default_factory=dict)


def fmt_time(t: Optional[float]) -> str:
    """Format time nicely."""
    if t is None:
        return "-"
    if t < 0.001:
        return f"{t*1e6:.1f} us"
    if t < 1.0:
        return f"{t*1000:.2f} ms"
    return f"{t:.3f} s"


# =============================================================================
# AADC Python Backend
# =============================================================================

def run_aadc_python(trades, market, num_portfolios, num_threads):
    """Run AADC Python backend (IR-only simplified SIMM)."""
    if not AADC_AVAILABLE:
        return None

    from model.trade_types import compute_crif_for_trades
    from model.pure_gpu_ir import IR_RISK_WEIGHTS, IR_CORRELATIONS, TENOR_LABELS

    T = len(trades)
    P = num_portfolios
    K = 12  # IR tenors only

    # Step 1: CRIF generation (use standard Python CRIF)
    t_crif_start = time.perf_counter()
    crif_df = compute_crif_for_trades(trades, market)
    crif_time = time.perf_counter() - t_crif_start

    # Build sensitivity matrix (IR tenors only)
    trade_ids = [t.trade_id for t in trades]
    trade_to_idx = {tid: i for i, tid in enumerate(trade_ids)}
    tenor_to_idx = {label: i for i, label in enumerate(TENOR_LABELS)}

    S = np.zeros((T, K), dtype=np.float64)
    for _, row in crif_df.iterrows():
        if row['RiskType'] == 'Risk_IRCurve':
            t_idx = trade_to_idx.get(row['TradeID'])
            k_idx = tenor_to_idx.get(row['Label1'])
            if t_idx is not None and k_idx is not None:
                S[t_idx, k_idx] += float(row['Amount'])

    # Record AADC kernel for IR-only SIMM
    t_rec_start = time.perf_counter()

    funcs = aadc.Functions()
    funcs.start_recording()

    # K inputs for aggregated sensitivities
    sens_inputs = []
    sens_handles = []
    for k in range(K):
        s = aadc.idouble(0.0)
        sens_handles.append(s.mark_as_input())
        sens_inputs.append(s)

    # Weighted sensitivities (CR=1 for simplicity)
    ws = [sens_inputs[k] * float(IR_RISK_WEIGHTS[k]) for k in range(K)]

    # K_ir^2 = sum_ij rho_ij * ws_i * ws_j
    k_ir_sq = aadc.idouble(0.0)
    for i in range(K):
        for j in range(K):
            rho = float(IR_CORRELATIONS[i, j])
            k_ir_sq = k_ir_sq + rho * ws[i] * ws[j]

    # K_ir = sqrt(K_ir^2) - use np.sqrt which works with idouble
    # Add small epsilon to avoid gradient singularity at 0
    k_ir_sq_safe = k_ir_sq + aadc.idouble(1e-30)
    k_ir = np.sqrt(k_ir_sq_safe)
    im_output = k_ir.mark_as_output()

    funcs.stop_recording()
    rec_time = time.perf_counter() - t_rec_start

    # Create allocation (round-robin)
    allocation = np.zeros((T, P), dtype=np.float64)
    for t in range(T):
        allocation[t, t % P] = 1.0

    # Aggregate: agg_S = S^T @ allocation -> (K, P)
    agg_S = S.T @ allocation

    # Evaluate for all P portfolios
    workers = aadc.ThreadPool(num_threads)

    t_simm_start = time.perf_counter()

    all_ims = np.zeros(P, dtype=np.float64)
    all_grads = np.zeros((K, P), dtype=np.float64)

    # Single batched evaluation
    inputs = {sens_handles[k]: agg_S[k, :] for k in range(K)}
    request = {im_output: sens_handles}

    results = aadc.evaluate(funcs, request, inputs, workers)

    all_ims = np.array(results[0][im_output])
    for k in range(K):
        all_grads[k, :] = np.array(results[1][im_output][sens_handles[k]])

    simm_time = time.perf_counter() - t_simm_start

    total_im = float(np.sum(all_ims))

    return BenchmarkResult(
        backend="AADC Python",
        num_trades=T,
        num_portfolios=P,
        num_threads=num_threads,
        crif_time_sec=crif_time,
        simm_time_sec=simm_time,
        total_time_sec=crif_time + rec_time + simm_time,
        total_im=total_im,
        details={
            "kernel_recording_time": rec_time,
            "per_portfolio_ims": list(all_ims),
            "K": K,
        }
    )


# =============================================================================
# Pure GPU Backend (IR only)
# =============================================================================

def run_pure_gpu(trades, market, num_portfolios, num_threads):
    """Run Pure GPU backend (CRIF + SIMM on GPU)."""
    if not CUDA_AVAILABLE:
        return None

    from model.pure_gpu_ir import PureGPUIRBackend

    T = len(trades)
    P = num_portfolios

    # Validate all trades are IR swaps
    for t in trades:
        if not isinstance(t, IRSwapTrade):
            raise ValueError(f"Pure GPU only supports IR swaps, got {type(t).__name__}")

    backend = PureGPUIRBackend()

    # Setup includes CRIF computation on GPU
    t_setup_start = time.perf_counter()
    timing = backend.setup(trades, market, trade_types=['ir_swap'])
    setup_time = time.perf_counter() - t_setup_start

    crif_time = timing.crif_time_sec

    # Create allocation
    allocation = np.zeros((T, P), dtype=np.float64)
    for t in range(T):
        allocation[t, t % P] = 1.0

    # Compute IM and gradient
    t_simm_start = time.perf_counter()
    im_values, gradients = backend.compute_im_and_gradient(allocation)
    simm_time = time.perf_counter() - t_simm_start

    total_im = float(np.sum(im_values))

    return BenchmarkResult(
        backend="Pure GPU",
        num_trades=T,
        num_portfolios=P,
        num_threads=1,  # GPU doesn't use CPU threads
        crif_time_sec=crif_time,
        simm_time_sec=simm_time,
        total_time_sec=setup_time + simm_time,
        total_im=total_im,
        details={
            "setup_time": setup_time,
            "per_portfolio_ims": list(im_values),
            "K": 12,  # IR tenors
        }
    )


# =============================================================================
# C++ AADC Backend
# =============================================================================

def run_cpp_aadc(num_trades, num_portfolios, num_threads, seed=42):
    """Run C++ AADC backend via subprocess.

    Note: C++ backend generates all 5 asset types, so IM won't match IR-only backends.
    This is included for timing comparison only. For IR-only C++ comparison,
    the C++ code would need a --trade-types flag.
    """
    if not CPP_AVAILABLE:
        return None

    cmd = [
        CPP_BINARY,
        "--trades", str(num_trades),
        "--portfolios", str(num_portfolios),
        "--threads", str(num_threads),
        "--max-iters", "1",  # Just one eval for benchmark
        "--seed", str(seed),
        "--method", "adam",
        "--no-greedy",  # Skip greedy refinement for timing benchmark
    ]

    try:
        t_start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        total_time = time.perf_counter() - t_start

        if result.returncode != 0:
            print(f"C++ AADC failed: {result.stderr}")
            return None

        # Parse output for timing
        import re
        output = result.stdout
        crif_time = 0.0
        simm_time = 0.0
        total_im = 0.0

        for line in output.split('\n'):
            if 'CRIF' in line and 'ms' in line:
                # Parse "Time: X.XX ms"
                match = re.search(r'Time:\s*([\d.]+)\s*ms', line)
                if match:
                    crif_time = float(match.group(1)) / 1000.0
            if 'Initial total IM' in line:
                match = re.search(r'\$([\d,.]+)', line)
                if match:
                    total_im = float(match.group(1).replace(',', ''))
            if 'Kernel recording' in line and 'ms' in line:
                match = re.search(r'([\d.]+)\s*ms', line)
                if match:
                    simm_time += float(match.group(1)) / 1000.0

        return BenchmarkResult(
            backend="C++ AADC",
            num_trades=num_trades,
            num_portfolios=num_portfolios,
            num_threads=num_threads,
            crif_time_sec=crif_time,
            simm_time_sec=simm_time,
            total_time_sec=total_time,
            total_im=total_im,
            details={"raw_output": output}
        )
    except subprocess.TimeoutExpired:
        print("C++ AADC timed out")
        return None
    except Exception as e:
        print(f"C++ AADC error: {e}")
        return None


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(num_trades, num_portfolios, num_threads, seed=42, verbose=True):
    """Run all backends and compare."""

    print("=" * 70)
    print("   Pure GPU vs AADC Benchmark (IR Swaps Only)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Trades:     {num_trades}")
    print(f"  Portfolios: {num_portfolios}")
    print(f"  Threads:    {num_threads}")
    print(f"  Seed:       {seed}")
    print()

    # Check available backends
    print("Available backends:")
    print(f"  AADC Python: {'Yes' if AADC_AVAILABLE else 'No'}")
    gpu_mode = "Yes (SIMULATOR)" if CUDA_SIMULATOR else ("Yes (GPU)" if CUDA_AVAILABLE else "No")
    print(f"  Pure GPU:    {gpu_mode}")
    print(f"  C++ AADC:    {'Yes' if CPP_AVAILABLE else 'No'}")
    if CUDA_SIMULATOR:
        print("  WARNING: CUDA simulator mode - Pure GPU timing not representative!")
    print()

    # Generate test data
    print("Generating trades and market data...")
    market = generate_market_environment(currencies=['USD'], seed=seed)
    trades = generate_trades_by_type('ir_swap', num_trades, currencies=['USD'], seed=seed)
    print(f"  Generated {len(trades)} IR swaps")
    print()

    results = []

    # Run AADC Python
    if AADC_AVAILABLE:
        print("Running AADC Python...")
        try:
            r = run_aadc_python(trades, market, num_portfolios, num_threads)
            if r:
                results.append(r)
                print(f"  CRIF:  {fmt_time(r.crif_time_sec)}")
                print(f"  SIMM:  {fmt_time(r.simm_time_sec)}")
                print(f"  Total: {fmt_time(r.total_time_sec)}")
                print(f"  IM:    ${r.total_im:,.2f}")
        except Exception as e:
            print(f"  Error: {e}")
        print()

    # Run Pure GPU
    if CUDA_AVAILABLE:
        print("Running Pure GPU...")
        try:
            r = run_pure_gpu(trades, market, num_portfolios, num_threads)
            if r:
                results.append(r)
                print(f"  CRIF:  {fmt_time(r.crif_time_sec)}")
                print(f"  SIMM:  {fmt_time(r.simm_time_sec)}")
                print(f"  Total: {fmt_time(r.total_time_sec)}")
                print(f"  IM:    ${r.total_im:,.2f}")
        except Exception as e:
            print(f"  Error: {e}")
        print()

    # Run C++ AADC
    if CPP_AVAILABLE:
        print("Running C++ AADC...")
        try:
            r = run_cpp_aadc(num_trades, num_portfolios, num_threads, seed)
            if r:
                results.append(r)
                print(f"  CRIF:  {fmt_time(r.crif_time_sec)}")
                print(f"  SIMM:  {fmt_time(r.simm_time_sec)}")
                print(f"  Total: {fmt_time(r.total_time_sec)}")
                print(f"  IM:    ${r.total_im:,.2f}")
        except Exception as e:
            print(f"  Error: {e}")
        print()

    # Summary table
    if len(results) >= 2:
        print("=" * 70)
        print("                         COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Backend':<15} {'CRIF':>12} {'SIMM':>12} {'Total':>12} {'Total IM':>18}")
        print("-" * 70)
        for r in results:
            print(f"{r.backend:<15} {fmt_time(r.crif_time_sec):>12} "
                  f"{fmt_time(r.simm_time_sec):>12} {fmt_time(r.total_time_sec):>12} "
                  f"${r.total_im:>15,.2f}")
        print("-" * 70)

        # Compute speedups relative to AADC Python
        aadc_py = next((r for r in results if r.backend == "AADC Python"), None)
        if aadc_py:
            print("\nSpeedups vs AADC Python:")
            for r in results:
                if r.backend != "AADC Python":
                    speedup = aadc_py.total_time_sec / r.total_time_sec if r.total_time_sec > 0 else 0
                    print(f"  {r.backend}: {speedup:.1f}x")

        # Verify IM values match (only for IR-only backends)
        ir_only_results = [r for r in results if r.backend != "C++ AADC"]
        if len(ir_only_results) >= 2:
            print("\nIM Validation (IR-only backends):")
            ref_im = ir_only_results[0].total_im
            for r in ir_only_results[1:]:
                diff = abs(r.total_im - ref_im)
                rel = diff / ref_im if ref_im > 0 else 0
                match = "MATCH" if rel < 0.01 else f"DIFF {rel:.2%}"
                print(f"  {ir_only_results[0].backend} vs {r.backend}: {match}")

        # Note about C++ AADC
        cpp_result = next((r for r in results if r.backend == "C++ AADC"), None)
        if cpp_result and len(ir_only_results) >= 1:
            print(f"\nNote: C++ AADC includes ALL 5 asset types (not IR-only), so its IM is"
                  f"\n      higher and not comparable to IR-only backends.")

    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: Pure GPU vs AADC (IR Swaps Only)"
    )
    parser.add_argument('--trades', '-t', type=int, default=1000)
    parser.add_argument('--portfolios', '-p', type=int, default=5)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    run_benchmark(
        num_trades=args.trades,
        num_portfolios=args.portfolios,
        num_threads=args.threads,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
