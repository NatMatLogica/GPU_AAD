#!/usr/bin/env python3
"""
Benchmark: AADC v1 vs v2 Optimization Performance Comparison.

Compares the performance of:
- v1: P separate aadc.evaluate() calls (one per portfolio)
- v2: Single aadc.evaluate() call for all P portfolios

Expected improvement: 10-200x for typical portfolio counts (P=5-20)

Usage:
    # Quick test (50 trades, 5 portfolios)
    python benchmark_aadc_v1_vs_v2.py --trades 50 --portfolios 5

    # Medium benchmark (500 trades, 5 portfolios)
    python benchmark_aadc_v1_vs_v2.py --trades 500 --portfolios 5 --optimize

    # Large benchmark (1000 trades, 10 portfolios)
    python benchmark_aadc_v1_vs_v2.py --trades 1000 --portfolios 10 --optimize --threads 8

    # Full comparison with optimization
    python benchmark_aadc_v1_vs_v2.py --trades 500 --portfolios 5 --threads 8 --optimize
"""

import numpy as np
import pandas as pd
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False
    print("ERROR: AADC not available. Install MatLogica AADC to run this benchmark.")
    sys.exit(1)

from model.trade_types import (
    MarketEnvironment, YieldCurve, VolSurface, InflationCurve,
    IRSwapTrade, EquityOptionTrade, FXOptionTrade,
    NUM_IR_TENORS, IR_TENORS,
)
from model.simm_portfolio_aadc import precompute_all_trade_crifs

# v1 imports
from model.simm_allocation_optimizer import (
    record_single_portfolio_simm_kernel,
    compute_allocation_gradient_chainrule,
    optimize_allocation_gradient_descent_efficient,
    _get_unique_risk_factors,
    _build_sensitivity_matrix,
    _get_factor_metadata,
    reallocate_trades_optimal,
)

# v2 imports
from model.simm_portfolio_aadc_v2 import (
    record_single_portfolio_simm_kernel_v2,
    compute_all_portfolios_im_gradient_v2,
    compute_allocation_gradient_chainrule_v2,
    optimize_allocation_gradient_descent_v2,
    _get_factor_metadata_v2,
)

from model.simm_allocation_optimizer_v2 import (
    reallocate_trades_optimal_v2,
    compare_v1_v2_results,
)


# =============================================================================
# Test Data Generation
# =============================================================================

def create_market_environment() -> MarketEnvironment:
    """Create standard market environment for testing."""
    np.random.seed(42)

    # Yield curves (USD, EUR, GBP)
    curves = {
        "USD": YieldCurve(zero_rates=np.linspace(0.02, 0.045, NUM_IR_TENORS)),
        "EUR": YieldCurve(zero_rates=np.linspace(0.01, 0.025, NUM_IR_TENORS)),
        "GBP": YieldCurve(zero_rates=np.linspace(0.015, 0.035, NUM_IR_TENORS)),
    }

    # Vol surfaces (VolSurface takes vols array, NUM_VEGA_EXPIRIES = 6)
    vol_surfaces = {
        "SPX": VolSurface(vols=np.full(6, 0.18)),
        "EURUSD": VolSurface(vols=np.full(6, 0.08)),
        "GBPUSD": VolSurface(vols=np.full(6, 0.10)),
    }

    return MarketEnvironment(
        curves=curves,
        vol_surfaces=vol_surfaces,
        equity_spots={"SPX": 4500.0, "AAPL": 180.0},
        fx_spots={"EURUSD": 1.08, "GBPUSD": 1.26},
        inflation=InflationCurve(),
    )


def generate_trades(num_trades: int, trade_types: List[str] = None) -> List:
    """Generate random trades for benchmarking."""
    if trade_types is None:
        trade_types = ["ir_swap", "equity_option", "fx_option"]

    np.random.seed(42)
    trades = []

    for i in range(num_trades):
        trade_type = trade_types[i % len(trade_types)]

        if trade_type == "ir_swap":
            trade = IRSwapTrade(
                trade_id=f"irs_{i}",
                currency=np.random.choice(["USD", "EUR", "GBP"]),
                notional=np.random.uniform(1e6, 1e8),
                fixed_rate=np.random.uniform(0.02, 0.05),
                maturity=np.random.choice([2, 3, 5, 7, 10]),
                frequency=np.random.choice([2, 4]),
                payer=np.random.choice([True, False]),
            )
        elif trade_type == "equity_option":
            trade = EquityOptionTrade(
                trade_id=f"eq_{i}",
                currency="USD",
                notional=np.random.uniform(1e5, 1e7),
                strike=np.random.uniform(4000, 5000),
                maturity=np.random.uniform(0.25, 2.0),
                is_call=np.random.choice([True, False]),
                underlying="SPX",
                dividend_yield=0.015,
                equity_bucket=np.random.randint(1, 12),
            )
        elif trade_type == "fx_option":
            trade = FXOptionTrade(
                trade_id=f"fx_{i}",
                domestic_ccy="USD",
                foreign_ccy=np.random.choice(["EUR", "GBP"]),
                notional=np.random.uniform(1e6, 1e8),
                strike=np.random.uniform(1.0, 1.3),
                maturity=np.random.uniform(0.25, 2.0),
                is_call=np.random.choice([True, False]),
            )
        else:
            continue

        trades.append(trade)

    return trades


def create_initial_allocation(num_trades: int, num_portfolios: int) -> np.ndarray:
    """Create initial allocation (round-robin assignment)."""
    allocation = np.zeros((num_trades, num_portfolios))
    for t in range(num_trades):
        allocation[t, t % num_portfolios] = 1.0
    return allocation


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_gradient_computation(
    funcs, sens_handles, im_output, S, allocation,
    num_threads: int, workers, num_iters: int = 20,
) -> Tuple[float, float]:
    """
    Benchmark gradient computation for v1 vs v2.

    Returns (v1_time, v2_time) in seconds.
    """
    # v1: P separate evaluate() calls per iteration
    v1_start = time.perf_counter()
    for _ in range(num_iters):
        compute_allocation_gradient_chainrule(
            funcs, sens_handles, im_output, S, allocation, num_threads, workers
        )
    v1_time = time.perf_counter() - v1_start

    # v2: 1 evaluate() call per iteration
    v2_start = time.perf_counter()
    for _ in range(num_iters):
        compute_allocation_gradient_chainrule_v2(
            funcs, sens_handles, im_output, S, allocation, num_threads, workers
        )
    v2_time = time.perf_counter() - v2_start

    return v1_time, v2_time


def benchmark_optimization(
    trades: List, market: MarketEnvironment,
    num_portfolios: int, initial_allocation: np.ndarray,
    num_threads: int, max_iters: int, verbose: bool,
) -> Tuple[Dict, Dict]:
    """
    Benchmark full optimization pipeline for v1 vs v2.

    Returns (v1_result, v2_result).
    """
    print("\n" + "=" * 70)
    print("Running v1 Optimization (P separate evaluate() calls)")
    print("=" * 70)

    v1_result = reallocate_trades_optimal(
        trades=trades,
        market=market,
        num_portfolios=num_portfolios,
        initial_allocation=initial_allocation,
        num_threads=num_threads,
        allow_partial=False,
        method='gradient_descent',
        max_iters=max_iters,
        verbose=verbose,
    )

    print("\n" + "=" * 70)
    print("Running v2 Optimization (SINGLE evaluate() call)")
    print("=" * 70)

    v2_result = reallocate_trades_optimal_v2(
        trades=trades,
        market=market,
        num_portfolios=num_portfolios,
        initial_allocation=initial_allocation,
        num_threads=num_threads,
        allow_partial=False,
        method='gradient_descent',
        max_iters=max_iters,
        verbose=verbose,
    )

    return v1_result, v2_result


# =============================================================================
# Results Display
# =============================================================================

def print_benchmark_results(
    args,
    v1_time: float, v2_time: float,
    v1_result: Dict = None, v2_result: Dict = None,
):
    """Print formatted benchmark results."""
    speedup = v1_time / max(v2_time, 1e-10)
    improvement_pct = (v1_time - v2_time) / v1_time * 100

    print("\n")
    print("=" * 80)
    print("                    AADC v1 vs v2 Benchmark Results")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Trades:           {args.trades}")
    print(f"  Portfolios:       {args.portfolios}")
    print(f"  Threads:          {args.threads}")
    print(f"  Max iterations:   {args.max_iters}")

    print("\n" + "-" * 80)
    print("Gradient Computation Performance:")
    print("-" * 80)
    print(f"  v1 (P evaluates): {v1_time*1000:.2f} ms")
    print(f"  v2 (1 evaluate):  {v2_time*1000:.2f} ms")
    print(f"  SPEEDUP:          {speedup:.1f}x")
    print(f"  Improvement:      {improvement_pct:.1f}%")

    if v1_result and v2_result:
        print("\n" + "-" * 80)
        print("Optimization Results:")
        print("-" * 80)
        print(f"  {'Metric':<25} {'v1':>15} {'v2':>15} {'Diff':>15}")
        print(f"  {'-'*70}")
        print(f"  {'Initial IM':.<25} ${v1_result['initial_im']:>14,.0f} ${v2_result['initial_im']:>14,.0f}")
        print(f"  {'Final IM':.<25} ${v1_result['final_im']:>14,.0f} ${v2_result['final_im']:>14,.0f}")

        im_diff = abs(v1_result['final_im'] - v2_result['final_im'])
        im_rel_diff = im_diff / max(v1_result['final_im'], 1e-10) * 100
        print(f"  {'IM Difference':.<25} ${im_diff:>14,.0f} ({im_rel_diff:.4f}%)")

        v1_reduction = (v1_result['initial_im'] - v1_result['final_im']) / v1_result['initial_im'] * 100
        v2_reduction = (v2_result['initial_im'] - v2_result['final_im']) / v2_result['initial_im'] * 100
        print(f"  {'IM Reduction (v1)':.<25} {v1_reduction:>14.2f}%")
        print(f"  {'IM Reduction (v2)':.<25} {v2_reduction:>14.2f}%")

        print(f"  {'Trades Moved':.<25} {v1_result['trades_moved']:>15} {v2_result['trades_moved']:>15}")
        print(f"  {'Iterations':.<25} {v1_result['num_iterations']:>15} {v2_result['num_iterations']:>15}")
        print(f"  {'Converged':.<25} {str(v1_result['converged']):>15} {str(v2_result['converged']):>15}")

        print("\n" + "-" * 80)
        print("Timing Breakdown:")
        print("-" * 80)
        print(f"  {'v1 Total Time':.<25} {v1_result['elapsed_time']:>14.3f}s")
        print(f"  {'v2 Total Time':.<25} {v2_result['elapsed_time']:>14.3f}s")
        time_speedup = v1_result['elapsed_time'] / max(v2_result['elapsed_time'], 1e-10)
        print(f"  {'Total Speedup':.<25} {time_speedup:>14.1f}x")

        if 'v2_metrics' in v2_result:
            m = v2_result['v2_metrics']
            print(f"\n  v2 Breakdown:")
            print(f"    CRIF time:      {m.get('crif_time', 0)*1000:.2f} ms")
            print(f"    Kernel time:    {m.get('kernel_time', 0)*1000:.2f} ms")
            print(f"    Opt time:       {m.get('opt_time', 0)*1000:.2f} ms")
            print(f"    Total eval:     {m.get('total_eval_time', 0)*1000:.2f} ms")
            print(f"    Avg eval/iter:  {m.get('avg_eval_time', 0)*1000:.2f} ms")

    print("\n" + "=" * 80)
    print("VERIFICATION:")
    if v1_result and v2_result:
        im_correct = im_rel_diff < 0.01  # < 0.01% difference
        print(f"  IM Correctness (<0.01% diff): {'PASS' if im_correct else 'FAIL'} ({im_rel_diff:.6f}%)")
    print(f"  Performance Gain (>10%):      {'PASS' if improvement_pct > 10 else 'FAIL'} ({improvement_pct:.1f}%)")
    print("=" * 80)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AADC v1 vs v2 Benchmark")
    parser.add_argument("--trades", type=int, default=100, help="Number of trades")
    parser.add_argument("--portfolios", type=int, default=5, help="Number of portfolios")
    parser.add_argument("--threads", type=int, default=4, help="Number of AADC threads")
    parser.add_argument("--max-iters", type=int, default=50, help="Max optimization iterations")
    parser.add_argument("--optimize", action="store_true", help="Run full optimization benchmark")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(f"\nAADC v1 vs v2 Benchmark")
    print(f"Trades: {args.trades}, Portfolios: {args.portfolios}, Threads: {args.threads}")

    # Setup
    market = create_market_environment()
    trades = generate_trades(args.trades)
    initial_allocation = create_initial_allocation(args.trades, args.portfolios)
    workers = aadc.ThreadPool(args.threads)

    # Precompute CRIFs and sensitivity matrix
    print("\nPrecomputing trade CRIFs...")
    crif_start = time.perf_counter()
    trade_crifs = precompute_all_trade_crifs(trades, market, args.threads, workers)
    crif_time = time.perf_counter() - crif_start
    print(f"  CRIF computation: {crif_time:.3f}s")

    trade_ids = [t.trade_id for t in trades if t.trade_id in trade_crifs]
    T = len(trade_ids)

    from model.simm_allocation_optimizer import _get_unique_risk_factors, _build_sensitivity_matrix
    risk_factors = _get_unique_risk_factors(trade_crifs)
    S = _build_sensitivity_matrix(trade_crifs, trade_ids, risk_factors)
    K = len(risk_factors)

    print(f"  Sensitivity matrix: {T} trades x {K} factors x {args.portfolios} portfolios")

    # Filter allocation
    trade_id_to_idx = {t.trade_id: i for i, t in enumerate(trades)}
    filtered_allocation = np.zeros((T, args.portfolios))
    for i, tid in enumerate(trade_ids):
        orig_idx = trade_id_to_idx.get(tid)
        if orig_idx is not None:
            filtered_allocation[i] = initial_allocation[orig_idx]

    # Record kernel
    print("\nRecording kernel...")
    factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = \
        _get_factor_metadata(risk_factors)

    kernel_start = time.perf_counter()
    funcs, sens_handles, im_output = record_single_portfolio_simm_kernel(
        K, factor_risk_classes, factor_weights,
        factor_risk_types, factor_labels, factor_buckets
    )
    kernel_time = time.perf_counter() - kernel_start
    print(f"  Kernel recording: {kernel_time*1000:.2f} ms")

    # Gradient computation benchmark
    print("\nBenchmarking gradient computation...")
    v1_time, v2_time = benchmark_gradient_computation(
        funcs, sens_handles, im_output, S, filtered_allocation,
        args.threads, workers, num_iters=20
    )

    # Full optimization benchmark (if requested)
    v1_result, v2_result = None, None
    if args.optimize:
        v1_result, v2_result = benchmark_optimization(
            trades, market, args.portfolios, initial_allocation,
            args.threads, args.max_iters, args.verbose
        )

    # Print results
    print_benchmark_results(args, v1_time, v2_time, v1_result, v2_result)


if __name__ == "__main__":
    main()
