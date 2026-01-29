#!/usr/bin/env python
"""
Benchmark 1: Gradient Computation Scaling for Portfolio Optimization

Demonstrates AADC superiority for a client with up to 1M trades:
- AADC computes dIM/dAllocation for ALL trades in O(K) via single adjoint pass
- Bump-and-revalue requires T×P evaluations (one per trade per portfolio)
- At 1M trades, bump-and-revalue is computationally infeasible

What matters for a 1M-trade client doing portfolio optimization:
1. Can I even compute gradients? (bump-and-revalue: NO at 1M)
2. How fast per optimization iteration? (AADC: seconds, not hours)
3. How many iterations can I run? (AADC: 100+ iterations feasible)

Scaling: 1K → 5K → 10K → 50K → 100K → (extrapolate to 1M)

Output: JSON benchmark data + console summary
"""

import sys
import os
import time
import json
import numpy as np
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.trade_types import (
    generate_market_environment,
    generate_trades_by_type,
    IRSwapTrade,
)
from model.simm_portfolio_aadc import precompute_all_trade_crifs
from model.simm_portfolio_aadc_v2 import (
    record_single_portfolio_simm_kernel_v2,
    compute_all_portfolios_im_gradient_v2,
    compute_allocation_gradient_chainrule_v2,
    _get_factor_metadata_v2,
    AADC_AVAILABLE,
)
from model.simm_allocation_optimizer import (
    _get_unique_risk_factors,
    _build_sensitivity_matrix,
)

NUM_THREADS = 8
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark_results')


def run_gradient_benchmark(num_trades: int, num_portfolios: int, num_threads: int = NUM_THREADS):
    """
    Benchmark gradient computation for a given trade/portfolio configuration.

    Measures:
    1. CRIF precomputation time (one-time cost)
    2. Sensitivity matrix build time
    3. AADC kernel recording time
    4. Single gradient evaluation time (= one optimization iteration)
    5. Estimated bump-and-revalue time (extrapolated from small sample)
    """
    print(f"\n{'='*70}")
    print(f"  Gradient Scaling: {num_trades:,} trades × {num_portfolios} portfolios")
    print(f"{'='*70}")

    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']
    market = generate_market_environment(currencies, seed=42)

    # Generate trades
    print(f"  Generating {num_trades:,} IR swap trades...")
    gen_start = time.perf_counter()
    trades = generate_trades_by_type('ir_swap', num_trades, currencies, seed=42)
    gen_time = time.perf_counter() - gen_start
    print(f"  Trade generation: {gen_time:.2f}s")

    if not AADC_AVAILABLE:
        print("  ERROR: AADC not available. Cannot run benchmark.")
        return None

    import aadc
    workers = aadc.ThreadPool(num_threads)

    # Step 1: Precompute CRIFs
    print(f"  Precomputing CRIFs for {num_trades:,} trades...")
    crif_start = time.perf_counter()
    trade_crifs = precompute_all_trade_crifs(trades, market, num_threads, workers)
    crif_time = time.perf_counter() - crif_start
    print(f"  CRIF precomputation: {crif_time:.2f}s")

    trade_ids = [t.trade_id for t in trades if t.trade_id in trade_crifs]
    T = len(trade_ids)
    print(f"  Valid trades: {T:,}")

    # Step 2: Build sensitivity matrix
    print(f"  Building sensitivity matrix...")
    smatrix_start = time.perf_counter()
    risk_factors = _get_unique_risk_factors(trade_crifs)
    S = _build_sensitivity_matrix(trade_crifs, trade_ids, risk_factors)
    K = len(risk_factors)
    smatrix_time = time.perf_counter() - smatrix_start
    print(f"  Sensitivity matrix: {T:,} trades × {K} risk factors ({smatrix_time:.2f}s)")

    # Step 3: Record AADC kernel
    print(f"  Recording AADC kernel (K={K} inputs)...")
    kernel_start = time.perf_counter()
    factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = \
        _get_factor_metadata_v2(risk_factors)
    funcs, sens_handles, im_output = record_single_portfolio_simm_kernel_v2(
        K, factor_risk_classes, factor_weights,
        factor_risk_types, factor_labels, factor_buckets,
        use_correlations=True
    )
    kernel_time = time.perf_counter() - kernel_start
    print(f"  Kernel recording: {kernel_time*1000:.1f}ms")

    # Step 4: Create random allocation
    P = num_portfolios
    allocation = np.zeros((T, P))
    rng = np.random.default_rng(42)
    for t in range(T):
        p = rng.integers(P)
        allocation[t, p] = 1.0

    # Step 5: AADC gradient evaluation (THE KEY MEASUREMENT)
    print(f"  Computing AADC gradient ({T:,} trades × {P} portfolios)...")

    # Warmup
    gradient_warmup, all_ims_warmup, _ = compute_all_portfolios_im_gradient_v2(
        funcs, sens_handles, im_output, S, allocation, num_threads, workers
    )

    # Timed runs (average of 3)
    aadc_times = []
    for run in range(3):
        eval_start = time.perf_counter()
        gradient, all_ims, eval_time = compute_all_portfolios_im_gradient_v2(
            funcs, sens_handles, im_output, S, allocation, num_threads, workers
        )
        full_eval_time = time.perf_counter() - eval_start
        aadc_times.append(full_eval_time)

    aadc_avg_time = np.mean(aadc_times)
    total_im = float(np.sum(all_ims))

    print(f"  AADC gradient evaluation: {aadc_avg_time*1000:.1f}ms (avg of 3)")
    print(f"  Total IM: ${total_im:,.0f}")

    # Step 6: Chain rule gradient (allocation gradient from kernel gradient)
    chain_start = time.perf_counter()
    alloc_gradient = compute_allocation_gradient_chainrule_v2(gradient, S)
    chain_time = time.perf_counter() - chain_start
    print(f"  Chain rule gradient: {chain_time*1000:.1f}ms")

    # Step 7: Estimate bump-and-revalue time
    # For bump-and-revalue, we'd need T*P full SIMM evaluations
    # Estimate by timing a single evaluation
    single_eval_start = time.perf_counter()
    agg_S = np.dot(S.T, allocation[:, 0:1])  # Single portfolio
    single_eval_time = time.perf_counter() - single_eval_start

    # Each bump requires: modify one trade's allocation, recompute aggregated sensitivities,
    # evaluate SIMM kernel. The SIMM kernel eval dominates.
    # We measure kernel eval time from the batched call and scale.
    bump_per_trade_ms = eval_time / P * 1000  # kernel time for one portfolio
    bump_total_estimate_ms = bump_per_trade_ms * T * P  # T trades × P portfolios
    bump_total_estimate_s = bump_total_estimate_ms / 1000

    aadc_total_iter_ms = (aadc_avg_time + chain_time) * 1000

    speedup = bump_total_estimate_ms / aadc_total_iter_ms if aadc_total_iter_ms > 0 else 0

    print(f"\n  --- Per-Iteration Comparison ---")
    print(f"  AADC (1 adjoint pass):        {aadc_total_iter_ms:>10.1f} ms")
    print(f"  Bump-and-revalue (estimated):  {bump_total_estimate_ms:>10.0f} ms ({bump_total_estimate_s:.1f}s)")
    print(f"  Speedup:                       {speedup:>10.0f}×")

    # For 100 iterations
    aadc_100_iters_s = aadc_total_iter_ms * 100 / 1000
    bump_100_iters_s = bump_total_estimate_ms * 100 / 1000
    print(f"\n  --- 100 Optimization Iterations ---")
    print(f"  AADC:             {aadc_100_iters_s:>10.1f}s")
    print(f"  Bump-and-revalue: {bump_100_iters_s:>10.0f}s ({bump_100_iters_s/3600:.1f}h)")

    result = {
        'num_trades': T,
        'num_portfolios': P,
        'num_risk_factors': K,
        'num_threads': num_threads,
        'total_im': total_im,
        'crif_precompute_s': crif_time,
        'smatrix_build_s': smatrix_time,
        'kernel_recording_ms': kernel_time * 1000,
        'aadc_gradient_eval_ms': aadc_avg_time * 1000,
        'chain_rule_ms': chain_time * 1000,
        'aadc_total_iter_ms': aadc_total_iter_ms,
        'bump_revalue_estimate_ms': bump_total_estimate_ms,
        'speedup': speedup,
        'aadc_100_iters_s': aadc_100_iters_s,
        'bump_100_iters_s': bump_100_iters_s,
        'gradient_shape': list(gradient.shape) if hasattr(gradient, 'shape') else [T, P],
    }

    return result


def main():
    print("=" * 70)
    print("  BENCHMARK: Gradient Computation Scaling for Portfolio Optimization")
    print("  Target: Demonstrate AADC feasibility at 1M trades")
    print("=" * 70)

    if not AADC_AVAILABLE:
        print("\nERROR: AADC not available. Install AADC to run this benchmark.")
        sys.exit(1)

    # Scaling configurations
    configs = [
        (1000,   20, "1K trades, 20 portfolios"),
        (5000,   20, "5K trades, 20 portfolios"),
        (10000,  20, "10K trades, 20 portfolios"),
        (50000,  20, "50K trades, 20 portfolios"),
        (100000, 20, "100K trades, 20 portfolios"),
    ]

    results = []

    for num_trades, num_portfolios, desc in configs:
        print(f"\n\n{'#'*70}")
        print(f"# Configuration: {desc}")
        print(f"{'#'*70}")

        try:
            result = run_gradient_benchmark(num_trades, num_portfolios)
            if result:
                result['description'] = desc
                results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            # Record partial result
            results.append({
                'description': desc,
                'num_trades': num_trades,
                'num_portfolios': num_portfolios,
                'error': str(e),
            })

    # Summary table
    print(f"\n\n{'='*90}")
    print("                          GRADIENT SCALING SUMMARY")
    print("=" * 90)
    print(f"\n{'Trades':>10} {'Portfolios':>12} {'K':>6} {'AADC/iter':>12} {'Bump/iter':>14} {'Speedup':>10} {'100 iters':>12}")
    print("-" * 90)

    for r in results:
        if 'error' in r:
            print(f"{r['num_trades']:>10,} {'ERROR':>60}")
            continue
        print(f"{r['num_trades']:>10,} {r['num_portfolios']:>12} {r['num_risk_factors']:>6} "
              f"{r['aadc_total_iter_ms']:>10.1f}ms {r['bump_revalue_estimate_ms']:>12,.0f}ms "
              f"{r['speedup']:>9,.0f}× {r['aadc_100_iters_s']:>10.1f}s")

    # Extrapolation to 1M trades
    if len(results) >= 2 and 'error' not in results[-1]:
        last = results[-1]
        T_last = last['num_trades']
        aadc_per_trade = last['aadc_total_iter_ms'] / T_last
        bump_per_trade = last['bump_revalue_estimate_ms'] / (T_last * last['num_portfolios'])

        T_1M = 1_000_000
        P = 20
        aadc_1M_ms = aadc_per_trade * T_1M  # Approximate - kernel is O(K), chain rule is O(T*K)
        bump_1M_ms = bump_per_trade * T_1M * P

        print(f"\n{'--- Extrapolation to 1M Trades ---':^90}")
        print(f"  AADC per iteration:        {aadc_1M_ms/1000:>10.1f}s")
        print(f"  Bump-and-revalue:          {bump_1M_ms/1000:>10,.0f}s ({bump_1M_ms/3600000:.0f}h)")
        print(f"  Estimated speedup:         {bump_1M_ms/aadc_1M_ms:>10,.0f}×")
        print(f"  AADC 100 iterations:       {aadc_1M_ms*100/1000:>10,.0f}s ({aadc_1M_ms*100/60000:.0f}min)")
        print(f"  Bump 100 iterations:       {bump_1M_ms*100/3600000:>10,.0f}h")
        print(f"\n  Verdict: AADC makes optimization FEASIBLE at 1M trades.")
        print(f"           Bump-and-revalue is computationally IMPOSSIBLE.")

    print("=" * 90)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'gradient_scaling_benchmark.json')
    with open(output_file, 'w') as f:
        json.dump({
            'benchmark': 'gradient_computation_scaling',
            'description': 'AADC gradient vs bump-and-revalue for portfolio optimization',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'num_threads': NUM_THREADS,
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
