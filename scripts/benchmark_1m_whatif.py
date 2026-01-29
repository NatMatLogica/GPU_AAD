#!/usr/bin/env python
"""
Benchmark 3: What-If / Margin Attribution at Scale

Demonstrates AADC superiority for a client with up to 1M trades:
- "Which of my trades are consuming the most margin?"
- "What happens to margin if I unwind my top 10 contributors?"

AADC approach (O(T×K)):
  1. Compute gradient dIM/dSensitivity ONCE via adjoint pass: O(K)
  2. For each trade, contribution = gradient · trade_sensitivities: O(K)
  3. Total: O(T×K) where K ~ 100 risk factors

Naive leave-one-out approach (O(T×N)):
  1. For each trade, remove it and recalculate full SIMM: O(N) per trade
  2. Total: T × O(N) where N = total sensitivities
  3. At 1M trades × ~100ms per SIMM = 100,000 seconds = 28 hours!

For a 1M-trade client, this is the difference between:
- AADC: Attribution report in seconds (actionable in real-time)
- Naive: Attribution report in hours/days (useless for decision-making)

Scaling: 100 → 500 → 1K → 5K → 10K → (extrapolate to 1M)
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.trade_types import (
    generate_market_environment,
    generate_trades_by_type,
)
from model.simm_portfolio_aadc import precompute_all_trade_crifs
from model.simm_portfolio_aadc_v2 import (
    record_single_portfolio_simm_kernel_v2,
    compute_all_portfolios_im_gradient_v2,
    _get_factor_metadata_v2,
    AADC_AVAILABLE,
)
from model.simm_allocation_optimizer import (
    _get_unique_risk_factors,
    _build_sensitivity_matrix,
)

NUM_THREADS = 8
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark_results')


def run_whatif_benchmark(num_trades: int, num_threads: int = NUM_THREADS):
    """
    Benchmark margin attribution computation.

    Single portfolio scenario (attribution doesn't need multi-portfolio).

    Measures:
    1. AADC gradient computation time
    2. AADC attribution computation time (T × dot product)
    3. Naive leave-one-out time (sampled, then extrapolated)
    4. Speedup
    """
    print(f"\n{'='*70}")
    print(f"  Margin Attribution: {num_trades:,} trades")
    print(f"{'='*70}")

    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']
    market = generate_market_environment(currencies, seed=42)

    # Generate trades
    print(f"  Generating {num_trades:,} trades...")
    gen_start = time.perf_counter()
    trades = generate_trades_by_type('ir_swap', num_trades, currencies, seed=42)
    gen_time = time.perf_counter() - gen_start

    if not AADC_AVAILABLE:
        print("  ERROR: AADC not available.")
        return None

    import aadc
    workers = aadc.ThreadPool(num_threads)

    # Precompute CRIFs
    print(f"  Precomputing CRIFs...")
    crif_start = time.perf_counter()
    trade_crifs = precompute_all_trade_crifs(trades, market, num_threads, workers)
    crif_time = time.perf_counter() - crif_start
    print(f"  CRIF precomputation: {crif_time:.2f}s")

    trade_ids = [t.trade_id for t in trades if t.trade_id in trade_crifs]
    T = len(trade_ids)

    # Build sensitivity matrix
    print(f"  Building sensitivity matrix...")
    smatrix_start = time.perf_counter()
    risk_factors = _get_unique_risk_factors(trade_crifs)
    S = _build_sensitivity_matrix(trade_crifs, trade_ids, risk_factors)
    K = len(risk_factors)
    smatrix_time = time.perf_counter() - smatrix_start
    print(f"  Sensitivity matrix: {T:,} trades × {K} risk factors ({smatrix_time:.2f}s)")

    # Record kernel
    factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = \
        _get_factor_metadata_v2(risk_factors)
    funcs, sens_handles, im_output = record_single_portfolio_simm_kernel_v2(
        K, factor_risk_classes, factor_weights,
        factor_risk_types, factor_labels, factor_buckets,
        use_correlations=True
    )

    # Single portfolio allocation (all trades in one portfolio)
    allocation = np.ones((T, 1))

    # -------------------------------------------------------------------------
    # AADC: Gradient computation + attribution
    # -------------------------------------------------------------------------
    print(f"  Computing AADC gradient...")

    # Warmup
    compute_all_portfolios_im_gradient_v2(
        funcs, sens_handles, im_output, S, allocation, num_threads, workers
    )

    # Timed gradient computation
    grad_times = []
    for _ in range(3):
        grad_start = time.perf_counter()
        gradient, all_ims, eval_time = compute_all_portfolios_im_gradient_v2(
            funcs, sens_handles, im_output, S, allocation, num_threads, workers
        )
        grad_times.append(time.perf_counter() - grad_start)

    grad_avg_time = np.mean(grad_times)
    total_im = float(np.sum(all_ims))
    print(f"  Gradient computation: {grad_avg_time*1000:.1f}ms (avg of 3)")
    print(f"  Total IM: ${total_im:,.0f}")

    # Attribution: for each trade, contribution = sum_k gradient[k] * S[trade, k]
    print(f"  Computing attribution for {T:,} trades...")
    attrib_start = time.perf_counter()

    # Vectorized: contributions = S @ gradient (for single portfolio)
    grad_vec = gradient[:, 0] if gradient.ndim == 2 else gradient
    contributions = S @ grad_vec  # (T,) vector of trade contributions

    attrib_time = time.perf_counter() - attrib_start
    print(f"  Attribution computation: {attrib_time*1000:.2f}ms")

    # Total AADC time
    aadc_total_ms = (grad_avg_time + attrib_time) * 1000

    # Validate: sum of contributions should ≈ total IM (Euler decomposition)
    total_contribution = float(np.sum(contributions))
    euler_error_pct = abs(total_contribution - total_im) / total_im * 100 if total_im > 0 else 0
    print(f"  Euler check: sum(contributions)=${total_contribution:,.0f} vs IM=${total_im:,.0f} (error: {euler_error_pct:.2f}%)")

    # Top contributors
    sorted_indices = np.argsort(-np.abs(contributions))
    print(f"\n  Top 5 margin consumers:")
    for rank, idx in enumerate(sorted_indices[:5]):
        c = contributions[idx]
        pct = c / total_im * 100 if total_im > 0 else 0
        print(f"    {rank+1}. {trade_ids[idx]}: ${c:>15,.0f} ({pct:>5.1f}%)")

    # Count additive vs reducing
    num_additive = int(np.sum(contributions > 0))
    num_reducing = int(np.sum(contributions <= 0))
    print(f"\n  Trades adding margin:    {num_additive:>6,}")
    print(f"  Trades reducing margin:  {num_reducing:>6,}")

    # -------------------------------------------------------------------------
    # Naive: Leave-one-out (sample a few trades, extrapolate)
    # -------------------------------------------------------------------------
    print(f"\n  Estimating naive leave-one-out time...")

    # Time a single SIMM evaluation
    agg_S_full = np.dot(S.T, allocation).flatten()
    inputs_full = {sens_handles[k]: np.array([float(agg_S_full[k])]) for k in range(K)}
    request = {im_output: sens_handles}

    eval_times = []
    for _ in range(10):
        ev_start = time.perf_counter()
        aadc.evaluate(funcs, request, inputs_full, workers)
        eval_times.append(time.perf_counter() - ev_start)
    single_simm_ms = np.mean(eval_times) * 1000

    # Actually run leave-one-out for a small sample to validate estimate
    sample_size = min(20, T)
    sample_indices = np.random.default_rng(42).choice(T, sample_size, replace=False)

    loo_times = []
    loo_contributions = []
    for idx in sample_indices:
        # Remove trade idx: set its allocation to 0
        modified_alloc = allocation.copy()
        modified_alloc[idx, 0] = 0.0

        agg_S_without = np.dot(S.T, modified_alloc).flatten()
        inputs_without = {sens_handles[k]: np.array([float(agg_S_without[k])]) for k in range(K)}

        loo_start = time.perf_counter()
        results = aadc.evaluate(funcs, request, inputs_without, workers)
        loo_time = time.perf_counter() - loo_start
        loo_times.append(loo_time)

        im_without = float(results[0][im_output][0])
        loo_contributions.append(total_im - im_without)

    avg_loo_time_ms = np.mean(loo_times) * 1000

    # Validate AADC attribution vs leave-one-out for sampled trades
    aadc_sample = contributions[sample_indices]
    loo_sample = np.array(loo_contributions)
    if np.max(np.abs(loo_sample)) > 0:
        max_diff = np.max(np.abs(aadc_sample - loo_sample))
        max_rel_diff = np.max(np.abs(aadc_sample - loo_sample) / (np.abs(loo_sample) + 1))
        print(f"  Validation (AADC vs leave-one-out on {sample_size} trades):")
        print(f"    Max absolute diff: ${max_diff:,.0f}")
        print(f"    Max relative diff: {max_rel_diff*100:.2f}%")

    # Extrapolate naive time for all T trades
    naive_total_ms = avg_loo_time_ms * T
    naive_total_s = naive_total_ms / 1000

    speedup = naive_total_ms / aadc_total_ms if aadc_total_ms > 0 else 0

    print(f"\n  --- Full Attribution Comparison ---")
    print(f"  AADC (gradient + dot products): {aadc_total_ms:>12.1f} ms")
    print(f"  Naive (T × leave-one-out):      {naive_total_ms:>12,.0f} ms ({naive_total_s:.1f}s)")
    print(f"  Speedup:                        {speedup:>12,.0f}×")

    # What-if scenario: unwind top 5 contributors
    print(f"\n  --- What-If: Unwind Top 5 ---")
    top5_indices = sorted_indices[:5]
    top5_saving_estimate = float(np.sum(contributions[top5_indices]))
    print(f"  Estimated IM reduction: ${top5_saving_estimate:,.0f} ({top5_saving_estimate/total_im*100:.1f}%)")

    # Actual what-if (remove top 5)
    whatif_alloc = allocation.copy()
    for idx in top5_indices:
        whatif_alloc[idx, 0] = 0.0
    agg_S_whatif = np.dot(S.T, whatif_alloc).flatten()
    inputs_whatif = {sens_handles[k]: np.array([float(agg_S_whatif[k])]) for k in range(K)}
    whatif_results = aadc.evaluate(funcs, request, inputs_whatif, workers)
    im_after_unwind = float(whatif_results[0][im_output][0])
    actual_saving = total_im - im_after_unwind
    print(f"  Actual IM reduction:    ${actual_saving:,.0f} ({actual_saving/total_im*100:.1f}%)")
    print(f"  Estimate accuracy:      {top5_saving_estimate/actual_saving*100:.1f}%" if actual_saving > 0 else "")

    result = {
        'num_trades': T,
        'num_risk_factors': K,
        'num_threads': num_threads,
        'total_im': total_im,
        'crif_precompute_s': crif_time,
        'smatrix_build_s': smatrix_time,
        'aadc_gradient_ms': grad_avg_time * 1000,
        'aadc_attribution_ms': attrib_time * 1000,
        'aadc_total_ms': aadc_total_ms,
        'naive_per_trade_ms': avg_loo_time_ms,
        'naive_total_ms': naive_total_ms,
        'speedup': speedup,
        'euler_error_pct': euler_error_pct,
        'num_additive_trades': num_additive,
        'num_reducing_trades': num_reducing,
        'top5_estimated_saving': top5_saving_estimate,
        'top5_actual_saving': actual_saving,
        'single_simm_eval_ms': single_simm_ms,
    }

    return result


def main():
    print("=" * 70)
    print("  BENCHMARK: What-If / Margin Attribution at Scale")
    print("  Target: Real-time attribution for 1M-trade portfolios")
    print("=" * 70)

    if not AADC_AVAILABLE:
        print("\nERROR: AADC not available.")
        sys.exit(1)

    # Scaling configurations
    # For attribution, the key bottleneck is T (number of trades)
    # We use smaller configs since naive leave-one-out is very slow
    configs = [
        (100,    "100 trades"),
        (500,    "500 trades"),
        (1000,   "1K trades"),
        (5000,   "5K trades"),
        (10000,  "10K trades"),
        (50000,  "50K trades"),
        (100000, "100K trades"),
    ]

    results = []

    for num_trades, desc in configs:
        print(f"\n\n{'#'*70}")
        print(f"# Configuration: {desc}")
        print(f"{'#'*70}")

        try:
            result = run_whatif_benchmark(num_trades)
            if result:
                result['description'] = desc
                results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                'description': desc,
                'num_trades': num_trades,
                'error': str(e),
            })

    # Summary
    print(f"\n\n{'='*100}")
    print("                       MARGIN ATTRIBUTION SCALING SUMMARY")
    print("=" * 100)
    print(f"\n{'Trades':>10} {'K':>6} {'AADC':>12} {'Naive':>14} {'Speedup':>10} {'Euler err':>10}")
    print("-" * 100)

    for r in results:
        if 'error' in r:
            print(f"{r['num_trades']:>10,} {'ERROR':>60}")
            continue
        naive_str = f"{r['naive_total_ms']:,.0f}ms"
        if r['naive_total_ms'] > 60000:
            naive_str = f"{r['naive_total_ms']/60000:.1f}min"
        elif r['naive_total_ms'] > 1000:
            naive_str = f"{r['naive_total_ms']/1000:.1f}s"
        print(f"{r['num_trades']:>10,} {r['num_risk_factors']:>6} "
              f"{r['aadc_total_ms']:>10.1f}ms {naive_str:>14} "
              f"{r['speedup']:>9,.0f}× {r['euler_error_pct']:>9.2f}%")

    # Extrapolation to 1M
    if len(results) >= 2 and 'error' not in results[-1]:
        last = results[-1]
        T_last = last['num_trades']

        # AADC gradient time is ~O(K) (constant), attribution is O(T*K) (linear in T)
        # Naive is O(T * single_eval) (linear in T)
        aadc_grad_ms = last['aadc_gradient_ms']
        aadc_attrib_per_trade_ms = last['aadc_attribution_ms'] / T_last
        naive_per_trade_ms = last['naive_per_trade_ms']

        T_1M = 1_000_000
        aadc_1M_ms = aadc_grad_ms + aadc_attrib_per_trade_ms * T_1M
        naive_1M_ms = naive_per_trade_ms * T_1M

        print(f"\n{'--- Extrapolation to 1M Trades ---':^100}")
        print(f"  AADC attribution:    {aadc_1M_ms/1000:>10.1f}s")
        print(f"  Naive leave-one-out: {naive_1M_ms/3600000:>10.1f}h ({naive_1M_ms/1000:,.0f}s)")
        print(f"  Estimated speedup:   {naive_1M_ms/aadc_1M_ms:>10,.0f}×")
        print(f"\n  AADC: Full attribution report in {aadc_1M_ms/1000:.0f} seconds")
        print(f"  Naive: Would take {naive_1M_ms/3600000:.0f} hours")
        print(f"  Verdict: Only AADC can provide real-time attribution at 1M trades")

    print("=" * 100)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'whatif_scaling_benchmark.json')
    with open(output_file, 'w') as f:
        json.dump({
            'benchmark': 'whatif_margin_attribution_scaling',
            'description': 'AADC gradient attribution vs naive leave-one-out',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'num_threads': NUM_THREADS,
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
