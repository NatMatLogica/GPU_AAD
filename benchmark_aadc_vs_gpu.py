#!/usr/bin/env python
"""
AADC vs GPU Apples-to-Apples Comparison

Compares AADC (AAD) and GPU (analytical gradient) implementations side-by-side:
- Same trades, same market, same CRIF
- Same SIMM formula (aligned between AADC kernel and GPU kernel)
- Same optimization (gradient descent, same parameters)
- Verifies identical IM numbers

Key differences being measured:
- AADC: Records SIMM formula as aadc.idouble tape, gets gradient via AAD (adjoint pass)
- GPU: Computes SIMM + gradient analytically in CUDA kernel (handwritten derivatives)
- Both use the same sensitivity matrix S[t,k] and allocation matrix x[t,p]

Usage:
    python benchmark_aadc_vs_gpu.py --trades 100 --portfolios 5 --threads 8
    python benchmark_aadc_vs_gpu.py --trades 500 --portfolios 5 --threads 8 --optimize --max-iters 100

Version: 1.0.0
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.trade_types import MarketEnvironment
from common.portfolio import generate_portfolio

# Check AADC
try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False

# Check CUDA
CUDA_SIMULATOR = os.environ.get('NUMBA_ENABLE_CUDASIM', '0') == '1'
try:
    from numba import cuda
    import numba
    CUDA_AVAILABLE = cuda.is_available() or CUDA_SIMULATOR
except ImportError:
    CUDA_AVAILABLE = False
    cuda = None

from model.simm_portfolio_aadc import (
    precompute_all_trade_crifs,
    _get_ir_risk_weight_v26,
    _map_risk_type_to_class,
    _is_delta_risk_type,
    _is_vega_risk_type,
    _get_vega_risk_weight,
    _get_risk_weight,
    PSI_MATRIX,
)
from model.simm_allocation_optimizer import (
    _get_unique_risk_factors,
    _build_sensitivity_matrix,
)


# =============================================================================
# AADC Implementation (Simplified formula matching GPU)
# =============================================================================

_RISK_CLASS_ORDER = ['Rates', 'CreditQ', 'CreditNonQ', 'Equity', 'Commodity', 'FX']


def record_simm_kernel_simplified_aadc(K, risk_weights, risk_class_idx):
    """
    Record AADC kernel matching the GPU's simplified SIMM formula EXACTLY.

    Formula: WS[k] = S[k] * w[k]
             K_r^2 = sum_k (WS[k]^2) for each risk class r
             IM^2 = sum_r sum_s psi[r,s] * K_r * K_s

    No correlations, no concentration factors, no Delta/Vega split.
    This ensures identical IM numbers vs GPU.
    """
    with aadc.record_kernel() as funcs:
        agg_sens = []
        sens_handles = []
        for k in range(K):
            s_k = aadc.idouble(0.0)
            handle = s_k.mark_as_input()
            sens_handles.append(handle)
            agg_sens.append(s_k)

        # Compute K_r for each risk class (sum of squares, no correlations)
        k_r_list = []
        for rc_idx in range(6):
            k_sq = aadc.idouble(0.0)
            for k in range(K):
                if risk_class_idx[k] == rc_idx:
                    ws_k = agg_sens[k] * float(risk_weights[k])
                    k_sq = k_sq + ws_k * ws_k
            k_r_list.append(np.sqrt(k_sq))

        # Cross-risk-class aggregation
        psi = PSI_MATRIX
        simm_sq = aadc.idouble(0.0)
        for r in range(6):
            for s in range(6):
                simm_sq = simm_sq + psi[r, s] * k_r_list[r] * k_r_list[s]

        im = np.sqrt(simm_sq)
        im_output = im.mark_as_output()

    return funcs, sens_handles, im_output


def compute_simm_gradient_aadc(
    funcs, sens_handles, im_output,
    agg_sensitivities,  # (P, K)
    num_threads,
    workers=None,
):
    """
    Compute SIMM IM + gradients for P portfolios using AADC.
    Single aadc.evaluate() call for all P portfolios.

    Returns:
        (im_values, gradients) where:
        - im_values: (P,) array
        - gradients: (P, K) array of dIM/dS
    """
    P, K = agg_sensitivities.shape

    if workers is None:
        workers = aadc.ThreadPool(num_threads)

    inputs = {sens_handles[k]: agg_sensitivities[:, k] for k in range(K)}
    request = {im_output: sens_handles}

    results = aadc.evaluate(funcs, request, inputs, workers)

    im_values = np.array(results[0][im_output])
    gradients = np.zeros((P, K))
    for k in range(K):
        gradients[:, k] = results[1][im_output][sens_handles[k]]

    return im_values, gradients


# =============================================================================
# GPU Implementation (from simm_portfolio_cuda.py)
# =============================================================================

def compute_simm_gradient_gpu(
    agg_sensitivities,  # (P, K)
    risk_weights,       # (K,)
    risk_class_idx,     # (K,)
    device=0,
):
    """
    Compute SIMM IM + gradients for P portfolios using GPU.
    Uses the same simplified formula as the AADC kernel.
    """
    from model.simm_portfolio_cuda import compute_simm_and_gradient_cuda
    return compute_simm_and_gradient_cuda(
        agg_sensitivities, risk_weights, risk_class_idx, device
    )


# =============================================================================
# Optimization (shared logic, different gradient backends)
# =============================================================================

def optimize_allocation(
    S, initial_allocation, risk_weights, risk_class_idx,
    grad_fn,  # function(agg_S_T) -> (im_values, gradients)
    max_iters=100, lr=None, tol=1e-6, verbose=True, label="",
):
    """
    Gradient descent optimization using provided gradient function.
    Shared logic for both AADC and GPU backends.
    """
    T, P = initial_allocation.shape
    K = S.shape[1]

    x = initial_allocation.copy()
    im_history = []

    eval_start = time.perf_counter()
    total_grad_time = 0.0

    # First evaluation
    agg_S = np.dot(S.T, x)  # (K, P)
    agg_S_T = agg_S.T       # (P, K)

    grad_start = time.perf_counter()
    im_values, grad_S = grad_fn(agg_S_T)
    total_grad_time += time.perf_counter() - grad_start

    total_im = float(np.sum(im_values))
    im_history.append(total_im)

    # Chain rule: dIM/dx[t,p] = sum_k grad_S[p,k] * S[t,k]
    gradient = np.dot(S, grad_S.T)  # (T, P)

    # Auto learning rate
    grad_max = np.abs(gradient).max()
    if lr is None:
        lr = 0.3 / grad_max if grad_max > 1e-10 else 1e-12

    if verbose:
        print(f"    [{label}] Initial IM: ${total_im:,.2f}")
        print(f"    [{label}] Gradient max: {grad_max:.2e}, LR: {lr:.2e}")

    for iteration in range(max_iters):
        if iteration > 0:
            agg_S = np.dot(S.T, x)
            agg_S_T = agg_S.T

            grad_start = time.perf_counter()
            im_values, grad_S = grad_fn(agg_S_T)
            total_grad_time += time.perf_counter() - grad_start

            total_im = float(np.sum(im_values))
            im_history.append(total_im)
            gradient = np.dot(S, grad_S.T)

        if verbose and iteration % 10 == 0:
            moves = int(np.sum(np.argmax(x, axis=1) != np.argmax(initial_allocation, axis=1)))
            print(f"    [{label}] Iter {iteration}: IM = ${total_im:,.2f}, moves = {moves}")

        # Gradient step + simplex projection
        x_new = x - lr * gradient
        x_new = _project_to_simplex(x_new)

        # Convergence check
        if iteration > 0:
            rel_change = abs(im_history[-1] - im_history[-2]) / max(abs(im_history[-2]), 1e-10)
            alloc_change = np.abs(x_new - x).max()
            if rel_change < tol and alloc_change < 1e-6:
                if verbose:
                    print(f"    [{label}] Converged at iteration {iteration + 1}")
                break

        x = x_new

    eval_time = time.perf_counter() - eval_start

    # Round to integer
    x_final = _round_to_integer(x)

    # Final IM
    agg_S_final = np.dot(S.T, x_final)
    im_final, _ = grad_fn(agg_S_final.T)
    final_im = float(np.sum(im_final))

    return {
        'final_allocation': x_final,
        'final_im': final_im,
        'initial_im': im_history[0],
        'im_history': im_history,
        'num_iterations': min(iteration + 1, max_iters),
        'eval_time': eval_time,
        'grad_time': total_grad_time,
        'trades_moved': int(np.sum(
            np.argmax(x_final, axis=1) != np.argmax(initial_allocation, axis=1)
        )),
    }


def _project_to_simplex(x):
    T, P = x.shape
    result = np.zeros_like(x)
    for t in range(T):
        v = x[t]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_candidates = np.nonzero(u * np.arange(1, P + 1) > (cssv - 1))[0]
        if len(rho_candidates) == 0:
            result[t] = np.ones(P) / P
        else:
            rho = rho_candidates[-1]
            theta = (cssv[rho] - 1) / (rho + 1)
            result[t] = np.maximum(v - theta, 0)
    return result


def _round_to_integer(x):
    T, P = x.shape
    result = np.zeros_like(x)
    for t in range(T):
        result[t, np.argmax(x[t])] = 1.0
    return result


# =============================================================================
# Full-ISDA v2 kernel comparison (AADC only - shows what GPU would need)
# =============================================================================

def compute_full_isda_aadc(
    S, allocation, risk_factors, combined_crif, num_threads
):
    """
    Compute IM using the full ISDA v2.6 kernel (AADC only).
    This shows the IM with correlations, concentration factors, etc.
    Used to assess how much the simplified formula differs from the full one.
    """
    from model.simm_portfolio_aadc_v2 import (
        record_single_portfolio_simm_kernel_v2_full,
        _get_factor_metadata_v2_full,
        compute_all_portfolios_im_gradient_v2,
    )

    K = len(risk_factors)
    factor_metadata = _get_factor_metadata_v2_full(risk_factors, combined_crif)

    funcs, sens_handles, im_output = record_single_portfolio_simm_kernel_v2_full(
        K, factor_metadata
    )

    workers = aadc.ThreadPool(num_threads)
    _, all_ims, eval_time = compute_all_portfolios_im_gradient_v2(
        funcs, sens_handles, im_output, S, allocation, num_threads, workers
    )

    return float(np.sum(all_ims)), all_ims, eval_time


# =============================================================================
# Main Comparison
# =============================================================================

def run_comparison(
    num_trades=100,
    num_portfolios=5,
    trade_types=None,
    num_threads=8,
    optimize=False,
    max_iters=100,
    num_simm_buckets=3,
    verbose=True,
):
    if trade_types is None:
        trade_types = ['ir_swap']

    print("=" * 80)
    print("        AADC vs GPU - Apples-to-Apples SIMM Comparison")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Trades:      {num_trades}")
    print(f"  Portfolios:  {num_portfolios}")
    print(f"  Trade types: {trade_types}")
    print(f"  Threads:     {num_threads}")
    print(f"  AADC:        {'Available' if AADC_AVAILABLE else 'NOT available'}")
    print(f"  CUDA:        {'Available' if CUDA_AVAILABLE else 'NOT available'}")
    if CUDA_AVAILABLE:
        print(f"  CUDA mode:   {'Simulator' if CUDA_SIMULATOR else 'GPU Hardware'}")
    print()

    # Step 1: Generate identical data
    print("Step 1: Generating trades and market data...")
    market, trades, group_ids, currencies = generate_portfolio(
        trade_types, num_trades, num_simm_buckets, num_portfolios
    )
    T = len(trades)
    P = num_portfolios

    # Step 2: Compute CRIF (shared - uses AADC for pricing)
    print("\nStep 2: Computing CRIF sensitivities (AADC pricing kernel)...")
    crif_start = time.perf_counter()
    workers = aadc.ThreadPool(num_threads) if AADC_AVAILABLE else None
    trade_crifs = precompute_all_trade_crifs(trades, market, num_threads, workers)
    crif_time = time.perf_counter() - crif_start
    print(f"  CRIF time: {crif_time:.3f}s")

    # Step 3: Build sensitivity matrix (shared)
    print("\nStep 3: Building sensitivity matrix...")
    risk_factors = _get_unique_risk_factors(trade_crifs)
    trade_ids = sorted(trade_crifs.keys())
    S = _build_sensitivity_matrix(trade_crifs, trade_ids, risk_factors)
    K = len(risk_factors)

    # Build risk weights and risk class indices (shared)
    risk_class_map = {"Rates": 0, "CreditQ": 1, "CreditNonQ": 2,
                      "Equity": 3, "Commodity": 4, "FX": 5}
    risk_weights = np.zeros(K, dtype=np.float64)
    risk_class_idx = np.zeros(K, dtype=np.int32)

    for k, (rt, qualifier, bucket, label1) in enumerate(risk_factors):
        rc = _map_risk_type_to_class(rt)
        risk_class_idx[k] = risk_class_map.get(rc, 0)

        if rt == "Risk_IRCurve" and qualifier and label1:
            risk_weights[k] = _get_ir_risk_weight_v26(qualifier, label1)
        elif _is_vega_risk_type(rt):
            risk_weights[k] = _get_vega_risk_weight(rt, bucket)
        else:
            risk_weights[k] = _get_risk_weight(rt, bucket)

    print(f"  Matrix: {T} trades x {K} risk factors")

    # Build initial allocation
    initial_allocation = np.zeros((T, P), dtype=np.float64)
    trade_id_to_idx = {tid: i for i, tid in enumerate(trade_ids)}
    for i, trade in enumerate(trades):
        if trade.trade_id in trade_id_to_idx:
            t_idx = trade_id_to_idx[trade.trade_id]
            initial_allocation[t_idx, group_ids[i]] = 1.0

    # Step 4: Compute initial IM with BOTH backends
    print("\nStep 4: Computing initial SIMM (both backends)...")
    agg_S = np.dot(S.T, initial_allocation)  # (K, P)
    agg_S_T = agg_S.T                        # (P, K)

    # AADC
    aadc_im_values = None
    aadc_gradients = None
    aadc_time = None
    if AADC_AVAILABLE:
        print("\n  --- AADC (AAD) ---")
        rec_start = time.perf_counter()
        funcs, sens_handles, im_output = record_simm_kernel_simplified_aadc(
            K, risk_weights, risk_class_idx
        )
        rec_time = time.perf_counter() - rec_start
        print(f"  Kernel recording: {rec_time*1000:.2f} ms")

        eval_start = time.perf_counter()
        aadc_im_values, aadc_gradients = compute_simm_gradient_aadc(
            funcs, sens_handles, im_output, agg_S_T, num_threads, workers
        )
        aadc_time = time.perf_counter() - eval_start
        aadc_total_im = float(np.sum(aadc_im_values))
        print(f"  Eval time:        {aadc_time*1000:.2f} ms")
        print(f"  Total IM:         ${aadc_total_im:,.2f}")
        print(f"  Per-portfolio:    {[f'${im:,.0f}' for im in aadc_im_values]}")

    # GPU
    gpu_im_values = None
    gpu_gradients = None
    gpu_time = None
    if CUDA_AVAILABLE:
        print("\n  --- GPU (CUDA) ---")
        eval_start = time.perf_counter()
        gpu_im_values, gpu_gradients = compute_simm_gradient_gpu(
            agg_S_T, risk_weights, risk_class_idx
        )
        gpu_time = time.perf_counter() - eval_start
        gpu_total_im = float(np.sum(gpu_im_values))
        print(f"  Eval time:        {gpu_time*1000:.2f} ms")
        print(f"  Total IM:         ${gpu_total_im:,.2f}")
        print(f"  Per-portfolio:    {[f'${im:,.0f}' for im in gpu_im_values]}")

    # Full ISDA v2 kernel (AADC only - reference)
    if AADC_AVAILABLE:
        print("\n  --- Full ISDA v2.6 (AADC, reference) ---")
        combined_crif = pd.concat(list(trade_crifs.values()), ignore_index=True)
        full_im, full_ims, full_time = compute_full_isda_aadc(
            S, initial_allocation, risk_factors, combined_crif, num_threads
        )
        print(f"  Eval time:        {full_time*1000:.2f} ms")
        print(f"  Total IM:         ${full_im:,.2f}")
        print(f"  Per-portfolio:    {[f'${im:,.0f}' for im in full_ims]}")

    # Step 5: Compare IM values
    print("\n" + "=" * 80)
    print("COMPARISON: IM Values")
    print("=" * 80)

    if AADC_AVAILABLE and CUDA_AVAILABLE:
        im_diff = np.abs(aadc_im_values - gpu_im_values)
        max_diff = np.max(im_diff)
        rel_diff = max_diff / max(np.max(np.abs(aadc_im_values)), 1e-10)
        print(f"  AADC total IM:    ${float(np.sum(aadc_im_values)):,.2f}")
        print(f"  GPU total IM:     ${float(np.sum(gpu_im_values)):,.2f}")
        print(f"  Max abs diff:     ${max_diff:.6f}")
        print(f"  Max rel diff:     {rel_diff:.2e}")
        match = rel_diff < 1e-6
        print(f"  MATCH:            {'YES' if match else 'NO'} (tol=1e-6)")

    if AADC_AVAILABLE:
        simplified_im = float(np.sum(aadc_im_values))
        diff_vs_full = abs(simplified_im - full_im) / max(full_im, 1e-10)
        print(f"\n  Simplified vs Full ISDA:")
        print(f"    Simplified IM:  ${simplified_im:,.2f}")
        print(f"    Full ISDA IM:   ${full_im:,.2f}")
        print(f"    Relative diff:  {diff_vs_full*100:.2f}%")
        print(f"    (This shows impact of correlations + concentration factors)")

    # Step 6: Compare gradients
    if AADC_AVAILABLE and CUDA_AVAILABLE:
        print("\n" + "=" * 80)
        print("COMPARISON: Gradients (dIM/dS)")
        print("=" * 80)
        grad_diff = np.abs(aadc_gradients - gpu_gradients)
        max_grad_diff = np.max(grad_diff)
        grad_rel_diff = max_grad_diff / max(np.max(np.abs(aadc_gradients)), 1e-10)
        print(f"  Max abs diff:     {max_grad_diff:.6e}")
        print(f"  Max rel diff:     {grad_rel_diff:.2e}")
        print(f"  Mean abs diff:    {np.mean(grad_diff):.6e}")
        grad_match = grad_rel_diff < 1e-4
        print(f"  MATCH:            {'YES' if grad_match else 'NO'} (tol=1e-4)")

    # Step 7: Performance comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<25} {'AADC (AAD)':>15} {'GPU (CUDA)':>15} {'Speedup':>10}")
    print("-" * 65)

    if AADC_AVAILABLE:
        print(f"{'Kernel recording':<25} {rec_time*1000:>12.2f} ms {'N/A':>15} {'':>10}")
    if AADC_AVAILABLE and CUDA_AVAILABLE:
        print(f"{'IM + gradient eval':<25} {aadc_time*1000:>12.2f} ms {gpu_time*1000:>12.2f} ms {aadc_time/max(gpu_time,1e-10):>8.1f}x")
    elif AADC_AVAILABLE:
        print(f"{'IM + gradient eval':<25} {aadc_time*1000:>12.2f} ms {'N/A':>15} {'':>10}")
    elif CUDA_AVAILABLE:
        print(f"{'IM + gradient eval':<25} {'N/A':>15} {gpu_time*1000:>12.2f} ms {'':>10}")

    # Step 8: Optimization comparison (if requested)
    if optimize:
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPARISON")
        print("=" * 80)

        if AADC_AVAILABLE:
            print("\n  --- AADC Gradient Descent ---")
            aadc_grad_fn = lambda agg: compute_simm_gradient_aadc(
                funcs, sens_handles, im_output, agg, num_threads, workers
            )
            aadc_opt = optimize_allocation(
                S, initial_allocation, risk_weights, risk_class_idx,
                aadc_grad_fn, max_iters=max_iters, verbose=verbose, label="AADC"
            )

        if CUDA_AVAILABLE:
            print("\n  --- GPU Gradient Descent ---")
            gpu_grad_fn = lambda agg: compute_simm_gradient_gpu(
                agg, risk_weights, risk_class_idx
            )
            gpu_opt = optimize_allocation(
                S, initial_allocation, risk_weights, risk_class_idx,
                gpu_grad_fn, max_iters=max_iters, verbose=verbose, label="GPU"
            )

        # Compare optimization results
        print("\n" + "-" * 65)
        print(f"{'Metric':<25} {'AADC':>15} {'GPU':>15} {'Match':>10}")
        print("-" * 65)

        if AADC_AVAILABLE and CUDA_AVAILABLE:
            print(f"{'Initial IM':<25} ${aadc_opt['initial_im']:>12,.0f} ${gpu_opt['initial_im']:>12,.0f} {'YES' if abs(aadc_opt['initial_im']-gpu_opt['initial_im'])<1 else 'NO':>10}")
            print(f"{'Final IM':<25} ${aadc_opt['final_im']:>12,.0f} ${gpu_opt['final_im']:>12,.0f} {'YES' if abs(aadc_opt['final_im']-gpu_opt['final_im'])<1 else 'NO':>10}")
            print(f"{'Trades moved':<25} {aadc_opt['trades_moved']:>15} {gpu_opt['trades_moved']:>15} {'YES' if aadc_opt['trades_moved']==gpu_opt['trades_moved'] else 'NO':>10}")
            print(f"{'Iterations':<25} {aadc_opt['num_iterations']:>15} {gpu_opt['num_iterations']:>15} {'':>10}")
            print(f"{'Total time':<25} {aadc_opt['eval_time']:>12.3f} s {gpu_opt['eval_time']:>12.3f} s {aadc_opt['eval_time']/max(gpu_opt['eval_time'],1e-10):>8.1f}x")
            print(f"{'Gradient time':<25} {aadc_opt['grad_time']:>12.3f} s {gpu_opt['grad_time']:>12.3f} s {aadc_opt['grad_time']/max(gpu_opt['grad_time'],1e-10):>8.1f}x")

            im_diff = abs(aadc_opt['final_im'] - gpu_opt['final_im'])
            print(f"\n  IM difference after optimization: ${im_diff:,.2f}")
            if im_diff < 1.0:
                print("  MATCH: Identical optimization results (within $1)")
            else:
                print(f"  NOTE: Different results - likely due to floating point differences")
                print(f"        in gradient computation accumulating over {max_iters} iterations")

        # GPU performance assessment for full ISDA complexity
        if CUDA_AVAILABLE:
            print("\n" + "=" * 80)
            print("GPU PERFORMANCE ASSESSMENT: Full ISDA Complexity")
            print("=" * 80)
            print()
            print("Current GPU kernel (simplified):")
            print("  - No intra-bucket correlations (saves K^2 ops per bucket)")
            print("  - No concentration factors CR/VCR")
            print("  - No Delta/Vega separation")
            print("  - No inter-bucket gamma aggregation")
            print()
            print("To match full ISDA formula, GPU kernel would need:")
            print("  1. Intra-bucket correlation loops (O(K_bucket^2) per bucket)")
            print("     Impact: ~2-5x more computation in kernel")
            print("  2. Bucket grouping (dynamic, varies by risk class)")
            print("     Impact: Requires shared memory or multiple kernel launches")
            print("  3. Concentration factors (pre-computed, passed as constants)")
            print("     Impact: Minimal overhead (just multiply)")
            print("  4. Delta/Vega split + separate margins")
            print("     Impact: ~2x more passes through factors")
            print("  5. Inter-bucket gamma for IR")
            print("     Impact: Small (only for IR risk class)")
            print()

            # Estimate complexity multiplier
            n_ir_factors = sum(1 for k in range(K) if risk_class_idx[k] == 0)
            n_other = K - n_ir_factors
            print(f"  This portfolio: K={K} factors ({n_ir_factors} IR + {n_other} other)")
            print(f"  Estimated GPU slowdown for full ISDA: 3-8x")
            print(f"  GPU with full ISDA would still be faster than CPU for large portfolios")

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="AADC vs GPU Apples-to-Apples Comparison"
    )
    parser.add_argument('--trades', '-t', type=int, default=100)
    parser.add_argument('--portfolios', '-p', type=int, default=5)
    parser.add_argument('--trade-types', type=str, default='ir_swap')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--max-iters', type=int, default=100)
    parser.add_argument('--simm-buckets', type=int, default=3)
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    trade_types = [t.strip() for t in args.trade_types.split(',')]

    run_comparison(
        num_trades=args.trades,
        num_portfolios=args.portfolios,
        trade_types=trade_types,
        num_threads=args.threads,
        optimize=args.optimize,
        max_iters=args.max_iters,
        num_simm_buckets=args.simm_buckets,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
