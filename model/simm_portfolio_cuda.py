#!/usr/bin/env python
"""
SIMM Portfolio Calculator - CUDA GPU Version

GPU-accelerated SIMM calculation and portfolio optimization using CUDA.
Provides the same interface as simm_portfolio_aadc.py for direct comparison.

Usage:
    python -m model.simm_portfolio_cuda --trades 1000 --portfolios 5 --threads 8 \
        --trade-types ir_swap,equity_option --optimize --method gradient_descent

    # Compare with AADC CPU version:
    python -m model.simm_portfolio_aadc --trades 1000 --portfolios 5 --threads 8 \
        --trade-types ir_swap,equity_option --optimize --method gradient_descent

Version: 3.1.0
"""

import math
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for CUDA availability
CUDA_SIMULATOR = os.environ.get('NUMBA_ENABLE_CUDASIM', '0') == '1'

try:
    from numba import cuda
    import numba
    CUDA_AVAILABLE = cuda.is_available() or CUDA_SIMULATOR
except ImportError:
    CUDA_AVAILABLE = False
    cuda = None

# Cross-risk-class correlation matrix (psi) - ISDA SIMM v2.6/v2.7
PSI_MATRIX = np.array([
    [1.00, 0.04, 0.04, 0.07, 0.37, 0.14],  # Rates
    [0.04, 1.00, 0.54, 0.70, 0.27, 0.37],  # CreditQ
    [0.04, 0.54, 1.00, 0.46, 0.24, 0.15],  # CreditNonQ
    [0.07, 0.70, 0.46, 1.00, 0.35, 0.39],  # Equity
    [0.37, 0.27, 0.24, 0.35, 1.00, 0.35],  # Commodity
    [0.14, 0.37, 0.15, 0.39, 0.35, 1.00],  # FX
])

# Import trade generators and market environment
try:
    from model.trade_types import (
        IRSwapTrade,
        EquityOptionTrade,
        FXOptionTrade,
        YieldCurve,
        VolSurface,
        MarketEnvironment,
        compute_crif_for_trades,
    )
    from common.portfolio import generate_portfolio
    from model.simm_portfolio_aadc import (
        _get_ir_risk_weight_v26,
        _get_concentration_threshold,
        _compute_concentration_risk,
        _map_risk_type_to_class,
        _is_delta_risk_type,
        _is_vega_risk_type,
        _get_vega_risk_weight,
        _get_risk_weight,
        _get_intra_correlation,
    )
    from Weights_and_Corr.v2_6 import (
        ir_gamma_diff_ccy,
        cr_gamma_diff_ccy,
        creditQ_corr_non_res,
        equity_corr_non_res,
        commodity_corr_non_res,
    )
    from common.logger import SIMMLogger, SIMMExecutionRecord
    TRADE_GENERATORS_AVAILABLE = True
except ImportError as e:
    print(f"Import warning: {e}")
    TRADE_GENERATORS_AVAILABLE = False

# Version
MODEL_VERSION = "3.2.0"  # v3.2: Add Adam optimizer


# =============================================================================
# CUDA Kernels for Optimization — Full ISDA SIMM v2.6
# =============================================================================
#
# Kernel implements:
#   1. Bucket-level aggregation: K_b = sqrt(Σ_ij ρ_ij × WS_i × WS_j)
#   2. Concentration risk: WS_k = S_k × RW_k × CR_k
#   3. Inter-bucket gamma with g_bc: K_rc² = Σ K_b² + Σ_{b≠c} γ_bc × S_b × S_c
#   4. Separate Delta + Vega margins: TotalMargin_r = Delta_r + Vega_r
#   5. Cross-risk-class: IM = sqrt(Σ_rs ψ_rs × Margin_r × Margin_s)
#   6. Analytical gradient via chain rule
#
# All bucket structure, correlations, gamma are pre-computed on CPU and
# passed as flat device arrays.
# =============================================================================

MAX_K = 200   # Max risk factors
MAX_B = 128   # Max buckets (across all RC × RM combinations)

if CUDA_AVAILABLE:
    @cuda.jit
    def _simm_gradient_kernel_full(
        sensitivities,        # (P, K)
        risk_weights,         # (K,)
        concentration,        # (K,) CR_k
        bucket_id,            # (K,) int32 -> 0..B-1
        risk_measure_idx,     # (K,) int32 -> 0=Delta, 1=Vega
        bucket_rc,            # (B,) int32 -> 0..5
        bucket_rm,            # (B,) int32 -> 0..1
        intra_corr_flat,      # (K*K,) intra-bucket correlation
        bucket_gamma_flat,    # (B*B,) inter-bucket gamma × g_bc
        psi_matrix,           # (6, 6)
        num_buckets,          # int scalar (passed as 1-element array)
        im_output,            # (P,)
        gradients,            # (P, K)
    ):
        p = cuda.grid(1)
        if p >= sensitivities.shape[0]:
            return

        K = sensitivities.shape[1]
        B = num_buckets[0]
        Kc = min(K, MAX_K)
        Bc = min(B, MAX_B)

        # Local arrays
        ws = cuda.local.array(MAX_K, dtype=numba.float64)
        K_b_sq = cuda.local.array(MAX_B, dtype=numba.float64)
        S_b = cuda.local.array(MAX_B, dtype=numba.float64)
        margin = cuda.local.array(12, dtype=numba.float64)     # 6 RC × 2 RM
        rc_margin = cuda.local.array(6, dtype=numba.float64)

        # Initialize bucket accumulators
        for b in range(Bc):
            K_b_sq[b] = 0.0
            S_b[b] = 0.0
        for i in range(12):
            margin[i] = 0.0

        # Step 1: Weighted sensitivities with concentration
        for k in range(Kc):
            ws[k] = sensitivities[p, k] * risk_weights[k] * concentration[k]

        # Step 2: Bucket sums S_b
        for k in range(Kc):
            b = bucket_id[k]
            S_b[b] += ws[k]

        # Step 3: Intra-bucket K_b^2 = Σ_{k,l} ρ_kl × WS_k × WS_l
        # corr matrix is 0 for cross-bucket pairs, so we iterate all K×K
        for k in range(Kc):
            b = bucket_id[k]
            for l in range(Kc):
                K_b_sq[b] += intra_corr_flat[k * K + l] * ws[k] * ws[l]

        # Step 4: Per (RC, RM) margin with inter-bucket gamma
        for b in range(Bc):
            rc_b = bucket_rc[b]
            rm_b = bucket_rm[b]
            rcrm = rc_b * 2 + rm_b
            margin[rcrm] += K_b_sq[b]

            for c in range(Bc):
                if c != b and bucket_rc[c] == rc_b and bucket_rm[c] == rm_b:
                    margin[rcrm] += bucket_gamma_flat[b * B + c] * S_b[b] * S_b[c]

        # sqrt each margin
        for i in range(12):
            margin[i] = math.sqrt(margin[i]) if margin[i] > 0.0 else 0.0

        # Step 5: Per risk class total = Delta + Vega
        for r in range(6):
            rc_margin[r] = margin[r * 2] + margin[r * 2 + 1]

        # Step 6: Cross-RC aggregation: IM = sqrt(Σ ψ_rs × M_r × M_s)
        im_sq = 0.0
        for r in range(6):
            for s in range(6):
                im_sq += psi_matrix[r, s] * rc_margin[r] * rc_margin[s]

        im_p = math.sqrt(im_sq) if im_sq > 0.0 else 0.0
        im_output[p] = im_p

        # =====================================================================
        # Gradient: dIM/dS_k via chain rule
        # =====================================================================
        if im_p < 1e-30:
            for k in range(K):
                gradients[p, k] = 0.0
            return

        # dIM/d(rc_margin_r) = Σ_s ψ_rs × rc_margin_s / IM
        dim_drcm = cuda.local.array(6, dtype=numba.float64)
        for r in range(6):
            dim_drcm[r] = 0.0
            for s in range(6):
                dim_drcm[r] += psi_matrix[r, s] * rc_margin[s]
            dim_drcm[r] /= im_p

        # Per factor gradient
        for k in range(Kc):
            b_k = bucket_id[k]
            rc_k = bucket_rc[b_k]
            rm_k = bucket_rm[b_k]
            rcrm_k = rc_k * 2 + rm_k
            margin_rcrm = margin[rcrm_k]

            if margin_rcrm < 1e-30:
                gradients[p, k] = 0.0
                continue

            # d(margin_rcrm)/dWS_k:
            #   from K_b^2: 2 × Σ_l ρ_kl × WS_l  (half because symmetric double-counted)
            #   Actually K_b^2 = Σ_k Σ_l ρ_kl WS_k WS_l, so dK_b^2/dWS_k = 2 Σ_l ρ_kl WS_l
            #   from inter-bucket S_b terms: 2 × Σ_{c≠b} γ_bc × S_c
            #   d(margin)/dWS_k = [intra_deriv + inter_deriv] / (2 × margin)
            intra_deriv = 0.0
            for l in range(Kc):
                intra_deriv += intra_corr_flat[k * K + l] * ws[l]
            intra_deriv *= 2.0

            inter_deriv = 0.0
            for c in range(Bc):
                if c != b_k and bucket_rc[c] == rc_k and bucket_rm[c] == rm_k:
                    inter_deriv += bucket_gamma_flat[b_k * B + c] * S_b[c]
            inter_deriv *= 2.0

            dmargin_dws = (intra_deriv + inter_deriv) / (2.0 * margin_rcrm)

            # d(rc_margin)/d(margin_rcrm) = 1 (linear sum)
            # dIM/dS_k = dIM/d(rc_margin) × dmargin/dWS_k × RW_k × CR_k
            gradients[p, k] = dim_drcm[rc_k] * dmargin_dws * risk_weights[k] * concentration[k]

        # Zero out any factors beyond Kc
        for k in range(Kc, K):
            gradients[p, k] = 0.0


# =============================================================================
# CUDA SIMM Calculator with Gradients — Full ISDA SIMM v2.6
# =============================================================================

def compute_simm_and_gradient_cuda(
    sensitivities: np.ndarray,
    risk_weights: np.ndarray,
    concentration: np.ndarray,
    bucket_id: np.ndarray,
    risk_measure_idx: np.ndarray,
    bucket_rc: np.ndarray,
    bucket_rm: np.ndarray,
    intra_corr_flat: np.ndarray,
    bucket_gamma_flat: np.ndarray,
    num_buckets: int,
    device: int = 0,
    gpu_arrays: dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute full ISDA SIMM v2.6 margin and gradients using CUDA.

    Args:
        sensitivities: (P, K) aggregated sensitivities per portfolio
        risk_weights: (K,) risk weights
        concentration: (K,) CR_k concentration factors
        bucket_id: (K,) int32 bucket index per factor
        risk_measure_idx: (K,) int32 0=Delta, 1=Vega
        bucket_rc: (B,) int32 risk class per bucket
        bucket_rm: (B,) int32 risk measure per bucket
        intra_corr_flat: (K*K,) flattened intra-bucket correlation
        bucket_gamma_flat: (B*B,) flattened inter-bucket gamma × g_bc
        num_buckets: total bucket count B
        device: GPU device ID
        gpu_arrays: optional pre-allocated device arrays (for optimizer reuse)

    Returns:
        (im_values, gradients) where im_values is (P,) and gradients is (P, K)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")

    P, K = sensitivities.shape

    sensitivities = np.ascontiguousarray(sensitivities, dtype=np.float64)

    # Allocate outputs
    im_output = np.zeros(P, dtype=np.float64)
    gradients_out = np.zeros((P, K), dtype=np.float64)

    if not CUDA_SIMULATOR:
        cuda.select_device(device)

    # Use pre-allocated GPU arrays if available (avoids repeated H2D transfers)
    if gpu_arrays is not None:
        d_weights = gpu_arrays['weights']
        d_conc = gpu_arrays['concentration']
        d_bucket_id = gpu_arrays['bucket_id']
        d_rm_idx = gpu_arrays['risk_measure_idx']
        d_bucket_rc = gpu_arrays['bucket_rc']
        d_bucket_rm = gpu_arrays['bucket_rm']
        d_intra_corr = gpu_arrays['intra_corr']
        d_bucket_gamma = gpu_arrays['bucket_gamma']
        d_psi = gpu_arrays['psi']
        d_num_buckets = gpu_arrays['num_buckets']
    else:
        d_weights = cuda.to_device(np.ascontiguousarray(risk_weights, dtype=np.float64))
        d_conc = cuda.to_device(np.ascontiguousarray(concentration, dtype=np.float64))
        d_bucket_id = cuda.to_device(np.ascontiguousarray(bucket_id, dtype=np.int32))
        d_rm_idx = cuda.to_device(np.ascontiguousarray(risk_measure_idx, dtype=np.int32))
        d_bucket_rc = cuda.to_device(np.ascontiguousarray(bucket_rc, dtype=np.int32))
        d_bucket_rm = cuda.to_device(np.ascontiguousarray(bucket_rm, dtype=np.int32))
        d_intra_corr = cuda.to_device(np.ascontiguousarray(intra_corr_flat, dtype=np.float64))
        d_bucket_gamma = cuda.to_device(np.ascontiguousarray(bucket_gamma_flat, dtype=np.float64))
        d_psi = cuda.to_device(PSI_MATRIX)
        d_num_buckets = cuda.to_device(np.array([num_buckets], dtype=np.int32))

    d_sens = cuda.to_device(sensitivities)
    d_im = cuda.to_device(im_output)
    d_grad = cuda.to_device(gradients_out)

    threads_per_block = 256
    blocks = (P + threads_per_block - 1) // threads_per_block

    _simm_gradient_kernel_full[blocks, threads_per_block](
        d_sens, d_weights, d_conc, d_bucket_id, d_rm_idx,
        d_bucket_rc, d_bucket_rm, d_intra_corr, d_bucket_gamma,
        d_psi, d_num_buckets, d_im, d_grad
    )

    if not CUDA_SIMULATOR:
        cuda.synchronize()

    d_im.copy_to_host(im_output)
    d_grad.copy_to_host(gradients_out)

    return im_output, gradients_out


def preallocate_gpu_arrays(
    risk_weights, concentration, bucket_id, risk_measure_idx,
    bucket_rc, bucket_rm, intra_corr_flat, bucket_gamma_flat,
    num_buckets, device=0,
):
    """Pre-allocate constant GPU arrays for reuse during optimization loop."""
    if not CUDA_SIMULATOR:
        cuda.select_device(device)
    return {
        'weights': cuda.to_device(np.ascontiguousarray(risk_weights, dtype=np.float64)),
        'concentration': cuda.to_device(np.ascontiguousarray(concentration, dtype=np.float64)),
        'bucket_id': cuda.to_device(np.ascontiguousarray(bucket_id, dtype=np.int32)),
        'risk_measure_idx': cuda.to_device(np.ascontiguousarray(risk_measure_idx, dtype=np.int32)),
        'bucket_rc': cuda.to_device(np.ascontiguousarray(bucket_rc, dtype=np.int32)),
        'bucket_rm': cuda.to_device(np.ascontiguousarray(bucket_rm, dtype=np.int32)),
        'intra_corr': cuda.to_device(np.ascontiguousarray(intra_corr_flat, dtype=np.float64)),
        'bucket_gamma': cuda.to_device(np.ascontiguousarray(bucket_gamma_flat, dtype=np.float64)),
        'psi': cuda.to_device(PSI_MATRIX),
        'num_buckets': cuda.to_device(np.array([num_buckets], dtype=np.int32)),
    }


# =============================================================================
# Portfolio Optimization using CUDA
# =============================================================================

def optimize_allocation_cuda(
    S: np.ndarray,
    initial_allocation: np.ndarray,
    risk_weights: np.ndarray,
    concentration: np.ndarray,
    bucket_id: np.ndarray,
    risk_measure_idx: np.ndarray,
    bucket_rc: np.ndarray,
    bucket_rm: np.ndarray,
    intra_corr_flat: np.ndarray,
    bucket_gamma_flat: np.ndarray,
    num_buckets: int,
    max_iters: int = 100,
    lr: float = None,
    tol: float = 1e-6,
    verbose: bool = True,
    device: int = 0,
    method: str = 'gradient_descent',
) -> Tuple[np.ndarray, List[float], int, float]:
    """Optimize trade allocation using CUDA-accelerated optimizer.

    Args:
        method: 'gradient_descent' or 'adam'
    """
    T, P = initial_allocation.shape
    K = S.shape[1]

    # Pre-allocate constant GPU arrays (avoids repeated H2D during loop)
    gpu = preallocate_gpu_arrays(
        risk_weights, concentration, bucket_id, risk_measure_idx,
        bucket_rc, bucket_rm, intra_corr_flat, bucket_gamma_flat,
        num_buckets, device,
    )

    def _eval(agg_S_T):
        return compute_simm_and_gradient_cuda(
            agg_S_T, risk_weights, concentration, bucket_id,
            risk_measure_idx, bucket_rc, bucket_rm, intra_corr_flat,
            bucket_gamma_flat, num_buckets, device, gpu_arrays=gpu,
        )

    eval_start = time.perf_counter()

    # Dispatch to optimizer
    if method == 'adam':
        best_x, im_history, num_iters = _optimize_adam(
            S, initial_allocation, _eval, max_iters, lr, tol, verbose,
        )
    else:
        best_x, im_history, num_iters = _optimize_gradient_descent(
            S, initial_allocation, _eval, max_iters, lr, tol, verbose,
        )

    # Round continuous allocation to integer for greedy search
    rounded_x = _round_to_integer(best_x)

    agg_S_r = (S.T @ rounded_x).T
    im_values_r, _ = _eval(agg_S_r)
    rounded_im = float(np.sum(im_values_r))

    if verbose:
        print(f"    Rounded IM: ${rounded_im:,.2f}")
        print(f"    Running greedy local search...")

    greedy_x, greedy_im, greedy_moves = _greedy_local_search_cuda(
        S, rounded_x, risk_weights, concentration, bucket_id,
        risk_measure_idx, bucket_rc, bucket_rm, intra_corr_flat,
        bucket_gamma_flat, num_buckets, device, gpu,
        max_rounds=max_iters, verbose=verbose,
    )

    # Always use greedy result — it's already integer and is the best integer solution
    best_x = greedy_x

    eval_time = time.perf_counter() - eval_start

    return best_x, im_history, num_iters, eval_time


def _optimize_gradient_descent(S, initial_allocation, _eval, max_iters, lr, tol, verbose):
    """Gradient descent with backtracking line search."""
    T, P = initial_allocation.shape
    x = initial_allocation.copy()
    best_x = x.copy()
    im_history = []

    LS_BETA = 0.5
    LS_MAX_TRIES = 10

    agg_S_T = (S.T @ x).T
    im_values, grad_S = _eval(agg_S_T)
    total_im = float(np.sum(im_values))
    im_history.append(total_im)
    best_im = total_im

    gradient = np.dot(S, grad_S.T)

    grad_max = np.abs(gradient).max()
    if lr is None:
        lr = 1.0 / grad_max if grad_max > 1e-10 else 1e-12

    if verbose:
        print(f"    [GD] Initial IM: ${total_im:,.2f}")
        print(f"    [GD] Gradient: max={grad_max:.2e}")
        print(f"    [GD] Learning rate: {lr:.2e}")

    stalled_count = 0

    for iteration in range(max_iters):
        if iteration > 0:
            agg_S_T = (S.T @ x).T
            im_values, grad_S = _eval(agg_S_T)
            total_im = float(np.sum(im_values))
            im_history.append(total_im)
            gradient = np.dot(S, grad_S.T)

        if total_im < best_im:
            best_im = total_im
            best_x = x.copy()
            stalled_count = 0
        else:
            stalled_count += 1

        if verbose and iteration % 10 == 0:
            moves = int(np.sum(np.argmax(x, axis=1) != np.argmax(initial_allocation, axis=1)))
            print(f"    [GD] Iter {iteration}: IM = ${total_im:,.2f}, best = ${best_im:,.2f}, moves = {moves}")

        if stalled_count >= 20:
            if verbose:
                print(f"    [GD] Stalled for {stalled_count} iterations, reverting to best")
            x = best_x.copy()
            break

        step_size = lr
        for _ in range(LS_MAX_TRIES):
            x_candidate = _project_to_simplex(x - step_size * gradient)
            agg_S_c = (S.T @ x_candidate).T
            im_values_c, _ = _eval(agg_S_c)
            candidate_im = float(np.sum(im_values_c))

            if candidate_im < total_im:
                x = x_candidate
                break
            step_size *= LS_BETA

        if iteration > 0 and len(im_history) >= 2:
            rel_change = abs(im_history[-1] - im_history[-2]) / max(abs(im_history[-2]), 1e-10)
            if rel_change < tol:
                if verbose:
                    print(f"    [GD] Converged at iteration {iteration + 1}")
                break

    if verbose and stalled_count < 20 and iteration == max_iters - 1:
        print(f"    [GD] Reached max iterations ({max_iters})")

    return best_x, im_history, iteration + 1


def _optimize_adam(S, initial_allocation, _eval, max_iters, lr, tol, verbose,
                   beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer with simplex projection and backtracking line search."""
    T, P = initial_allocation.shape
    x = initial_allocation.copy()
    best_x = x.copy()
    im_history = []

    # Adam moment estimates
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    LS_BETA = 0.5
    LS_MAX_TRIES = 10

    agg_S_T = (S.T @ x).T
    im_values, grad_S = _eval(agg_S_T)
    total_im = float(np.sum(im_values))
    im_history.append(total_im)
    best_im = total_im

    gradient = np.dot(S, grad_S.T)

    grad_max = np.abs(gradient).max()
    if lr is None:
        lr = 1.0 / grad_max if grad_max > 1e-10 else 1e-12

    if verbose:
        print(f"    [Adam] Initial IM: ${total_im:,.2f}")
        print(f"    [Adam] Gradient: max={grad_max:.2e}")
        print(f"    [Adam] Learning rate: {lr:.2e}")

    stalled_count = 0

    for iteration in range(max_iters):
        if iteration > 0:
            agg_S_T = (S.T @ x).T
            im_values, grad_S = _eval(agg_S_T)
            total_im = float(np.sum(im_values))
            im_history.append(total_im)
            gradient = np.dot(S, grad_S.T)

        if total_im < best_im:
            best_im = total_im
            best_x = x.copy()
            stalled_count = 0
        else:
            stalled_count += 1

        if verbose and iteration % 10 == 0:
            moves = int(np.sum(np.argmax(x, axis=1) != np.argmax(initial_allocation, axis=1)))
            print(f"    [Adam] Iter {iteration}: IM = ${total_im:,.2f}, best = ${best_im:,.2f}, moves = {moves}")

        if stalled_count >= 20:
            if verbose:
                print(f"    [Adam] Stalled for {stalled_count} iterations, reverting to best")
            x = best_x.copy()
            break

        # Adam moment updates (1-indexed for bias correction)
        t_step = iteration + 1
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        # Bias-corrected estimates
        m_hat = m / (1 - beta1 ** t_step)
        v_hat = v / (1 - beta2 ** t_step)

        # Adam direction
        adam_step = m_hat / (np.sqrt(v_hat) + eps)

        # Backtracking line search
        step_size = lr
        for _ in range(LS_MAX_TRIES):
            x_candidate = _project_to_simplex(x - step_size * adam_step)
            agg_S_c = (S.T @ x_candidate).T
            im_values_c, _ = _eval(agg_S_c)
            candidate_im = float(np.sum(im_values_c))

            if candidate_im < total_im:
                x = x_candidate
                break
            step_size *= LS_BETA

        if iteration > 0 and len(im_history) >= 2:
            rel_change = abs(im_history[-1] - im_history[-2]) / max(abs(im_history[-2]), 1e-10)
            if rel_change < tol:
                if verbose:
                    print(f"    [Adam] Converged at iteration {iteration + 1}")
                break

    if verbose and stalled_count < 20 and iteration == max_iters - 1:
        print(f"    [Adam] Reached max iterations ({max_iters})")

    return best_x, im_history, iteration + 1


def _project_to_simplex(x: np.ndarray) -> np.ndarray:
    """Project each row to probability simplex (sum=1, all>=0)."""
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


def _greedy_local_search_cuda(
    S, integer_allocation, risk_weights, concentration, bucket_id,
    risk_measure_idx, bucket_rc, bucket_rm, intra_corr_flat,
    bucket_gamma_flat, num_buckets, device, gpu_arrays,
    max_rounds=50, verbose=True,
):
    """Gradient-guided greedy local search for CUDA optimizer."""
    T, P = integer_allocation.shape
    x = integer_allocation.copy()
    total_moves = 0

    def _eval(agg_S_T):
        return compute_simm_and_gradient_cuda(
            agg_S_T, risk_weights, concentration, bucket_id,
            risk_measure_idx, bucket_rc, bucket_rm, intra_corr_flat,
            bucket_gamma_flat, num_buckets, device, gpu_arrays=gpu_arrays,
        )

    agg_S_T = (S.T @ x).T
    im_values, grad_S = _eval(agg_S_T)
    current_im = float(np.sum(im_values))
    gradient = np.dot(S, grad_S.T)

    for round_idx in range(max_rounds):
        current_assignments = np.argmax(x, axis=1)
        current_grads = gradient[np.arange(T), current_assignments]
        best_targets = np.argmin(gradient, axis=1)

        mask = best_targets != current_assignments
        if not np.any(mask):
            break

        candidate_indices = np.where(mask)[0]
        expected_improvement = current_grads - np.min(gradient, axis=1)
        sorted_candidates = candidate_indices[
            np.argsort(expected_improvement[candidate_indices])[::-1]
        ]

        accepted = False
        max_tries = min(len(sorted_candidates), T // 5 + 5)

        for try_idx in range(max_tries):
            t = sorted_candidates[try_idx]
            from_p = current_assignments[t]
            to_p = best_targets[t]

            x[t, :] = 0.0
            x[t, to_p] = 1.0

            agg_S_T_c = (S.T @ x).T
            im_values_c, _ = _eval(agg_S_T_c)
            candidate_im = float(np.sum(im_values_c))

            if candidate_im < current_im:
                improvement = current_im - candidate_im
                current_im = candidate_im
                total_moves += 1
                accepted = True

                if verbose:
                    print(f"    Greedy round {round_idx+1}: move trade {t} "
                          f"(p{from_p}->p{to_p}), IM -${improvement:,.0f}")

                agg_S_T = (S.T @ x).T
                im_values, grad_S = _eval(agg_S_T)
                gradient = np.dot(S, grad_S.T)
                break
            else:
                x[t, :] = 0.0
                x[t, from_p] = 1.0

        if not accepted:
            break

    if verbose and total_moves > 0:
        print(f"    Greedy search: {total_moves} moves, final IM ${current_im:,.2f}")

    return x, current_im, total_moves


def _round_to_integer(x: np.ndarray) -> np.ndarray:
    """Round continuous allocation to integer (each trade to one portfolio)."""
    T, P = x.shape
    result = np.zeros_like(x)
    for t in range(T):
        best_p = np.argmax(x[t])
        result[t, best_p] = 1.0
    return result


# =============================================================================
# Main Pipeline
# =============================================================================

def run_portfolio_cuda(
    num_trades: int = 1000,
    num_portfolios: int = 5,
    trade_types: List[str] = None,
    num_threads: int = 8,  # Not used for GPU, kept for interface compatibility
    optimize: bool = False,
    method: str = 'gradient_descent',
    max_iters: int = 100,
    verbose: bool = True,
    device: int = 0,
    num_simm_buckets: int = 3,
) -> Dict:
    """
    Run SIMM portfolio calculation and optimization using CUDA.

    Args:
        num_trades: Number of trades to generate
        num_portfolios: Number of portfolios
        trade_types: List of trade types to generate
        num_threads: (Unused, for API compatibility)
        optimize: Whether to run allocation optimization
        method: Optimization method ('gradient_descent')
        max_iters: Max optimization iterations
        verbose: Print progress
        device: GPU device ID

    Returns:
        Dict with results and timing
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available. Set NUMBA_ENABLE_CUDASIM=1 for testing.")

    if not TRADE_GENERATORS_AVAILABLE:
        raise RuntimeError("Trade generators not available. Check simm_portfolio_aadc.py")

    if trade_types is None:
        trade_types = ['ir_swap', 'equity_option']

    results = {
        'num_trades': num_trades,
        'num_portfolios': num_portfolios,
        'trade_types': trade_types,
        'device': 'GPU (CUDA Simulator)' if CUDA_SIMULATOR else f'GPU {device}',
        'timings': {},
    }

    total_start = time.perf_counter()

    # Step 1: Generate trades
    if verbose:
        print(f"\n{'='*70}")
        print(f"SIMM Portfolio Calculator - CUDA GPU Version")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  Trades:      {num_trades}")
        print(f"  Portfolios:  {num_portfolios}")
        print(f"  Trade types: {trade_types}")
        print(f"  Device:      {results['device']}")
        print()

    if verbose:
        print("Step 1: Generating trades and market data...")
    gen_start = time.perf_counter()

    # Use common portfolio generator (returns market, trades, group_ids, currencies)
    market, trades, group_ids, currencies = generate_portfolio(
        trade_types, num_trades, num_simm_buckets, num_portfolios
    )

    results['timings']['trade_generation'] = time.perf_counter() - gen_start
    results['actual_trades'] = len(trades)

    if verbose:
        print(f"  Generated {len(trades)} trades in {results['timings']['trade_generation']:.3f}s")

    # Step 2: Compute CRIF (analytical bump-and-revalue, no AADC)
    if verbose:
        print("\nStep 2: Computing CRIF sensitivities (analytical)...")
    crif_start = time.perf_counter()

    crif_df = compute_crif_for_trades(trades, market)
    crif_records = crif_df.to_dict('records') if len(crif_df) > 0 else []

    results['timings']['crif_computation'] = time.perf_counter() - crif_start
    results['num_sensitivities'] = len(crif_records)

    if verbose:
        print(f"  Computed {len(crif_records)} sensitivities in {results['timings']['crif_computation']:.3f}s")

    # Step 3: Build sensitivity matrix and full SIMM v2.6 structure
    if verbose:
        print("\nStep 3: Building sensitivity matrix and SIMM structure...")

    # Extract unique risk factors
    risk_factors = []
    for rec in crif_records:
        rf = (rec['RiskType'], rec['Qualifier'], rec.get('Bucket', ''), rec.get('Label1', ''))
        if rf not in risk_factors:
            risk_factors.append(rf)

    T = len(trades)
    K = len(risk_factors)
    P = num_portfolios

    # Build sensitivity matrix S[t, k]
    S = np.zeros((T, K), dtype=np.float64)
    trade_id_to_idx = {t.trade_id: i for i, t in enumerate(trades)}
    rf_to_idx = {rf: i for i, rf in enumerate(risk_factors)}

    for rec in crif_records:
        t_idx = trade_id_to_idx.get(rec['TradeID'])
        rf = (rec['RiskType'], rec['Qualifier'], rec.get('Bucket', ''), rec.get('Label1', ''))
        k_idx = rf_to_idx.get(rf)
        if t_idx is not None and k_idx is not None:
            S[t_idx, k_idx] += rec['AmountUSD']

    # --- Per-factor metadata ---
    risk_class_map = {
        "Rates": 0, "CreditQ": 1, "CreditNonQ": 2,
        "Equity": 3, "Commodity": 4, "FX": 5,
    }
    risk_weights = np.zeros(K, dtype=np.float64)
    risk_class_idx = np.zeros(K, dtype=np.int32)
    risk_measure_idx = np.zeros(K, dtype=np.int32)

    # Per-factor: rc name, bucket_key, bucket number for lookups
    factor_rc_name = []
    factor_bucket_key = []
    factor_bucket_num = []

    for k, rf in enumerate(risk_factors):
        rt, qualifier, bucket, label1 = rf
        rc = _map_risk_type_to_class(rt)
        rc_idx = risk_class_map.get(rc, 0)
        risk_class_idx[k] = rc_idx

        is_vega = _is_vega_risk_type(rt)
        risk_measure_idx[k] = 1 if is_vega else 0

        if rt == "Risk_IRCurve" and qualifier and label1:
            risk_weights[k] = _get_ir_risk_weight_v26(qualifier, label1)
        elif is_vega:
            risk_weights[k] = _get_vega_risk_weight(rt, bucket)
        else:
            risk_weights[k] = _get_risk_weight(rt, bucket)

        # Bucket key for grouping (currency for Rates/FX, bucket number for others)
        if rc in ("Rates", "FX"):
            bkey = qualifier if qualifier else ""
        else:
            bkey = str(bucket) if bucket else "0"

        factor_rc_name.append(rc)
        factor_bucket_key.append(bkey)
        try:
            factor_bucket_num.append(int(bucket) if bucket else 0)
        except (ValueError, TypeError):
            factor_bucket_num.append(0)

    # --- Assign unique bucket IDs per (rc, rm, bucket_key) ---
    bucket_map = {}  # (rc_idx, rm, bucket_key) -> bucket_id
    for k in range(K):
        key = (risk_class_idx[k], risk_measure_idx[k], factor_bucket_key[k])
        if key not in bucket_map:
            bucket_map[key] = len(bucket_map)

    B = len(bucket_map)
    bucket_id = np.zeros(K, dtype=np.int32)
    bucket_rc = np.zeros(B, dtype=np.int32)
    bucket_rm = np.zeros(B, dtype=np.int32)
    bucket_num = np.zeros(B, dtype=np.int32)  # Original bucket number for inter-bucket lookup

    for k in range(K):
        key = (risk_class_idx[k], risk_measure_idx[k], factor_bucket_key[k])
        bid = bucket_map[key]
        bucket_id[k] = bid
        bucket_rc[bid] = risk_class_idx[k]
        bucket_rm[bid] = risk_measure_idx[k]
        bucket_num[bid] = factor_bucket_num[k]

    # --- Concentration factors from CRIF ---
    from model.simm_portfolio_aadc import _precompute_concentration_factors
    delta_cr = _precompute_concentration_factors(crif_df, "Delta")
    vega_cr = _precompute_concentration_factors(crif_df, "Vega")
    concentration = np.ones(K, dtype=np.float64)
    for k in range(K):
        rt, qualifier, bucket, label1 = risk_factors[k]
        rc = factor_rc_name[k]
        bkey = factor_bucket_key[k]
        cr_lookup_key = (rc, bkey)
        if risk_measure_idx[k] == 0:  # Delta
            concentration[k] = delta_cr.get(cr_lookup_key, 1.0)
        else:  # Vega
            concentration[k] = vega_cr.get(cr_lookup_key, 1.0)

    # --- Intra-bucket correlation matrix (K × K, 0 for cross-bucket) ---
    intra_corr = np.zeros(K * K, dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if bucket_id[i] != bucket_id[j]:
                continue  # 0 for cross-bucket
            if i == j:
                intra_corr[i * K + j] = 1.0
            else:
                rho = _get_intra_correlation(
                    factor_rc_name[i],
                    risk_factors[i][0], risk_factors[j][0],  # risk_type1, risk_type2
                    risk_factors[i][3], risk_factors[j][3],  # label1_1, label1_2
                    factor_bucket_key[i],
                )
                intra_corr[i * K + j] = rho

    # --- Inter-bucket gamma matrix (B × B) with g_bc correction ---
    # Convert v2_6.py zip-tuple format to numpy arrays
    _eq_inter = np.array([list(row) for row in equity_corr_non_res])      # (12, 12)
    _cm_inter = np.array([list(row) for row in commodity_corr_non_res])    # (17, 17)
    _cq_inter = np.array([list(row) for row in creditQ_corr_non_res])      # (12, 12)

    # Per-bucket concentration (representative CR for g_bc computation)
    bucket_cr_rep = np.ones(B, dtype=np.float64)
    for k in range(K):
        bid = bucket_id[k]
        bucket_cr_rep[bid] = concentration[k]  # All factors in same bucket have same CR

    bucket_gamma = np.zeros(B * B, dtype=np.float64)
    for bi in range(B):
        for bj in range(B):
            if bi == bj:
                continue
            if bucket_rc[bi] != bucket_rc[bj] or bucket_rm[bi] != bucket_rm[bj]:
                continue  # Different RC or RM — no gamma
            rc = bucket_rc[bi]
            b1 = bucket_num[bi]
            b2 = bucket_num[bj]

            # Look up gamma
            gamma = 0.0
            if rc == 0:  # Rates
                gamma = ir_gamma_diff_ccy
            elif rc == 1:  # CreditQ
                if 1 <= b1 <= 12 and 1 <= b2 <= 12:
                    gamma = _cq_inter[b1 - 1, b2 - 1]
                else:
                    gamma = 0.5  # Residual
            elif rc == 2:  # CreditNonQ
                gamma = cr_gamma_diff_ccy
            elif rc == 3:  # Equity
                if 1 <= b1 <= 12 and 1 <= b2 <= 12:
                    gamma = _eq_inter[b1 - 1, b2 - 1]
                # Residual = 0
            elif rc == 4:  # Commodity
                if 1 <= b1 <= 17 and 1 <= b2 <= 17:
                    gamma = _cm_inter[b1 - 1, b2 - 1]
                # Residual = 0
            # FX: single bucket per currency, no inter-bucket

            if gamma != 0.0:
                # g_bc concentration adjustment
                cr_b = bucket_cr_rep[bi]
                cr_c = bucket_cr_rep[bj]
                g_bc = min(cr_b, cr_c) / max(cr_b, cr_c) if max(cr_b, cr_c) > 0 else 1.0
                bucket_gamma[bi * B + bj] = gamma * g_bc

    results['num_risk_factors'] = K

    if verbose:
        print(f"  Matrix: {T} trades × {K} risk factors")
        print(f"  Buckets: {B} (across all RC × RM)")
        non_trivial_corr = np.sum(intra_corr != 0) - K  # Exclude diagonal
        print(f"  Intra-bucket correlations: {non_trivial_corr} non-trivial pairs")
        print(f"  Concentration factors: min={concentration.min():.2f}, max={concentration.max():.2f}")

    # Step 4: Initial allocation (from generate_portfolio group assignments)
    initial_allocation = np.zeros((T, P), dtype=np.float64)
    for t in range(T):
        initial_allocation[t, group_ids[t]] = 1.0

    # Step 5: Compute initial SIMM
    if verbose:
        print("\nStep 4: Computing initial SIMM (CUDA)...")
    simm_start = time.perf_counter()

    agg_S_T = (S.T @ initial_allocation).T

    im_values, gradients = compute_simm_and_gradient_cuda(
        agg_S_T, risk_weights, concentration, bucket_id, risk_measure_idx,
        bucket_rc, bucket_rm, intra_corr, bucket_gamma, B, device,
    )
    initial_im = float(np.sum(im_values))

    results['timings']['initial_simm'] = time.perf_counter() - simm_start
    results['initial_im'] = initial_im

    if verbose:
        print(f"  Initial total IM: ${initial_im:,.2f}")
        print(f"  CUDA eval time: {results['timings']['initial_simm']*1000:.2f} ms")

    # Step 6: Optimization (if requested)
    if optimize:
        if verbose:
            print(f"\nStep 5: Optimizing allocation ({method})...")
        opt_start = time.perf_counter()

        final_allocation, im_history, num_iters, eval_time = optimize_allocation_cuda(
            S, initial_allocation, risk_weights, concentration,
            bucket_id, risk_measure_idx, bucket_rc, bucket_rm,
            intra_corr, bucket_gamma, B,
            max_iters=max_iters, verbose=verbose, device=device,
            method=method,
        )

        # Round to integer allocation
        final_allocation = _round_to_integer(final_allocation)

        # Compute final IM
        agg_S_final = (S.T @ final_allocation).T
        im_final, _ = compute_simm_and_gradient_cuda(
            agg_S_final, risk_weights, concentration, bucket_id,
            risk_measure_idx, bucket_rc, bucket_rm, intra_corr,
            bucket_gamma, B, device,
        )
        final_im = float(np.sum(im_final))

        results['timings']['optimization'] = time.perf_counter() - opt_start
        results['timings']['cuda_eval'] = eval_time
        results['final_im'] = final_im
        results['im_reduction'] = initial_im - final_im
        results['im_reduction_pct'] = (initial_im - final_im) / initial_im * 100
        results['num_iterations'] = num_iters
        results['trades_moved'] = int(np.sum(
            np.argmax(final_allocation, axis=1) != np.argmax(initial_allocation, axis=1)
        ))

        if verbose:
            print(f"\n  Final IM: ${final_im:,.2f}")
            print(f"  IM reduction: ${results['im_reduction']:,.2f} ({results['im_reduction_pct']:.2f}%)")
            print(f"  Trades moved: {results['trades_moved']}")
            print(f"  Iterations: {num_iters}")
            print(f"  Optimization time: {results['timings']['optimization']:.3f}s")
            print(f"  CUDA eval time: {eval_time:.3f}s")
    else:
        results['final_im'] = initial_im

    results['timings']['total'] = time.perf_counter() - total_start

    if verbose:
        print(f"\n{'='*70}")
        print(f"Total time: {results['timings']['total']:.3f}s")
        print(f"{'='*70}")

    return results


def log_to_execution_log(results: Dict):
    """Log results using the common SIMMLogger for consistent schema."""
    logger = SIMMLogger()

    eval_time = results['timings'].get('cuda_eval', results['timings'].get('initial_simm', 0))

    record = SIMMExecutionRecord(
        model_name='simm_portfolio_cuda',
        model_version=MODEL_VERSION,
        mode='margin_with_optimization' if 'optimization' in results['timings'] else 'margin_only',
        num_trades=results['num_trades'],
        num_risk_factors=results.get('num_risk_factors', 0),
        num_sensitivities=results.get('num_sensitivities', 0),
        num_threads=1,  # GPU
        simm_total=results.get('final_im', results.get('initial_im', 0)),
        eval_time_sec=eval_time,
        recording_time_sec=0,
        kernel_execution_time_sec=results['timings'].get('initial_simm', 0),
        language='CUDA',
        uses_aadc=False,
        status='success',
    )

    logger.log(record)
    print(f"Results logged to {logger.log_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SIMM Portfolio Calculator - CUDA GPU Version"
    )
    parser.add_argument('--trades', '-t', type=int, default=1000,
                        help='Number of trades')
    parser.add_argument('--portfolios', '-p', type=int, default=5,
                        help='Number of portfolios')
    parser.add_argument('--trade-types', type=str, default='ir_swap,equity_option',
                        help='Comma-separated trade types')
    parser.add_argument('--threads', type=int, default=8,
                        help='(Unused for GPU, kept for API compatibility)')
    parser.add_argument('--optimize', action='store_true',
                        help='Run allocation optimization')
    parser.add_argument('--method', choices=['gradient_descent', 'adam'], default='gradient_descent',
                        help='Optimization method')
    parser.add_argument('--max-iters', type=int, default=100,
                        help='Max optimization iterations')
    parser.add_argument('--simm-buckets', type=int, default=3,
                        help='Number of currencies (SIMM buckets)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--log', action='store_true',
                        help='Log results to execution_log.csv')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode')

    args = parser.parse_args()

    trade_types = [t.strip() for t in args.trade_types.split(',')]

    try:
        results = run_portfolio_cuda(
            num_trades=args.trades,
            num_portfolios=args.portfolios,
            trade_types=trade_types,
            num_threads=args.threads,
            optimize=args.optimize,
            method=args.method,
            max_iters=args.max_iters,
            verbose=not args.quiet,
            device=args.device,
            num_simm_buckets=args.simm_buckets,
        )

        if args.log:
            log_to_execution_log(results)

        # Print summary
        print(f"\nSummary:")
        print(f"  Initial IM:  ${results['initial_im']:,.2f}")
        if args.optimize:
            print(f"  Final IM:    ${results['final_im']:,.2f}")
            print(f"  Reduction:   ${results['im_reduction']:,.2f} ({results['im_reduction_pct']:.2f}%)")
            print(f"  CUDA time:   {results['timings'].get('cuda_eval', 0)*1000:.2f} ms")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
