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

Version: 1.0.0
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

# Import ISDA SIMM v2.6 PSI matrix (same as AADC module)
from model.simm_portfolio_aadc import PSI_MATRIX

# Import from existing modules (simm_cuda only used for standalone kernel tests)
try:
    from model.simm_cuda import (
        compute_simm_cuda,
        compute_simm_gradient_cuda,
    )
    SIMM_CUDA_AVAILABLE = True
except ImportError:
    SIMM_CUDA_AVAILABLE = False

# Import trade generators and market environment
try:
    from model.trade_types import (
        IRSwapTrade,
        EquityOptionTrade,
        FXOptionTrade,
        YieldCurve,
        VolSurface,
        MarketEnvironment,
    )
    from common.portfolio import generate_portfolio
    from model.simm_portfolio_aadc import (
        precompute_all_trade_crifs,
        _get_ir_risk_weight_v26,
        _get_concentration_threshold,
        _compute_concentration_risk,
        _map_risk_type_to_class,
        _is_delta_risk_type,
        _is_vega_risk_type,
        _get_vega_risk_weight,
        _get_risk_weight,
    )
    from common.logger import SIMMLogger, SIMMExecutionRecord
    TRADE_GENERATORS_AVAILABLE = True
except ImportError as e:
    print(f"Import warning: {e}")
    TRADE_GENERATORS_AVAILABLE = False

# Version
MODEL_VERSION = "2.0.0"  # v2: ISDA v2.6 aligned (PSI, risk weights, concentration)


# =============================================================================
# CUDA Kernels for Optimization
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def _simm_gradient_kernel_full(
        sensitivities,      # (P, K) - aggregated sensitivities per portfolio
        risk_weights,       # (K,)
        risk_class_idx,     # (K,)
        psi_matrix,         # (6, 6)
        im_output,          # (P,) - output IM values
        gradients,          # (P, K) - output gradients dIM/dS
    ):
        """
        CUDA kernel for SIMM + gradient computation.
        One thread per portfolio, computes both IM and dIM/dS.
        """
        p = cuda.grid(1)
        if p >= sensitivities.shape[0]:
            return

        K = sensitivities.shape[1]

        # Local arrays
        ws = cuda.local.array(200, dtype=numba.float64)
        k_r = cuda.local.array(6, dtype=numba.float64)
        k_r_sq = cuda.local.array(6, dtype=numba.float64)

        # Initialize
        for r in range(6):
            k_r[r] = 0.0
            k_r_sq[r] = 0.0

        # Compute weighted sensitivities and K_r^2
        for k in range(min(K, 200)):
            ws[k] = sensitivities[p, k] * risk_weights[k]
            rc = risk_class_idx[k]
            k_r_sq[rc] += ws[k] * ws[k]

        # K_r = sqrt(K_r^2)
        for r in range(6):
            k_r[r] = math.sqrt(k_r_sq[r]) if k_r_sq[r] > 0 else 0.0

        # Cross-RC aggregation: IM^2 = sum_rs psi[r,s] * K_r * K_s
        im_sq = 0.0
        for r in range(6):
            for s in range(6):
                im_sq += psi_matrix[r, s] * k_r[r] * k_r[s]

        im_p = math.sqrt(im_sq) if im_sq > 0 else 0.0
        im_output[p] = im_p

        # Compute gradients
        if im_p < 1e-10:
            for k in range(K):
                gradients[p, k] = 0.0
            return

        # dIM/dK_r = (1/IM) * sum_s psi[r,s] * K_s
        dim_dk = cuda.local.array(6, dtype=numba.float64)
        for r in range(6):
            dim_dk[r] = 0.0
            for s in range(6):
                dim_dk[r] += psi_matrix[r, s] * k_r[s]
            dim_dk[r] /= im_p

        # dIM/dS[k] = dIM/dK_r * dK_r/dWS * dWS/dS
        #           = dIM/dK_r * (WS[k]/K_r) * w[k]
        for k in range(K):
            rc = risk_class_idx[k]
            w_k = risk_weights[k]
            if k_r[rc] > 1e-10:
                dk_dws = ws[k] / k_r[rc]
                gradients[p, k] = dim_dk[rc] * dk_dws * w_k
            else:
                gradients[p, k] = 0.0


# =============================================================================
# CUDA SIMM Calculator with Gradients
# =============================================================================

def compute_simm_and_gradient_cuda(
    sensitivities: np.ndarray,
    risk_weights: np.ndarray,
    risk_class_idx: np.ndarray,
    device: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SIMM margin and gradients for multiple portfolios using CUDA.

    Args:
        sensitivities: (P, K) array of aggregated sensitivities
        risk_weights: (K,) array of risk weights
        risk_class_idx: (K,) array of risk class indices (0-5)
        device: GPU device ID

    Returns:
        (im_values, gradients) where:
        - im_values: (P,) array of IM values
        - gradients: (P, K) array of dIM/dS
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")

    P, K = sensitivities.shape

    # Ensure contiguous arrays
    sensitivities = np.ascontiguousarray(sensitivities, dtype=np.float64)
    risk_weights = np.ascontiguousarray(risk_weights, dtype=np.float64)
    risk_class_idx = np.ascontiguousarray(risk_class_idx, dtype=np.int32)

    # Allocate outputs
    im_output = np.zeros(P, dtype=np.float64)
    gradients = np.zeros((P, K), dtype=np.float64)

    # Transfer to GPU
    if not CUDA_SIMULATOR:
        cuda.select_device(device)

    d_sens = cuda.to_device(sensitivities)
    d_weights = cuda.to_device(risk_weights)
    d_rc_idx = cuda.to_device(risk_class_idx)
    d_psi = cuda.to_device(PSI_MATRIX)
    d_im = cuda.to_device(im_output)
    d_grad = cuda.to_device(gradients)

    # Launch kernel
    threads_per_block = 256
    blocks = (P + threads_per_block - 1) // threads_per_block

    _simm_gradient_kernel_full[blocks, threads_per_block](
        d_sens, d_weights, d_rc_idx, d_psi, d_im, d_grad
    )

    # Synchronize and copy back
    if not CUDA_SIMULATOR:
        cuda.synchronize()

    d_im.copy_to_host(im_output)
    d_grad.copy_to_host(gradients)

    return im_output, gradients


# =============================================================================
# Portfolio Optimization using CUDA
# =============================================================================

def optimize_allocation_cuda(
    S: np.ndarray,                    # (T, K) sensitivity matrix
    initial_allocation: np.ndarray,  # (T, P) initial allocation
    risk_weights: np.ndarray,        # (K,) risk weights
    risk_class_idx: np.ndarray,      # (K,) risk class indices
    max_iters: int = 100,
    lr: float = None,
    tol: float = 1e-6,
    verbose: bool = True,
    device: int = 0,
) -> Tuple[np.ndarray, List[float], int, float]:
    """
    Optimize trade allocation using CUDA-accelerated gradient descent.

    Args:
        S: Sensitivity matrix (T trades × K risk factors)
        initial_allocation: Starting allocation (T trades × P portfolios)
        risk_weights: Risk weight per factor
        risk_class_idx: Risk class (0-5) per factor
        max_iters: Maximum iterations
        lr: Learning rate (auto if None)
        tol: Convergence tolerance
        verbose: Print progress
        device: GPU device

    Returns:
        (final_allocation, im_history, num_iterations, eval_time)
    """
    T, P = initial_allocation.shape
    K = S.shape[1]

    x = initial_allocation.copy()
    best_x = x.copy()
    im_history = []

    # Backtracking line search parameters
    LS_BETA = 0.5
    LS_MAX_TRIES = 10

    eval_start = time.perf_counter()

    # First evaluation to get gradient scale
    # Aggregate sensitivities: agg_S[k,p] = sum_t x[t,p] * S[t,k]
    agg_S = np.dot(S.T, x)  # (K, P) -> transpose to (P, K) for CUDA
    agg_S_T = agg_S.T  # (P, K)

    im_values, grad_S = compute_simm_and_gradient_cuda(
        agg_S_T, risk_weights, risk_class_idx, device
    )
    total_im = float(np.sum(im_values))
    im_history.append(total_im)
    best_im = total_im

    # Chain rule: dIM/dx[t,p] = sum_k (dIM_p/dS_p[k]) * S[t,k]
    # grad_S is (P, K), S is (T, K)
    # gradient[t,p] = sum_k grad_S[p,k] * S[t,k] = S @ grad_S.T
    gradient = np.dot(S, grad_S.T)  # (T, P)

    # Auto learning rate (conservative)
    grad_max = np.abs(gradient).max()
    if lr is None:
        lr = 1.0 / grad_max if grad_max > 1e-10 else 1e-12

    if verbose:
        print(f"    Initial IM: ${total_im:,.2f}")
        print(f"    Gradient: max={grad_max:.2e}")
        print(f"    Learning rate: {lr:.2e}")

    stalled_count = 0

    for iteration in range(max_iters):
        if iteration > 0:
            # Recompute gradient
            agg_S = np.dot(S.T, x)
            agg_S_T = agg_S.T

            im_values, grad_S = compute_simm_and_gradient_cuda(
                agg_S_T, risk_weights, risk_class_idx, device
            )
            total_im = float(np.sum(im_values))
            im_history.append(total_im)

            gradient = np.dot(S, grad_S.T)

        # Track best solution
        if total_im < best_im:
            best_im = total_im
            best_x = x.copy()
            stalled_count = 0
        else:
            stalled_count += 1

        if verbose and iteration % 10 == 0:
            moves = int(np.sum(np.argmax(x, axis=1) != np.argmax(initial_allocation, axis=1)))
            print(f"    Iter {iteration}: IM = ${total_im:,.2f}, best = ${best_im:,.2f}, moves = {moves}")

        # Early exit if stalled
        if stalled_count >= 20:
            if verbose:
                print(f"    Stalled for {stalled_count} iterations, reverting to best")
            x = best_x.copy()
            break

        # Monotone backtracking line search
        step_size = lr
        accepted = False

        for _ in range(LS_MAX_TRIES):
            x_candidate = _project_to_simplex(x - step_size * gradient)

            agg_S_c = np.dot(S.T, x_candidate)
            im_values_c, _ = compute_simm_and_gradient_cuda(
                agg_S_c.T, risk_weights, risk_class_idx, device
            )
            candidate_im = float(np.sum(im_values_c))

            if candidate_im < total_im:
                x = x_candidate
                accepted = True
                break

            step_size *= LS_BETA

        # Check convergence
        if iteration > 0 and len(im_history) >= 2:
            rel_change = abs(im_history[-1] - im_history[-2]) / max(abs(im_history[-2]), 1e-10)
            if rel_change < tol:
                if verbose:
                    print(f"    Converged at iteration {iteration + 1}")
                break

    # Round to integer allocation
    rounded_x = np.zeros_like(best_x)
    for t in range(T):
        rounded_x[t, np.argmax(best_x[t])] = 1.0

    # Greedy local search on integer allocation
    if verbose:
        print(f"    Running greedy local search...")
    greedy_x, greedy_im, greedy_moves = _greedy_local_search_cuda(
        S, rounded_x, risk_weights, risk_class_idx, device,
        max_rounds=max_iters, verbose=verbose,
    )

    if greedy_im < best_im:
        best_x = greedy_x
        best_im = greedy_im

    eval_time = time.perf_counter() - eval_start

    if verbose and stalled_count < 20 and iteration == max_iters - 1:
        print(f"    Reached max iterations ({max_iters})")

    # Always return best solution seen
    return best_x, im_history, iteration + 1, eval_time


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
    S: np.ndarray,
    integer_allocation: np.ndarray,
    risk_weights: np.ndarray,
    risk_class_idx: np.ndarray,
    device: int = 0,
    max_rounds: int = 50,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """
    Gradient-guided greedy local search for CUDA optimizer.

    Uses gradient to identify promising single-trade moves, verifies each
    with actual IM evaluation.

    Returns:
        (improved_allocation, final_im, total_moves)
    """
    T, P = integer_allocation.shape
    x = integer_allocation.copy()
    total_moves = 0

    # Initial evaluation
    agg_S_T = (S.T @ x).T
    im_values, grad_S = compute_simm_and_gradient_cuda(
        agg_S_T, risk_weights, risk_class_idx, device
    )
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
            im_values_c, _ = compute_simm_and_gradient_cuda(
                agg_S_T_c, risk_weights, risk_class_idx, device
            )
            candidate_im = float(np.sum(im_values_c))

            if candidate_im < current_im:
                improvement = current_im - candidate_im
                current_im = candidate_im
                total_moves += 1
                accepted = True

                if verbose:
                    print(f"    Greedy round {round_idx+1}: move trade {t} "
                          f"(p{from_p}->p{to_p}), IM -${improvement:,.0f}")

                # Recompute gradient
                agg_S_T = (S.T @ x).T
                im_values, grad_S = compute_simm_and_gradient_cuda(
                    agg_S_T, risk_weights, risk_class_idx, device
                )
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

    # Step 2: Compute CRIF
    if verbose:
        print("\nStep 2: Computing CRIF sensitivities...")
    crif_start = time.perf_counter()

    # Use precompute_all_trade_crifs from AADC module (same CRIF, different eval)
    trade_crifs = precompute_all_trade_crifs(trades, market, num_threads=8)

    # Collect all CRIF records
    crif_records = []
    for trade_id, crif_df in trade_crifs.items():
        for _, row in crif_df.iterrows():
            crif_records.append(row.to_dict())

    results['timings']['crif_computation'] = time.perf_counter() - crif_start
    results['num_sensitivities'] = len(crif_records)

    if verbose:
        print(f"  Computed {len(crif_records)} sensitivities in {results['timings']['crif_computation']:.3f}s")

    # Step 3: Build sensitivity matrix
    if verbose:
        print("\nStep 3: Building sensitivity matrix...")

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

    # Risk weights — ISDA SIMM v2.6 (tenor-specific for IR)
    risk_class_map = {
        "Rates": 0, "CreditQ": 1, "CreditNonQ": 2,
        "Equity": 3, "Commodity": 4, "FX": 5,
    }
    risk_weights = np.zeros(K, dtype=np.float64)
    risk_class_idx = np.zeros(K, dtype=np.int32)

    for k, rf in enumerate(risk_factors):
        rt, qualifier, bucket, label1 = rf
        rc = _map_risk_type_to_class(rt)
        risk_class_idx[k] = risk_class_map.get(rc, 0)

        # Use v2.6 currency-specific weights for IR
        if rt == "Risk_IRCurve" and qualifier and label1:
            risk_weights[k] = _get_ir_risk_weight_v26(qualifier, label1)
        elif _is_vega_risk_type(rt):
            risk_weights[k] = _get_vega_risk_weight(rt, bucket)
        else:
            risk_weights[k] = _get_risk_weight(rt, bucket)

    results['num_risk_factors'] = K

    if verbose:
        print(f"  Matrix: {T} trades × {K} risk factors")

    # Step 4: Initial allocation (from generate_portfolio group assignments)
    initial_allocation = np.zeros((T, P), dtype=np.float64)
    for t in range(T):
        initial_allocation[t, group_ids[t]] = 1.0

    # Step 5: Compute initial SIMM
    if verbose:
        print("\nStep 4: Computing initial SIMM (CUDA)...")
    simm_start = time.perf_counter()

    # Aggregate sensitivities for each portfolio
    agg_S = np.dot(S.T, initial_allocation)  # (K, P)
    agg_S_T = agg_S.T  # (P, K)

    im_values, gradients = compute_simm_and_gradient_cuda(
        agg_S_T, risk_weights, risk_class_idx, device
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
            S, initial_allocation, risk_weights, risk_class_idx,
            max_iters=max_iters,
            verbose=verbose,
            device=device,
        )

        # Round to integer allocation
        final_allocation = _round_to_integer(final_allocation)

        # Compute final IM
        agg_S_final = np.dot(S.T, final_allocation)
        im_final, _ = compute_simm_and_gradient_cuda(
            agg_S_final.T, risk_weights, risk_class_idx, device
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
    parser.add_argument('--method', choices=['gradient_descent'], default='gradient_descent',
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
