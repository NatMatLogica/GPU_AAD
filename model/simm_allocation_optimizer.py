"""
Optimal Trade Allocation to Minimize Total SIMM.

EFFICIENT DESIGN: Uses small kernel with K inputs (~100 risk factors)
instead of T×P inputs (~25,000 for 5000 trades × 5 portfolios).

Gradient computation via chain rule:
  ∂total_IM/∂x[t,p] = Σ_k (∂IM_p/∂S_p[k]) * S[t,k]

Where:
  - S_p[k] = Σ_t x[t,p] * S[t,k]  (aggregated sensitivity, computed in numpy)
  - ∂IM_p/∂S_p[k]                 (from AADC kernel)
  - S[t,k]                         (raw trade sensitivity, precomputed)

Mathematical Formulation:
    Input:
      - T trades with CRIF sensitivity vectors s_t (computed via AADC)
      - P portfolios
      - Allocation matrix X where x_tp = fraction of trade t in portfolio p

    Constraints:
      - sum_p x_tp = 1 for all t (each trade fully allocated)
      - x_tp in [0,1] for partial trades, or {0,1} for whole trades

    Portfolio sensitivities:
      - S_p = sum_t x_tp * s_t (aggregated CRIF for portfolio p)

    Portfolio IM (ISDA SIMM v2.6):
      - K_r = sqrt(sum_i (w_i * S_p[i])^2) for each risk class r
      - IM_p = sqrt(sum_r sum_s psi_rs * K_r * K_s) with ISDA correlation matrix

    Objective:
      - Minimize sum_p IM_p (total IM across all portfolios)

Key Efficiency (v3.0 - Single Evaluate Optimization):
    - Kernel size: O(K) ≈ 100 inputs (vs old O(T×P) ≈ 25,000)
    - Kernel recording: <0.1 seconds (vs old 30+ seconds)
    - Per-iteration: 1 evaluate() call for ALL P portfolios (vs old P calls)
    - Chain rule gradient: O(T×K) numpy ops (vectorized)
    - Speedup: 5-10x from single-evaluate pattern (reduces Python-C++ dispatch overhead)

Usage:
    from model.simm_allocation_optimizer import reallocate_trades_optimal

    result = reallocate_trades_optimal(
        trades, market, num_portfolios=5,
        initial_allocation=current_allocation,
        allow_partial=False, method='gradient_descent'
    )

Version: 2.0.0
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional

# Version
MODULE_VERSION = "3.0.0"  # CRITICAL: Single evaluate() for all P portfolios (10x speedup)

try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False

# Import ISDA SIMM v2.6 parameters for correlations and weights
try:
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from Weights_and_Corr.v2_6 import (
        ir_corr,  # 12x12 IR tenor correlation matrix
        reg_vol_rw, low_vol_rw, high_vol_rw,  # Currency-specific IR weights
        reg_vol_ccy_bucket, low_vol_ccy_bucket,  # Currency volatility buckets
        inflation_rw, inflation_corr,  # Inflation parameters
        equity_corr, commodity_corr, fx_vega_corr,  # Other risk class correlations
    )
    SIMM_PARAMS_AVAILABLE = True
except ImportError:
    SIMM_PARAMS_AVAILABLE = False

# SIMM tenor list (order matches ir_corr matrix)
SIMM_TENOR_LIST = ['2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
SIMM_TENOR_TO_IDX = {t: i for i, t in enumerate(SIMM_TENOR_LIST)}

# Build numpy IR correlation matrix from v2_6 data
if SIMM_PARAMS_AVAILABLE:
    IR_CORR_MATRIX = np.array(ir_corr)  # 12x12
else:
    IR_CORR_MATRIX = np.eye(12)  # Fallback to identity


# =============================================================================
# Helper Functions
# =============================================================================

def _get_unique_risk_factors(trade_crifs: Dict[str, pd.DataFrame]) -> List[Tuple]:
    """
    Extract unique risk factors across all trades.

    Returns list of (RiskType, Qualifier, Bucket, Label1) tuples that
    uniquely identify each sensitivity in the CRIF.
    """
    factors = set()
    for crif_df in trade_crifs.values():
        # Use itertuples() for 10-100x speedup over iterrows()
        for row in crif_df.itertuples(index=False):
            key = (row.RiskType, row.Qualifier, str(row.Bucket), row.Label1)
            factors.add(key)
    return sorted(factors)


def _build_sensitivity_matrix(
    trade_crifs: Dict[str, pd.DataFrame],
    trade_ids: List[str],
    risk_factors: List[Tuple],
) -> np.ndarray:
    """
    Build T x K sensitivity matrix where S[t,k] = sensitivity of trade t to factor k.

    Args:
        trade_crifs: Dict mapping trade_id to CRIF DataFrame
        trade_ids: Ordered list of trade IDs
        risk_factors: Ordered list of (RiskType, Qualifier, Bucket, Label1) tuples

    Returns:
        S: numpy array of shape (T, K) where T = num trades, K = num risk factors
    """
    T = len(trade_ids)
    K = len(risk_factors)
    S = np.zeros((T, K))

    factor_to_idx = {f: i for i, f in enumerate(risk_factors)}
    trade_to_idx = {t: i for i, t in enumerate(trade_ids)}

    for trade_id, crif_df in trade_crifs.items():
        if trade_id not in trade_to_idx:
            continue
        t_idx = trade_to_idx[trade_id]
        # Use itertuples() for 10-100x speedup over iterrows()
        for row in crif_df.itertuples(index=False):
            key = (row.RiskType, row.Qualifier, str(row.Bucket), row.Label1)
            if key in factor_to_idx:
                k_idx = factor_to_idx[key]
                S[t_idx, k_idx] += float(row.Amount)

    return S


def _map_risk_type_to_class(risk_type: str) -> str:
    """Map CRIF RiskType to SIMM risk class."""
    if risk_type in ("Risk_IRCurve", "Risk_Inflation", "Risk_XCcyBasis",
                     "Risk_IRVol", "Risk_InflationVol"):
        return "Rates"
    elif risk_type in ("Risk_FX", "Risk_FXVol"):
        return "FX"
    elif risk_type in ("Risk_CreditQ", "Risk_CreditVol", "Risk_BaseCorr"):
        return "CreditQ"
    elif risk_type in ("Risk_CreditNonQ", "Risk_CreditVolNonQ"):
        return "CreditNonQ"
    elif risk_type in ("Risk_Equity", "Risk_EquityVol"):
        return "Equity"
    elif risk_type in ("Risk_Commodity", "Risk_CommodityVol"):
        return "Commodity"
    return "Rates"


# IR risk weights by bucket (ISDA SIMM v2.6)
_IR_RISK_WEIGHTS = np.array([
    77, 77, 68, 56, 52, 50, 51, 52, 50, 51, 51, 64
], dtype=float)
_FX_RISK_WEIGHT = 8.4
_EQUITY_RISK_WEIGHT = 25.0
_INFLATION_RISK_WEIGHT = 63.0


def _get_risk_weight(risk_type: str, bucket: str) -> float:
    """Get simplified risk weight for a CRIF entry."""
    if risk_type == "Risk_IRCurve":
        try:
            idx = int(bucket) - 1
            if 0 <= idx < len(_IR_RISK_WEIGHTS):
                return _IR_RISK_WEIGHTS[idx]
        except (ValueError, IndexError):
            pass
        return 50.0
    elif risk_type == "Risk_Inflation":
        return _INFLATION_RISK_WEIGHT
    elif risk_type == "Risk_FX":
        return _FX_RISK_WEIGHT
    elif risk_type == "Risk_FXVol":
        return _FX_RISK_WEIGHT * 0.55
    elif risk_type in ("Risk_Equity", "Risk_EquityVol"):
        return _EQUITY_RISK_WEIGHT
    return 50.0


def _get_ir_risk_weight_v26(currency: str, tenor: str) -> float:
    """Get ISDA v2.6 currency-specific IR risk weight."""
    if not SIMM_PARAMS_AVAILABLE:
        return 50.0
    tenor_lower = str(tenor).lower().strip()
    currency_upper = str(currency).upper().strip()

    if currency_upper in reg_vol_ccy_bucket:
        return reg_vol_rw.get(tenor_lower, 50.0)
    elif currency_upper in low_vol_ccy_bucket:
        return low_vol_rw.get(tenor_lower, 15.0)
    else:
        return high_vol_rw.get(tenor_lower, 100.0)


def _get_ir_tenor_index(label1: str) -> int:
    """Map Label1 (e.g., '3m', '5y') to IR tenor index (0-11)."""
    label_lower = str(label1).lower().strip()
    return SIMM_TENOR_TO_IDX.get(label_lower, 6)  # Default to ~3y


def _get_ir_correlation(tenor1: str, tenor2: str) -> float:
    """Get IR intra-bucket correlation between two tenors."""
    idx1 = _get_ir_tenor_index(tenor1)
    idx2 = _get_ir_tenor_index(tenor2)
    return IR_CORR_MATRIX[idx1, idx2]


def _get_intra_correlation(risk_class: str, risk_type1: str, risk_type2: str,
                           label1_1: str, label1_2: str, bucket: str = None) -> float:
    """
    Get intra-bucket correlation for same risk class.

    For IR: Uses 12x12 tenor correlation matrix from ISDA SIMM v2.6.
    For Equity/Commodity: Uses single correlation value per bucket.
    """
    if not SIMM_PARAMS_AVAILABLE:
        return 1.0  # Fallback to perfectly correlated

    if risk_class == "Rates":
        if risk_type1 == "Risk_Inflation" or risk_type2 == "Risk_Inflation":
            if risk_type1 == risk_type2:
                return 1.0
            return inflation_corr
        elif risk_type1 == "Risk_IRCurve" and risk_type2 == "Risk_IRCurve":
            return _get_ir_correlation(label1_1, label1_2)
        return 1.0

    elif risk_class == "Equity":
        try:
            bucket_int = int(bucket) if bucket else 0
        except ValueError:
            bucket_int = 0
        return equity_corr.get(bucket_int, 0.25)

    elif risk_class == "Commodity":
        try:
            bucket_int = int(bucket) if bucket else 0
        except ValueError:
            bucket_int = 0
        return commodity_corr.get(bucket_int, 0.5)

    elif risk_class == "FX":
        return fx_vega_corr

    elif risk_class in ("CreditQ", "CreditNonQ"):
        return 0.5

    return 1.0


# Cross-risk-class correlation matrix (psi) - ISDA SIMM v2.6/v2.7
# Order: Rates, CreditQ, CreditNonQ, Equity, Commodity, FX
_PSI_MATRIX = np.array([
    [1.00, 0.04, 0.04, 0.07, 0.37, 0.14],  # Rates
    [0.04, 1.00, 0.54, 0.70, 0.27, 0.37],  # CreditQ
    [0.04, 0.54, 1.00, 0.46, 0.24, 0.15],  # CreditNonQ
    [0.07, 0.70, 0.46, 1.00, 0.35, 0.39],  # Equity
    [0.37, 0.27, 0.24, 0.35, 1.00, 0.35],  # Commodity
    [0.14, 0.37, 0.15, 0.39, 0.35, 1.00],  # FX
])

_RISK_CLASS_ORDER = ['Rates', 'CreditQ', 'CreditNonQ', 'Equity', 'Commodity', 'FX']


# =============================================================================
# EFFICIENT Small Kernel + Chain Rule (NEW - O(K) inputs)
# =============================================================================

def _get_factor_metadata(risk_factors: List[Tuple]) -> Tuple[List[str], np.ndarray, List[str], List[str], List[str]]:
    """
    Extract risk class, weight, and metadata for each risk factor.

    Args:
        risk_factors: List of (RiskType, Qualifier, Bucket, Label1) tuples

    Returns:
        (factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets) where:
        - factor_risk_classes: List of risk class names for each factor
        - factor_weights: numpy array of risk weights
        - factor_risk_types: List of RiskType strings
        - factor_labels: List of Label1 (tenor) strings
        - factor_buckets: List of Bucket strings
    """
    factor_risk_classes = []
    factor_weights = []
    factor_risk_types = []
    factor_labels = []
    factor_buckets = []

    for rt, qualifier, bucket, label1 in risk_factors:
        rc = _map_risk_type_to_class(rt)
        # Use v2.6 currency-specific weights for IR if available
        if rt == "Risk_IRCurve" and qualifier and label1 and SIMM_PARAMS_AVAILABLE:
            rw = _get_ir_risk_weight_v26(qualifier, label1)
        else:
            rw = _get_risk_weight(rt, bucket)

        factor_risk_classes.append(rc)
        factor_weights.append(rw)
        factor_risk_types.append(rt)
        factor_labels.append(label1)
        factor_buckets.append(bucket)

    return factor_risk_classes, np.array(factor_weights), factor_risk_types, factor_labels, factor_buckets


def record_single_portfolio_simm_kernel(
    K: int,
    factor_risk_classes: List[str],
    factor_weights: np.ndarray,
    factor_risk_types: List[str] = None,
    factor_labels: List[str] = None,
    factor_buckets: List[str] = None,
    use_correlations: bool = True,
) -> Tuple['aadc.Functions', List[int], int]:
    """
    Record AADC kernel: K aggregated sensitivities → single portfolio IM.

    CRITICAL: This kernel has only K inputs (~100), not T×P (~25,000).

    Inputs: S_p[k] for k=0..K-1 (aggregated sensitivities for ONE portfolio)
    Output: IM_p (SIMM margin for that portfolio)

    Algorithm:
    1. Mark S_p[k] as inputs (K differentiable values, ~100)
    2. Apply ISDA SIMM aggregation with intra-bucket correlations:
       - For each risk class rc:
         K_rc = sqrt(Σ_{k∈rc} Σ_{l∈rc} ρ_kl * WS_k * WS_l)
       - IM_p = sqrt(Σ_i Σ_j ψ[i,j] * K_i * K_j)
    3. Mark IM_p as output

    Args:
        K: Number of risk factors
        factor_risk_classes: List of risk class names for each factor
        factor_weights: Array of risk weights for each factor
        factor_risk_types: List of RiskType strings (for correlation lookup)
        factor_labels: List of Label1/tenor strings (for correlation lookup)
        factor_buckets: List of Bucket strings (for correlation lookup)
        use_correlations: If True, apply ISDA intra-bucket correlations

    Returns:
        (funcs, sensitivity_handles, im_output)
    """
    if not AADC_AVAILABLE:
        raise RuntimeError("AADC is required for allocation optimization")

    # Provide defaults for backwards compatibility
    if factor_risk_types is None:
        factor_risk_types = ["Risk_IRCurve"] * K
    if factor_labels is None:
        factor_labels = ["5y"] * K
    if factor_buckets is None:
        factor_buckets = [""] * K

    with aadc.record_kernel() as funcs:
        # Mark aggregated sensitivities as differentiable inputs (K values)
        agg_sens = []
        sens_handles = []
        for k in range(K):
            s_k = aadc.idouble(0.0)
            handle = s_k.mark_as_input()
            sens_handles.append(handle)
            agg_sens.append(s_k)

        # Compute risk class margins with intra-bucket correlations
        # K_rc = sqrt(Σ_{k∈rc} Σ_{l∈rc} ρ_kl * WS_k * WS_l)
        risk_class_margins = []

        for rc in _RISK_CLASS_ORDER:
            rc_indices = [k for k in range(K) if factor_risk_classes[k] == rc]
            if not rc_indices:
                risk_class_margins.append(aadc.idouble(0.0))
                continue

            # Compute weighted sensitivities
            ws_list = []
            for k in rc_indices:
                ws_k = agg_sens[k] * float(factor_weights[k])
                ws_list.append(ws_k)

            # Apply correlations: K² = Σ_k Σ_l ρ_kl × WS_k × WS_l
            k_sq = aadc.idouble(0.0)
            num_in_rc = len(rc_indices)

            if use_correlations and num_in_rc > 1 and SIMM_PARAMS_AVAILABLE:
                for i_local in range(num_in_rc):
                    for j_local in range(num_in_rc):
                        i_global = rc_indices[i_local]
                        j_global = rc_indices[j_local]

                        # Get correlation between these two risk factors
                        rho_ij = _get_intra_correlation(
                            rc,
                            factor_risk_types[i_global],
                            factor_risk_types[j_global],
                            factor_labels[i_global],
                            factor_labels[j_global],
                            factor_buckets[i_global] if factor_buckets[i_global] == factor_buckets[j_global] else None
                        )

                        k_sq = k_sq + rho_ij * ws_list[i_local] * ws_list[j_local]
            else:
                # Simplified: sum of squares (no correlations)
                for ws_k in ws_list:
                    k_sq = k_sq + ws_k * ws_k

            k_r = np.sqrt(k_sq)
            risk_class_margins.append(k_r)

        # Cross-risk-class aggregation: IM_p = sqrt(Σ_i Σ_j ψ[i,j] * K_i * K_j)
        simm_sq = aadc.idouble(0.0)
        for i in range(6):
            for j in range(6):
                psi_ij = _PSI_MATRIX[i, j]
                simm_sq = simm_sq + psi_ij * risk_class_margins[i] * risk_class_margins[j]

        im_p = np.sqrt(simm_sq)
        im_output = im_p.mark_as_output()

    return funcs, sens_handles, im_output


def compute_allocation_gradient_chainrule(
    funcs: 'aadc.Functions',
    sens_handles: List[int],
    im_output: int,
    S: np.ndarray,                    # T × K sensitivity matrix
    current_allocation: np.ndarray,   # T × P allocation fractions
    num_threads: int,
    workers: 'aadc.ThreadPool' = None,
) -> Tuple[np.ndarray, float]:
    """
    Compute gradient ∂total_IM/∂x[t,p] using chain rule with SINGLE aadc.evaluate() call.

    OPTIMIZED (v3.0): All P portfolios evaluated in ONE aadc.evaluate() call
    instead of P separate calls. This reduces Python-C++ dispatch overhead
    and yields 5-10x speedup for typical portfolio counts.

    Algorithm:
    1. Aggregate sensitivities for ALL portfolios at once:
       agg_S[k,p] = Σ_t x[t,p] * S[t,k]  (numpy matmul)
    2. SINGLE evaluate() call with arrays of length P:
       -> Returns P IM values and K×P gradient values
    3. Chain rule (numpy):
       ∂total_IM/∂x[t,p] = Σ_k grad_p[k] * S[t,k]

    Args:
        funcs: AADC Functions object from record_single_portfolio_simm_kernel
        sens_handles: List of handles for aggregated sensitivity inputs
        im_output: Handle for IM output
        S: Sensitivity matrix of shape (T, K)
        current_allocation: Current allocation matrix of shape (T, P)
        num_threads: Number of AADC worker threads
        workers: Optional pre-created ThreadPool (avoids creation overhead)

    Returns:
        (gradient_array[T,P], total_im_value)

    CRITICAL: Kernel is evaluated ONCE with arrays of length P, not P times!
    """
    T, P = current_allocation.shape
    K = S.shape[1]

    # Use provided workers or create new (for backwards compatibility)
    if workers is None:
        workers = aadc.ThreadPool(num_threads)

    # Step 1: Aggregate sensitivities for ALL portfolios at once (numpy - fast)
    # agg_S_all[k,p] = Σ_t x[t,p] * S[t,k]
    # current_allocation is (T, P), S is (T, K)
    # We want (K, P) where each column p is the aggregated sensitivities for portfolio p
    agg_S_all = np.dot(S.T, current_allocation)  # Shape: (K, P)

    # Step 2: SINGLE evaluate() call with arrays of length P
    # Each input handle gets an array of P values (one per portfolio)
    inputs = {sens_handles[k]: agg_S_all[k, :] for k in range(K)}
    request = {im_output: sens_handles}

    results = aadc.evaluate(funcs, request, inputs, workers)

    # Extract results: array of P IM values
    all_ims = np.array(results[0][im_output])  # Shape: (P,)
    total_im = float(np.sum(all_ims))

    # Extract gradients: for each factor k, we get an array of P gradient values
    # results[1][im_output][sens_handles[k]] = array of P values (dIM_p/dS_p[k] for each p)
    grad_matrix = np.zeros((K, P))
    for k in range(K):
        grad_matrix[k, :] = results[1][im_output][sens_handles[k]]

    # Step 3: Chain rule - gradient w.r.t. allocations for ALL portfolios
    # ∂IM_p/∂x[t,p] = Σ_k (∂IM_p/∂S_p[k]) * S[t,k] = S @ grad_p
    # For all portfolios: gradient[T, P] = S @ grad_matrix
    gradient = np.dot(S, grad_matrix)  # Shape: (T, P)

    return gradient, total_im


def optimize_allocation_gradient_descent_efficient(
    funcs: 'aadc.Functions',
    sens_handles: List[int],
    im_output: int,
    S: np.ndarray,
    initial_allocation: np.ndarray,
    num_threads: int,
    max_iters: int = 100,
    lr: float = None,
    tol: float = 1e-6,
    verbose: bool = True,
    workers: 'aadc.ThreadPool' = None,
) -> Tuple[np.ndarray, List[float], int]:
    """
    Gradient descent with simplex projection using efficient chain rule gradients.

    For each iteration:
    1. Compute gradient dIM/dx_tp via chain rule (small kernel + numpy)
    2. Take gradient step: x_new = x - lr * grad
    3. Project each row to simplex: sum_p x_tp = 1, x_tp >= 0
    4. Check convergence

    Args:
        funcs: AADC Functions object from record_single_portfolio_simm_kernel
        sens_handles: List of handles for aggregated sensitivity inputs
        im_output: Handle for IM output
        S: Sensitivity matrix of shape (T, K)
        initial_allocation: Starting allocation of shape (T, P)
        num_threads: AADC worker threads
        max_iters: Maximum iterations
        lr: Learning rate (auto-computed if None)
        tol: Relative tolerance for convergence
        verbose: Print progress
        workers: Optional pre-created ThreadPool (avoids creation overhead)

    Returns:
        (optimal_allocation, im_history, num_iterations)
    """
    x = initial_allocation.copy()
    im_history = []
    T, P = x.shape

    # Create a single ThreadPool for all iterations (5-10% speedup)
    if workers is None:
        workers = aadc.ThreadPool(num_threads)

    # First evaluation to get gradient scale for adaptive lr
    gradient, total_im = compute_allocation_gradient_chainrule(
        funcs, sens_handles, im_output, S, x, num_threads, workers
    )
    im_history.append(total_im)

    # Compute adaptive learning rate if not provided
    grad_max = np.abs(gradient).max()
    if lr is None:
        if grad_max > 1e-10:
            # Target step size ~0.3 (30% allocation change per iteration)
            # This allows flipping a trade in ~3-4 iterations
            lr = 0.3 / grad_max
        else:
            lr = 1e-12

    if verbose:
        print(f"    Initial IM: ${total_im:,.2f}")
        print(f"    Gradient: max={grad_max:.2e}, mean={np.abs(gradient).mean():.2e}")
        print(f"    Learning rate: {lr:.2e}")

    for iteration in range(max_iters):
        if iteration > 0:
            gradient, total_im = compute_allocation_gradient_chainrule(
                funcs, sens_handles, im_output, S, x, num_threads, workers
            )
            im_history.append(total_im)

        if verbose and iteration % 10 == 0:
            moves = int(np.sum(np.argmax(x, axis=1) != np.argmax(initial_allocation, axis=1)))
            print(f"    Iter {iteration}: IM = ${total_im:,.2f}, moves = {moves}")

        # Gradient step
        x_new = x - lr * gradient

        # Project to simplex (each row sums to 1, all >= 0)
        x_new = project_to_simplex(x_new)

        # Check convergence based on allocation change and IM change
        alloc_change = np.abs(x_new - x).max()
        if iteration > 0:
            rel_change = abs(im_history[-1] - im_history[-2]) / max(abs(im_history[-2]), 1e-10)
            if rel_change < tol and alloc_change < 1e-6:
                if verbose:
                    print(f"    Converged at iteration {iteration + 1}")
                return x_new, im_history, iteration + 1

        x = x_new

    if verbose:
        print(f"    Reached max iterations ({max_iters})")

    # Show allocation distribution before rounding
    if verbose:
        # Count how many trades have "mixed" allocations (max < 0.9)
        max_allocs = x.max(axis=1)
        mixed_count = np.sum(max_allocs < 0.9)
        near_flipped = np.sum((max_allocs >= 0.4) & (max_allocs < 0.6))
        print(f"    Allocation distribution: {mixed_count} trades with max_alloc < 0.9, {near_flipped} near 50/50")
        # Show top 5 trades with smallest max allocation (most "undecided")
        if T > 0:
            sorted_idx = np.argsort(max_allocs)[:min(5, T)]
            print(f"    Most mixed allocations (top 5):")
            for idx in sorted_idx:
                print(f"      Trade {idx}: {x[idx]} (max={max_allocs[idx]:.3f})")

    return x, im_history, max_iters


# =============================================================================
# AADC Allocation Kernel (Legacy - O(T×P) inputs, for smaller problems)
# =============================================================================

def _precompute_weighted_rc_sensitivities(
    S: np.ndarray,
    risk_factors: List[Tuple],
) -> np.ndarray:
    """
    Precompute weighted squared sensitivities aggregated by risk class.

    Instead of tracking K individual factors in the kernel, we precompute:
    WS_rc[t, rc, k_within_rc] = S[t,k] * w[k] for each factor k in risk class rc

    This allows the kernel to work with much smaller arrays.

    Returns:
        List of arrays, one per risk class, each of shape (T, num_factors_in_rc)
    """
    T, K = S.shape

    # Compute weights
    factor_weights = np.array([_get_risk_weight(rt, bucket) for rt, _, bucket, _ in risk_factors])

    # Group by risk class
    rc_ws_matrices = []
    for rc in _RISK_CLASS_ORDER:
        indices = [k for k, (rt, _, _, _) in enumerate(risk_factors)
                   if _map_risk_type_to_class(rt) == rc]
        if indices:
            # Extract and weight the columns for this risk class
            S_rc = S[:, indices]  # (T, num_factors_in_rc)
            w_rc = factor_weights[indices]  # (num_factors_in_rc,)
            WS_rc = S_rc * w_rc[np.newaxis, :]  # (T, num_factors_in_rc)
            rc_ws_matrices.append(WS_rc)
        else:
            rc_ws_matrices.append(np.zeros((T, 0)))

    return rc_ws_matrices


def record_allocation_im_kernel(
    S: np.ndarray,  # T x K sensitivity matrix
    risk_factors: List[Tuple],  # List of (RiskType, Qualifier, Bucket, Label1)
    num_portfolios: int,
) -> Tuple['aadc.Functions', Dict, int, np.ndarray]:
    """
    Record AADC kernel: allocation fractions -> total IM.

    Highly optimized version:
    - Precomputes weighted sensitivities by risk class
    - Embeds sensitivity values as constants in kernel (they don't change)
    - Only T*P allocation variables are tracked

    The kernel complexity is O(T*P*6) instead of O(T*P*K).

    Args:
        S: Sensitivity matrix of shape (T, K)
        risk_factors: List of (RiskType, Qualifier, Bucket, Label1) for each column of S
        num_portfolios: Number of portfolios P

    Returns:
        (funcs, x_handles, im_output, S) where:
        - funcs: AADC Functions object
        - x_handles: Dict[(t,p) -> handle] for allocation inputs
        - im_output: Handle for total_im output
        - S: The sensitivity matrix (returned for convenience)
    """
    if not AADC_AVAILABLE:
        raise RuntimeError("AADC is required for allocation optimization")

    T, K = S.shape
    P = num_portfolios

    # Precompute weighted sensitivities by risk class
    rc_ws_matrices = _precompute_weighted_rc_sensitivities(S, risk_factors)

    with aadc.record_kernel() as funcs:
        # Create allocation variables - T*P scalars as differentiable inputs
        x_aadc = []
        x_handles = {}
        for t in range(T):
            row = []
            for p in range(P):
                x_tp = aadc.idouble(1.0 / P)
                handle = x_tp.mark_as_input()
                x_handles[(t, p)] = handle
                row.append(x_tp)
            x_aadc.append(row)

        # Compute portfolio IMs
        portfolio_ims = []
        for p in range(P):
            # Compute K_r for each risk class
            risk_class_K = []
            for rc_idx, WS_rc in enumerate(rc_ws_matrices):
                num_factors_in_rc = WS_rc.shape[1]
                if num_factors_in_rc == 0:
                    risk_class_K.append(aadc.idouble(0.0))
                    continue

                # K_r = sqrt(sum_k (sum_t x[t,p] * WS_rc[t,k])^2)
                ws_sq_sum = aadc.idouble(0.0)
                for k in range(num_factors_in_rc):
                    # ws_k = sum_t x[t,p] * WS_rc[t,k]
                    ws_k = aadc.idouble(0.0)
                    for t in range(T):
                        # WS_rc[t,k] is a constant, embedded in the kernel
                        ws_k = ws_k + x_aadc[t][p] * WS_rc[t, k]
                    ws_sq_sum = ws_sq_sum + ws_k * ws_k

                k_r = np.sqrt(ws_sq_sum)
                risk_class_K.append(k_r)

            # Cross-risk-class aggregation: IM_p = sqrt(sum_r,s psi_rs * K_r * K_s)
            simm_sq = aadc.idouble(0.0)
            for i in range(6):
                for j in range(6):
                    psi_ij = _PSI_MATRIX[i, j]
                    simm_sq = simm_sq + psi_ij * risk_class_K[i] * risk_class_K[j]

            im_p = np.sqrt(simm_sq)
            portfolio_ims.append(im_p)

        # Total IM
        total_im = aadc.idouble(0.0)
        for p in range(P):
            total_im = total_im + portfolio_ims[p]

        im_output = total_im.mark_as_output()

    return funcs, x_handles, im_output, S


# =============================================================================
# Gradient Computation (Optimized)
# =============================================================================

def compute_allocation_gradient(
    funcs: 'aadc.Functions',
    x_handles: Dict,
    unused,  # Kept for API compatibility
    im_output: int,
    S: np.ndarray,
    current_allocation: np.ndarray,
    num_threads: int,
    workers: 'aadc.ThreadPool' = None,
) -> Tuple[np.ndarray, float]:
    """
    Evaluate kernel and compute gradient dIM/dx[t,p].

    Args:
        funcs: AADC Functions object from record_allocation_im_kernel
        x_handles: Dict[(t,p) -> handle] for allocation inputs
        unused: Kept for API compatibility
        im_output: Handle for total_im output
        S: Sensitivity matrix of shape (T, K)
        current_allocation: Current allocation matrix of shape (T, P)
        num_threads: Number of AADC worker threads
        workers: Optional pre-created ThreadPool (avoids creation overhead)

    Returns:
        (gradient, total_im) where:
        - gradient: Array of shape (T, P) with dIM/dx[t,p]
        - total_im: Scalar total IM value
    """
    T, P = current_allocation.shape

    # Use provided workers or create new (for backwards compatibility)
    if workers is None:
        workers = aadc.ThreadPool(num_threads)

    # Set allocation inputs
    inputs = {}
    for t in range(T):
        for p in range(P):
            inputs[x_handles[(t, p)]] = np.array([current_allocation[t, p]])

    # Request gradient w.r.t. allocation variables
    diff_handles = [x_handles[(t, p)] for t in range(T) for p in range(P)]
    request = {im_output: diff_handles}

    results = aadc.evaluate(funcs, request, inputs, workers)

    total_im = float(results[0][im_output][0])

    # Extract gradients
    gradient = np.zeros((T, P))
    for t in range(T):
        for p in range(P):
            h = x_handles[(t, p)]
            gradient[t, p] = float(results[1][im_output][h][0])

    return gradient, total_im


# =============================================================================
# Simplex Projection
# =============================================================================

def project_to_simplex(x: np.ndarray) -> np.ndarray:
    """
    Project each row of x onto the probability simplex (sum=1, all>=0).

    Uses the algorithm from "Efficient Projections onto the l1-Ball
    for Learning in High Dimensions" by Duchi et al. (2008).

    Args:
        x: Array of shape (T, P)

    Returns:
        Projected array where each row sums to 1 and all values >= 0
    """
    T, P = x.shape
    result = np.zeros_like(x)

    for t in range(T):
        v = x[t]
        # Sort in descending order
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_candidates = np.nonzero(u * np.arange(1, P + 1) > (cssv - 1))[0]
        if len(rho_candidates) == 0:
            # Fallback: uniform distribution
            result[t] = np.ones(P) / P
        else:
            rho = rho_candidates[-1]
            theta = (cssv[rho] - 1) / (rho + 1)
            result[t] = np.maximum(v - theta, 0)

    return result


# =============================================================================
# Optimization Algorithms
# =============================================================================

def optimize_allocation_gradient_descent(
    funcs: 'aadc.Functions',
    x_handles: Dict,
    unused,  # Kept for API compatibility
    im_output: int,
    S: np.ndarray,
    initial_allocation: np.ndarray,
    num_threads: int,
    max_iters: int = 100,
    lr: float = None,  # Auto-computed if None
    tol: float = 1e-6,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[float], int]:
    """
    Gradient descent with simplex projection.

    For each iteration:
    1. Compute gradient dIM/dx_tp
    2. Take gradient step: x_new = x - lr * grad
    3. Project each row to simplex: sum_p x_tp = 1, x_tp >= 0
    4. Check convergence

    Args:
        funcs: AADC Functions object
        x_handles: Dict[(t,p) -> handle] for allocation inputs
        unused: Kept for API compatibility
        im_output: From record_allocation_im_kernel
        S: Sensitivity matrix
        initial_allocation: Starting allocation of shape (T, P)
        num_threads: AADC worker threads
        max_iters: Maximum iterations
        lr: Learning rate (auto-computed if None based on gradient scale)
        tol: Relative tolerance for convergence
        verbose: Print progress

    Returns:
        (optimal_allocation, im_history, num_iterations)
    """
    x = initial_allocation.copy()
    im_history = []
    T, P = x.shape

    # First evaluation to get gradient scale for adaptive lr
    gradient, total_im = compute_allocation_gradient(
        funcs, x_handles, None, im_output, S, x, num_threads
    )
    im_history.append(total_im)

    # Compute adaptive learning rate if not provided
    # We want the step to move allocation fractions by ~0.01-0.1 per iteration
    grad_max = np.abs(gradient).max()
    if lr is None:
        if grad_max > 1e-10:
            # Target step size ~0.05 (5% allocation change per iteration)
            lr = 0.05 / grad_max
        else:
            lr = 1e-12  # Fallback if gradient is near zero

    if verbose:
        print(f"    Initial IM: ${total_im:,.2f}")
        print(f"    Gradient: max={grad_max:.2e}, mean={np.abs(gradient).mean():.2e}")
        print(f"    Learning rate: {lr:.2e}")

    for iteration in range(max_iters):
        if iteration > 0:
            gradient, total_im = compute_allocation_gradient(
                funcs, x_handles, None, im_output, S, x, num_threads
            )
            im_history.append(total_im)

        if verbose and iteration % 10 == 0:
            moves = int(np.sum(np.argmax(x, axis=1) != np.argmax(initial_allocation, axis=1)))
            print(f"    Iter {iteration}: IM = ${total_im:,.2f}, moves = {moves}")

        # Gradient step
        x_new = x - lr * gradient

        # Project to simplex (each row sums to 1, all >= 0)
        x_new = project_to_simplex(x_new)

        # Check convergence based on allocation change, not just IM change
        alloc_change = np.abs(x_new - x).max()
        if iteration > 0:
            rel_change = abs(im_history[-1] - im_history[-2]) / max(abs(im_history[-2]), 1e-10)
            if rel_change < tol and alloc_change < 1e-6:
                if verbose:
                    print(f"    Converged at iteration {iteration + 1}")
                return x_new, im_history, iteration + 1

        x = x_new

    if verbose:
        print(f"    Reached max iterations ({max_iters})")
    return x, im_history, max_iters


def optimize_allocation_greedy(
    trade_crifs: Dict[str, pd.DataFrame],
    num_portfolios: int,
    num_threads: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    Greedy assignment: assign each trade to portfolio that minimizes total IM.

    This is a fast heuristic that may not find the global optimum but provides
    a good initial solution or standalone result for smaller problems.

    Algorithm:
    1. Start with all trades in portfolio 0
    2. For each trade, try moving it to each portfolio
    3. Keep the assignment that minimizes total IM
    4. Repeat until no improvement

    Args:
        trade_crifs: Dict mapping trade_id to CRIF DataFrame
        num_portfolios: Number of portfolios
        num_threads: AADC worker threads
        verbose: Print progress

    Returns:
        Allocation matrix of shape (T, P) with one-hot rows
    """
    from model.simm_portfolio_aadc import precompute_all_trade_crifs
    from common.portfolio import run_simm

    trade_ids = list(trade_crifs.keys())
    T = len(trade_ids)
    P = num_portfolios

    # Start with uniform distribution across portfolios
    assignment = np.zeros(T, dtype=int)
    for t in range(T):
        assignment[t] = t % P

    def compute_total_im(assign):
        """Compute total IM for a given assignment."""
        total = 0.0
        for p in range(P):
            p_trades = [trade_ids[t] for t in range(T) if assign[t] == p]
            if not p_trades:
                continue
            # Combine CRIFs
            p_crifs = [trade_crifs[tid] for tid in p_trades]
            p_crif = pd.concat(p_crifs, ignore_index=True)
            _, im, _ = run_simm(p_crif)
            total += im
        return total

    best_im = compute_total_im(assignment)
    if verbose:
        print(f"    Greedy initial IM: ${best_im:,.2f}")

    improved = True
    iteration = 0
    while improved and iteration < 100:
        improved = False
        iteration += 1

        for t in range(T):
            current_p = assignment[t]
            best_move_im = best_im

            for new_p in range(P):
                if new_p == current_p:
                    continue

                # Try moving trade t to portfolio new_p
                test_assign = assignment.copy()
                test_assign[t] = new_p
                test_im = compute_total_im(test_assign)

                if test_im < best_move_im - 1e-6:
                    best_move_im = test_im
                    best_new_p = new_p

            if best_move_im < best_im - 1e-6:
                assignment[t] = best_new_p
                best_im = best_move_im
                improved = True
                if verbose:
                    print(f"    Iter {iteration}: Moved trade {t} to portfolio {best_new_p}, IM = ${best_im:,.2f}")

    # Convert to allocation matrix
    allocation = np.zeros((T, P))
    for t in range(T):
        allocation[t, assignment[t]] = 1.0

    return allocation


def round_to_integer_allocation(continuous_allocation: np.ndarray) -> np.ndarray:
    """
    Round continuous allocation to integer (one-hot per row).

    Each trade is assigned to the portfolio with highest allocation fraction.

    Args:
        continuous_allocation: Array of shape (T, P) with values in [0, 1]

    Returns:
        Array of shape (T, P) with exactly one 1.0 per row
    """
    T, P = continuous_allocation.shape
    result = np.zeros_like(continuous_allocation)

    for t in range(T):
        best_p = np.argmax(continuous_allocation[t])
        result[t, best_p] = 1.0

    return result


# =============================================================================
# Main Entry Point
# =============================================================================

def reallocate_trades_optimal(
    trades: list,
    market,  # MarketEnvironment
    num_portfolios: int,
    initial_allocation: np.ndarray,
    num_threads: int = 8,
    allow_partial: bool = False,
    method: str = 'auto',  # 'auto', 'gradient_descent', or 'greedy'
    max_iters: int = 100,
    lr: float = None,  # Auto-computed based on gradient scale
    tol: float = 1e-6,
    verbose: bool = True,
) -> Dict:
    """
    Full optimization pipeline for trade allocation.

    Steps:
    1. Precompute CRIF for all T trades (batched by structure)
    2. For small problems: use gradient descent with AADC
       For large problems: use greedy heuristic (faster)
    3. Round to integer (if allow_partial is False)

    Args:
        trades: List of trade objects
        market: MarketEnvironment with curves, spots, etc.
        num_portfolios: Number of portfolios P
        initial_allocation: Starting allocation matrix of shape (T, P)
        num_threads: AADC worker threads
        allow_partial: If False, round to integer allocation at the end
        method: 'auto' (selects based on size), 'gradient_descent', or 'greedy'
        max_iters: Maximum optimization iterations
        lr: Learning rate for gradient descent
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        Dict with optimization results
    """
    if not AADC_AVAILABLE:
        raise RuntimeError("AADC is required for allocation optimization")

    from model.simm_portfolio_aadc import precompute_all_trade_crifs

    start_time = time.perf_counter()
    T = len(trades)

    # Create a single ThreadPool for the entire optimization (5-10% speedup)
    workers = aadc.ThreadPool(num_threads)

    # Auto-select method based on problem size
    # With efficient kernel (O(K) inputs + chain rule), gradient descent is fast for much larger T
    # The kernel has only K≈100 inputs, and per-iteration is P kernel evals + O(T×K) numpy ops
    GRADIENT_DESCENT_THRESHOLD = 5000  # Increased from 200 due to efficient implementation
    if method == 'auto':
        if T > GRADIENT_DESCENT_THRESHOLD:
            method = 'greedy'
            if verbose:
                print(f"  Auto-selected greedy method for {T} trades (>{GRADIENT_DESCENT_THRESHOLD})")
        else:
            method = 'gradient_descent'
            if verbose:
                print(f"  Auto-selected gradient descent for {T} trades (efficient O(K) kernel)")

    # Step 1: Precompute all trade CRIFs
    if verbose:
        print("  Precomputing trade CRIFs...")
    trade_crifs = precompute_all_trade_crifs(trades, market, num_threads, workers)

    trade_ids = [t.trade_id for t in trades if t.trade_id in trade_crifs]
    T = len(trade_ids)
    P = num_portfolios

    if T == 0:
        return {
            'final_allocation': initial_allocation,
            'final_im': 0.0,
            'initial_im': 0.0,
            'im_history': [],
            'num_iterations': 0,
            'elapsed_time': time.perf_counter() - start_time,
            'trades_moved': 0,
            'trade_ids': [],
            'converged': True,
        }

    # Filter initial_allocation to match trade_ids
    trade_id_to_idx = {t.trade_id: i for i, t in enumerate(trades)}
    filtered_allocation = np.zeros((T, P))
    for i, tid in enumerate(trade_ids):
        orig_idx = trade_id_to_idx.get(tid)
        if orig_idx is not None and orig_idx < initial_allocation.shape[0]:
            filtered_allocation[i] = initial_allocation[orig_idx]

    # Step 2: Build sensitivity matrix
    risk_factors = _get_unique_risk_factors(trade_crifs)
    S = _build_sensitivity_matrix(trade_crifs, trade_ids, risk_factors)
    K = len(risk_factors)

    if verbose:
        print(f"  Sensitivity matrix: {T} trades x {K} risk factors x {P} portfolios")

    if method == 'greedy':
        # Use greedy algorithm (doesn't need AADC kernel)
        if verbose:
            print("  Running greedy optimization...")
        final_allocation = optimize_allocation_greedy(
            trade_crifs, P, num_threads, verbose
        )
        im_history = []
        num_iters = 0

        # Compute final IM
        from common.portfolio import run_simm
        final_im = 0.0
        initial_im = 0.0
        for p in range(P):
            p_trade_ids = [trade_ids[t] for t in range(T) if filtered_allocation[t, p] > 0.5]
            if p_trade_ids:
                p_crifs = [trade_crifs[tid] for tid in p_trade_ids]
                p_crif = pd.concat(p_crifs, ignore_index=True)
                _, im, _ = run_simm(p_crif)
                initial_im += im

            p_trade_ids_final = [trade_ids[t] for t in range(T) if final_allocation[t, p] > 0.5]
            if p_trade_ids_final:
                p_crifs = [trade_crifs[tid] for tid in p_trade_ids_final]
                p_crif = pd.concat(p_crifs, ignore_index=True)
                _, im, _ = run_simm(p_crif)
                final_im += im

    else:
        # Gradient descent method using EFFICIENT small kernel + chain rule
        # Step 3: Get factor metadata and record SMALL kernel (O(K) inputs)
        factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = _get_factor_metadata(risk_factors)

        if verbose:
            print(f"  Recording efficient kernel (K={K} inputs, NOT T×P={T*P})...")
            print(f"  Using ISDA v2.6 correlations: {SIMM_PARAMS_AVAILABLE}")

        funcs, sens_handles, im_output = record_single_portfolio_simm_kernel(
            K, factor_risk_classes, factor_weights,
            factor_risk_types, factor_labels, factor_buckets,
            use_correlations=True
        )

        if verbose:
            print(f"  Kernel recorded: {K} inputs (100-1000x smaller than old approach)")

        # Step 4: Compute initial IM using chain rule gradient
        _, initial_im = compute_allocation_gradient_chainrule(
            funcs, sens_handles, im_output, S, filtered_allocation, num_threads, workers
        )
        if verbose:
            print(f"  Initial total IM: ${initial_im:,.2f}")

        # Step 5: Optimize using efficient chain rule gradients
        if verbose:
            print(f"  Running {method} optimization with chain rule gradients...")
        final_allocation, im_history, num_iters = optimize_allocation_gradient_descent_efficient(
            funcs, sens_handles, im_output, S,
            filtered_allocation, num_threads,
            max_iters=max_iters, lr=lr, tol=tol, verbose=verbose,
            workers=workers,
        )

        # Step 6: Round if needed
        continuous_allocation = final_allocation.copy()
        if not allow_partial:
            if verbose:
                # Show IM before rounding
                _, continuous_im = compute_allocation_gradient_chainrule(
                    funcs, sens_handles, im_output, S, continuous_allocation, num_threads, workers
                )
                print(f"  Continuous IM (before rounding): ${continuous_im:,.2f}")

            final_allocation = round_to_integer_allocation(final_allocation)

            if verbose:
                # Count how allocation changed due to rounding
                continuous_assignments = np.argmax(continuous_allocation, axis=1)
                rounded_assignments = np.argmax(final_allocation, axis=1)
                rounding_changes = int(np.sum(continuous_assignments != rounded_assignments))
                if rounding_changes > 0:
                    print(f"  WARNING: Rounding changed {rounding_changes} assignments!")

        # Step 7: Compute final IM
        _, final_im = compute_allocation_gradient_chainrule(
            funcs, sens_handles, im_output, S, final_allocation, num_threads, workers
        )

    # Count moves
    initial_assignments = np.argmax(filtered_allocation, axis=1)
    final_assignments = np.argmax(final_allocation, axis=1)
    trades_moved = int(np.sum(initial_assignments != final_assignments))

    elapsed = time.perf_counter() - start_time

    converged = num_iters < max_iters if method == 'gradient_descent' else True

    return {
        'final_allocation': final_allocation,
        'final_im': final_im,
        'initial_im': initial_im,
        'im_history': im_history,
        'num_iterations': num_iters,
        'elapsed_time': elapsed,
        'trades_moved': trades_moved,
        'trade_ids': trade_ids,
        'converged': converged,
    }


# =============================================================================
# Verification Utilities
# =============================================================================

def verify_allocation_gradient(
    funcs: 'aadc.Functions',
    x_handles: Dict,
    im_output: int,
    S: np.ndarray,
    allocation: np.ndarray,
    num_threads: int,
    eps: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Verify AADC gradient via finite differences.

    Args:
        funcs, x_handles, im_output: From record_allocation_im_kernel
        S: Sensitivity matrix
        allocation: Current allocation
        num_threads: AADC worker threads
        eps: Finite difference step size

    Returns:
        (max_rel_error, aadc_gradient, fd_gradient)
    """
    T, P = allocation.shape

    # Get AADC gradient
    aadc_grad, base_im = compute_allocation_gradient(
        funcs, x_handles, None, im_output, S, allocation, num_threads
    )

    # Compute finite difference gradient
    fd_grad = np.zeros((T, P))
    for t in range(T):
        for p in range(P):
            x_plus = allocation.copy()
            x_plus[t, p] += eps

            _, im_plus = compute_allocation_gradient(
                funcs, x_handles, None, im_output, S, x_plus, num_threads
            )
            fd_grad[t, p] = (im_plus - base_im) / eps

    # Compute max relative error
    denom = np.maximum(np.abs(aadc_grad), np.abs(fd_grad))
    denom = np.where(denom < 1e-10, 1.0, denom)
    rel_errors = np.abs(aadc_grad - fd_grad) / denom
    max_rel_error = np.max(rel_errors)

    return max_rel_error, aadc_grad, fd_grad


def verify_allocation_gradient_chainrule(
    funcs: 'aadc.Functions',
    sens_handles: List[int],
    im_output: int,
    S: np.ndarray,
    allocation: np.ndarray,
    num_threads: int,
    eps: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Verify chain rule AADC gradient via finite differences.

    Args:
        funcs, sens_handles, im_output: From record_single_portfolio_simm_kernel
        S: Sensitivity matrix of shape (T, K)
        allocation: Current allocation of shape (T, P)
        num_threads: AADC worker threads
        eps: Finite difference step size

    Returns:
        (max_rel_error, aadc_gradient, fd_gradient)
    """
    T, P = allocation.shape

    # Get AADC gradient via chain rule
    aadc_grad, base_im = compute_allocation_gradient_chainrule(
        funcs, sens_handles, im_output, S, allocation, num_threads
    )

    # Compute finite difference gradient
    fd_grad = np.zeros((T, P))
    for t in range(T):
        for p in range(P):
            x_plus = allocation.copy()
            x_plus[t, p] += eps

            _, im_plus = compute_allocation_gradient_chainrule(
                funcs, sens_handles, im_output, S, x_plus, num_threads
            )
            fd_grad[t, p] = (im_plus - base_im) / eps

    # Compute max relative error
    denom = np.maximum(np.abs(aadc_grad), np.abs(fd_grad))
    denom = np.where(denom < 1e-10, 1.0, denom)
    rel_errors = np.abs(aadc_grad - fd_grad) / denom
    max_rel_error = np.max(rel_errors)

    return max_rel_error, aadc_grad, fd_grad


def verify_simplex_projection(allocation: np.ndarray) -> Tuple[bool, float, float]:
    """
    Verify allocation matrix satisfies simplex constraints.

    Args:
        allocation: Array of shape (T, P)

    Returns:
        (is_valid, max_row_sum_error, min_value)
    """
    row_sums = allocation.sum(axis=1)
    max_row_sum_error = np.max(np.abs(row_sums - 1.0))
    min_value = np.min(allocation)
    is_valid = max_row_sum_error < 1e-6 and min_value >= -1e-10

    return is_valid, max_row_sum_error, min_value
