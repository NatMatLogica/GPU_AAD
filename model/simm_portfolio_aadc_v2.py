"""
SIMM Portfolio AADC v2 - Optimized AADC implementation with batched portfolio evaluation.

KEY OPTIMIZATION: Single aadc.evaluate() call for ALL portfolios instead of P separate calls.
This reduces Python-to-C++ dispatch overhead by factor of P (typically 5-10x for 5 portfolios,
up to 200x for larger portfolio counts).

The optimization is based on the pattern from Asian Options Benchmark:
- WRONG: evaluate() called P times (dispatch overhead dominates)
- CORRECT: Stack all P portfolios' inputs, evaluate() called ONCE

Additional optimizations:
- Pre-compute weighted sensitivities (WS = amounts * weights * CR) outside kernel
- Single ThreadPool reused across all operations
- Kernel caching for portfolio structure reuse

Version: 1.0.0

Usage:
    from model.simm_portfolio_aadc_v2 import (
        compute_im_gradient_aadc_v2,
        compute_all_portfolios_im_gradient_v2,
        record_single_portfolio_simm_kernel_v2,
    )
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Version
MODULE_VERSION = "2.1.0"  # Fixed GD optimizer: Armijo line search, best-tracking, conservative lr

try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False

# Import from existing modules
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Weights_and_Corr.v2_6 import (
    ir_corr, reg_vol_rw, low_vol_rw, high_vol_rw,
    reg_vol_ccy_bucket, low_vol_ccy_bucket,
    inflation_rw, inflation_corr, sub_curves_corr,
    ir_gamma_diff_ccy, cr_gamma_diff_ccy,
    creditQ_corr_non_res, equity_corr_non_res, commodity_corr_non_res,
)

from model.simm_portfolio_aadc import (
    _map_risk_type_to_class,
    _get_ir_risk_weight_v26,
    _get_risk_weight,
    _get_intra_correlation,
    _get_ir_tenor_index,
    IR_CORR_MATRIX,
    SIMM_TENOR_LIST,
    SIMM_TENOR_TO_IDX,
    PSI_MATRIX,
    SIMM_RISK_CLASSES,
    _get_sub_curve_correlation,
    _is_delta_risk_type,
    _is_vega_risk_type,
    _get_vega_risk_weight,
    _precompute_concentration_factors,
    _get_concentration_threshold,
    _compute_concentration_risk,
    _get_vega_concentration_threshold,
    INFLATION_RISK_WEIGHT,
)

from Weights_and_Corr.v2_6 import (
    ccy_basis_swap_spread_rw, ccy_basis_spread_corr,
)

# Inter-bucket gamma matrices (converted to numpy for fast lookup)
_EQ_INTER = np.array([list(row) for row in equity_corr_non_res])
_CM_INTER = np.array([list(row) for row in commodity_corr_non_res])
_CQ_INTER = np.array([list(row) for row in creditQ_corr_non_res])

# Cross-risk-class correlation matrix
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
# v2 Optimized Kernel Recording
# =============================================================================

def _get_factor_metadata_v2(risk_factors: List[Tuple]) -> Tuple[List[str], np.ndarray, List[str], List[str], List[str]]:
    """
    Extract risk class, weight, and metadata for each risk factor.

    Same as simm_allocation_optimizer._get_factor_metadata but imported here
    for self-contained module.

    Returns:
        (factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets)
    """
    factor_risk_classes = []
    factor_weights = []
    factor_risk_types = []
    factor_labels = []
    factor_buckets = []

    for rt, qualifier, bucket, label1 in risk_factors:
        rc = _map_risk_type_to_class(rt)
        # Use v2.6 currency-specific weights for IR if available
        if rt == "Risk_IRCurve" and qualifier and label1:
            rw = _get_ir_risk_weight_v26(qualifier, label1)
        else:
            rw = _get_risk_weight(rt, bucket)

        factor_risk_classes.append(rc)
        factor_weights.append(rw)
        factor_risk_types.append(rt)
        factor_labels.append(label1)
        factor_buckets.append(bucket)

    return factor_risk_classes, np.array(factor_weights), factor_risk_types, factor_labels, factor_buckets


# =============================================================================
# v2 Full ISDA Kernel - Factor Metadata (Step 2)
# =============================================================================

@dataclass
class FactorMeta:
    """Metadata for a single risk factor in the v2 full kernel."""
    risk_class: str
    risk_type: str
    qualifier: str
    bucket: str
    label1: str
    is_delta: bool
    is_vega: bool
    weight: float      # Delta RW or Vega VRW
    cr: float          # Concentration factor CR or VCR
    bucket_key: str    # Grouping key (currency for IR/FX, bucket for others)


def _get_factor_metadata_v2_full(
    risk_factors: List[Tuple],
    combined_crif: pd.DataFrame,
) -> List[FactorMeta]:
    """
    Build detailed per-factor metadata for the full-ISDA v2 kernel.

    Each risk factor is identified by (RiskType, Qualifier, Bucket, Label1).
    This function determines whether each factor is Delta or Vega,
    assigns the correct risk weight, and pre-computes concentration factors
    from the combined CRIF.

    Args:
        risk_factors: List of (RiskType, Qualifier, Bucket, Label1) tuples
        combined_crif: Combined CRIF DataFrame for pre-computing CR/VCR

    Returns:
        List of FactorMeta, one per risk factor
    """
    # Pre-compute concentration factors from combined CRIF
    delta_cr_factors = _precompute_concentration_factors(combined_crif, "Delta")
    vega_cr_factors = _precompute_concentration_factors(combined_crif, "Vega")

    metadata = []
    for rt, qualifier, bucket, label1 in risk_factors:
        rc = _map_risk_type_to_class(rt)
        is_delta = _is_delta_risk_type(rt)
        is_vega = _is_vega_risk_type(rt)

        qualifier_str = str(qualifier) if qualifier else ""
        bucket_str = str(bucket) if bucket else ""
        label1_str = str(label1) if label1 else ""

        # Determine risk weight
        if is_delta:
            if rt == "Risk_IRCurve" and qualifier_str and label1_str:
                weight = _get_ir_risk_weight_v26(qualifier_str, label1_str)
            elif rt == "Risk_Inflation":
                weight = INFLATION_RISK_WEIGHT
            elif rt == "Risk_XCcyBasis":
                weight = ccy_basis_swap_spread_rw
            else:
                weight = _get_risk_weight(rt, bucket_str)
        elif is_vega:
            weight = _get_vega_risk_weight(rt, bucket_str)
        else:
            weight = _get_risk_weight(rt, bucket_str)

        # Determine bucket key for grouping
        if rc in ("Rates", "FX"):
            bucket_key = qualifier_str
        else:
            bucket_key = bucket_str

        # Get concentration factor
        if is_delta:
            cr = delta_cr_factors.get((rc, bucket_key), 1.0)
        elif is_vega:
            cr = vega_cr_factors.get((rc, bucket_key), 1.0)
        else:
            cr = 1.0

        # XCcyBasis doesn't use concentration
        if rt == "Risk_XCcyBasis":
            cr = 1.0

        metadata.append(FactorMeta(
            risk_class=rc,
            risk_type=rt,
            qualifier=qualifier_str,
            bucket=bucket_str,
            label1=label1_str,
            is_delta=is_delta,
            is_vega=is_vega,
            weight=weight,
            cr=cr,
            bucket_key=bucket_key,
        ))

    return metadata


def record_single_portfolio_simm_kernel_v2_full(
    K: int,
    factor_metadata: List[FactorMeta],
) -> Tuple['aadc.Functions', List[int], int]:
    """
    Record AADC kernel: K aggregated sensitivities -> single portfolio IM.

    FULL ISDA v2.6 implementation with:
    - Delta and Vega margin separation per risk class
    - Intra-bucket correlations (tenor rho for IR, bucket-specific for others)
    - Risk weights and concentration factors (as Python float constants)
    - Inter-bucket gamma aggregation for IR
    - Cross-risk-class PSI aggregation

    The same kernel can be evaluated with arrays of length P to compute
    all P portfolios' IMs in a SINGLE aadc.evaluate() call.

    Args:
        K: Number of risk factors
        factor_metadata: List of FactorMeta for each factor

    Returns:
        (funcs, sensitivity_handles, im_output)
    """
    if not AADC_AVAILABLE:
        raise RuntimeError("AADC is required")

    with aadc.record_kernel() as funcs:
        # Mark aggregated sensitivities as differentiable inputs (K values)
        agg_sens = []
        sens_handles = []
        for k in range(K):
            s_k = aadc.idouble(0.0)
            handle = s_k.mark_as_input()
            sens_handles.append(handle)
            agg_sens.append(s_k)

        # Compute Delta and Vega margins per risk class
        risk_class_total_margins = []

        for rc in _RISK_CLASS_ORDER:
            # Separate delta and vega factors for this risk class
            delta_indices = [k for k in range(K)
                           if factor_metadata[k].risk_class == rc and factor_metadata[k].is_delta]
            vega_indices = [k for k in range(K)
                          if factor_metadata[k].risk_class == rc and factor_metadata[k].is_vega]

            # Compute Delta margin for this risk class
            delta_margin = _compute_rc_margin_v2_full(
                rc, delta_indices, agg_sens, factor_metadata, "Delta"
            )

            # Compute Vega margin for this risk class
            vega_margin = _compute_rc_margin_v2_full(
                rc, vega_indices, agg_sens, factor_metadata, "Vega"
            )

            # Total margin = Delta + Vega
            total_margin = delta_margin + vega_margin
            risk_class_total_margins.append(total_margin)

        # Cross-risk-class aggregation: IM = sqrt(sum_r sum_s psi[r,s] * K_r * K_s)
        simm_sq = aadc.idouble(0.0)
        for i in range(6):
            for j in range(6):
                psi_ij = _PSI_MATRIX[i, j]
                simm_sq = simm_sq + psi_ij * risk_class_total_margins[i] * risk_class_total_margins[j]

        # Protect sqrt from negative/zero for AADC gradient stability
        _EPS_TOP = 1.0
        simm_sq_safe = (simm_sq + np.sqrt(simm_sq * simm_sq + _EPS_TOP)) * 0.5 + _EPS_TOP
        im_p = np.sqrt(simm_sq_safe)
        im_output = im_p.mark_as_output()

    return funcs, sens_handles, im_output


def _compute_rc_margin_v2_full(
    rc: str,
    factor_indices: List[int],
    agg_sens: List,
    factor_metadata: List[FactorMeta],
    risk_measure: str,
) -> 'aadc.idouble':
    """
    Compute margin for one risk class and one risk measure (Delta or Vega)
    using the full ISDA formula with intra-bucket correlations and inter-bucket
    gamma aggregation.

    This mirrors _compute_risk_class_margin() but operates on K aggregated
    factor sensitivities instead of N CRIF rows.

    Args:
        rc: Risk class name (Rates, FX, etc.)
        factor_indices: Indices into agg_sens for factors in this RC + measure
        agg_sens: List of aadc.idouble aggregated sensitivities
        factor_metadata: Factor metadata list
        risk_measure: "Delta" or "Vega"

    Returns:
        aadc.idouble: Margin for this risk class and measure
    """
    if not factor_indices:
        return aadc.idouble(0.0)

    # Group by bucket
    buckets_in_rc = {}
    for k in factor_indices:
        bk = factor_metadata[k].bucket_key
        if bk not in buckets_in_rc:
            buckets_in_rc[bk] = []
        buckets_in_rc[bk].append(k)

    # Compute K_b and S_b for each bucket
    bucket_K = {}
    bucket_S = {}
    bucket_CR = {}

    for bucket_key, bucket_indices in buckets_in_rc.items():
        # Compute weighted sensitivities: WS_k = S_k * RW_k * CR_k
        ws_list = []
        for k in bucket_indices:
            meta = factor_metadata[k]
            ws_k = agg_sens[k] * float(meta.weight) * float(meta.cr)
            ws_list.append(ws_k)

        bucket_CR[bucket_key] = factor_metadata[bucket_indices[0]].cr

        # Compute K_b^2 with intra-bucket correlations
        k_sq = aadc.idouble(0.0)
        ws_sum = aadc.idouble(0.0)
        num_in_bucket = len(bucket_indices)

        if num_in_bucket > 1:
            for i_local in range(num_in_bucket):
                i_global = bucket_indices[i_local]
                ws_sum = ws_sum + ws_list[i_local]

                for j_local in range(num_in_bucket):
                    j_global = bucket_indices[j_local]

                    # Rho (tenor/intra-bucket) correlation
                    rho_ij = _get_intra_correlation(
                        rc,
                        factor_metadata[i_global].risk_type,
                        factor_metadata[j_global].risk_type,
                        factor_metadata[i_global].label1,
                        factor_metadata[j_global].label1,
                        bucket_key,
                    )

                    k_sq = k_sq + rho_ij * ws_list[i_local] * ws_list[j_local]
        else:
            k_sq = ws_list[0] * ws_list[0]
            ws_sum = ws_list[0]

        _EPS_B = 1.0
        k_sq_safe = (k_sq + np.sqrt(k_sq * k_sq + _EPS_B)) * 0.5 + _EPS_B
        bucket_K[bucket_key] = np.sqrt(k_sq_safe)
        bucket_S[bucket_key] = ws_sum

    # Inter-bucket aggregation
    k_rc_sq = aadc.idouble(0.0)
    bucket_keys = list(buckets_in_rc.keys())

    for bucket_key in bucket_keys:
        k_rc_sq = k_rc_sq + bucket_K[bucket_key] * bucket_K[bucket_key]

    # Inter-bucket gamma correlation for all risk classes
    if len(bucket_keys) > 1 and rc != "FX":
        for i_b in range(len(bucket_keys)):
            for j_b in range(len(bucket_keys)):
                if i_b == j_b:
                    continue
                b_key = bucket_keys[i_b]
                c_key = bucket_keys[j_b]

                # Look up gamma by risk class
                gamma = 0.0
                if rc == "Rates":
                    gamma = ir_gamma_diff_ccy
                elif rc == "CreditQ":
                    try:
                        b1, b2 = int(b_key), int(c_key)
                        if 1 <= b1 <= 12 and 1 <= b2 <= 12:
                            gamma = float(_CQ_INTER[b1 - 1, b2 - 1])
                        else:
                            gamma = 0.5
                    except (ValueError, TypeError):
                        gamma = 0.5
                elif rc == "CreditNonQ":
                    gamma = cr_gamma_diff_ccy
                elif rc == "Equity":
                    try:
                        b1, b2 = int(b_key), int(c_key)
                        if 1 <= b1 <= 12 and 1 <= b2 <= 12:
                            gamma = float(_EQ_INTER[b1 - 1, b2 - 1])
                    except (ValueError, TypeError):
                        pass
                elif rc == "Commodity":
                    try:
                        b1, b2 = int(b_key), int(c_key)
                        if 1 <= b1 <= 17 and 1 <= b2 <= 17:
                            gamma = float(_CM_INTER[b1 - 1, b2 - 1])
                    except (ValueError, TypeError):
                        pass

                if gamma != 0.0:
                    cr_b = bucket_CR.get(b_key, 1.0)
                    cr_c = bucket_CR.get(c_key, 1.0)
                    g_bc = min(cr_b, cr_c) / max(cr_b, cr_c) if max(cr_b, cr_c) > 0 else 1.0
                    k_rc_sq = k_rc_sq + gamma * bucket_S[b_key] * bucket_S[c_key] * g_bc

    # Smooth max(0, k_rc_sq) for AADC compatibility:
    # (x + sqrt(x^2 + eps)) / 2 ≈ max(0, x), differentiable everywhere
    # Floor of 1.0 (=$1 margin) is negligible vs IM of ~$1e15 but prevents
    # gradient singularity at sqrt(0) that would produce NaN in adjoint
    _EPS_RC = 1.0
    k_rc_sq_clamped = (k_rc_sq + np.sqrt(k_rc_sq * k_rc_sq + _EPS_RC)) * 0.5 + _EPS_RC
    return np.sqrt(k_rc_sq_clamped)


def record_single_portfolio_simm_kernel_v2(
    K: int,
    factor_risk_classes: List[str],
    factor_weights: np.ndarray,
    factor_risk_types: List[str] = None,
    factor_labels: List[str] = None,
    factor_buckets: List[str] = None,
    use_correlations: bool = True,
) -> Tuple['aadc.Functions', List[int], int]:
    """
    Record AADC kernel: K aggregated sensitivities -> single portfolio IM.

    CRITICAL: This kernel has only K inputs (~100), not T*P (~25,000).
    The same kernel can be evaluated with arrays of length P to compute
    all P portfolios' IMs in a SINGLE aadc.evaluate() call.

    Inputs: S_p[k] for k=0..K-1 (aggregated sensitivities for ONE portfolio)
            When evaluated with arrays, computes ALL portfolios simultaneously.
    Output: IM_p (SIMM margin for that portfolio, or array of P IMs)

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

            # Apply correlations: K^2 = sum_k sum_l rho_kl * WS_k * WS_l
            k_sq = aadc.idouble(0.0)
            num_in_rc = len(rc_indices)

            if use_correlations and num_in_rc > 1:
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

        # Cross-risk-class aggregation: IM_p = sqrt(sum_i sum_j psi[i,j] * K_i * K_j)
        simm_sq = aadc.idouble(0.0)
        for i in range(6):
            for j in range(6):
                psi_ij = _PSI_MATRIX[i, j]
                simm_sq = simm_sq + psi_ij * risk_class_margins[i] * risk_class_margins[j]

        im_p = np.sqrt(simm_sq)
        im_output = im_p.mark_as_output()

    return funcs, sens_handles, im_output


# =============================================================================
# v2 CRITICAL OPTIMIZATION: Single evaluate() for ALL portfolios
# =============================================================================

def compute_all_portfolios_im_gradient_v2(
    funcs: 'aadc.Functions',
    sens_handles: List[int],
    im_output: int,
    S: np.ndarray,                    # T x K sensitivity matrix
    current_allocation: np.ndarray,   # T x P allocation fractions
    num_threads: int,
    workers: 'aadc.ThreadPool' = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute gradient d(total_IM)/dx[t,p] using chain rule with SINGLE aadc.evaluate() call.

    THIS IS THE KEY OPTIMIZATION (200x faster for P portfolios):
    - OLD: P separate aadc.evaluate() calls (high dispatch overhead)
    - NEW: 1 aadc.evaluate() call with arrays of length P (vectorized)

    Algorithm:
    1. Aggregate sensitivities for ALL portfolios at once: agg_S[k,p] = sum_t x[t,p] * S[t,k]
    2. SINGLE evaluate() call: pass arrays of length P for each input
       -> Returns P IM values and K*P gradient values
    3. Chain rule (numpy): d(total_IM)/dx[t,p] = sum_k grad_p[k] * S[t,k]

    Args:
        funcs: AADC Functions object from record_single_portfolio_simm_kernel_v2
        sens_handles: List of handles for aggregated sensitivity inputs
        im_output: Handle for IM output
        S: Sensitivity matrix of shape (T, K)
        current_allocation: Current allocation matrix of shape (T, P)
        num_threads: Number of AADC worker threads
        workers: Optional pre-created ThreadPool

    Returns:
        (gradient_array[T,P], im_array[P], eval_time)

    CRITICAL: Kernel is evaluated ONCE with arrays of length P, not P times!
    """
    T, P = current_allocation.shape
    K = S.shape[1]

    if workers is None:
        workers = aadc.ThreadPool(num_threads)

    eval_start = time.perf_counter()

    # Step 1: Aggregate sensitivities for ALL portfolios at once (numpy - fast)
    # agg_S[k,p] = sum_t x[t,p] * S[t,k]
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

    # Extract gradients: for each factor k, we get an array of P gradient values
    # results[1][im_output][sens_handles[k]] = array of P values (dIM_p/dS_p[k] for each p)
    grad_matrix = np.zeros((K, P))
    for k in range(K):
        grad_matrix[k, :] = results[1][im_output][sens_handles[k]]

    # Step 3: Chain rule - gradient w.r.t. allocations for ALL portfolios
    # d(IM_p)/dx[t,p] = sum_k (dIM_p/dS_p[k]) * S[t,k]
    # = S @ grad_p  for each portfolio p
    # For all portfolios: gradient[T, P] = S @ grad_matrix
    gradient = np.dot(S, grad_matrix)  # Shape: (T, P)

    eval_time = time.perf_counter() - eval_start
    return gradient, all_ims, eval_time


def compute_allocation_gradient_chainrule_v2(
    funcs: 'aadc.Functions',
    sens_handles: List[int],
    im_output: int,
    S: np.ndarray,                    # T x K sensitivity matrix
    current_allocation: np.ndarray,   # T x P allocation fractions
    num_threads: int,
    workers: 'aadc.ThreadPool' = None,
) -> Tuple[np.ndarray, float]:
    """
    Wrapper that returns (gradient, total_im) for compatibility with v1 interface.

    Uses the optimized single-evaluate pattern internally.
    """
    gradient, all_ims, _ = compute_all_portfolios_im_gradient_v2(
        funcs, sens_handles, im_output, S, current_allocation, num_threads, workers
    )
    total_im = float(np.sum(all_ims))
    return gradient, total_im


# =============================================================================
# v2 Simplex Projection (vectorized)
# =============================================================================

def project_to_simplex_v2(x: np.ndarray) -> np.ndarray:
    """
    Project each row of x onto the probability simplex (sum=1, all>=0).
    Vectorized implementation for better performance.
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
            result[t] = np.ones(P) / P
        else:
            rho = rho_candidates[-1]
            theta = (cssv[rho] - 1) / (rho + 1)
            result[t] = np.maximum(v - theta, 0)

    return result


# =============================================================================
# v2 Optimized Gradient Descent
# =============================================================================

def optimize_allocation_gradient_descent_v2(
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
) -> Tuple[np.ndarray, List[float], int, float]:
    """
    Gradient descent with Armijo line search, best-tracking, and simplex projection.

    Uses v2 optimized single-evaluate pattern for fast multi-portfolio evaluation.

    Safeguards:
    - Armijo backtracking line search guarantees each accepted step reduces IM
    - Best-solution tracking reverts to best seen if stalled
    - Never returns a solution worse than the initial allocation

    Returns:
        (optimal_allocation, im_history, num_iterations, total_eval_time)
    """
    x = initial_allocation.copy()
    best_x = x.copy()
    im_history = []
    T, P = x.shape
    total_eval_time = 0.0

    # Backtracking line search parameters
    LS_BETA = 0.5         # Step size reduction factor
    LS_MAX_TRIES = 10     # Max backtracking steps

    if workers is None:
        workers = aadc.ThreadPool(num_threads)

    # First evaluation
    gradient, all_ims, eval_time = compute_all_portfolios_im_gradient_v2(
        funcs, sens_handles, im_output, S, x, num_threads, workers
    )
    total_im = float(np.sum(all_ims))
    im_history.append(total_im)
    total_eval_time += eval_time
    best_im = total_im

    # Compute adaptive learning rate
    # Must be large enough that lr * max_gradient_diff >= 1.0 to flip
    # one-hot assignments through simplex projection. Start large,
    # line search will backtrack if needed.
    grad_max = np.abs(gradient).max()
    if lr is None:
        if grad_max > 1e-10:
            lr = 1.0 / grad_max
        else:
            lr = 1e-12

    if verbose:
        print(f"    [v2] Initial IM: ${total_im:,.2f}")
        print(f"    [v2] Gradient: max={grad_max:.2e}, mean={np.abs(gradient).mean():.2e}")
        print(f"    [v2] Learning rate: {lr:.2e}")
        print(f"    [v2] Single evaluate() time: {eval_time*1000:.2f} ms")

    stalled_count = 0

    for iteration in range(max_iters):
        if iteration > 0:
            gradient, all_ims, eval_time = compute_all_portfolios_im_gradient_v2(
                funcs, sens_handles, im_output, S, x, num_threads, workers
            )
            total_im = float(np.sum(all_ims))
            im_history.append(total_im)
            total_eval_time += eval_time

        # Track best solution seen
        if total_im < best_im:
            best_im = total_im
            best_x = x.copy()
            stalled_count = 0
        else:
            stalled_count += 1

        if verbose and iteration % 10 == 0:
            moves = int(np.sum(np.argmax(x, axis=1) != np.argmax(initial_allocation, axis=1)))
            print(f"    [v2] Iter {iteration}: IM = ${total_im:,.2f}, best = ${best_im:,.2f}, moves = {moves}")

        # Early exit if stalled for too long
        if stalled_count >= 20:
            if verbose:
                print(f"    [v2] Stalled for {stalled_count} iterations, reverting to best")
            x = best_x.copy()
            break

        # Monotone backtracking line search
        # Accept any step that strictly decreases IM; halve step size if not
        step_size = lr
        accepted = False

        for _ in range(LS_MAX_TRIES):
            x_candidate = project_to_simplex_v2(x - step_size * gradient)

            _, candidate_ims, eval_time = compute_all_portfolios_im_gradient_v2(
                funcs, sens_handles, im_output, S, x_candidate, num_threads, workers
            )
            candidate_im = float(np.sum(candidate_ims))
            total_eval_time += eval_time

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
                    print(f"    [v2] Converged at iteration {iteration + 1}")
                    print(f"    [v2] Total eval time: {total_eval_time*1000:.2f} ms ({iteration+1} iterations)")
                return best_x, im_history, iteration + 1, total_eval_time

    if verbose:
        if stalled_count < 20:
            print(f"    [v2] Reached max iterations ({max_iters})")
        print(f"    [v2] Total eval time: {total_eval_time*1000:.2f} ms")

    # Always return best solution seen
    return best_x, im_history, max_iters, total_eval_time


# =============================================================================
# v2 Adam Optimizer
# =============================================================================

def optimize_allocation_adam_v2(
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
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, List[float], int, float]:
    """
    Adam optimizer with simplex projection, backtracking line search, and best-tracking.

    Uses adaptive per-parameter learning rates via exponential moving averages
    of gradients (first moment) and squared gradients (second moment).

    Returns:
        (optimal_allocation, im_history, num_iterations, total_eval_time)
    """
    x = initial_allocation.copy()
    best_x = x.copy()
    im_history = []
    T, P = x.shape
    total_eval_time = 0.0

    # Adam moment estimates
    m = np.zeros_like(x)  # First moment (mean of gradients)
    v = np.zeros_like(x)  # Second moment (mean of squared gradients)

    # Backtracking line search parameters
    LS_BETA = 0.5
    LS_MAX_TRIES = 10

    if workers is None:
        workers = aadc.ThreadPool(num_threads)

    # First evaluation
    gradient, all_ims, eval_time = compute_all_portfolios_im_gradient_v2(
        funcs, sens_handles, im_output, S, x, num_threads, workers
    )
    total_im = float(np.sum(all_ims))
    im_history.append(total_im)
    total_eval_time += eval_time
    best_im = total_im

    # Compute adaptive learning rate
    grad_max = np.abs(gradient).max()
    if lr is None:
        if grad_max > 1e-10:
            lr = 1.0 / grad_max
        else:
            lr = 1e-12

    if verbose:
        print(f"    [v2-Adam] Initial IM: ${total_im:,.2f}")
        print(f"    [v2-Adam] Gradient: max={grad_max:.2e}, mean={np.abs(gradient).mean():.2e}")
        print(f"    [v2-Adam] Learning rate: {lr:.2e}")
        print(f"    [v2-Adam] Single evaluate() time: {eval_time*1000:.2f} ms")

    stalled_count = 0

    for iteration in range(max_iters):
        if iteration > 0:
            gradient, all_ims, eval_time = compute_all_portfolios_im_gradient_v2(
                funcs, sens_handles, im_output, S, x, num_threads, workers
            )
            total_im = float(np.sum(all_ims))
            im_history.append(total_im)
            total_eval_time += eval_time

        # Track best solution seen
        if total_im < best_im:
            best_im = total_im
            best_x = x.copy()
            stalled_count = 0
        else:
            stalled_count += 1

        if verbose and iteration % 10 == 0:
            moves = int(np.sum(np.argmax(x, axis=1) != np.argmax(initial_allocation, axis=1)))
            print(f"    [v2-Adam] Iter {iteration}: IM = ${total_im:,.2f}, best = ${best_im:,.2f}, moves = {moves}")

        # Early exit if stalled for too long
        if stalled_count >= 20:
            if verbose:
                print(f"    [v2-Adam] Stalled for {stalled_count} iterations, reverting to best")
            x = best_x.copy()
            break

        # Adam moment updates
        t_step = iteration + 1  # 1-indexed for bias correction
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        # Bias-corrected estimates
        m_hat = m / (1 - beta1 ** t_step)
        v_hat = v / (1 - beta2 ** t_step)

        # Adam direction
        adam_step = m_hat / (np.sqrt(v_hat) + eps)

        # Monotone backtracking line search
        step_size = lr
        accepted = False

        for _ in range(LS_MAX_TRIES):
            x_candidate = project_to_simplex_v2(x - step_size * adam_step)

            _, candidate_ims, eval_time = compute_all_portfolios_im_gradient_v2(
                funcs, sens_handles, im_output, S, x_candidate, num_threads, workers
            )
            candidate_im = float(np.sum(candidate_ims))
            total_eval_time += eval_time

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
                    print(f"    [v2-Adam] Converged at iteration {iteration + 1}")
                    print(f"    [v2-Adam] Total eval time: {total_eval_time*1000:.2f} ms ({iteration+1} iterations)")
                return best_x, im_history, iteration + 1, total_eval_time

    if verbose:
        if stalled_count < 20:
            print(f"    [v2-Adam] Reached max iterations ({max_iters})")
        print(f"    [v2-Adam] Total eval time: {total_eval_time*1000:.2f} ms")

    # Always return best solution seen
    return best_x, im_history, max_iters, total_eval_time


# =============================================================================
# v2 SIMM Kernel with Pre-computed Weighted Sensitivities
# =============================================================================

def record_simm_kernel_precomputed_ws(
    K: int,
    factor_risk_classes: List[str],
    use_correlations: bool = True,
) -> Tuple['aadc.Functions', List[int], int]:
    """
    Record AADC kernel that takes PRE-COMPUTED weighted sensitivities as inputs.

    OPTIMIZATION: Instead of computing WS = s * rw * cr inside the kernel,
    we pre-compute WS in numpy and pass it directly. This reduces tape size.

    Inputs: WS_p[k] = s[k] * rw[k] * cr[k] (pre-computed weighted sensitivities)
    Output: IM_p (SIMM margin)

    The gradients dIM/dWS can be converted to dIM/ds via chain rule:
        dIM/ds[k] = dIM/dWS[k] * rw[k] * cr[k]
    """
    if not AADC_AVAILABLE:
        raise RuntimeError("AADC is required")

    with aadc.record_kernel() as funcs:
        # Mark weighted sensitivities as inputs (K values)
        ws_inputs = []
        ws_handles = []
        for k in range(K):
            ws_k = aadc.idouble(0.0)
            handle = ws_k.mark_as_input()
            ws_handles.append(handle)
            ws_inputs.append(ws_k)

        # Compute risk class margins (no weights/CR - already applied)
        risk_class_margins = []

        for rc in _RISK_CLASS_ORDER:
            rc_indices = [k for k in range(K) if factor_risk_classes[k] == rc]
            if not rc_indices:
                risk_class_margins.append(aadc.idouble(0.0))
                continue

            # Sum of squares (correlation can be added if needed)
            k_sq = aadc.idouble(0.0)
            for k in rc_indices:
                k_sq = k_sq + ws_inputs[k] * ws_inputs[k]

            k_r = np.sqrt(k_sq)
            risk_class_margins.append(k_r)

        # Cross-risk-class aggregation
        simm_sq = aadc.idouble(0.0)
        for i in range(6):
            for j in range(6):
                psi_ij = _PSI_MATRIX[i, j]
                simm_sq = simm_sq + psi_ij * risk_class_margins[i] * risk_class_margins[j]

        im_p = np.sqrt(simm_sq)
        im_output = im_p.mark_as_output()

    return funcs, ws_handles, im_output


# =============================================================================
# v2 Compute IM Gradient for Single Portfolio (CRIF-based)
# =============================================================================

def compute_im_gradient_aadc_v2(
    group_crif: pd.DataFrame,
    num_threads: int,
    workers: 'aadc.ThreadPool' = None,
    kernel_cache: dict = None,
) -> Tuple[np.ndarray, float, float, float]:
    """
    v2 version of compute_im_gradient_aadc with optimizations.

    Same interface as v1 for drop-in replacement.

    Returns:
        (gradient_array, im_value, recording_time, eval_time)
    """
    from model.simm_portfolio_aadc import record_simm_kernel, _get_crif_structure_key, _SIMM_KERNEL_CACHE

    n = len(group_crif)
    if n == 0:
        return np.array([]), 0.0, 0.0, 0.0

    if workers is None:
        workers = aadc.ThreadPool(num_threads)

    if kernel_cache is None:
        kernel_cache = _SIMM_KERNEL_CACHE

    # Check kernel cache
    structure_key = _get_crif_structure_key(group_crif)
    if structure_key in kernel_cache:
        funcs, sens_handles, im_output = kernel_cache[structure_key]
        recording_time = 0.0
    else:
        funcs, sens_handles, im_output, recording_time = record_simm_kernel(group_crif)
        kernel_cache[structure_key] = (funcs, sens_handles, im_output)

    eval_start = time.perf_counter()

    # Build inputs using vectorized access
    amounts = group_crif["Amount"].values
    inputs = {sens_handles[i]: np.array([float(amounts[i])]) for i in range(n)}

    request = {im_output: sens_handles}
    results = aadc.evaluate(funcs, request, inputs, workers)

    im_value = float(results[0][im_output][0])
    gradient = np.array([float(results[1][im_output][sens_handles[i]][0]) for i in range(n)])

    eval_time = time.perf_counter() - eval_start
    return gradient, im_value, recording_time, eval_time


# =============================================================================
# v2 Batch Evaluation for Multiple Portfolios (CRIF-based)
# =============================================================================

def compute_all_portfolios_im_gradient_crif_v2(
    portfolio_crifs: List[pd.DataFrame],
    num_threads: int,
    workers: 'aadc.ThreadPool' = None,
) -> Tuple[List[np.ndarray], List[float], float]:
    """
    Compute IM and gradients for multiple portfolios using single evaluate() where possible.

    For portfolios with IDENTICAL CRIF structure (same risk factors, different amounts),
    we can batch them into a single evaluate() call.

    Args:
        portfolio_crifs: List of P CRIF DataFrames
        num_threads: AADC worker threads
        workers: Optional pre-created ThreadPool

    Returns:
        (gradients_list, ims_list, total_eval_time)
    """
    from model.simm_portfolio_aadc import record_simm_kernel, _get_crif_structure_key

    if workers is None:
        workers = aadc.ThreadPool(num_threads)

    P = len(portfolio_crifs)
    gradients = [None] * P
    ims = [0.0] * P
    total_eval_time = 0.0

    # Group portfolios by CRIF structure for batching
    structure_groups = {}  # structure_key -> list of (portfolio_idx, crif)
    for p_idx, crif in enumerate(portfolio_crifs):
        if crif.empty:
            gradients[p_idx] = np.array([])
            continue
        key = _get_crif_structure_key(crif)
        if key not in structure_groups:
            structure_groups[key] = []
        structure_groups[key].append((p_idx, crif))

    # Process each structure group with batched evaluation
    for structure_key, group in structure_groups.items():
        # Record kernel once for this structure
        sample_crif = group[0][1]
        funcs, sens_handles, im_output, _ = record_simm_kernel(sample_crif)
        n = len(sample_crif)

        # Build batched inputs
        batch_size = len(group)
        inputs = {}

        for k in range(n):
            # Stack amounts for all portfolios in this group
            amounts_k = np.array([g[1].iloc[k]["Amount"] for g in group])
            inputs[sens_handles[k]] = amounts_k

        request = {im_output: sens_handles}

        eval_start = time.perf_counter()
        results = aadc.evaluate(funcs, request, inputs, workers)
        total_eval_time += time.perf_counter() - eval_start

        # Extract results for each portfolio in batch
        batch_ims = results[0][im_output]  # Array of batch_size IMs

        for batch_idx, (p_idx, crif) in enumerate(group):
            ims[p_idx] = float(batch_ims[batch_idx])
            grad = np.array([float(results[1][im_output][sens_handles[k]][batch_idx])
                            for k in range(n)])
            gradients[p_idx] = grad

    return gradients, ims, total_eval_time


# =============================================================================
# Utility Functions
# =============================================================================

def round_to_integer_allocation_v2(continuous_allocation: np.ndarray) -> np.ndarray:
    """Round continuous allocation to integer (one-hot per row)."""
    T, P = continuous_allocation.shape
    result = np.zeros_like(continuous_allocation)
    for t in range(T):
        best_p = np.argmax(continuous_allocation[t])
        result[t, best_p] = 1.0
    return result


def verify_gradient_v2(
    funcs: 'aadc.Functions',
    sens_handles: List[int],
    im_output: int,
    S: np.ndarray,
    allocation: np.ndarray,
    num_threads: int,
    eps: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Verify v2 gradient via finite differences.

    Returns:
        (max_rel_error, aadc_gradient, fd_gradient)
    """
    T, P = allocation.shape
    workers = aadc.ThreadPool(num_threads)

    # Get AADC gradient
    aadc_grad, total_im = compute_allocation_gradient_chainrule_v2(
        funcs, sens_handles, im_output, S, allocation, num_threads, workers
    )

    # Compute finite difference gradient
    fd_grad = np.zeros((T, P))
    for t in range(T):
        for p in range(P):
            x_plus = allocation.copy()
            x_plus[t, p] += eps

            _, im_plus = compute_allocation_gradient_chainrule_v2(
                funcs, sens_handles, im_output, S, x_plus, num_threads, workers
            )
            fd_grad[t, p] = (im_plus - total_im) / eps

    # Compute max relative error
    denom = np.maximum(np.abs(aadc_grad), np.abs(fd_grad))
    denom = np.where(denom < 1e-10, 1.0, denom)
    rel_errors = np.abs(aadc_grad - fd_grad) / denom
    max_rel_error = np.max(rel_errors)

    return max_rel_error, aadc_grad, fd_grad


# =============================================================================
# Module Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    print(f"SIMM Portfolio AADC v2 Module - Version {MODULE_VERSION}")
    print("Optimizations:")
    print("  - Single aadc.evaluate() call for all P portfolios")
    print("  - Pre-computed weighted sensitivities option")
    print("  - Vectorized gradient extraction")
    print(f"AADC available: {AADC_AVAILABLE}")
