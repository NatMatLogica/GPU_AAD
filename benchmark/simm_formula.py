"""
Shared ISDA SIMM v2.6 formula — NumPy reference implementation.

This module provides the canonical SIMM calculation that all backends must match.
Uses the exact same PSI matrix, risk weights, correlations, and concentration
factors as model/simm_portfolio_aadc.py.

Functions:
    compute_simm_im:       Single-portfolio SIMM IM calculation.
    compute_simm_batch:    Batched IM for P portfolios.
    compute_simm_gradient: Analytical gradient dIM/dS for P portfolios.
"""

import numpy as np
from typing import Tuple

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.backends.base import FactorMetadata

# =============================================================================
# ISDA SIMM v2.6 Cross-Risk-Class Correlation Matrix (PSI)
# =============================================================================
# Order: Rates, CreditQ, CreditNonQ, Equity, Commodity, FX
PSI_MATRIX = np.array([
    [1.00, 0.04, 0.04, 0.07, 0.37, 0.14],  # Rates
    [0.04, 1.00, 0.54, 0.70, 0.27, 0.37],  # CreditQ
    [0.04, 0.54, 1.00, 0.46, 0.24, 0.15],  # CreditNonQ
    [0.07, 0.70, 0.46, 1.00, 0.35, 0.39],  # Equity
    [0.37, 0.27, 0.24, 0.35, 1.00, 0.35],  # Commodity
    [0.14, 0.37, 0.15, 0.39, 0.35, 1.00],  # FX
], dtype=np.float64)

RISK_CLASS_ORDER = ['Rates', 'CreditQ', 'CreditNonQ', 'Equity', 'Commodity', 'FX']
RISK_CLASS_TO_IDX = {rc: i for i, rc in enumerate(RISK_CLASS_ORDER)}
NUM_RISK_CLASSES = 6


def compute_simm_im(
    agg_sensitivities: np.ndarray,    # (K,) aggregated sensitivities
    factor_meta: FactorMetadata,
) -> float:
    """
    Reference SIMM calculation for one portfolio.

    Steps:
    1. WS_k = s_k * rw_k * cr_k   (weighted sensitivity with concentration)
    2. Per risk class rc: K_rc = sqrt(Σ_{k∈rc} Σ_{l∈rc} rho_kl * WS_k * WS_l)
    3. IM = sqrt(Σ_r Σ_s psi_rs * K_r * K_s)

    Args:
        agg_sensitivities: (K,) vector of aggregated sensitivities for one portfolio.
        factor_meta: Metadata describing each risk factor.

    Returns:
        Scalar IM value.
    """
    K = len(agg_sensitivities)
    ws = agg_sensitivities * factor_meta.risk_weights * factor_meta.concentration_factors

    # Risk class margins
    k_r = np.zeros(NUM_RISK_CLASSES)

    for rc_idx in range(NUM_RISK_CLASSES):
        mask = factor_meta.risk_class_idx == rc_idx
        if not np.any(mask):
            continue

        ws_rc = ws[mask]

        if factor_meta.intra_corr_matrix is not None:
            # Full correlation: K² = Σ_k Σ_l rho_kl * WS_k * WS_l
            # Extract sub-matrix of correlations for this risk class
            rc_indices = np.where(mask)[0]
            n_rc = len(rc_indices)
            corr_sub = np.empty((n_rc, n_rc))
            for i in range(n_rc):
                for j in range(n_rc):
                    corr_sub[i, j] = factor_meta.intra_corr_matrix[rc_indices[i], rc_indices[j]]
            k_sq = float(ws_rc @ corr_sub @ ws_rc)
        else:
            # No correlations: K² = Σ_k WS_k²
            k_sq = float(np.dot(ws_rc, ws_rc))

        k_r[rc_idx] = np.sqrt(max(k_sq, 0.0))

    # Cross-risk-class aggregation
    im_sq = float(k_r @ PSI_MATRIX @ k_r)
    return np.sqrt(max(im_sq, 0.0))


def compute_simm_batch(
    agg_S: np.ndarray,               # (P, K) aggregated sensitivities
    factor_meta: FactorMetadata,
) -> np.ndarray:
    """
    Batched SIMM IM for P portfolios.

    Args:
        agg_S: (P, K) aggregated sensitivities.
        factor_meta: Metadata describing each risk factor.

    Returns:
        (P,) array of IM values.
    """
    P = agg_S.shape[0]
    im_values = np.zeros(P)

    rw = factor_meta.risk_weights
    cr = factor_meta.concentration_factors
    rc_idx = factor_meta.risk_class_idx
    corr = factor_meta.intra_corr_matrix

    for p in range(P):
        ws = agg_S[p] * rw * cr

        k_r = np.zeros(NUM_RISK_CLASSES)
        for rc in range(NUM_RISK_CLASSES):
            mask = rc_idx == rc
            if not np.any(mask):
                continue
            ws_rc = ws[mask]

            if corr is not None:
                rc_indices = np.where(mask)[0]
                n_rc = len(rc_indices)
                corr_sub = corr[np.ix_(rc_indices, rc_indices)]
                k_sq = float(ws_rc @ corr_sub @ ws_rc)
            else:
                k_sq = float(np.dot(ws_rc, ws_rc))

            k_r[rc] = np.sqrt(max(k_sq, 0.0))

        im_sq = float(k_r @ PSI_MATRIX @ k_r)
        im_values[p] = np.sqrt(max(im_sq, 0.0))

    return im_values


def compute_simm_gradient(
    agg_S: np.ndarray,               # (P, K) aggregated sensitivities
    factor_meta: FactorMetadata,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytical gradient dIM/dS for P portfolios.

    Chain rule:
        dIM/dS_k = dIM/dK_r * dK_r/dWS_k * dWS_k/dS_k

    where:
        dIM/dK_r = (1/IM) * Σ_s psi_rs * K_s
        dK_r/dWS_k = (1/K_r) * Σ_l rho_kl * WS_l      [with correlations]
                   = WS_k / K_r                          [without correlations]
        dWS_k/dS_k = rw_k * cr_k

    Args:
        agg_S: (P, K) aggregated sensitivities.
        factor_meta: Metadata describing each risk factor.

    Returns:
        im_values: (P,) array of IM values.
        gradients: (P, K) array of dIM/dS.
    """
    P, K = agg_S.shape
    im_values = np.zeros(P)
    gradients = np.zeros((P, K))

    rw = factor_meta.risk_weights
    cr = factor_meta.concentration_factors
    rc_idx = factor_meta.risk_class_idx
    corr = factor_meta.intra_corr_matrix

    for p in range(P):
        ws = agg_S[p] * rw * cr

        # Forward pass: compute K_r per risk class
        k_r = np.zeros(NUM_RISK_CLASSES)
        # Store dK_r/dWS for each factor (needed for gradient)
        dk_dws = np.zeros(K)

        for rc in range(NUM_RISK_CLASSES):
            mask = rc_idx == rc
            if not np.any(mask):
                continue
            ws_rc = ws[mask]
            rc_indices = np.where(mask)[0]
            n_rc = len(rc_indices)

            if corr is not None and n_rc > 1:
                corr_sub = corr[np.ix_(rc_indices, rc_indices)]
                k_sq = float(ws_rc @ corr_sub @ ws_rc)
                k_r[rc] = np.sqrt(max(k_sq, 0.0))

                if k_r[rc] > 1e-15:
                    # dK_r/dWS_k = (1/K_r) * Σ_l rho_kl * WS_l
                    # = (1/K_r) * (corr_sub @ ws_rc)
                    d_ws_rc = corr_sub @ ws_rc / k_r[rc]
                    for i, gi in enumerate(rc_indices):
                        dk_dws[gi] = d_ws_rc[i]
            else:
                k_sq = float(np.dot(ws_rc, ws_rc))
                k_r[rc] = np.sqrt(max(k_sq, 0.0))

                if k_r[rc] > 1e-15:
                    for i, gi in enumerate(rc_indices):
                        dk_dws[gi] = ws_rc[i] / k_r[rc]

        # Cross-risk-class: IM = sqrt(k_r @ PSI @ k_r)
        im_sq = float(k_r @ PSI_MATRIX @ k_r)
        im_p = np.sqrt(max(im_sq, 0.0))
        im_values[p] = im_p

        if im_p < 1e-15:
            continue

        # dIM/dK_r = (1/IM) * Σ_s psi_rs * K_s = (1/IM) * PSI @ k_r
        dim_dk = PSI_MATRIX @ k_r / im_p

        # Full gradient: dIM/dS_k = dim_dk[rc_k] * dk_dws[k] * rw_k * cr_k
        for k in range(K):
            rc = rc_idx[k]
            gradients[p, k] = dim_dk[rc] * dk_dws[k] * rw[k] * cr[k]

    return im_values, gradients
