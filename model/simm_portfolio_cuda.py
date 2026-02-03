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
MODEL_VERSION = "3.5.0"  # v3.5: Optimized GPU kernels with shared memory and parallel reduction


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

# Constants for CRIF generation (matching trade_types.py)
NUM_IR_TENORS = 12
NUM_VEGA_EXPIRIES = 6
BUMP_SIZE = 0.0001  # 1bp for IR/inflation
SPOT_BUMP = 1.0     # 1 unit for spot
VOL_BUMP = 0.01     # 1% for vol


# =============================================================================
# CUDA Device Functions for Pricing (GPU bump-and-revalue CRIF generation)
# =============================================================================
# These device functions mirror the Python pricing functions in trade_types.py
# They are used by the bump-and-revalue kernel to compute sensitivities on GPU.
# =============================================================================

if CUDA_AVAILABLE:
    # IR tenor fractions (years)
    IR_TENORS_DEVICE = np.array([2/52, 1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0], dtype=np.float64)
    VEGA_EXPIRIES_DEVICE = np.array([0.5, 1.0, 3.0, 5.0, 10.0, 30.0], dtype=np.float64)

    @cuda.jit(device=True)
    def _interp_rate(t, tenors, rates, n_tenors):
        """Linear interpolation for yield curve rate at time t."""
        if t <= tenors[0]:
            return rates[0]
        if t >= tenors[n_tenors - 1]:
            return rates[n_tenors - 1]
        for i in range(n_tenors - 1):
            if tenors[i] <= t < tenors[i + 1]:
                alpha = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
                return rates[i] + alpha * (rates[i + 1] - rates[i])
        return rates[n_tenors - 1]

    @cuda.jit(device=True)
    def _discount(t, tenors, rates, n_tenors):
        """Discount factor at time t."""
        if t <= 0.0:
            return 1.0
        r = _interp_rate(t, tenors, rates, n_tenors)
        return math.exp(-r * t)

    @cuda.jit(device=True)
    def _forward_rate(t1, t2, tenors, rates, n_tenors):
        """Forward rate from t1 to t2."""
        if t2 <= t1:
            return _interp_rate(t1, tenors, rates, n_tenors)
        df1 = _discount(t1, tenors, rates, n_tenors)
        df2 = _discount(t2, tenors, rates, n_tenors)
        if df2 <= 0.0:
            return 0.0
        return math.log(df1 / df2) / (t2 - t1)

    @cuda.jit(device=True)
    def _normal_cdf(x):
        """Standard normal CDF approximation."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @cuda.jit(device=True)
    def _price_irs_device(notional, fixed_rate, maturity, frequency, payer,
                          tenors, rates, n_tenors):
        """Price vanilla IRS on device."""
        dt = 1.0 / frequency
        num_periods = int(maturity * frequency)

        fixed_leg = 0.0
        floating_leg = 0.0

        for i in range(1, num_periods + 1):
            t = i * dt
            df = _discount(t, tenors, rates, n_tenors)
            fixed_leg += notional * fixed_rate * dt * df

            t_prev = (i - 1) * dt
            fwd = _forward_rate(t_prev, t, tenors, rates, n_tenors)
            floating_leg += notional * dt * fwd * df

        npv = floating_leg - fixed_leg
        if not payer:
            npv = fixed_leg - floating_leg
        return npv

    @cuda.jit(device=True)
    def _price_equity_option_device(notional, strike, maturity, dividend_yield, is_call,
                                     spot, vol, tenors, rates, n_tenors):
        """Price equity option (Black-Scholes) on device."""
        r = _interp_rate(maturity, tenors, rates, n_tenors)
        q = dividend_yield
        tau = maturity
        sqrt_tau = math.sqrt(tau)

        if vol <= 0.0 or tau <= 0.0:
            return 0.0

        d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
        d2 = d1 - vol * sqrt_tau

        df = math.exp(-r * tau)
        dq = math.exp(-q * tau)

        num_contracts = notional / strike

        if is_call:
            price = num_contracts * (spot * dq * _normal_cdf(d1) - strike * df * _normal_cdf(d2))
        else:
            price = num_contracts * (strike * df * _normal_cdf(-d2) - spot * dq * _normal_cdf(-d1))

        return price

    @cuda.jit(device=True)
    def _price_fx_option_device(notional, strike, maturity, is_call,
                                 spot, vol, dom_tenors, dom_rates, fgn_tenors, fgn_rates, n_tenors):
        """Price FX option (Garman-Kohlhagen) on device."""
        rd = _interp_rate(maturity, dom_tenors, dom_rates, n_tenors)
        rf = _interp_rate(maturity, fgn_tenors, fgn_rates, n_tenors)
        tau = maturity
        sqrt_tau = math.sqrt(tau)

        if vol <= 0.0 or tau <= 0.0:
            return 0.0

        d1 = (math.log(spot / strike) + (rd - rf + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
        d2 = d1 - vol * sqrt_tau

        df_dom = math.exp(-rd * tau)
        df_fgn = math.exp(-rf * tau)

        fgn_notional = notional / strike

        if is_call:
            price = fgn_notional * (spot * df_fgn * _normal_cdf(d1) - strike * df_dom * _normal_cdf(d2))
        else:
            price = fgn_notional * (strike * df_dom * _normal_cdf(-d2) - spot * df_fgn * _normal_cdf(-d1))

        return price

    @cuda.jit(device=True)
    def _price_inflation_swap_device(notional, fixed_rate, maturity,
                                      tenors, rates, inflation_rates, base_cpi, n_tenors):
        """Price inflation swap on device."""
        tau = maturity
        df = _discount(tau, tenors, rates, n_tenors)

        fixed_leg = notional * (math.exp(fixed_rate * tau) - 1.0) * df

        infl_rate = _interp_rate(tau, tenors, inflation_rates, n_tenors)
        cpi_ratio = math.exp(infl_rate * tau)  # projected_cpi / base_cpi
        inflation_leg = notional * (cpi_ratio - 1.0) * df

        return inflation_leg - fixed_leg

    @cuda.jit(device=True)
    def _price_xccy_swap_device(dom_notional, fgn_notional, dom_fixed_rate, fgn_fixed_rate,
                                 maturity, frequency, exchange_notional, fx_spot,
                                 dom_tenors, dom_rates, fgn_tenors, fgn_rates, n_tenors):
        """Price cross-currency swap on device."""
        dt = 1.0 / frequency
        num_periods = int(maturity * frequency)

        dom_leg = 0.0
        for i in range(1, num_periods + 1):
            t = i * dt
            df = _discount(t, dom_tenors, dom_rates, n_tenors)
            dom_leg += dom_notional * dom_fixed_rate * dt * df

        fgn_leg = 0.0
        for i in range(1, num_periods + 1):
            t = i * dt
            df = _discount(t, fgn_tenors, fgn_rates, n_tenors)
            fgn_leg += fgn_notional * fgn_fixed_rate * dt * df

        if exchange_notional:
            dom_df_mat = _discount(maturity, dom_tenors, dom_rates, n_tenors)
            fgn_df_mat = _discount(maturity, fgn_tenors, fgn_rates, n_tenors)
            dom_leg += dom_notional * dom_df_mat - dom_notional
            fgn_leg += fgn_notional * fgn_df_mat - fgn_notional

        return dom_leg - fgn_leg * fx_spot


# =============================================================================
# CUDA Bump-and-Revalue Kernel for CRIF Generation
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def _bump_reval_irs_kernel(
        # Trade parameters (T trades)
        notionals, fixed_rates, maturities, frequencies, payers,
        # Market data (12 tenors)
        curve_rates, tenors,
        # Output: sensitivities (T, 12)
        sensitivities,
    ):
        """Compute IR sensitivities for IRS trades via bump-and-revalue."""
        t_idx = cuda.grid(1)
        if t_idx >= notionals.shape[0]:
            return

        n_tenors = 12

        # Trade parameters
        notional = notionals[t_idx]
        fixed_rate = fixed_rates[t_idx]
        maturity = maturities[t_idx]
        frequency = frequencies[t_idx]
        payer = payers[t_idx]

        # Base price
        base_pv = _price_irs_device(
            notional, fixed_rate, maturity, frequency, payer,
            tenors, curve_rates, n_tenors
        )

        # Local array for bumped rates
        bumped_rates = cuda.local.array(12, dtype=numba.float64)

        # Bump each tenor and compute delta
        for i in range(n_tenors):
            # Copy rates
            for j in range(n_tenors):
                bumped_rates[j] = curve_rates[j]

            # Bump tenor i
            bumped_rates[i] += BUMP_SIZE

            # Bumped price
            bumped_pv = _price_irs_device(
                notional, fixed_rate, maturity, frequency, payer,
                tenors, bumped_rates, n_tenors
            )

            # Delta = (bumped - base) / bump_size
            delta = (bumped_pv - base_pv) / BUMP_SIZE
            sensitivities[t_idx, i] = delta

    @cuda.jit
    def _bump_reval_equity_option_kernel(
        # Trade parameters (T trades)
        notionals, strikes, maturities, dividend_yields, is_calls, equity_buckets,
        # Market data
        curve_rates, tenors, spot, vol_surface, vega_expiries,
        # Output: sensitivities (T, K) where K = 12 IR + 1 spot + 6 vega = 19
        sensitivities,
    ):
        """Compute sensitivities for equity option trades via bump-and-revalue."""
        t_idx = cuda.grid(1)
        if t_idx >= notionals.shape[0]:
            return

        n_tenors = 12
        n_vega = 6

        # Trade parameters
        notional = notionals[t_idx]
        strike = strikes[t_idx]
        maturity = maturities[t_idx]
        div_yield = dividend_yields[t_idx]
        is_call = is_calls[t_idx]

        # Get vol at trade maturity
        vol = _interp_rate(maturity, vega_expiries, vol_surface, n_vega)

        # Base price
        base_pv = _price_equity_option_device(
            notional, strike, maturity, div_yield, is_call,
            spot[0], vol, tenors, curve_rates, n_tenors
        )

        # Local arrays
        bumped_rates = cuda.local.array(12, dtype=numba.float64)
        bumped_vols = cuda.local.array(6, dtype=numba.float64)

        # IR Delta (12 tenors)
        for i in range(n_tenors):
            for j in range(n_tenors):
                bumped_rates[j] = curve_rates[j]
            bumped_rates[i] += BUMP_SIZE

            bumped_pv = _price_equity_option_device(
                notional, strike, maturity, div_yield, is_call,
                spot[0], vol, tenors, bumped_rates, n_tenors
            )
            sensitivities[t_idx, i] = (bumped_pv - base_pv) / BUMP_SIZE

        # Equity spot delta (index 12)
        bumped_spot = spot[0] + SPOT_BUMP
        bumped_pv = _price_equity_option_device(
            notional, strike, maturity, div_yield, is_call,
            bumped_spot, vol, tenors, curve_rates, n_tenors
        )
        # Sensitivity in notional terms: (dPV/dS) * S
        sensitivities[t_idx, 12] = (bumped_pv - base_pv) / SPOT_BUMP * spot[0]

        # Equity vega (6 expiries, indices 13-18)
        for i in range(n_vega):
            for j in range(n_vega):
                bumped_vols[j] = vol_surface[j]
            bumped_vols[i] += VOL_BUMP

            bumped_vol = _interp_rate(maturity, vega_expiries, bumped_vols, n_vega)
            bumped_pv = _price_equity_option_device(
                notional, strike, maturity, div_yield, is_call,
                spot[0], bumped_vol, tenors, curve_rates, n_tenors
            )
            sensitivities[t_idx, 13 + i] = (bumped_pv - base_pv) / VOL_BUMP

    @cuda.jit
    def _bump_reval_fx_option_kernel(
        # Trade parameters (T trades)
        notionals, strikes, maturities, is_calls,
        # Market data
        dom_rates, fgn_rates, tenors, fx_spot, vol_surface, vega_expiries,
        # Output: sensitivities (T, K) where K = 12 dom IR + 12 fgn IR + 1 FX + 6 vega = 31
        sensitivities,
    ):
        """Compute sensitivities for FX option trades via bump-and-revalue."""
        t_idx = cuda.grid(1)
        if t_idx >= notionals.shape[0]:
            return

        n_tenors = 12
        n_vega = 6

        # Trade parameters
        notional = notionals[t_idx]
        strike = strikes[t_idx]
        maturity = maturities[t_idx]
        is_call = is_calls[t_idx]

        spot = fx_spot[0]
        vol = _interp_rate(maturity, vega_expiries, vol_surface, n_vega)

        # Base price
        base_pv = _price_fx_option_device(
            notional, strike, maturity, is_call,
            spot, vol, tenors, dom_rates, tenors, fgn_rates, n_tenors
        )

        # Local arrays
        bumped_dom = cuda.local.array(12, dtype=numba.float64)
        bumped_fgn = cuda.local.array(12, dtype=numba.float64)
        bumped_vols = cuda.local.array(6, dtype=numba.float64)

        # Domestic IR Delta (12 tenors, indices 0-11)
        for i in range(n_tenors):
            for j in range(n_tenors):
                bumped_dom[j] = dom_rates[j]
            bumped_dom[i] += BUMP_SIZE

            bumped_pv = _price_fx_option_device(
                notional, strike, maturity, is_call,
                spot, vol, tenors, bumped_dom, tenors, fgn_rates, n_tenors
            )
            sensitivities[t_idx, i] = (bumped_pv - base_pv) / BUMP_SIZE

        # Foreign IR Delta (12 tenors, indices 12-23)
        for i in range(n_tenors):
            for j in range(n_tenors):
                bumped_fgn[j] = fgn_rates[j]
            bumped_fgn[i] += BUMP_SIZE

            bumped_pv = _price_fx_option_device(
                notional, strike, maturity, is_call,
                spot, vol, tenors, dom_rates, tenors, bumped_fgn, n_tenors
            )
            sensitivities[t_idx, 12 + i] = (bumped_pv - base_pv) / BUMP_SIZE

        # FX spot delta (index 24)
        bumped_spot = spot + SPOT_BUMP * 0.01
        bumped_pv = _price_fx_option_device(
            notional, strike, maturity, is_call,
            bumped_spot, vol, tenors, dom_rates, tenors, fgn_rates, n_tenors
        )
        sensitivities[t_idx, 24] = (bumped_pv - base_pv) / (SPOT_BUMP * 0.01) * spot

        # FX vega (6 expiries, indices 25-30)
        for i in range(n_vega):
            for j in range(n_vega):
                bumped_vols[j] = vol_surface[j]
            bumped_vols[i] += VOL_BUMP

            bumped_vol = _interp_rate(maturity, vega_expiries, bumped_vols, n_vega)
            bumped_pv = _price_fx_option_device(
                notional, strike, maturity, is_call,
                spot, bumped_vol, tenors, dom_rates, tenors, fgn_rates, n_tenors
            )
            sensitivities[t_idx, 25 + i] = (bumped_pv - base_pv) / VOL_BUMP

    @cuda.jit
    def _bump_reval_inflation_swap_kernel(
        # Trade parameters (T trades)
        notionals, fixed_rates, maturities,
        # Market data
        curve_rates, inflation_rates, base_cpi, tenors,
        # Output: sensitivities (T, K) where K = 12 IR + 12 inflation = 24
        sensitivities,
    ):
        """Compute sensitivities for inflation swap trades via bump-and-revalue."""
        t_idx = cuda.grid(1)
        if t_idx >= notionals.shape[0]:
            return

        n_tenors = 12

        # Trade parameters
        notional = notionals[t_idx]
        fixed_rate = fixed_rates[t_idx]
        maturity = maturities[t_idx]
        cpi = base_cpi[0]

        # Base price
        base_pv = _price_inflation_swap_device(
            notional, fixed_rate, maturity,
            tenors, curve_rates, inflation_rates, cpi, n_tenors
        )

        # Local arrays
        bumped_rates = cuda.local.array(12, dtype=numba.float64)
        bumped_infl = cuda.local.array(12, dtype=numba.float64)

        # IR Delta (12 tenors, indices 0-11)
        for i in range(n_tenors):
            for j in range(n_tenors):
                bumped_rates[j] = curve_rates[j]
            bumped_rates[i] += BUMP_SIZE

            bumped_pv = _price_inflation_swap_device(
                notional, fixed_rate, maturity,
                tenors, bumped_rates, inflation_rates, cpi, n_tenors
            )
            sensitivities[t_idx, i] = (bumped_pv - base_pv) / BUMP_SIZE

        # Inflation Delta (12 tenors, indices 12-23)
        for i in range(n_tenors):
            for j in range(n_tenors):
                bumped_infl[j] = inflation_rates[j]
            bumped_infl[i] += BUMP_SIZE

            bumped_pv = _price_inflation_swap_device(
                notional, fixed_rate, maturity,
                tenors, curve_rates, bumped_infl, cpi, n_tenors
            )
            sensitivities[t_idx, 12 + i] = (bumped_pv - base_pv) / BUMP_SIZE

    @cuda.jit
    def _bump_reval_xccy_swap_kernel(
        # Trade parameters (T trades)
        dom_notionals, fgn_notionals, dom_fixed_rates, fgn_fixed_rates,
        maturities, frequencies, exchange_notionals,
        # Market data
        dom_rates, fgn_rates, fx_spot, tenors,
        # Output: sensitivities (T, K) where K = 12 dom IR + 12 fgn IR + 1 FX = 25
        sensitivities,
    ):
        """Compute sensitivities for XCCY swap trades via bump-and-revalue."""
        t_idx = cuda.grid(1)
        if t_idx >= dom_notionals.shape[0]:
            return

        n_tenors = 12

        # Trade parameters
        dom_not = dom_notionals[t_idx]
        fgn_not = fgn_notionals[t_idx]
        dom_fixed = dom_fixed_rates[t_idx]
        fgn_fixed = fgn_fixed_rates[t_idx]
        maturity = maturities[t_idx]
        freq = frequencies[t_idx]
        exch_not = exchange_notionals[t_idx]
        spot = fx_spot[0]

        # Base price
        base_pv = _price_xccy_swap_device(
            dom_not, fgn_not, dom_fixed, fgn_fixed,
            maturity, freq, exch_not, spot,
            tenors, dom_rates, tenors, fgn_rates, n_tenors
        )

        # Local arrays
        bumped_dom = cuda.local.array(12, dtype=numba.float64)
        bumped_fgn = cuda.local.array(12, dtype=numba.float64)

        # Domestic IR Delta (12 tenors, indices 0-11)
        for i in range(n_tenors):
            for j in range(n_tenors):
                bumped_dom[j] = dom_rates[j]
            bumped_dom[i] += BUMP_SIZE

            bumped_pv = _price_xccy_swap_device(
                dom_not, fgn_not, dom_fixed, fgn_fixed,
                maturity, freq, exch_not, spot,
                tenors, bumped_dom, tenors, fgn_rates, n_tenors
            )
            sensitivities[t_idx, i] = (bumped_pv - base_pv) / BUMP_SIZE

        # Foreign IR Delta (12 tenors, indices 12-23)
        for i in range(n_tenors):
            for j in range(n_tenors):
                bumped_fgn[j] = fgn_rates[j]
            bumped_fgn[i] += BUMP_SIZE

            bumped_pv = _price_xccy_swap_device(
                dom_not, fgn_not, dom_fixed, fgn_fixed,
                maturity, freq, exch_not, spot,
                tenors, dom_rates, tenors, bumped_fgn, n_tenors
            )
            sensitivities[t_idx, 12 + i] = (bumped_pv - base_pv) / BUMP_SIZE

        # FX spot delta (index 24)
        bumped_spot = spot + SPOT_BUMP * 0.01
        bumped_pv = _price_xccy_swap_device(
            dom_not, fgn_not, dom_fixed, fgn_fixed,
            maturity, freq, exch_not, bumped_spot,
            tenors, dom_rates, tenors, fgn_rates, n_tenors
        )
        sensitivities[t_idx, 24] = (bumped_pv - base_pv) / (SPOT_BUMP * 0.01) * spot


# =============================================================================
# GPU CRIF Generation Functions
# =============================================================================

def compute_crif_gpu(trades: list, market, device: int = 0) -> Tuple[pd.DataFrame, float]:
    """
    Compute CRIF sensitivities using GPU bump-and-revalue for all trade types.

    Returns:
        (crif_df, sensies_time_sec) - CRIF DataFrame and sensitivity computation time
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available for GPU CRIF generation")

    if not CUDA_SIMULATOR:
        cuda.select_device(device)

    # Group trades by type
    trades_by_type = {}
    for i, trade in enumerate(trades):
        tt = trade.trade_type
        if tt not in trades_by_type:
            trades_by_type[tt] = []
        trades_by_type[tt].append((i, trade))

    all_crif_rows = []
    total_sensies_time = 0.0

    tenors_d = cuda.to_device(IR_TENORS_DEVICE)
    vega_exp_d = cuda.to_device(VEGA_EXPIRIES_DEVICE)

    threads_per_block = 256

    # Process each trade type
    for trade_type, trade_list in trades_by_type.items():
        n_trades = len(trade_list)
        if n_trades == 0:
            continue

        sensies_start = time.perf_counter()

        if trade_type == "ir_swap":
            # Extract trade parameters
            notionals = np.array([t.notional for _, t in trade_list], dtype=np.float64)
            fixed_rates = np.array([t.fixed_rate for _, t in trade_list], dtype=np.float64)
            maturities = np.array([t.maturity for _, t in trade_list], dtype=np.float64)
            frequencies = np.array([t.frequency for _, t in trade_list], dtype=np.int32)
            payers = np.array([1 if t.payer else 0 for _, t in trade_list], dtype=np.int32)

            # Get curve for first trade's currency (simplified - assumes single currency)
            ccy = trade_list[0][1].currency
            curve = market.curves.get(ccy)
            if curve is None:
                continue
            curve_rates = np.ascontiguousarray(curve.zero_rates, dtype=np.float64)

            # Allocate output
            sensitivities = np.zeros((n_trades, NUM_IR_TENORS), dtype=np.float64)

            # Launch kernel
            blocks = (n_trades + threads_per_block - 1) // threads_per_block
            d_sensitivities = cuda.to_device(sensitivities)

            _bump_reval_irs_kernel[blocks, threads_per_block](
                cuda.to_device(notionals),
                cuda.to_device(fixed_rates),
                cuda.to_device(maturities),
                cuda.to_device(frequencies),
                cuda.to_device(payers),
                cuda.to_device(curve_rates),
                tenors_d,
                d_sensitivities,
            )

            if not CUDA_SIMULATOR:
                cuda.synchronize()
            d_sensitivities.copy_to_host(sensitivities)

            total_sensies_time += time.perf_counter() - sensies_start

            # Convert to CRIF rows
            tenor_labels = ["2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"]
            for idx, (orig_idx, trade) in enumerate(trade_list):
                for i in range(NUM_IR_TENORS):
                    delta = sensitivities[idx, i]
                    if abs(delta) > 1e-10:
                        all_crif_rows.append({
                            "TradeID": trade.trade_id,
                            "ProductClass": "RatesFX",
                            "RiskType": "Risk_IRCurve",
                            "Qualifier": trade.currency,
                            "Bucket": str(i + 1),
                            "Label1": tenor_labels[i],
                            "Label2": "OIS",
                            "Amount": delta,
                            "AmountCurrency": trade.currency,
                            "AmountUSD": delta,
                        })

        elif trade_type == "equity_option":
            # Extract trade parameters
            notionals = np.array([t.notional for _, t in trade_list], dtype=np.float64)
            strikes = np.array([t.strike for _, t in trade_list], dtype=np.float64)
            maturities = np.array([t.maturity for _, t in trade_list], dtype=np.float64)
            div_yields = np.array([t.dividend_yield for _, t in trade_list], dtype=np.float64)
            is_calls = np.array([1 if t.is_call else 0 for _, t in trade_list], dtype=np.int32)
            eq_buckets = np.array([t.equity_bucket for _, t in trade_list], dtype=np.int32)

            # Market data
            ccy = trade_list[0][1].currency
            underlying = trade_list[0][1].underlying
            curve = market.curves.get(ccy)
            if curve is None:
                continue
            curve_rates = np.ascontiguousarray(curve.zero_rates, dtype=np.float64)

            spot = np.array([market.equity_spots.get(underlying, 100.0)], dtype=np.float64)
            vol_surf = market.vol_surfaces.get(underlying)
            if vol_surf is None:
                vol_surf_arr = np.full(NUM_VEGA_EXPIRIES, 0.2, dtype=np.float64)
            else:
                vol_surf_arr = np.ascontiguousarray(vol_surf.vols, dtype=np.float64)

            # Allocate output (12 IR + 1 spot + 6 vega = 19)
            sensitivities = np.zeros((n_trades, 19), dtype=np.float64)

            blocks = (n_trades + threads_per_block - 1) // threads_per_block
            d_sensitivities = cuda.to_device(sensitivities)

            _bump_reval_equity_option_kernel[blocks, threads_per_block](
                cuda.to_device(notionals),
                cuda.to_device(strikes),
                cuda.to_device(maturities),
                cuda.to_device(div_yields),
                cuda.to_device(is_calls),
                cuda.to_device(eq_buckets),
                cuda.to_device(curve_rates),
                tenors_d,
                cuda.to_device(spot),
                cuda.to_device(vol_surf_arr),
                vega_exp_d,
                d_sensitivities,
            )

            if not CUDA_SIMULATOR:
                cuda.synchronize()
            d_sensitivities.copy_to_host(sensitivities)

            total_sensies_time += time.perf_counter() - sensies_start

            # Convert to CRIF rows
            tenor_labels = ["2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"]
            vega_labels = ["0.5y", "1.0y", "3.0y", "5.0y", "10.0y", "30.0y"]
            for idx, (orig_idx, trade) in enumerate(trade_list):
                # IR deltas
                for i in range(NUM_IR_TENORS):
                    delta = sensitivities[idx, i]
                    if abs(delta) > 1e-10:
                        all_crif_rows.append({
                            "TradeID": trade.trade_id,
                            "ProductClass": "RatesFX",
                            "RiskType": "Risk_IRCurve",
                            "Qualifier": trade.currency,
                            "Bucket": str(i + 1),
                            "Label1": tenor_labels[i],
                            "Label2": "OIS",
                            "Amount": delta,
                            "AmountCurrency": trade.currency,
                            "AmountUSD": delta,
                        })
                # Equity delta
                eq_delta = sensitivities[idx, 12]
                if abs(eq_delta) > 1e-10:
                    all_crif_rows.append({
                        "TradeID": trade.trade_id,
                        "ProductClass": "Equity",
                        "RiskType": "Risk_Equity",
                        "Qualifier": trade.underlying,
                        "Bucket": str(trade.equity_bucket + 1),
                        "Label1": "",
                        "Label2": "spot",
                        "Amount": eq_delta,
                        "AmountCurrency": trade.currency,
                        "AmountUSD": eq_delta,
                    })
                # Equity vega
                for i in range(NUM_VEGA_EXPIRIES):
                    vega = sensitivities[idx, 13 + i]
                    if abs(vega) > 1e-10:
                        all_crif_rows.append({
                            "TradeID": trade.trade_id,
                            "ProductClass": "Equity",
                            "RiskType": "Risk_EquityVol",
                            "Qualifier": trade.underlying,
                            "Bucket": str(trade.equity_bucket + 1),
                            "Label1": vega_labels[i],
                            "Label2": "",
                            "Amount": vega,
                            "AmountCurrency": trade.currency,
                            "AmountUSD": vega,
                        })

        elif trade_type == "fx_option":
            # Extract trade parameters
            notionals = np.array([t.notional for _, t in trade_list], dtype=np.float64)
            strikes = np.array([t.strike for _, t in trade_list], dtype=np.float64)
            maturities = np.array([t.maturity for _, t in trade_list], dtype=np.float64)
            is_calls = np.array([1 if t.is_call else 0 for _, t in trade_list], dtype=np.int32)

            # Market data (use first trade's currencies for simplified implementation)
            dom_ccy = trade_list[0][1].domestic_ccy
            fgn_ccy = trade_list[0][1].foreign_ccy
            pair = f"{fgn_ccy}{dom_ccy}"

            dom_curve = market.curves.get(dom_ccy)
            fgn_curve = market.curves.get(fgn_ccy)
            if dom_curve is None:
                continue
            if fgn_curve is None:
                fgn_rates = np.full(NUM_IR_TENORS, 0.02, dtype=np.float64)
            else:
                fgn_rates = np.ascontiguousarray(fgn_curve.zero_rates, dtype=np.float64)

            dom_rates = np.ascontiguousarray(dom_curve.zero_rates, dtype=np.float64)
            fx_spot = np.array([market.fx_spots.get(pair, trade_list[0][1].strike)], dtype=np.float64)

            vol_surf = market.vol_surfaces.get(pair)
            if vol_surf is None:
                vol_surf_arr = np.full(NUM_VEGA_EXPIRIES, 0.1, dtype=np.float64)
            else:
                vol_surf_arr = np.ascontiguousarray(vol_surf.vols, dtype=np.float64)

            # Allocate output (12 dom IR + 12 fgn IR + 1 FX + 6 vega = 31)
            sensitivities = np.zeros((n_trades, 31), dtype=np.float64)

            blocks = (n_trades + threads_per_block - 1) // threads_per_block
            d_sensitivities = cuda.to_device(sensitivities)

            _bump_reval_fx_option_kernel[blocks, threads_per_block](
                cuda.to_device(notionals),
                cuda.to_device(strikes),
                cuda.to_device(maturities),
                cuda.to_device(is_calls),
                cuda.to_device(dom_rates),
                cuda.to_device(fgn_rates),
                tenors_d,
                cuda.to_device(fx_spot),
                cuda.to_device(vol_surf_arr),
                vega_exp_d,
                d_sensitivities,
            )

            if not CUDA_SIMULATOR:
                cuda.synchronize()
            d_sensitivities.copy_to_host(sensitivities)

            total_sensies_time += time.perf_counter() - sensies_start

            # Convert to CRIF rows
            tenor_labels = ["2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"]
            vega_labels = ["0.5y", "1.0y", "3.0y", "5.0y", "10.0y", "30.0y"]
            for idx, (orig_idx, trade) in enumerate(trade_list):
                # Domestic IR deltas
                for i in range(NUM_IR_TENORS):
                    delta = sensitivities[idx, i]
                    if abs(delta) > 1e-10:
                        all_crif_rows.append({
                            "TradeID": trade.trade_id,
                            "ProductClass": "RatesFX",
                            "RiskType": "Risk_IRCurve",
                            "Qualifier": trade.domestic_ccy,
                            "Bucket": str(i + 1),
                            "Label1": tenor_labels[i],
                            "Label2": "OIS",
                            "Amount": delta,
                            "AmountCurrency": trade.domestic_ccy,
                            "AmountUSD": delta,
                        })
                # Foreign IR deltas
                for i in range(NUM_IR_TENORS):
                    delta = sensitivities[idx, 12 + i]
                    if abs(delta) > 1e-10:
                        all_crif_rows.append({
                            "TradeID": trade.trade_id,
                            "ProductClass": "RatesFX",
                            "RiskType": "Risk_IRCurve",
                            "Qualifier": trade.foreign_ccy,
                            "Bucket": str(i + 1),
                            "Label1": tenor_labels[i],
                            "Label2": "OIS",
                            "Amount": delta,
                            "AmountCurrency": trade.domestic_ccy,
                            "AmountUSD": delta,
                        })
                # FX delta
                fx_delta = sensitivities[idx, 24]
                if abs(fx_delta) > 1e-10:
                    all_crif_rows.append({
                        "TradeID": trade.trade_id,
                        "ProductClass": "RatesFX",
                        "RiskType": "Risk_FX",
                        "Qualifier": f"{trade.foreign_ccy}{trade.domestic_ccy}",
                        "Bucket": "",
                        "Label1": "",
                        "Label2": "",
                        "Amount": fx_delta,
                        "AmountCurrency": trade.domestic_ccy,
                        "AmountUSD": fx_delta,
                    })
                # FX vega
                for i in range(NUM_VEGA_EXPIRIES):
                    vega = sensitivities[idx, 25 + i]
                    if abs(vega) > 1e-10:
                        all_crif_rows.append({
                            "TradeID": trade.trade_id,
                            "ProductClass": "RatesFX",
                            "RiskType": "Risk_FXVol",
                            "Qualifier": f"{trade.foreign_ccy}{trade.domestic_ccy}",
                            "Bucket": "",
                            "Label1": vega_labels[i],
                            "Label2": "",
                            "Amount": vega,
                            "AmountCurrency": trade.domestic_ccy,
                            "AmountUSD": vega,
                        })

        elif trade_type == "inflation_swap":
            # Extract trade parameters
            notionals = np.array([t.notional for _, t in trade_list], dtype=np.float64)
            fixed_rates = np.array([t.fixed_rate for _, t in trade_list], dtype=np.float64)
            maturities = np.array([t.maturity for _, t in trade_list], dtype=np.float64)

            # Market data
            ccy = trade_list[0][1].currency
            curve = market.curves.get(ccy)
            if curve is None:
                continue
            curve_rates = np.ascontiguousarray(curve.zero_rates, dtype=np.float64)

            inflation = market.inflation
            if inflation is None:
                infl_rates = np.full(NUM_IR_TENORS, 0.025, dtype=np.float64)
                base_cpi = np.array([100.0], dtype=np.float64)
            else:
                infl_rates = np.ascontiguousarray(inflation.inflation_rates, dtype=np.float64)
                base_cpi = np.array([inflation.base_cpi], dtype=np.float64)

            # Allocate output (12 IR + 12 inflation = 24)
            sensitivities = np.zeros((n_trades, 24), dtype=np.float64)

            blocks = (n_trades + threads_per_block - 1) // threads_per_block
            d_sensitivities = cuda.to_device(sensitivities)

            _bump_reval_inflation_swap_kernel[blocks, threads_per_block](
                cuda.to_device(notionals),
                cuda.to_device(fixed_rates),
                cuda.to_device(maturities),
                cuda.to_device(curve_rates),
                cuda.to_device(infl_rates),
                cuda.to_device(base_cpi),
                tenors_d,
                d_sensitivities,
            )

            if not CUDA_SIMULATOR:
                cuda.synchronize()
            d_sensitivities.copy_to_host(sensitivities)

            total_sensies_time += time.perf_counter() - sensies_start

            # Convert to CRIF rows
            tenor_labels = ["2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"]
            for idx, (orig_idx, trade) in enumerate(trade_list):
                # IR deltas
                for i in range(NUM_IR_TENORS):
                    delta = sensitivities[idx, i]
                    if abs(delta) > 1e-10:
                        all_crif_rows.append({
                            "TradeID": trade.trade_id,
                            "ProductClass": "RatesFX",
                            "RiskType": "Risk_IRCurve",
                            "Qualifier": trade.currency,
                            "Bucket": str(i + 1),
                            "Label1": tenor_labels[i],
                            "Label2": "OIS",
                            "Amount": delta,
                            "AmountCurrency": trade.currency,
                            "AmountUSD": delta,
                        })
                # Inflation deltas
                for i in range(NUM_IR_TENORS):
                    delta = sensitivities[idx, 12 + i]
                    if abs(delta) > 1e-10:
                        all_crif_rows.append({
                            "TradeID": trade.trade_id,
                            "ProductClass": "RatesFX",
                            "RiskType": "Risk_Inflation",
                            "Qualifier": trade.currency,
                            "Bucket": str(i + 1),
                            "Label1": tenor_labels[i],
                            "Label2": "",
                            "Amount": delta,
                            "AmountCurrency": trade.currency,
                            "AmountUSD": delta,
                        })

        elif trade_type == "xccy_swap":
            # Extract trade parameters
            dom_notionals = np.array([t.dom_notional for _, t in trade_list], dtype=np.float64)
            fgn_notionals = np.array([t.fgn_notional for _, t in trade_list], dtype=np.float64)
            dom_fixed_rates = np.array([t.dom_fixed_rate for _, t in trade_list], dtype=np.float64)
            fgn_fixed_rates = np.array([t.fgn_fixed_rate for _, t in trade_list], dtype=np.float64)
            maturities = np.array([t.maturity for _, t in trade_list], dtype=np.float64)
            frequencies = np.array([t.frequency for _, t in trade_list], dtype=np.int32)
            exch_notionals = np.array([1 if t.exchange_notional else 0 for _, t in trade_list], dtype=np.int32)

            # Market data
            dom_ccy = trade_list[0][1].domestic_ccy
            fgn_ccy = trade_list[0][1].foreign_ccy
            pair = f"{fgn_ccy}{dom_ccy}"

            dom_curve = market.curves.get(dom_ccy)
            fgn_curve = market.curves.get(fgn_ccy)
            if dom_curve is None:
                continue
            if fgn_curve is None:
                fgn_rates = np.full(NUM_IR_TENORS, 0.02, dtype=np.float64)
            else:
                fgn_rates = np.ascontiguousarray(fgn_curve.zero_rates, dtype=np.float64)

            dom_rates = np.ascontiguousarray(dom_curve.zero_rates, dtype=np.float64)
            t0 = trade_list[0][1]
            fx_spot = np.array([market.fx_spots.get(pair, t0.dom_notional / t0.fgn_notional)], dtype=np.float64)

            # Allocate output (12 dom IR + 12 fgn IR + 1 FX = 25)
            sensitivities = np.zeros((n_trades, 25), dtype=np.float64)

            blocks = (n_trades + threads_per_block - 1) // threads_per_block
            d_sensitivities = cuda.to_device(sensitivities)

            _bump_reval_xccy_swap_kernel[blocks, threads_per_block](
                cuda.to_device(dom_notionals),
                cuda.to_device(fgn_notionals),
                cuda.to_device(dom_fixed_rates),
                cuda.to_device(fgn_fixed_rates),
                cuda.to_device(maturities),
                cuda.to_device(frequencies),
                cuda.to_device(exch_notionals),
                cuda.to_device(dom_rates),
                cuda.to_device(fgn_rates),
                cuda.to_device(fx_spot),
                tenors_d,
                d_sensitivities,
            )

            if not CUDA_SIMULATOR:
                cuda.synchronize()
            d_sensitivities.copy_to_host(sensitivities)

            total_sensies_time += time.perf_counter() - sensies_start

            # Convert to CRIF rows
            tenor_labels = ["2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"]
            for idx, (orig_idx, trade) in enumerate(trade_list):
                # Domestic IR deltas
                for i in range(NUM_IR_TENORS):
                    delta = sensitivities[idx, i]
                    if abs(delta) > 1e-10:
                        all_crif_rows.append({
                            "TradeID": trade.trade_id,
                            "ProductClass": "RatesFX",
                            "RiskType": "Risk_IRCurve",
                            "Qualifier": trade.domestic_ccy,
                            "Bucket": str(i + 1),
                            "Label1": tenor_labels[i],
                            "Label2": "OIS",
                            "Amount": delta,
                            "AmountCurrency": trade.domestic_ccy,
                            "AmountUSD": delta,
                        })
                # Foreign IR deltas
                for i in range(NUM_IR_TENORS):
                    delta = sensitivities[idx, 12 + i]
                    if abs(delta) > 1e-10:
                        all_crif_rows.append({
                            "TradeID": trade.trade_id,
                            "ProductClass": "RatesFX",
                            "RiskType": "Risk_IRCurve",
                            "Qualifier": trade.foreign_ccy,
                            "Bucket": str(i + 1),
                            "Label1": tenor_labels[i],
                            "Label2": "OIS",
                            "Amount": delta,
                            "AmountCurrency": trade.domestic_ccy,
                            "AmountUSD": delta,
                        })
                # FX delta
                fx_delta = sensitivities[idx, 24]
                if abs(fx_delta) > 1e-10:
                    all_crif_rows.append({
                        "TradeID": trade.trade_id,
                        "ProductClass": "RatesFX",
                        "RiskType": "Risk_FX",
                        "Qualifier": f"{trade.foreign_ccy}{trade.domestic_ccy}",
                        "Bucket": "",
                        "Label1": "",
                        "Label2": "",
                        "Amount": fx_delta,
                        "AmountCurrency": trade.domestic_ccy,
                        "AmountUSD": fx_delta,
                    })

    crif_df = pd.DataFrame(all_crif_rows) if all_crif_rows else pd.DataFrame()
    return crif_df, total_sensies_time


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

    # Use v2 optimized kernel when K ≤ 64 (fits in shared memory)
    use_v2 = K <= V2_MAX_K_SHARED

    if use_v2:
        # v2 kernel: 1 block per portfolio, 64 threads per block
        # Uses shared memory for correlation matrix
        blocks = P
        threads_per_block = V2_THREADS_FULL

        # Get 2D correlation matrix (either from gpu_arrays or create it)
        if gpu_arrays is not None and 'intra_corr_2d' in gpu_arrays:
            d_intra_corr_2d = gpu_arrays['intra_corr_2d']
        else:
            intra_corr_2d = intra_corr_flat.reshape(K, K)
            d_intra_corr_2d = cuda.to_device(np.ascontiguousarray(intra_corr_2d, dtype=np.float64))

        _simm_gradient_kernel_v2[blocks, threads_per_block](
            d_sens, d_weights, d_conc, d_bucket_id, d_rm_idx,
            d_bucket_rc, d_bucket_rm, d_intra_corr_2d, d_bucket_gamma,
            d_psi, d_num_buckets, d_im, d_grad
        )
    else:
        # v1 kernel: 1 thread per portfolio
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

    K = len(risk_weights)
    # Reshape flat correlation to 2D for v2 kernels
    intra_corr_2d = intra_corr_flat.reshape(K, K)

    return {
        'weights': cuda.to_device(np.ascontiguousarray(risk_weights, dtype=np.float64)),
        'concentration': cuda.to_device(np.ascontiguousarray(concentration, dtype=np.float64)),
        'bucket_id': cuda.to_device(np.ascontiguousarray(bucket_id, dtype=np.int32)),
        'risk_measure_idx': cuda.to_device(np.ascontiguousarray(risk_measure_idx, dtype=np.int32)),
        'bucket_rc': cuda.to_device(np.ascontiguousarray(bucket_rc, dtype=np.int32)),
        'bucket_rm': cuda.to_device(np.ascontiguousarray(bucket_rm, dtype=np.int32)),
        'intra_corr': cuda.to_device(np.ascontiguousarray(intra_corr_flat, dtype=np.float64)),
        'intra_corr_2d': cuda.to_device(np.ascontiguousarray(intra_corr_2d, dtype=np.float64)),
        'bucket_gamma': cuda.to_device(np.ascontiguousarray(bucket_gamma_flat, dtype=np.float64)),
        'psi': cuda.to_device(PSI_MATRIX),
        'num_buckets': cuda.to_device(np.array([num_buckets], dtype=np.int32)),
        'K': K,  # Store K for v2 kernel selection
    }


# =============================================================================
# Forward-Only SIMM Kernel (no gradient) — for brute-force search
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def _simm_forward_kernel(
        sensitivities,        # (N, K)
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
        im_output,            # (N,)
    ):
        """Forward-only SIMM: computes IM without gradients (~50% less work per thread)."""
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
        margin = cuda.local.array(12, dtype=numba.float64)
        rc_margin = cuda.local.array(6, dtype=numba.float64)

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

        # Step 3: Intra-bucket K_b^2
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

        for i in range(12):
            margin[i] = math.sqrt(margin[i]) if margin[i] > 0.0 else 0.0

        # Step 5: Per risk class total = Delta + Vega
        for r in range(6):
            rc_margin[r] = margin[r * 2] + margin[r * 2 + 1]

        # Step 6: Cross-RC aggregation
        im_sq = 0.0
        for r in range(6):
            for s in range(6):
                im_sq += psi_matrix[r, s] * rc_margin[r] * rc_margin[s]

        im_output[p] = math.sqrt(im_sq) if im_sq > 0.0 else 0.0


# =============================================================================
# Optimized CUDA Kernels v2 - Shared Memory + Parallel Reduction
# =============================================================================
#
# Optimization strategy for variable K (12-200):
#   - 1 thread-block per portfolio, 64 threads per block
#   - Shared memory for correlation matrix tiles (up to 32KB)
#   - Parallel reduction for K×K correlation computation
#   - For K ≤ 64: Load full correlation matrix to shared memory
#   - For K > 64: Use tiled approach or fall back to v1
#
# Occupancy analysis (H100):
#   - 64 threads/block, ~35KB shared memory for K=64
#   - Theoretical: ~2 blocks/SM (96KB shared limit)
#   - Better than v1 which uses large local arrays
# =============================================================================

# Configuration for v2 kernels
V2_THREADS_FULL = 64    # Threads per block for full SIMM kernel
V2_MAX_K_SHARED = 64    # Max K that fits in shared memory (64×64×8 = 32KB)

if CUDA_AVAILABLE:
    @cuda.jit
    def _simm_gradient_kernel_v2(
        sensitivities,        # (P, K)
        risk_weights,         # (K,)
        concentration,        # (K,)
        bucket_id,            # (K,) int32
        risk_measure_idx,     # (K,) int32
        bucket_rc,            # (B,) int32
        bucket_rm,            # (B,) int32
        intra_corr_2d,        # (K, K) 2D correlation matrix
        bucket_gamma_flat,    # (B*B,)
        psi_matrix,           # (6, 6)
        num_buckets,          # (1,) int32
        im_output,            # (P,)
        gradients,            # (P, K)
    ):
        """
        Optimized SIMM gradient kernel using shared memory.

        Grid: P blocks (1 per portfolio)
        Block: 64 threads

        For K ≤ V2_MAX_K_SHARED: Uses shared memory for correlation matrix
        Note: This kernel requires K ≤ 64. For K > 64, use v1 kernel.
        """
        p = cuda.blockIdx.x
        tid = cuda.threadIdx.x

        if p >= sensitivities.shape[0]:
            return

        K = sensitivities.shape[1]
        B = num_buckets[0]

        # Shared memory for correlation matrix tile (K×K), ws, bucket accumulators
        # For K=64: 64×64×8 = 32KB correlation + 64×8 = 512B ws + misc = ~33KB
        shared_corr = cuda.shared.array((V2_MAX_K_SHARED, V2_MAX_K_SHARED), dtype=numba.float64)
        shared_ws = cuda.shared.array(V2_MAX_K_SHARED, dtype=numba.float64)
        shared_K_b_sq = cuda.shared.array(MAX_B, dtype=numba.float64)
        shared_S_b = cuda.shared.array(MAX_B, dtype=numba.float64)
        shared_margin = cuda.shared.array(12, dtype=numba.float64)
        shared_rc_margin = cuda.shared.array(6, dtype=numba.float64)
        shared_im = cuda.shared.array(1, dtype=numba.float64)

        Kc = min(K, V2_MAX_K_SHARED)
        Bc = min(B, MAX_B)

        # Step 0: Initialize shared arrays (cooperative)
        for b in range(tid, Bc, V2_THREADS_FULL):
            shared_K_b_sq[b] = 0.0
            shared_S_b[b] = 0.0
        for i in range(tid, 12, V2_THREADS_FULL):
            shared_margin[i] = 0.0
        cuda.syncthreads()

        # Step 1: Cooperative load of correlation matrix (64 threads load K×K elements)
        total_corr = Kc * Kc
        for idx in range(tid, total_corr, V2_THREADS_FULL):
            i = idx // Kc
            j = idx % Kc
            if i < K and j < K:
                shared_corr[i, j] = intra_corr_2d[i, j]
        cuda.syncthreads()

        # Step 2: Compute weighted sensitivities (threads 0..K-1)
        if tid < Kc:
            shared_ws[tid] = sensitivities[p, tid] * risk_weights[tid] * concentration[tid]
        cuda.syncthreads()

        # Step 3: Compute bucket sums S_b (parallel)
        # Each thread handles subset of factors
        for k in range(tid, Kc, V2_THREADS_FULL):
            b = bucket_id[k]
            cuda.atomic.add(shared_S_b, b, shared_ws[k])
        cuda.syncthreads()

        # Step 4: Intra-bucket K_b^2 computation (parallel reduction)
        # Each thread computes partial sum of correlation terms
        # Then atomic add to shared bucket accumulator
        for k in range(Kc):
            b_k = bucket_id[k]
            # Each thread handles a subset of l values
            partial_sum = 0.0
            for l in range(tid, Kc, V2_THREADS_FULL):
                b_l = bucket_id[l]
                if b_k == b_l:  # Only add intra-bucket terms
                    partial_sum += shared_corr[k, l] * shared_ws[k] * shared_ws[l]
            if partial_sum != 0.0:
                cuda.atomic.add(shared_K_b_sq, b_k, partial_sum)
        cuda.syncthreads()

        # Step 5: Per (RC, RM) margin with inter-bucket gamma (thread 0)
        if tid == 0:
            for b in range(Bc):
                rc_b = bucket_rc[b]
                rm_b = bucket_rm[b]
                rcrm = rc_b * 2 + rm_b
                shared_margin[rcrm] += shared_K_b_sq[b]

                for c in range(Bc):
                    if c != b and bucket_rc[c] == rc_b and bucket_rm[c] == rm_b:
                        shared_margin[rcrm] += bucket_gamma_flat[b * B + c] * shared_S_b[b] * shared_S_b[c]

            # sqrt each margin
            for i in range(12):
                shared_margin[i] = math.sqrt(shared_margin[i]) if shared_margin[i] > 0.0 else 0.0

            # Per risk class total = Delta + Vega
            for r in range(6):
                shared_rc_margin[r] = shared_margin[r * 2] + shared_margin[r * 2 + 1]

            # Cross-RC aggregation: IM = sqrt(Σ ψ_rs × M_r × M_s)
            im_sq = 0.0
            for r in range(6):
                for s in range(6):
                    im_sq += psi_matrix[r, s] * shared_rc_margin[r] * shared_rc_margin[s]

            im_p = math.sqrt(im_sq) if im_sq > 0.0 else 0.0
            im_output[p] = im_p
            shared_im[0] = im_p
        cuda.syncthreads()

        im_p = shared_im[0]

        # Step 6: Gradient computation (parallel across K factors)
        if tid < Kc:
            if im_p < 1e-30:
                gradients[p, tid] = 0.0
            else:
                k = tid
                b_k = bucket_id[k]
                rc_k = bucket_rc[b_k]
                rm_k = bucket_rm[b_k]
                rcrm_k = rc_k * 2 + rm_k
                margin_rcrm = shared_margin[rcrm_k]

                if margin_rcrm < 1e-30:
                    gradients[p, k] = 0.0
                else:
                    # dIM/d(rc_margin_r) = Σ_s ψ_rs × rc_margin_s / IM
                    dim_drcm = 0.0
                    for s in range(6):
                        dim_drcm += psi_matrix[rc_k, s] * shared_rc_margin[s]
                    dim_drcm /= im_p

                    # Intra-bucket derivative
                    intra_deriv = 0.0
                    for l in range(Kc):
                        if bucket_id[l] == b_k:
                            intra_deriv += shared_corr[k, l] * shared_ws[l]
                    intra_deriv *= 2.0

                    # Inter-bucket derivative
                    inter_deriv = 0.0
                    for c in range(Bc):
                        if c != b_k and bucket_rc[c] == rc_k and bucket_rm[c] == rm_k:
                            inter_deriv += bucket_gamma_flat[b_k * B + c] * shared_S_b[c]
                    inter_deriv *= 2.0

                    dmargin_dws = (intra_deriv + inter_deriv) / (2.0 * margin_rcrm)
                    gradients[p, k] = dim_drcm * dmargin_dws * risk_weights[k] * concentration[k]

        # Zero out factors beyond Kc
        for k in range(Kc + tid, K, V2_THREADS_FULL):
            gradients[p, k] = 0.0

    @cuda.jit
    def _simm_forward_kernel_v2(
        sensitivities,        # (N, K)
        risk_weights,         # (K,)
        concentration,        # (K,)
        bucket_id,            # (K,) int32
        risk_measure_idx,     # (K,) int32
        bucket_rc,            # (B,) int32
        bucket_rm,            # (B,) int32
        intra_corr_2d,        # (K, K) 2D correlation matrix
        bucket_gamma_flat,    # (B*B,)
        psi_matrix,           # (6, 6)
        num_buckets,          # (1,) int32
        im_output,            # (N,)
    ):
        """
        Optimized forward-only SIMM kernel using shared memory.
        ~50% less work than gradient version.
        """
        p = cuda.blockIdx.x
        tid = cuda.threadIdx.x

        if p >= sensitivities.shape[0]:
            return

        K = sensitivities.shape[1]
        B = num_buckets[0]

        # Shared memory
        shared_corr = cuda.shared.array((V2_MAX_K_SHARED, V2_MAX_K_SHARED), dtype=numba.float64)
        shared_ws = cuda.shared.array(V2_MAX_K_SHARED, dtype=numba.float64)
        shared_K_b_sq = cuda.shared.array(MAX_B, dtype=numba.float64)
        shared_S_b = cuda.shared.array(MAX_B, dtype=numba.float64)
        shared_margin = cuda.shared.array(12, dtype=numba.float64)
        shared_rc_margin = cuda.shared.array(6, dtype=numba.float64)

        Kc = min(K, V2_MAX_K_SHARED)
        Bc = min(B, MAX_B)

        # Initialize
        for b in range(tid, Bc, V2_THREADS_FULL):
            shared_K_b_sq[b] = 0.0
            shared_S_b[b] = 0.0
        for i in range(tid, 12, V2_THREADS_FULL):
            shared_margin[i] = 0.0
        cuda.syncthreads()

        # Load correlation matrix
        total_corr = Kc * Kc
        for idx in range(tid, total_corr, V2_THREADS_FULL):
            i = idx // Kc
            j = idx % Kc
            if i < K and j < K:
                shared_corr[i, j] = intra_corr_2d[i, j]
        cuda.syncthreads()

        # Compute ws
        if tid < Kc:
            shared_ws[tid] = sensitivities[p, tid] * risk_weights[tid] * concentration[tid]
        cuda.syncthreads()

        # Bucket sums
        for k in range(tid, Kc, V2_THREADS_FULL):
            b = bucket_id[k]
            cuda.atomic.add(shared_S_b, b, shared_ws[k])
        cuda.syncthreads()

        # Intra-bucket K_b^2
        for k in range(Kc):
            b_k = bucket_id[k]
            partial_sum = 0.0
            for l in range(tid, Kc, V2_THREADS_FULL):
                b_l = bucket_id[l]
                if b_k == b_l:
                    partial_sum += shared_corr[k, l] * shared_ws[k] * shared_ws[l]
            if partial_sum != 0.0:
                cuda.atomic.add(shared_K_b_sq, b_k, partial_sum)
        cuda.syncthreads()

        # Final aggregation (thread 0)
        if tid == 0:
            for b in range(Bc):
                rc_b = bucket_rc[b]
                rm_b = bucket_rm[b]
                rcrm = rc_b * 2 + rm_b
                shared_margin[rcrm] += shared_K_b_sq[b]

                for c in range(Bc):
                    if c != b and bucket_rc[c] == rc_b and bucket_rm[c] == rm_b:
                        shared_margin[rcrm] += bucket_gamma_flat[b * B + c] * shared_S_b[b] * shared_S_b[c]

            for i in range(12):
                shared_margin[i] = math.sqrt(shared_margin[i]) if shared_margin[i] > 0.0 else 0.0

            for r in range(6):
                shared_rc_margin[r] = shared_margin[r * 2] + shared_margin[r * 2 + 1]

            im_sq = 0.0
            for r in range(6):
                for s in range(6):
                    im_sq += psi_matrix[r, s] * shared_rc_margin[r] * shared_rc_margin[s]

            im_output[p] = math.sqrt(im_sq) if im_sq > 0.0 else 0.0


def compute_simm_im_only_cuda(
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
    max_batch: int = 500_000,
) -> np.ndarray:
    """
    Compute SIMM IM only (no gradients) using forward-only CUDA kernel.

    Supports chunking for large N (> max_batch rows).

    Returns:
        im_values: (N,) array of IM per row
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")

    N, K = sensitivities.shape
    sensitivities = np.ascontiguousarray(sensitivities, dtype=np.float64)

    if not CUDA_SIMULATOR:
        cuda.select_device(device)

    # Resolve constant arrays
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
        d_intra_corr_2d = gpu_arrays.get('intra_corr_2d')
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
        d_intra_corr_2d = None

    # Use v2 optimized kernel when K ≤ 64 (fits in shared memory)
    use_v2 = K <= V2_MAX_K_SHARED

    # Prepare 2D correlation for v2 kernel if needed
    if use_v2 and d_intra_corr_2d is None:
        intra_corr_2d = intra_corr_flat.reshape(K, K)
        d_intra_corr_2d = cuda.to_device(np.ascontiguousarray(intra_corr_2d, dtype=np.float64))

    # Chunked evaluation for large N
    if N <= max_batch:
        im_output = np.zeros(N, dtype=np.float64)
        d_sens = cuda.to_device(sensitivities)
        d_im = cuda.to_device(im_output)

        if use_v2:
            # v2 kernel: 1 block per scenario, 64 threads per block
            blocks = N
            threads_per_block = V2_THREADS_FULL

            _simm_forward_kernel_v2[blocks, threads_per_block](
                d_sens, d_weights, d_conc, d_bucket_id, d_rm_idx,
                d_bucket_rc, d_bucket_rm, d_intra_corr_2d, d_bucket_gamma,
                d_psi, d_num_buckets, d_im
            )
        else:
            # v1 kernel: 1 thread per scenario
            threads_per_block = 256
            blocks = (N + threads_per_block - 1) // threads_per_block

            _simm_forward_kernel[blocks, threads_per_block](
                d_sens, d_weights, d_conc, d_bucket_id, d_rm_idx,
                d_bucket_rc, d_bucket_rm, d_intra_corr, d_bucket_gamma,
                d_psi, d_num_buckets, d_im
            )

        if not CUDA_SIMULATOR:
            cuda.synchronize()
        d_im.copy_to_host(im_output)
        return im_output
    else:
        # Chunked
        im_all = np.zeros(N, dtype=np.float64)
        for start in range(0, N, max_batch):
            end = min(start + max_batch, N)
            chunk = sensitivities[start:end]
            chunk_n = end - start
            im_chunk = np.zeros(chunk_n, dtype=np.float64)
            d_sens = cuda.to_device(chunk)
            d_im = cuda.to_device(im_chunk)

            if use_v2:
                blocks = chunk_n
                threads_per_block = V2_THREADS_FULL

                _simm_forward_kernel_v2[blocks, threads_per_block](
                    d_sens, d_weights, d_conc, d_bucket_id, d_rm_idx,
                    d_bucket_rc, d_bucket_rm, d_intra_corr_2d, d_bucket_gamma,
                    d_psi, d_num_buckets, d_im
                )
            else:
                threads_per_block = 256
                blocks = (chunk_n + threads_per_block - 1) // threads_per_block

                _simm_forward_kernel[blocks, threads_per_block](
                    d_sens, d_weights, d_conc, d_bucket_id, d_rm_idx,
                    d_bucket_rc, d_bucket_rm, d_intra_corr, d_bucket_gamma,
                    d_psi, d_num_buckets, d_im
                )

            if not CUDA_SIMULATOR:
                cuda.synchronize()
            d_im.copy_to_host(im_chunk)
            im_all[start:end] = im_chunk

        return im_all


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
    """Project each row to probability simplex (sum=1, all>=0), vectorized."""
    T, P = x.shape
    u = np.sort(x, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1)
    arange = np.arange(1, P + 1)
    mask = u * arange > (cssv - 1)
    rho = P - 1 - np.argmax(mask[:, ::-1], axis=1)
    theta = (cssv[np.arange(T), rho] - 1.0) / (rho + 1)
    return np.maximum(x - theta[:, None], 0.0)


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
# Pre-Trade Analytics: Counterparty Routing via Brute-Force
# =============================================================================

@dataclass
class PreTradeRoutingResult:
    """Result of pre-trade counterparty routing analysis."""
    marginal_ims: np.ndarray      # Marginal IM for each portfolio (P,)
    base_ims: np.ndarray          # Current IM for each portfolio before new trade (P,)
    new_ims: np.ndarray           # IM after adding new trade to each portfolio (P,)
    best_portfolio: int           # Portfolio index with lowest marginal IM
    best_marginal_im: float       # Lowest marginal IM value
    worst_portfolio: int          # Portfolio index with highest marginal IM
    worst_marginal_im: float      # Highest marginal IM value
    sensies_time_sec: float       # Time to compute new trade sensitivities
    eval_time_sec: float          # Time to evaluate all 2P SIMM scenarios


def pretrade_routing_gpu(
    S: np.ndarray,                       # (T, K) sensitivity matrix
    allocation: np.ndarray,              # (T, P) current allocation
    new_trade_sens: np.ndarray,          # (K,) new trade sensitivities
    risk_weights: np.ndarray,            # (K,)
    concentration: np.ndarray,           # (K,)
    bucket_id: np.ndarray,               # (K,)
    risk_measure_idx: np.ndarray,        # (K,)
    bucket_rc: np.ndarray,               # (B,)
    bucket_rm: np.ndarray,               # (B,)
    intra_corr_flat: np.ndarray,         # (K*K,)
    bucket_gamma_flat: np.ndarray,       # (B*B,)
    num_buckets: int,
    device: int = 0,
    gpu_arrays: dict = None,
) -> PreTradeRoutingResult:
    """
    Compute marginal IM for routing a new trade to each portfolio using GPU brute-force.

    For each portfolio p:
      - base_im[p] = SIMM(current portfolio p)
      - new_im[p] = SIMM(portfolio p + new trade)
      - marginal_im[p] = new_im[p] - base_im[p]

    Returns the portfolio with lowest marginal IM (best routing decision).

    This is the GPU brute-force equivalent of AADC's gradient-based marginal IM:
      AADC: marginal ≈ ∂IM/∂S · new_trade_sens (first-order approximation)
      GPU:  marginal = IM(with) - IM(without) (exact, but 2P evaluations)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available for pre-trade routing")

    T, K = S.shape
    P = allocation.shape[1]

    eval_start = time.perf_counter()

    # Compute current aggregated sensitivities per portfolio
    agg_S_base = (S.T @ allocation).T  # (P, K)

    # Compute aggregated sensitivities with new trade added to each portfolio
    # new_trade_sens is broadcast-added to each portfolio
    agg_S_with_new = agg_S_base + new_trade_sens[np.newaxis, :]  # (P, K)

    # Stack both scenarios for a single GPU kernel launch: [base_0, ..., base_P-1, new_0, ..., new_P-1]
    all_scenarios = np.vstack([agg_S_base, agg_S_with_new])  # (2P, K)

    # Evaluate all 2P scenarios in one kernel launch
    all_ims = compute_simm_im_only_cuda(
        all_scenarios, risk_weights, concentration, bucket_id, risk_measure_idx,
        bucket_rc, bucket_rm, intra_corr_flat, bucket_gamma_flat, num_buckets,
        device, gpu_arrays,
    )

    eval_time = time.perf_counter() - eval_start

    # Split results
    base_ims = all_ims[:P]
    new_ims = all_ims[P:]
    marginal_ims = new_ims - base_ims

    # Find best and worst portfolios
    best_p = int(np.argmin(marginal_ims))
    worst_p = int(np.argmax(marginal_ims))

    return PreTradeRoutingResult(
        marginal_ims=marginal_ims,
        base_ims=base_ims,
        new_ims=new_ims,
        best_portfolio=best_p,
        best_marginal_im=float(marginal_ims[best_p]),
        worst_portfolio=worst_p,
        worst_marginal_im=float(marginal_ims[worst_p]),
        sensies_time_sec=0.0,  # Will be set by caller if new trade CRIF was computed
        eval_time_sec=eval_time,
    )


def pretrade_routing_with_crif_gpu(
    trades: list,
    market,
    new_trade,
    S: np.ndarray,
    allocation: np.ndarray,
    risk_weights: np.ndarray,
    concentration: np.ndarray,
    bucket_id: np.ndarray,
    risk_measure_idx: np.ndarray,
    bucket_rc: np.ndarray,
    bucket_rm: np.ndarray,
    intra_corr_flat: np.ndarray,
    bucket_gamma_flat: np.ndarray,
    num_buckets: int,
    risk_factors: list,
    device: int = 0,
    gpu_arrays: dict = None,
    crif_method: str = 'gpu',
) -> PreTradeRoutingResult:
    """
    Full pre-trade routing: compute new trade CRIF, then route via brute-force.

    This includes sensitivity computation time (crif_sensies) in the result.
    """
    K = S.shape[1]

    # Step 1: Compute CRIF for new trade
    sensies_start = time.perf_counter()

    if crif_method == 'gpu':
        crif_df, _ = compute_crif_gpu([new_trade], market, device)
    else:
        crif_df = compute_crif_for_trades([new_trade], market)

    sensies_time = time.perf_counter() - sensies_start

    # Step 2: Map CRIF to K-dimensional sensitivity vector
    new_trade_sens = np.zeros(K, dtype=np.float64)
    rf_to_idx = {rf: i for i, rf in enumerate(risk_factors)}

    for _, row in crif_df.iterrows():
        rf = (row['RiskType'], row['Qualifier'], row.get('Bucket', ''), row.get('Label1', ''))
        k_idx = rf_to_idx.get(rf)
        if k_idx is not None:
            new_trade_sens[k_idx] += row['AmountUSD']

    # Step 3: Run routing
    result = pretrade_routing_gpu(
        S, allocation, new_trade_sens,
        risk_weights, concentration, bucket_id, risk_measure_idx,
        bucket_rc, bucket_rm, intra_corr_flat, bucket_gamma_flat,
        num_buckets, device, gpu_arrays,
    )

    # Add sensies time
    result.sensies_time_sec = sensies_time

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
    crif_method: str = 'gpu',  # 'gpu' or 'cpu' - GPU uses CUDA bump-and-revalue
    pretrade: bool = False,   # Run pre-trade routing analysis
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
        crif_method: 'gpu' for CUDA bump-and-revalue, 'cpu' for Python bump-and-revalue
        pretrade: Run pre-trade routing analysis for a random new trade

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

    # Step 2: Compute CRIF (bump-and-revalue)
    if verbose:
        method_label = "GPU CUDA" if crif_method == 'gpu' else "CPU Python"
        print(f"\nStep 2: Computing CRIF sensitivities ({method_label} bump-and-revalue)...")
    crif_start = time.perf_counter()

    if crif_method == 'gpu':
        # GPU bump-and-revalue - sensies_time tracked separately
        crif_df, sensies_time = compute_crif_gpu(trades, market, device)
        crif_records = crif_df.to_dict('records') if len(crif_df) > 0 else []
        results['timings']['crif_sensies'] = sensies_time  # GPU kernel time for sensitivities
    else:
        # CPU bump-and-revalue (fallback)
        sensies_start = time.perf_counter()
        crif_df = compute_crif_for_trades(trades, market)
        sensies_time = time.perf_counter() - sensies_start
        crif_records = crif_df.to_dict('records') if len(crif_df) > 0 else []
        results['timings']['crif_sensies'] = sensies_time  # CPU time for sensitivities

    results['timings']['crif_computation'] = time.perf_counter() - crif_start
    results['num_sensitivities'] = len(crif_records)
    results['crif_method'] = crif_method

    if verbose:
        print(f"  Computed {len(crif_records)} sensitivities in {results['timings']['crif_computation']:.3f}s")
        print(f"  Sensitivity kernel time: {results['timings']['crif_sensies']:.4f}s")

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

    # Step 7: Pre-trade routing (if requested)
    if pretrade:
        if verbose:
            print(f"\nStep 6: Pre-trade routing (brute-force {2*P} SIMM evaluations)...")

        # Generate a random new trade of the first type
        new_trade_type = trade_types[0] if trade_types else 'ir_swap'
        from model.trade_types import generate_trades_by_type
        new_trades = generate_trades_by_type(new_trade_type, 1, currencies, seed=9999)
        new_trade = new_trades[0]
        new_trade.trade_id = "NEW_TRADE_001"

        if verbose:
            print(f"  New trade: {new_trade.trade_id} ({new_trade_type})")

        # Pre-allocate GPU arrays for efficiency
        gpu_arrays = preallocate_gpu_arrays(
            risk_weights, concentration, bucket_id, risk_measure_idx,
            bucket_rc, bucket_rm, intra_corr, bucket_gamma, B, device,
        )

        # Run pre-trade routing with CRIF computation
        pt_result = pretrade_routing_with_crif_gpu(
            trades, market, new_trade,
            S, initial_allocation,
            risk_weights, concentration, bucket_id, risk_measure_idx,
            bucket_rc, bucket_rm, intra_corr, bucket_gamma, B,
            risk_factors, device, gpu_arrays, crif_method,
        )

        results['pretrade'] = {
            'best_portfolio': pt_result.best_portfolio,
            'best_marginal_im': pt_result.best_marginal_im,
            'worst_portfolio': pt_result.worst_portfolio,
            'worst_marginal_im': pt_result.worst_marginal_im,
            'sensies_time_sec': pt_result.sensies_time_sec,
            'eval_time_sec': pt_result.eval_time_sec,
            'marginal_ims': pt_result.marginal_ims.tolist(),
        }
        results['timings']['pretrade_sensies'] = pt_result.sensies_time_sec
        results['timings']['pretrade_eval'] = pt_result.eval_time_sec

        if verbose:
            print(f"  Routing results:")
            for p in range(P):
                marker = " <-- BEST" if p == pt_result.best_portfolio else (" <-- WORST" if p == pt_result.worst_portfolio else "")
                print(f"    Portfolio {p}: base=${pt_result.base_ims[p]:,.0f}, "
                      f"with_new=${pt_result.new_ims[p]:,.0f}, "
                      f"marginal=${pt_result.marginal_ims[p]:,.0f}{marker}")
            print(f"  Recommendation: Route to portfolio {pt_result.best_portfolio} "
                  f"(marginal IM: ${pt_result.best_marginal_im:,.0f})")
            print(f"  Sensies time: {pt_result.sensies_time_sec*1000:.2f} ms")
            print(f"  SIMM eval time: {pt_result.eval_time_sec*1000:.2f} ms")

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
    parser.add_argument('--crif-method', choices=['gpu', 'cpu'], default='gpu',
                        help='CRIF generation method: gpu (CUDA bump-and-revalue) or cpu (Python)')
    parser.add_argument('--pretrade', action='store_true',
                        help='Run pre-trade routing analysis for a random new trade')
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
            crif_method=args.crif_method,
            pretrade=args.pretrade,
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
        if 'pretrade' in results:
            pt = results['pretrade']
            print(f"\nPre-Trade Routing (brute-force):")
            print(f"  Best portfolio:  {pt['best_portfolio']} (marginal IM: ${pt['best_marginal_im']:,.2f})")
            print(f"  Worst portfolio: {pt['worst_portfolio']} (marginal IM: ${pt['worst_marginal_im']:,.2f})")
            print(f"  Sensies time:    {pt['sensies_time_sec']*1000:.2f} ms")
            print(f"  SIMM eval time:  {pt['eval_time_sec']*1000:.2f} ms (2×{args.portfolios} scenarios)")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
