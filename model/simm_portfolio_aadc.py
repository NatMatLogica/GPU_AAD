"""
SIMM Portfolio AADC - AADC-enabled SIMM implementation with portfolio allocation.

Same portfolio creation logic as simm_portfolio_baseline.py, but uses AADC for:
- Pricing & risk kernel recording per trade type (AAD instead of bump-and-revalue)
- CRIF sensitivity computation via AADC (single forward + adjoint pass)
- SIMM kernel recording per group (dIM/dsensitivity via AADC)
- Per-trade SIMM contribution via Euler decomposition
- Trade reallocation using gradient info (--reallocate N)

Key differences from baseline:
- Baseline: O(N) pricing passes per trade for N risk factors (bump & revalue)
- AADC: O(1) pricing passes per trade (~2-4x overhead, single adjoint pass)
- Baseline: O(M) SIMM evaluations for M CRIF entries (gradient via bump)
- AADC: O(1) SIMM evaluation for gradient (single adjoint pass)

Reallocation (--reallocate N):
  When specified, after computing per-trade contributions, the N trades with
  highest absolute margin contribution are candidates for reallocation. Each
  trade is moved to the group where its marginal impact (estimated via dIM/ds
  gradient) is lowest. After moves, SIMM is recomputed for affected groups
  and compared against the gradient-based estimate.

Version: 2.2.0

Usage:
    python -m model.simm_portfolio_aadc \
        --trades 10 --simm-buckets 2 --portfolios 5 --threads 8 \
        --trade-types ir_swap,equity_option,fx_option,inflation_swap,xccy_swap

    # With reallocation:
    python -m model.simm_portfolio_aadc \
        --trades 10 --simm-buckets 2 --portfolios 5 --threads 8 \
        --trade-types ir_swap,equity_option --reallocate 3
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Version
MODEL_VERSION = "2.6.0"  # Iterative gradient refresh during reallocation
MODEL_NAME = "simm_portfolio_aadc_py"

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.trade_types import (
    MarketEnvironment,
    IRSwapTrade, EquityOptionTrade, FXOptionTrade,
    InflationSwapTrade, XCCYSwapTrade,
    YieldCurve, VolSurface, InflationCurve,
    IR_TENORS, IR_TENOR_LABELS, NUM_IR_TENORS,
)
from common.portfolio import (
    parse_common_args, generate_portfolio, run_simm,
    make_empty_group_result,
    save_crif_log, write_log,
    print_results_table, build_log_rows,
    compute_trade_contributions, print_trade_contributions,
    save_trade_contributions_log,
    SIMM_RISK_CLASSES, LOG_FILE,
)

# Import ISDA SIMM v2.6 parameters for correlations and weights
from Weights_and_Corr.v2_6 import (
    ir_corr,  # 12x12 IR tenor correlation matrix
    reg_vol_rw, low_vol_rw, high_vol_rw,  # Currency-specific IR weights
    reg_vol_ccy_bucket, low_vol_ccy_bucket,  # Currency volatility buckets
    inflation_rw, inflation_corr,  # Inflation parameters
    sub_curves_corr,  # Correlation between sub-curves of same currency
    equity_corr, commodity_corr, fx_vega_corr,  # Other risk class correlations
)

# Try to import AADC
try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False
    print("Warning: AADC not available. Install MatLogica AADC for AAD support.")

# Cross-risk-class correlation matrix (psi) - ISDA SIMM v2.6/v2.7
PSI_MATRIX = np.array([
    [1.00, 0.04, 0.04, 0.07, 0.37, 0.14],  # Rates
    [0.04, 1.00, 0.54, 0.70, 0.27, 0.37],  # CreditQ
    [0.04, 0.54, 1.00, 0.46, 0.24, 0.15],  # CreditNonQ
    [0.07, 0.70, 0.46, 1.00, 0.35, 0.39],  # Equity
    [0.37, 0.27, 0.24, 0.35, 1.00, 0.35],  # Commodity
    [0.14, 0.37, 0.15, 0.39, 0.35, 1.00],  # FX
])

# Simplified risk weights for AADC SIMM kernel (legacy - now using v2_6 params)
IR_RISK_WEIGHTS = np.array([
    77, 77, 68, 56, 52, 50, 51, 52, 50, 51, 51, 64
], dtype=float)
FX_RISK_WEIGHT = 8.4
EQUITY_RISK_WEIGHT = 25.0
INFLATION_RISK_WEIGHT = 63.0

# SIMM tenor list (order matches ir_corr matrix)
SIMM_TENOR_LIST = ['2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
SIMM_TENOR_TO_IDX = {t: i for i, t in enumerate(SIMM_TENOR_LIST)}

# Build numpy IR correlation matrix from v2_6 data
IR_CORR_MATRIX = np.array(ir_corr)  # 12x12

def _get_ir_tenor_index(label1: str) -> int:
    """Map Label1 (e.g., '3m', '5y') to IR tenor index (0-11)."""
    label_lower = str(label1).lower().strip()
    return SIMM_TENOR_TO_IDX.get(label_lower, 6)  # Default to ~3y if not found


def _get_ir_correlation(tenor1: str, tenor2: str) -> float:
    """Get IR intra-bucket correlation between two tenors."""
    idx1 = _get_ir_tenor_index(tenor1)
    idx2 = _get_ir_tenor_index(tenor2)
    return IR_CORR_MATRIX[idx1, idx2]


def _get_ir_risk_weight_v26(currency: str, tenor: str) -> float:
    """Get ISDA v2.6 currency-specific IR risk weight."""
    tenor_lower = str(tenor).lower().strip()
    currency_upper = str(currency).upper().strip()

    if currency_upper in reg_vol_ccy_bucket:
        return reg_vol_rw.get(tenor_lower, 50.0)
    elif currency_upper in low_vol_ccy_bucket:
        return low_vol_rw.get(tenor_lower, 15.0)
    else:
        # High volatility currencies
        return high_vol_rw.get(tenor_lower, 100.0)


def _get_intra_correlation(risk_class: str, risk_type1: str, risk_type2: str,
                           label1_1: str, label1_2: str, bucket: str = None) -> float:
    """
    Get intra-bucket correlation for same risk class.

    For IR: Uses 12x12 tenor correlation matrix from ISDA SIMM v2.6.
    For Equity/Commodity: Uses single correlation value per bucket.
    For FX: Assumes 0.5 vega correlation.
    """
    if risk_class == "Rates":
        # IR Delta: tenor-based correlations
        if risk_type1 == "Risk_Inflation" or risk_type2 == "Risk_Inflation":
            if risk_type1 == risk_type2:
                return 1.0  # Same inflation curve
            return inflation_corr  # Inflation vs IR curve
        elif risk_type1 == "Risk_IRCurve" and risk_type2 == "Risk_IRCurve":
            # Same currency, potentially different tenors or sub-curves
            # Sub-curve correlation is 0.993, tenor correlation from matrix
            tenor_corr = _get_ir_correlation(label1_1, label1_2)
            # If different sub-curves (Label2), multiply by sub_curves_corr
            # For now, we don't track Label2 in the kernel, use tenor corr only
            return tenor_corr
        else:
            return 1.0  # Same risk type

    elif risk_class == "Equity":
        # Equity intra-bucket correlation depends on bucket
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
        return fx_vega_corr  # 0.5 for FX

    elif risk_class in ("CreditQ", "CreditNonQ"):
        # Credit has complex correlation structure; simplified to 0.5 for same bucket
        return 0.5

    return 1.0  # Default: perfectly correlated


# =============================================================================
# AADC-Compatible Pricing Functions
# =============================================================================

def normal_cdf_aadc(x):
    """AADC-compatible normal CDF using Abramowitz & Stegun 26.2.17 (branchless).

    Max absolute error < 7.5e-8 vs exact erfc-based CDF.
    Uses only exp, sqrt, *, +, / — all AADC-compatible operations.
    Branchless formulation: sign(x) computed via x/|x| so the recorded
    kernel works correctly regardless of the sign of x at recording time.
    """
    # Abramowitz & Stegun constants
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    # Branchless |x| and sign(x)
    abs_x = np.sqrt(x * x + 1e-30)  # smooth abs, epsilon avoids 0/0 at x=0
    sign_x = x / abs_x

    # Standard normal PDF at |x|
    phi = np.exp(-0.5 * x * x) / 2.5066282746310002  # sqrt(2*pi) = 2.50662827...

    # Polynomial in t = 1/(1 + p*|x|)
    t = 1.0 / (1.0 + p * abs_x)
    poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))

    # Tail probability (valid for x >= 0: Phi(x) = 1 - phi*poly)
    tail = phi * poly

    # Branchless combination using symmetry:
    # x >= 0: Phi(x) = 1 - tail  →  0.5 + (0.5 - tail) * 1
    # x <  0: Phi(x) = tail      →  0.5 + (0.5 - tail) * (-1)
    return 0.5 + (0.5 - tail) * sign_x

def interp_aadc(query_times, tenors, rates_aadc):
    """Linear interpolation on aadc.idouble rate values at given query times."""
    n_tenors = len(tenors)
    results = []
    for t in query_times:
        if t <= tenors[0]:
            results.append(rates_aadc[0])
        elif t >= tenors[-1]:
            results.append(rates_aadc[-1])
        else:
            idx = int(np.searchsorted(tenors, t, side='right')) - 1
            idx = min(idx, n_tenors - 2)
            w = (t - tenors[idx]) / (tenors[idx + 1] - tenors[idx])
            rate = rates_aadc[idx] * (1.0 - w) + rates_aadc[idx + 1] * w
            results.append(rate)
    return results


def discount_factors_aadc(payment_times, rates_aadc):
    """Compute discount factors for a vector of payment times."""
    interpolated_rates = interp_aadc(payment_times, IR_TENORS, rates_aadc)
    dfs = []
    for i, t in enumerate(payment_times):
        dfs.append(np.exp(-interpolated_rates[i] * t))
    return dfs


def forward_rates_aadc(t_starts, t_ends, rates_aadc):
    """Compute forward rates for vectors of start/end times."""
    r_starts = interp_aadc(t_starts, IR_TENORS, rates_aadc)
    r_ends = interp_aadc(t_ends, IR_TENORS, rates_aadc)
    fwd_rates = []
    for i in range(len(t_starts)):
        dt = t_ends[i] - t_starts[i]
        fwd = (r_ends[i] * t_ends[i] - r_starts[i] * t_starts[i]) / dt
        fwd_rates.append(fwd)
    return fwd_rates


def price_irs_aadc(notional, fixed_rate, maturity, frequency, is_payer, rates_aadc):
    """Price IR swap using AADC-tracked rates."""
    dt = 1.0 / frequency
    num_periods = int(maturity * frequency)
    payment_times = np.array([(i + 1) * dt for i in range(num_periods)])
    start_times = np.array([i * dt for i in range(num_periods)])

    dfs = discount_factors_aadc(payment_times, rates_aadc)
    fwd_rates = forward_rates_aadc(start_times, payment_times, rates_aadc)

    fixed_leg = notional * fixed_rate * dt * dfs[0]
    floating_leg = notional * fwd_rates[0] * dt * dfs[0]
    for i in range(1, num_periods):
        fixed_leg = fixed_leg + notional * fixed_rate * dt * dfs[i]
        floating_leg = floating_leg + notional * fwd_rates[i] * dt * dfs[i]

    npv = floating_leg - fixed_leg
    if not is_payer:
        npv = fixed_leg - floating_leg
    return npv


def price_equity_option_aadc(notional, strike, maturity, dividend_yield,
                              is_call, rates_aadc, spot, vol):
    """Price equity option using AADC-tracked inputs."""
    r = interp_aadc(np.array([maturity]), IR_TENORS, rates_aadc)[0]
    q = dividend_yield
    tau = maturity
    sqrt_tau = np.sqrt(tau)

    d1 = (np.log(spot / strike) + (r - q + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau

    df = np.exp(-r * tau)
    dq = np.exp(-q * tau)
    num_contracts = notional / strike

    nd1 = normal_cdf_aadc(d1)
    nd2 = normal_cdf_aadc(d2)

    if is_call:
        price = num_contracts * (spot * dq * nd1 - strike * df * nd2)
    else:
        price = num_contracts * (strike * df * (1.0 - nd2) - spot * dq * (1.0 - nd1))
    return price


def price_fx_option_aadc(notional, strike, maturity, is_call,
                          dom_rates_aadc, fgn_rates_aadc, spot, vol):
    """Price FX option using AADC-tracked inputs (Garman-Kohlhagen)."""
    rd = interp_aadc(np.array([maturity]), IR_TENORS, dom_rates_aadc)[0]
    rf = interp_aadc(np.array([maturity]), IR_TENORS, fgn_rates_aadc)[0]
    tau = maturity
    sqrt_tau = np.sqrt(tau)

    d1 = (np.log(spot / strike) + (rd - rf + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau

    df_dom = np.exp(-rd * tau)
    df_fgn = np.exp(-rf * tau)
    fgn_notional = notional / strike

    nd1 = normal_cdf_aadc(d1)
    nd2 = normal_cdf_aadc(d2)

    if is_call:
        price = fgn_notional * (spot * df_fgn * nd1 - strike * df_dom * nd2)
    else:
        price = fgn_notional * (strike * df_dom * (1.0 - nd2) - spot * df_fgn * (1.0 - nd1))
    return price


def price_inflation_swap_aadc(notional, fixed_rate, maturity,
                               rates_aadc, inflation_rates_aadc, base_cpi):
    """Price zero-coupon inflation swap using AADC-tracked inputs."""
    df = discount_factors_aadc(np.array([maturity]), rates_aadc)[0]
    infl_rate = interp_aadc(np.array([maturity]), IR_TENORS, inflation_rates_aadc)[0]
    fixed_leg = notional * (np.exp(fixed_rate * maturity) - 1.0) * df
    cpi_ratio = np.exp(infl_rate * maturity)
    inflation_leg = notional * (cpi_ratio - 1.0) * df
    return inflation_leg - fixed_leg


def price_xccy_swap_aadc(dom_notional, fgn_notional, dom_fixed_rate, fgn_fixed_rate,
                          maturity, frequency, exchange_notional,
                          dom_rates_aadc, fgn_rates_aadc, fx_spot):
    """Price cross-currency swap using AADC-tracked inputs."""
    dt = 1.0 / frequency
    num_periods = int(maturity * frequency)
    payment_times = np.array([(i + 1) * dt for i in range(num_periods)])

    dom_dfs = discount_factors_aadc(payment_times, dom_rates_aadc)
    dom_leg = dom_notional * dom_fixed_rate * dt * dom_dfs[0]
    for i in range(1, num_periods):
        dom_leg = dom_leg + dom_notional * dom_fixed_rate * dt * dom_dfs[i]

    fgn_dfs = discount_factors_aadc(payment_times, fgn_rates_aadc)
    fgn_leg = fgn_notional * fgn_fixed_rate * dt * fgn_dfs[0]
    for i in range(1, num_periods):
        fgn_leg = fgn_leg + fgn_notional * fgn_fixed_rate * dt * fgn_dfs[i]

    if exchange_notional:
        dom_leg = dom_leg + dom_notional * dom_dfs[-1] - dom_notional
        fgn_leg = fgn_leg + fgn_notional * fgn_dfs[-1] - fgn_notional

    return dom_leg - fgn_leg * fx_spot


# =============================================================================
# AADC Kernel Recording Per Trade Type
# =============================================================================

def _get_trade_structure_key(trade) -> tuple:
    """Generate cache key from trade structure for kernel reuse."""
    if isinstance(trade, IRSwapTrade):
        return ("ir_swap", trade.currency, trade.maturity, trade.frequency, trade.payer)
    elif isinstance(trade, EquityOptionTrade):
        return ("equity_option", trade.currency, trade.maturity, trade.is_call, trade.underlying)
    elif isinstance(trade, FXOptionTrade):
        return ("fx_option", trade.domestic_ccy, trade.foreign_ccy, trade.maturity, trade.is_call)
    elif isinstance(trade, InflationSwapTrade):
        return ("inflation_swap", trade.currency, trade.maturity)
    elif isinstance(trade, XCCYSwapTrade):
        return ("xccy_swap", trade.domestic_ccy, trade.foreign_ccy,
                trade.maturity, trade.frequency, trade.exchange_notional)
    else:
        raise ValueError(f"Unknown trade type: {type(trade)}")


def record_pricing_kernel(trade, market: MarketEnvironment):
    """
    Record AADC pricing kernel for a single trade.

    Returns (funcs, all_handles, pv_output, crif_metadata, nodiff_inputs)
    """
    if isinstance(trade, IRSwapTrade):
        return _record_irs_kernel(trade, market)
    elif isinstance(trade, EquityOptionTrade):
        return _record_equity_option_kernel(trade, market)
    elif isinstance(trade, FXOptionTrade):
        return _record_fx_option_kernel(trade, market)
    elif isinstance(trade, InflationSwapTrade):
        return _record_inflation_swap_kernel(trade, market)
    elif isinstance(trade, XCCYSwapTrade):
        return _record_xccy_swap_kernel(trade, market)
    else:
        raise ValueError(f"Unknown trade type: {type(trade)}")


def _record_irs_kernel(trade: IRSwapTrade, market: MarketEnvironment):
    """Record AADC kernel for IR swap pricing."""
    curve = market.curves[trade.currency]
    nodiff_inputs = {}

    with aadc.record_kernel() as funcs:
        rates_aadc = []
        rate_handles = []
        for i in range(NUM_IR_TENORS):
            r = aadc.idouble(float(curve.zero_rates[i]))
            handle = r.mark_as_input()
            rates_aadc.append(r)
            rate_handles.append(handle)

        notional = aadc.idouble(trade.notional)
        notional_h = notional.mark_as_input_no_diff()
        nodiff_inputs[notional_h] = trade.notional
        fixed_rate = aadc.idouble(trade.fixed_rate)
        fixed_rate_h = fixed_rate.mark_as_input_no_diff()
        nodiff_inputs[fixed_rate_h] = trade.fixed_rate

        pv = price_irs_aadc(
            notional, fixed_rate, trade.maturity, trade.frequency,
            trade.payer, rates_aadc
        )
        pv_output = pv.mark_as_output()

    crif_meta = []
    for i in range(NUM_IR_TENORS):
        crif_meta.append({
            "handle": rate_handles[i],
            "risk_type": "Risk_IRCurve",
            "qualifier": trade.currency,
            "bucket": str(i + 1),
            "label1": IR_TENOR_LABELS[i],
            "label2": "OIS",
            "scale": 1.0,
            "product_class": "RatesFX",
            "risk_class": "Rates",
        })

    return funcs, rate_handles, pv_output, crif_meta, nodiff_inputs


def _record_equity_option_kernel(trade: EquityOptionTrade, market: MarketEnvironment):
    """Record AADC kernel for equity option pricing."""
    curve = market.curves[trade.currency]
    spot = market.equity_spots.get(trade.underlying, 100.0)
    vol_surface = market.vol_surfaces.get(trade.underlying, VolSurface())
    vol_val = vol_surface.vol(trade.maturity)
    nodiff_inputs = {}

    with aadc.record_kernel() as funcs:
        rates_aadc = []
        rate_handles = []
        for i in range(NUM_IR_TENORS):
            r = aadc.idouble(float(curve.zero_rates[i]))
            handle = r.mark_as_input()
            rates_aadc.append(r)
            rate_handles.append(handle)

        spot_v = aadc.idouble(float(spot))
        spot_handle = spot_v.mark_as_input()
        vol_v = aadc.idouble(float(vol_val))
        vol_handle = vol_v.mark_as_input()

        notional = aadc.idouble(trade.notional)
        notional_h = notional.mark_as_input_no_diff()
        nodiff_inputs[notional_h] = trade.notional
        strike = aadc.idouble(trade.strike)
        strike_h = strike.mark_as_input_no_diff()
        nodiff_inputs[strike_h] = trade.strike

        pv = price_equity_option_aadc(
            notional, strike, trade.maturity, trade.dividend_yield,
            trade.is_call, rates_aadc, spot_v, vol_v
        )
        pv_output = pv.mark_as_output()

    all_handles = rate_handles + [spot_handle, vol_handle]
    crif_meta = []
    for i in range(NUM_IR_TENORS):
        crif_meta.append({
            "handle": rate_handles[i],
            "risk_type": "Risk_IRCurve",
            "qualifier": trade.currency,
            "bucket": str(i + 1),
            "label1": IR_TENOR_LABELS[i],
            "label2": "OIS",
            "scale": 1.0,
            "product_class": "RatesFX",
            "risk_class": "Rates",
        })
    crif_meta.append({
        "handle": spot_handle,
        "risk_type": "Risk_Equity",
        "qualifier": trade.underlying,
        "bucket": str(trade.equity_bucket + 1),
        "label1": "",
        "label2": "spot",
        "scale": float(spot),
        "product_class": "Equity",
        "risk_class": "Equity",
    })
    crif_meta.append({
        "handle": vol_handle,
        "risk_type": "Risk_EquityVol",
        "qualifier": trade.underlying,
        "bucket": str(trade.equity_bucket + 1),
        "label1": f"{trade.maturity:.1f}y",
        "label2": "",
        "scale": 1.0,
        "product_class": "Equity",
        "risk_class": "Equity",
    })

    return funcs, all_handles, pv_output, crif_meta, nodiff_inputs


def _record_fx_option_kernel(trade: FXOptionTrade, market: MarketEnvironment):
    """Record AADC kernel for FX option pricing."""
    dom_curve = market.curves[trade.domestic_ccy]
    fgn_curve = market.curves.get(trade.foreign_ccy,
                                   YieldCurve(zero_rates=np.full(NUM_IR_TENORS, 0.02)))
    pair = f"{trade.foreign_ccy}{trade.domestic_ccy}"
    spot = market.fx_spots.get(pair, trade.strike)
    vol_surface = market.vol_surfaces.get(pair, VolSurface())
    vol_val = vol_surface.vol(trade.maturity)
    nodiff_inputs = {}

    with aadc.record_kernel() as funcs:
        dom_rates_aadc = []
        dom_handles = []
        for i in range(NUM_IR_TENORS):
            r = aadc.idouble(float(dom_curve.zero_rates[i]))
            handle = r.mark_as_input()
            dom_rates_aadc.append(r)
            dom_handles.append(handle)

        fgn_rates_aadc = []
        fgn_handles = []
        for i in range(NUM_IR_TENORS):
            r = aadc.idouble(float(fgn_curve.zero_rates[i]))
            handle = r.mark_as_input()
            fgn_rates_aadc.append(r)
            fgn_handles.append(handle)

        spot_v = aadc.idouble(float(spot))
        spot_handle = spot_v.mark_as_input()
        vol_v = aadc.idouble(float(vol_val))
        vol_handle = vol_v.mark_as_input()

        notional = aadc.idouble(trade.notional)
        notional_h = notional.mark_as_input_no_diff()
        nodiff_inputs[notional_h] = trade.notional
        strike = aadc.idouble(trade.strike)
        strike_h = strike.mark_as_input_no_diff()
        nodiff_inputs[strike_h] = trade.strike

        pv = price_fx_option_aadc(
            notional, strike, trade.maturity, trade.is_call,
            dom_rates_aadc, fgn_rates_aadc, spot_v, vol_v
        )
        pv_output = pv.mark_as_output()

    all_handles = dom_handles + fgn_handles + [spot_handle, vol_handle]
    crif_meta = []
    for i in range(NUM_IR_TENORS):
        crif_meta.append({
            "handle": dom_handles[i],
            "risk_type": "Risk_IRCurve",
            "qualifier": trade.domestic_ccy,
            "bucket": str(i + 1),
            "label1": IR_TENOR_LABELS[i],
            "label2": "OIS",
            "scale": 1.0,
            "product_class": "RatesFX",
            "risk_class": "Rates",
        })
    for i in range(NUM_IR_TENORS):
        crif_meta.append({
            "handle": fgn_handles[i],
            "risk_type": "Risk_IRCurve",
            "qualifier": trade.foreign_ccy,
            "bucket": str(i + 1),
            "label1": IR_TENOR_LABELS[i],
            "label2": "OIS",
            "scale": 1.0,
            "product_class": "RatesFX",
            "risk_class": "Rates",
        })
    crif_meta.append({
        "handle": spot_handle,
        "risk_type": "Risk_FX",
        "qualifier": pair,
        "bucket": "",
        "label1": "",
        "label2": "",
        "scale": float(spot),
        "product_class": "RatesFX",
        "risk_class": "FX",
    })
    crif_meta.append({
        "handle": vol_handle,
        "risk_type": "Risk_FXVol",
        "qualifier": pair,
        "bucket": "",
        "label1": f"{trade.maturity:.1f}y",
        "label2": "",
        "scale": 1.0,
        "product_class": "RatesFX",
        "risk_class": "FX",
    })

    return funcs, all_handles, pv_output, crif_meta, nodiff_inputs


def _record_inflation_swap_kernel(trade: InflationSwapTrade, market: MarketEnvironment):
    """Record AADC kernel for inflation swap pricing."""
    curve = market.curves[trade.currency]
    inflation = market.inflation or InflationCurve()
    nodiff_inputs = {}

    with aadc.record_kernel() as funcs:
        rates_aadc = []
        rate_handles = []
        for i in range(NUM_IR_TENORS):
            r = aadc.idouble(float(curve.zero_rates[i]))
            handle = r.mark_as_input()
            rates_aadc.append(r)
            rate_handles.append(handle)

        infl_rates_aadc = []
        infl_handles = []
        for i in range(NUM_IR_TENORS):
            r = aadc.idouble(float(inflation.inflation_rates[i]))
            handle = r.mark_as_input()
            infl_rates_aadc.append(r)
            infl_handles.append(handle)

        notional = aadc.idouble(trade.notional)
        notional_h = notional.mark_as_input_no_diff()
        nodiff_inputs[notional_h] = trade.notional
        fixed_rate = aadc.idouble(trade.fixed_rate)
        fixed_rate_h = fixed_rate.mark_as_input_no_diff()
        nodiff_inputs[fixed_rate_h] = trade.fixed_rate

        pv = price_inflation_swap_aadc(
            notional, fixed_rate, trade.maturity,
            rates_aadc, infl_rates_aadc, inflation.base_cpi
        )
        pv_output = pv.mark_as_output()

    all_handles = rate_handles + infl_handles
    crif_meta = []
    for i in range(NUM_IR_TENORS):
        crif_meta.append({
            "handle": rate_handles[i],
            "risk_type": "Risk_IRCurve",
            "qualifier": trade.currency,
            "bucket": str(i + 1),
            "label1": IR_TENOR_LABELS[i],
            "label2": "OIS",
            "scale": 1.0,
            "product_class": "RatesFX",
            "risk_class": "Rates",
        })
    for i in range(NUM_IR_TENORS):
        crif_meta.append({
            "handle": infl_handles[i],
            "risk_type": "Risk_Inflation",
            "qualifier": trade.currency,
            "bucket": str(i + 1),
            "label1": IR_TENOR_LABELS[i],
            "label2": "",
            "scale": 1.0,
            "product_class": "RatesFX",
            "risk_class": "Rates",
        })

    return funcs, all_handles, pv_output, crif_meta, nodiff_inputs


def _record_xccy_swap_kernel(trade: XCCYSwapTrade, market: MarketEnvironment):
    """Record AADC kernel for cross-currency swap pricing."""
    dom_curve = market.curves[trade.domestic_ccy]
    fgn_curve = market.curves.get(trade.foreign_ccy,
                                   YieldCurve(zero_rates=np.full(NUM_IR_TENORS, 0.02)))
    pair = f"{trade.foreign_ccy}{trade.domestic_ccy}"
    fx_spot = market.fx_spots.get(pair, trade.dom_notional / trade.fgn_notional)
    nodiff_inputs = {}

    with aadc.record_kernel() as funcs:
        dom_rates_aadc = []
        dom_handles = []
        for i in range(NUM_IR_TENORS):
            r = aadc.idouble(float(dom_curve.zero_rates[i]))
            handle = r.mark_as_input()
            dom_rates_aadc.append(r)
            dom_handles.append(handle)

        fgn_rates_aadc = []
        fgn_handles = []
        for i in range(NUM_IR_TENORS):
            r = aadc.idouble(float(fgn_curve.zero_rates[i]))
            handle = r.mark_as_input()
            fgn_rates_aadc.append(r)
            fgn_handles.append(handle)

        fx_v = aadc.idouble(float(fx_spot))
        fx_handle = fx_v.mark_as_input()

        dom_not = aadc.idouble(trade.dom_notional)
        dom_not_h = dom_not.mark_as_input_no_diff()
        nodiff_inputs[dom_not_h] = trade.dom_notional
        fgn_not = aadc.idouble(trade.fgn_notional)
        fgn_not_h = fgn_not.mark_as_input_no_diff()
        nodiff_inputs[fgn_not_h] = trade.fgn_notional
        dom_rate = aadc.idouble(trade.dom_fixed_rate)
        dom_rate_h = dom_rate.mark_as_input_no_diff()
        nodiff_inputs[dom_rate_h] = trade.dom_fixed_rate
        fgn_rate = aadc.idouble(trade.fgn_fixed_rate)
        fgn_rate_h = fgn_rate.mark_as_input_no_diff()
        nodiff_inputs[fgn_rate_h] = trade.fgn_fixed_rate

        pv = price_xccy_swap_aadc(
            dom_not, fgn_not, dom_rate, fgn_rate,
            trade.maturity, trade.frequency, trade.exchange_notional,
            dom_rates_aadc, fgn_rates_aadc, fx_v
        )
        pv_output = pv.mark_as_output()

    all_handles = dom_handles + fgn_handles + [fx_handle]
    crif_meta = []
    for i in range(NUM_IR_TENORS):
        crif_meta.append({
            "handle": dom_handles[i],
            "risk_type": "Risk_IRCurve",
            "qualifier": trade.domestic_ccy,
            "bucket": str(i + 1),
            "label1": IR_TENOR_LABELS[i],
            "label2": "OIS",
            "scale": 1.0,
            "product_class": "RatesFX",
            "risk_class": "Rates",
        })
    for i in range(NUM_IR_TENORS):
        crif_meta.append({
            "handle": fgn_handles[i],
            "risk_type": "Risk_IRCurve",
            "qualifier": trade.foreign_ccy,
            "bucket": str(i + 1),
            "label1": IR_TENOR_LABELS[i],
            "label2": "OIS",
            "scale": 1.0,
            "product_class": "RatesFX",
            "risk_class": "Rates",
        })
    crif_meta.append({
        "handle": fx_handle,
        "risk_type": "Risk_FX",
        "qualifier": pair,
        "bucket": "",
        "label1": "",
        "label2": "",
        "scale": float(fx_spot),
        "product_class": "RatesFX",
        "risk_class": "FX",
    })

    return funcs, all_handles, pv_output, crif_meta, nodiff_inputs


# =============================================================================
# AADC CRIF Computation
# =============================================================================

def compute_crif_aadc(
    trades: list,
    market: MarketEnvironment,
    num_threads: int,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Compute CRIF sensitivities via AADC (single forward + adjoint pass).

    Returns:
        (crif_df, kernel_recording_time, evaluation_time)
    """
    workers = aadc.ThreadPool(num_threads)
    kernel_cache = {}
    all_crif_rows = []

    # Phase 1: Record kernels
    record_start = time.perf_counter()
    trade_kernels = []
    for trade in trades:
        key = _get_trade_structure_key(trade)
        if key not in kernel_cache:
            result = record_pricing_kernel(trade, market)
            kernel_cache[key] = result
        trade_kernels.append((trade, kernel_cache[key]))
    recording_time = time.perf_counter() - record_start

    # Phase 2: Evaluate kernels
    eval_start = time.perf_counter()

    for trade, (funcs, all_handles, pv_output, crif_meta, nodiff_inputs) in trade_kernels:
        inputs = {}
        diff_handles = []
        for meta in crif_meta:
            h = meta["handle"]
            diff_handles.append(h)
            val = _get_market_value_for_handle(trade, meta, market)
            inputs[h] = np.array([val])

        # Add non-differentiable inputs
        for h, val in nodiff_inputs.items():
            inputs[h] = np.array([float(val)])

        request = {pv_output: diff_handles}
        results = aadc.evaluate(funcs, request, inputs, workers)

        for meta in crif_meta:
            h = meta["handle"]
            deriv = float(results[1][pv_output][h][0])
            sensitivity = deriv * meta["scale"]

            if abs(sensitivity) > 1e-10:
                all_crif_rows.append({
                    "TradeID": trade.trade_id,
                    "ProductClass": meta["product_class"],
                    "RiskType": meta["risk_type"],
                    "Qualifier": meta["qualifier"],
                    "Bucket": meta["bucket"],
                    "Label1": meta["label1"],
                    "Label2": meta["label2"],
                    "Amount": sensitivity,
                    "AmountCurrency": getattr(trade, 'currency', 'USD'),
                    "AmountUSD": sensitivity,
                })

    eval_time = time.perf_counter() - eval_start
    crif_df = pd.DataFrame(all_crif_rows) if all_crif_rows else pd.DataFrame()
    return crif_df, recording_time, eval_time


def compute_trade_crif_standalone(
    trade,
    market: MarketEnvironment,
    num_threads: int = 1,
) -> pd.DataFrame:
    """
    Compute CRIF for a single trade independently using AADC.

    This is a thin wrapper around compute_crif_aadc() for single-trade use.
    Returns a DataFrame with columns: TradeID, ProductClass, RiskType,
    Qualifier, Bucket, Label1, Label2, Amount, AmountCurrency, AmountUSD
    """
    crif_df, _, _ = compute_crif_aadc([trade], market, num_threads)
    return crif_df


def precompute_all_trade_crifs(
    trades: list,
    market: MarketEnvironment,
    num_threads: int,
) -> Dict[str, pd.DataFrame]:
    """
    Precompute CRIF for all trades independently.

    Returns Dict[trade_id, crif_df] for O(1) lookup during allocation optimization.
    Each crif_df contains sensitivities for that trade only.

    Optimized: Batches trades by structure type for efficient kernel reuse.
    """
    workers = aadc.ThreadPool(num_threads)
    trade_crifs = {}

    # Group trades by structure key for batched evaluation
    from collections import defaultdict
    trades_by_structure = defaultdict(list)
    for trade in trades:
        key = _get_trade_structure_key(trade)
        trades_by_structure[key].append(trade)

    # Cache kernels
    kernel_cache = {}

    # Process each structure group
    for structure_key, group_trades in trades_by_structure.items():
        # Record kernel once for this structure (using first trade as template)
        if structure_key not in kernel_cache:
            result = record_pricing_kernel(group_trades[0], market)
            kernel_cache[structure_key] = result

        funcs, all_handles, pv_output, crif_meta, nodiff_inputs = kernel_cache[structure_key]

        # Batch evaluate all trades in this group
        batch_size = len(group_trades)

        # Build batched inputs - each handle maps to array of values
        inputs = {}
        diff_handles = []

        for meta in crif_meta:
            h = meta["handle"]
            diff_handles.append(h)
            # Get value for each trade in batch
            vals = np.array([_get_market_value_for_handle(t, meta, market) for t in group_trades])
            inputs[h] = vals

        # Add non-differentiable inputs (same for all trades of same structure)
        for h, val in nodiff_inputs.items():
            inputs[h] = np.full(batch_size, float(val))

        request = {pv_output: diff_handles}
        results = aadc.evaluate(funcs, request, inputs, workers)

        # Extract CRIF for each trade in batch
        for batch_idx, trade in enumerate(group_trades):
            crif_rows = []
            for meta in crif_meta:
                h = meta["handle"]
                deriv = float(results[1][pv_output][h][batch_idx])
                sensitivity = deriv * meta["scale"]

                if abs(sensitivity) > 1e-10:
                    crif_rows.append({
                        "TradeID": trade.trade_id,
                        "RiskType": meta["risk_type"],
                        "Qualifier": meta["qualifier"],
                        "Bucket": meta["bucket"],
                        "Label1": meta.get("label1", ""),
                        "Label2": meta.get("label2", ""),
                        "Amount": sensitivity,
                        "AmountCurrency": meta.get("currency", "USD"),
                        "AmountUSD": sensitivity,
                    })

            if crif_rows:
                trade_crifs[trade.trade_id] = pd.DataFrame(crif_rows)

    return trade_crifs


def _get_market_value_for_handle(trade, meta, market: MarketEnvironment) -> float:
    """Get the actual market value corresponding to a CRIF handle."""
    risk_type = meta["risk_type"]
    qualifier = meta["qualifier"]

    if risk_type == "Risk_IRCurve":
        bucket_idx = int(meta["bucket"]) - 1
        curve = market.curves.get(qualifier, YieldCurve())
        return float(curve.zero_rates[bucket_idx])
    elif risk_type == "Risk_Inflation":
        bucket_idx = int(meta["bucket"]) - 1
        inflation = market.inflation or InflationCurve()
        return float(inflation.inflation_rates[bucket_idx])
    elif risk_type == "Risk_Equity":
        return float(market.equity_spots.get(qualifier, 100.0))
    elif risk_type == "Risk_EquityVol":
        vol_surface = market.vol_surfaces.get(qualifier, VolSurface())
        maturity = getattr(trade, 'maturity', 1.0)
        return float(vol_surface.vol(maturity))
    elif risk_type == "Risk_FX":
        return float(market.fx_spots.get(qualifier, 1.0))
    elif risk_type == "Risk_FXVol":
        vol_surface = market.vol_surfaces.get(qualifier, VolSurface())
        maturity = getattr(trade, 'maturity', 1.0)
        return float(vol_surface.vol(maturity))
    return 0.0


# =============================================================================
# AADC SIMM Kernel (dIM/dsensitivity)
# =============================================================================

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


def _get_risk_weight(risk_type: str, bucket: str) -> float:
    """Get simplified risk weight for a CRIF entry."""
    if risk_type == "Risk_IRCurve":
        try:
            idx = int(bucket) - 1
            if 0 <= idx < len(IR_RISK_WEIGHTS):
                return IR_RISK_WEIGHTS[idx]
        except (ValueError, IndexError):
            pass
        return 50.0
    elif risk_type == "Risk_Inflation":
        return INFLATION_RISK_WEIGHT
    elif risk_type == "Risk_FX":
        return FX_RISK_WEIGHT
    elif risk_type == "Risk_FXVol":
        return FX_RISK_WEIGHT * 0.55
    elif risk_type in ("Risk_Equity", "Risk_EquityVol"):
        return EQUITY_RISK_WEIGHT
    return 50.0


def record_simm_kernel(group_crif: pd.DataFrame, use_correlations: bool = True):
    """
    Record AADC kernel for SIMM margin computation.

    Args:
        group_crif: DataFrame with CRIF sensitivities (RiskType, Qualifier, Bucket, Label1, Amount)
        use_correlations: If True, apply ISDA intra-bucket correlations; if False, use sum-of-squares

    Returns:
        (funcs, sens_handles, im_output, recording_time)

    ISDA SIMM Formula:
        K_r = sqrt(Σ_i Σ_j ρ_ij × WS_i × WS_j)
        IM = sqrt(Σ_r Σ_s ψ_rs × K_r × K_s)

    Where:
        - WS_i = RiskWeight_i × Sensitivity_i
        - ρ_ij = intra-bucket correlation (from ir_corr for IR, equity_corr for Equity, etc.)
        - ψ_rs = cross-risk-class correlation (from PSI_MATRIX)
    """
    n = len(group_crif)
    risk_class_order = SIMM_RISK_CLASSES

    # Pre-extract CRIF metadata for correlation lookups
    crif_risk_classes = []
    crif_risk_types = []
    crif_labels = []
    crif_buckets = []
    crif_qualifiers = []
    crif_weights = []

    for _, row in group_crif.iterrows():
        rt = row["RiskType"]
        rc = _map_risk_type_to_class(rt)
        qualifier = str(row.get("Qualifier", ""))
        bucket = str(row.get("Bucket", ""))
        label1 = str(row.get("Label1", ""))

        # Get risk weight (use v2.6 currency-specific for IR if available)
        if rt == "Risk_IRCurve" and qualifier and label1:
            rw = _get_ir_risk_weight_v26(qualifier, label1)
        else:
            rw = _get_risk_weight(rt, bucket)

        crif_risk_classes.append(rc)
        crif_risk_types.append(rt)
        crif_labels.append(label1)
        crif_buckets.append(bucket)
        crif_qualifiers.append(qualifier)
        crif_weights.append(rw)

    record_start = time.perf_counter()

    with aadc.record_kernel() as funcs:
        # Mark sensitivities as inputs
        sens_inputs = []
        sens_handles = []
        for i in range(n):
            s = aadc.idouble(float(group_crif.iloc[i]["Amount"]))
            handle = s.mark_as_input()
            sens_inputs.append(s)
            sens_handles.append(handle)

        # Compute risk class margins with intra-bucket correlations
        risk_class_margins = []
        for rc in risk_class_order:
            rc_indices = [i for i in range(n) if crif_risk_classes[i] == rc]
            if not rc_indices:
                risk_class_margins.append(aadc.idouble(0.0))
                continue

            # Compute weighted sensitivities: WS_i = w_i * s_i
            ws_list = []
            for i in rc_indices:
                ws_i = sens_inputs[i] * crif_weights[i]
                ws_list.append(ws_i)

            # Compute K_r² = Σ_i Σ_j ρ_ij × WS_i × WS_j
            k_sq = aadc.idouble(0.0)
            num_in_rc = len(rc_indices)

            if use_correlations and num_in_rc > 1:
                # Apply ISDA intra-bucket correlations
                for i_local in range(num_in_rc):
                    for j_local in range(num_in_rc):
                        i_global = rc_indices[i_local]
                        j_global = rc_indices[j_local]

                        # Get correlation between these two sensitivities
                        rho_ij = _get_intra_correlation(
                            rc,
                            crif_risk_types[i_global],
                            crif_risk_types[j_global],
                            crif_labels[i_global],
                            crif_labels[j_global],
                            crif_buckets[i_global] if crif_buckets[i_global] == crif_buckets[j_global] else None
                        )

                        k_sq = k_sq + rho_ij * ws_list[i_local] * ws_list[j_local]
            else:
                # Simplified: sum of squares (no correlations)
                for ws_i in ws_list:
                    k_sq = k_sq + ws_i * ws_i

            k_r = np.sqrt(k_sq)
            risk_class_margins.append(k_r)

        # Cross-risk-class aggregation: IM = sqrt(Σ_r Σ_s ψ_rs × K_r × K_s)
        simm_sq = aadc.idouble(0.0)
        for i in range(6):
            for j in range(6):
                psi_ij = PSI_MATRIX[i, j]
                simm_sq = simm_sq + psi_ij * risk_class_margins[i] * risk_class_margins[j]

        im = np.sqrt(simm_sq)
        im_output = im.mark_as_output()

    recording_time = time.perf_counter() - record_start
    return funcs, sens_handles, im_output, recording_time


def compute_im_gradient_aadc(
    group_crif: pd.DataFrame,
    num_threads: int,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Compute dIM/dsensitivity for each CRIF row via AADC (single adjoint pass).

    Returns:
        (gradient_array, im_value, recording_time, eval_time)
    """
    n = len(group_crif)
    if n == 0:
        return np.array([]), 0.0, 0.0, 0.0

    workers = aadc.ThreadPool(num_threads)
    funcs, sens_handles, im_output, recording_time = record_simm_kernel(group_crif)

    eval_start = time.perf_counter()
    inputs = {}
    for i in range(n):
        val = float(group_crif.iloc[i]["Amount"])
        inputs[sens_handles[i]] = np.array([val])

    request = {im_output: sens_handles}
    results = aadc.evaluate(funcs, request, inputs, workers)

    im_value = float(results[0][im_output][0])
    gradient = np.zeros(n)
    for i in range(n):
        gradient[i] = float(results[1][im_output][sens_handles[i]][0])

    eval_time = time.perf_counter() - eval_start
    return gradient, im_value, recording_time, eval_time


# =============================================================================
# Reallocation via Gradient Info
# =============================================================================

def estimate_marginal_impact(trade_crif: pd.DataFrame, group_crif: pd.DataFrame,
                              group_gradient: np.ndarray, group_im: float = None) -> float:
    """
    Estimate the marginal impact of adding a trade's sensitivities to a group
    using the group's existing dIM/ds gradient (first-order Taylor approximation).

    For each sensitivity in the trade's CRIF, find the matching entry in the
    group's CRIF (same RiskType + Qualifier + Bucket + Label1). If found,
    use that entry's gradient. If not found, use a principled estimate
    derived from SIMM structure.

    Args:
        trade_crif: CRIF for the trade to add
        group_crif: Current CRIF for the group
        group_gradient: dIM/ds gradient array from AADC
        group_im: Current IM for the group (optional, used for new factor estimate)

    Returns estimated change in group IM from adding this trade.

    For new (uncorrelated) risk factors, the marginal impact is derived from SIMM structure:
        K_r_new² = K_r² + (rw * s)²    (assuming no correlation)
        ΔK_r ≈ (rw * s)² / (2 * K_r)   (first-order Taylor)
        ΔIM ≈ (ψ * K_r * ΔK_r) / IM    (chain rule through cross-RC aggregation)

    For large K_r and IM, this simplifies to: ΔIM ≈ (rw * s)² / (2 * IM)
    """
    if group_crif.empty or len(group_gradient) == 0:
        # No existing sensitivities in group; estimate via standalone IM
        if trade_crif.empty:
            return 0.0
        # Compute standalone SIMM for the trade
        _, standalone_im, _ = run_simm(trade_crif)
        return standalone_im

    # Get group IM if not provided (for new factor estimation)
    if group_im is None or group_im <= 0:
        group_im = 1e6  # Default to $1M if unknown (conservative)

    impact = 0.0
    for _, trade_row in trade_crif.iterrows():
        sensitivity = float(trade_row["Amount"])
        # Find matching entry in group CRIF
        mask = (
            (group_crif["RiskType"] == trade_row["RiskType"]) &
            (group_crif["Qualifier"] == trade_row["Qualifier"]) &
            (group_crif["Bucket"] == trade_row["Bucket"]) &
            (group_crif["Label1"] == trade_row["Label1"])
        )
        matching_indices = group_crif.index[mask].tolist()

        if matching_indices:
            # Use gradient of matching entry: dIM/ds * delta_s
            idx_in_group = group_crif.index.get_loc(matching_indices[0])
            impact += group_gradient[idx_in_group] * sensitivity
        else:
            # No matching risk factor in group
            # For new uncorrelated factor: ΔIM ≈ (rw * s)² / (2 * IM)
            # This derives from SIMM structure: new factor adds standalone K_r contribution
            rt = trade_row["RiskType"]
            qualifier = str(trade_row.get("Qualifier", ""))
            label1 = str(trade_row.get("Label1", ""))
            bucket = str(trade_row.get("Bucket", ""))

            # Use v2.6 weights where applicable
            if rt == "Risk_IRCurve" and qualifier and label1:
                rw = _get_ir_risk_weight_v26(qualifier, label1)
            else:
                rw = _get_risk_weight(rt, bucket)

            # Principled estimate for uncorrelated new factor
            # ΔIM ≈ (rw * s)² / (2 * IM), sign matches sensitivity sign
            weighted_sens_sq = (rw * abs(sensitivity)) ** 2
            delta_im = weighted_sens_sq / (2 * group_im + 1e-10)

            # Apply sign: if sensitivity is positive, IM typically increases
            impact += np.sign(sensitivity) * delta_im

    return impact


def reallocate_trades(
    n_reallocate: int,
    trades: list,
    group_ids: np.ndarray,
    group_crifs: Dict[int, pd.DataFrame],
    group_gradients: Dict[int, np.ndarray],
    group_ims: Dict[int, float],
    group_contributions: Dict[int, Dict[str, float]],
    market: MarketEnvironment,
    num_threads: int,
    num_portfolios: int,
    refresh_gradients: bool = True,
) -> Tuple[Dict, float]:
    """
    Reallocate N trades to minimize total SIMM using gradient info.

    Algorithm:
    1. Rank all trades by absolute Euler contribution (highest first)
    2. For top N trades, estimate marginal impact on each other group
    3. Move each trade to the group where its marginal impact is lowest
    4. (v2.6.0) After each move, refresh gradients for affected groups (if refresh_gradients=True)
    5. Recompute SIMM for affected groups using AADC
    6. Compare recomputed total vs gradient-based estimate

    Args:
        refresh_gradients: If True, recompute gradients after each move decision.
                          This uses AADC's fast evaluation (~1ms per portfolio) to
                          ensure subsequent move decisions use accurate gradients.
                          Default True for better allocation quality.

    Returns:
        (realloc_result_dict, realloc_time_sec)
    """
    realloc_start = time.perf_counter()

    # Build trade_id -> index mapping
    trade_id_to_idx = {}
    for i, trade in enumerate(trades):
        trade_id_to_idx[trade.trade_id] = i

    # Collect all trades with their contributions across groups
    trade_contrib_list = []  # (trade_id, group_id, abs_contribution)
    for group_id, contribs in group_contributions.items():
        for trade_id, contrib in contribs.items():
            trade_contrib_list.append((trade_id, group_id, contrib))

    # Sort by absolute contribution (highest first = biggest margin consumers)
    trade_contrib_list.sort(key=lambda x: abs(x[2]), reverse=True)

    # Select top N trades to reallocate
    n_available = min(n_reallocate, len(trade_contrib_list))
    trades_to_move = trade_contrib_list[:n_available]

    if not trades_to_move:
        return {}, time.perf_counter() - realloc_start

    # Track moves: (trade_id, from_group, to_group, old_contrib, signed_impact, standalone_im)
    moves = []
    new_group_ids = group_ids.copy()

    # Working state that gets updated after each move (v2.6.0)
    # This solves the stale gradient problem: gradients are refreshed after each move
    current_group_crifs = {g: crif.copy() for g, crif in group_crifs.items()}
    current_group_gradients = {g: grad.copy() if isinstance(grad, np.ndarray) else np.array([])
                               for g, grad in group_gradients.items()}
    current_group_ims = dict(group_ims)
    gradient_refresh_count = 0

    for trade_id, from_group, old_contrib in trades_to_move:
        trade_idx = trade_id_to_idx.get(trade_id)
        if trade_idx is None:
            continue

        # Get this trade's CRIF rows from the source group (use current state)
        from_crif = current_group_crifs.get(from_group, pd.DataFrame())
        if from_crif.empty:
            continue
        trade_crif = from_crif[from_crif["TradeID"] == trade_id].copy()
        if trade_crif.empty:
            continue

        # Compute standalone SIMM for this trade (used in quadratic estimate)
        _, trade_standalone_im, _ = run_simm(trade_crif)

        # Estimate marginal impact on each candidate group (signed)
        # Use CURRENT state (gradients may have been refreshed after previous moves)
        current_from_im = current_group_ims.get(from_group, 0.0)
        current_from_gradient = current_group_gradients.get(from_group, np.array([]))

        # Recompute old_contrib using current gradient (may differ from initial)
        if len(current_from_gradient) > 0:
            # Compute Euler contribution with current gradient
            current_old_contrib = estimate_marginal_impact(
                trade_crif, from_crif, current_from_gradient, current_from_im
            )
        else:
            current_old_contrib = old_contrib

        best_group = from_group
        best_abs_impact = abs(current_old_contrib)  # staying put = current contribution
        best_signed_impact = current_old_contrib

        for candidate_group in range(num_portfolios):
            if candidate_group == from_group:
                continue
            # Use CURRENT state (v2.6.0 - gradients refreshed after each move)
            cand_crif = current_group_crifs.get(candidate_group, pd.DataFrame())
            cand_gradient = current_group_gradients.get(candidate_group, np.array([]))
            cand_im = current_group_ims.get(candidate_group, 0.0)

            impact = estimate_marginal_impact(trade_crif, cand_crif, cand_gradient, cand_im)

            if abs(impact) < best_abs_impact:
                best_abs_impact = abs(impact)
                best_signed_impact = impact
                best_group = candidate_group

        if best_group != from_group:
            moves.append((trade_id, from_group, best_group, current_old_contrib,
                          best_signed_impact, trade_standalone_im))
            new_group_ids[trade_idx] = best_group

            # v2.6.0: Update CRIF state and refresh gradients for affected groups
            if refresh_gradients:
                # Update CRIF state: remove trade from source, add to destination
                current_group_crifs[from_group] = from_crif[
                    from_crif["TradeID"] != trade_id
                ].copy()
                if best_group in current_group_crifs and not current_group_crifs[best_group].empty:
                    current_group_crifs[best_group] = pd.concat([
                        current_group_crifs[best_group], trade_crif
                    ], ignore_index=True)
                else:
                    current_group_crifs[best_group] = trade_crif.copy()

                # Refresh gradients for affected groups using AADC (~1ms each)
                for affected_group in [from_group, best_group]:
                    affected_crif = current_group_crifs.get(affected_group, pd.DataFrame())
                    if affected_crif.empty or len(affected_crif) == 0:
                        current_group_gradients[affected_group] = np.array([])
                        current_group_ims[affected_group] = 0.0
                    else:
                        try:
                            gradient, im_val, _, _ = compute_im_gradient_aadc(
                                affected_crif, num_threads
                            )
                            current_group_gradients[affected_group] = gradient
                            current_group_ims[affected_group] = im_val
                            gradient_refresh_count += 1
                        except Exception as e:
                            # If AADC fails, keep old gradient (degraded accuracy)
                            print(f"    Warning: gradient refresh failed for group {affected_group}: {e}")
                            pass

    if not moves:
        elapsed = time.perf_counter() - realloc_start
        return {
            "n_moves": 0,
            "estimated_im_change": 0.0,
            "im_before": sum(group_ims.values()),
            "im_after": sum(group_ims.values()),
            "im_estimate": sum(group_ims.values()),
            "matches": True,
        }, elapsed

    # Compute estimated new total IM using quadratic formula.
    # SIMM = sqrt(s^T M s), so after adding/removing Δs:
    #   IM_new = sqrt(IM² ± 2*IM*marginal_impact + standalone²)
    # where standalone = SIMM of the moved trade alone.
    # This is exact for a single risk factor and second-order accurate in general.
    estimated_group_ims = dict(group_ims)
    for trade_id, from_group, to_group, old_contrib, signed_impact, standalone_im in moves:
        # Source group: remove trade (quadratic correction)
        im_from = estimated_group_ims[from_group]
        im_from_sq = im_from ** 2 - 2.0 * im_from * old_contrib + standalone_im ** 2
        estimated_group_ims[from_group] = np.sqrt(max(0.0, im_from_sq))

        # Target group: add trade (quadratic correction, signed impact)
        im_to = estimated_group_ims[to_group]
        im_to_sq = im_to ** 2 + 2.0 * im_to * signed_impact + standalone_im ** 2
        estimated_group_ims[to_group] = np.sqrt(max(0.0, im_to_sq))

    estimated_total_im = sum(estimated_group_ims.values())

    # Recompute SIMM for affected groups using AADC
    affected_groups = set()
    for _, from_g, to_g, _, _, _ in moves:
        affected_groups.add(from_g)
        affected_groups.add(to_g)

    recomputed_ims = dict(group_ims)
    recomputed_crifs = {}
    recomputed_gradients = {}

    for group in affected_groups:
        # Get trades now assigned to this group
        group_trade_indices = [i for i in range(len(trades)) if new_group_ids[i] == group]
        group_trades = [trades[i] for i in group_trade_indices]

        if not group_trades:
            recomputed_ims[group] = 0.0
            continue

        # Recompute CRIF via AADC
        new_crif, _, _ = compute_crif_aadc(group_trades, market, num_threads)

        if new_crif.empty:
            recomputed_ims[group] = 0.0
            continue

        recomputed_crifs[group] = new_crif

        # Recompute SIMM
        _, new_im, _ = run_simm(new_crif)
        recomputed_ims[group] = new_im

        # Recompute gradient for the new group composition
        if new_im > 0.0:
            gradient, _, _, _ = compute_im_gradient_aadc(new_crif, num_threads)
            recomputed_gradients[group] = gradient

    recomputed_total_im = sum(recomputed_ims.values())
    im_before = sum(group_ims.values())

    # Check if gradient estimate matches recomputation
    # Use relative tolerance of 10% (first-order estimate is approximate)
    if im_before > 0:
        rel_error = abs(estimated_total_im - recomputed_total_im) / im_before
    else:
        rel_error = 0.0
    matches = rel_error < 0.10  # 10% tolerance for first-order approximation

    elapsed = time.perf_counter() - realloc_start

    # Print reallocation summary
    print()
    print(f"  Reallocation Summary ({n_available} trades considered, {len(moves)} moved):")
    print(f"    {'TradeID':<20} {'From':>5} {'To':>5} {'Old Contrib':>18} {'Signed Impact':>18} {'Standalone IM':>18}")
    print(f"    {'-'*90}")
    for trade_id, from_g, to_g, old_c, signed_i, standalone in moves:
        print(f"    {trade_id:<20} {from_g:>5} {to_g:>5} {old_c:>18,.2f} {signed_i:>18,.2f} {standalone:>18,.2f}")
    print()
    print(f"    Total IM before:     {im_before:>18,.2f}")
    print(f"    Total IM after:      {recomputed_total_im:>18,.2f}")
    print(f"    Quadratic estimate:  {estimated_total_im:>18,.2f}")
    print(f"    Estimate rel error:  {rel_error*100:.4f}%")
    print(f"    Estimate matches:    {'YES' if matches else 'NO'} (< 10% tolerance)")
    if refresh_gradients:
        print(f"    Gradient refreshes:  {gradient_refresh_count:>18} (v2.6.0 iterative)")
    print(f"    Reallocation time:   {elapsed:.3f} s")
    print()

    return {
        "n_moves": len(moves),
        "estimated_im_change": estimated_total_im - im_before,
        "im_before": im_before,
        "im_after": recomputed_total_im,
        "im_estimate": estimated_total_im,
        "matches": matches,
        "rel_error": rel_error,
    }, elapsed


# =============================================================================
# Main
# =============================================================================

def main():
    if not AADC_AVAILABLE:
        print("Error: AADC is required for this implementation.")
        print("Install MatLogica AADC: pip install aadc")
        sys.exit(1)

    args = parse_common_args("SIMM Portfolio AADC Benchmark")

    trade_types = [t.strip() for t in args.trade_types.split(",")]
    num_trades = args.trades
    num_simm_buckets = args.simm_buckets
    num_portfolios = args.portfolios
    num_threads = args.threads
    n_reallocate = args.reallocate
    trade_types_str = ",".join(trade_types)

    print("=" * 80)
    print("           SIMM Portfolio AADC")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model:           {MODEL_NAME} v{MODEL_VERSION}")
    print(f"  Trade Types:     {trade_types_str}")
    print(f"  Trades:          {num_trades}")
    print(f"  SIMM Buckets:    {num_simm_buckets}")
    print(f"  Portfolios:      {num_portfolios}")
    print(f"  Threads:         {num_threads} (AADC workers)")
    if n_reallocate is not None:
        print(f"  Reallocate:      {n_reallocate} trades")
    print()

    # Generate portfolio (same as baseline)
    market, trades, group_ids, currencies = generate_portfolio(
        trade_types, num_trades, num_simm_buckets, num_portfolios
    )
    num_trades_actual = len(trades)

    # Process each group
    print(f"Processing groups (AADC CRIF -> SIMM -> AADC dIM/ds -> trade contributions)...")
    print()

    group_results = []
    all_crifs = []
    all_contrib_rows = []

    # Collect per-group data for reallocation
    group_crifs = {}       # group_id -> CRIF DataFrame
    group_gradients = {}   # group_id -> gradient array
    group_ims = {}         # group_id -> IM value
    group_contributions = {}  # group_id -> {trade_id: contribution}

    for group in range(num_portfolios):
        group_trade_indices = [i for i in range(len(trades)) if group_ids[i] == group]
        group_trades = [trades[i] for i in group_trade_indices]
        num_group_trades = len(group_trades)

        if num_group_trades == 0:
            group_results.append(make_empty_group_result(group))
            group_ims[group] = 0.0
            continue

        # 1. Compute CRIF via AADC
        group_crif, crif_record_time, crif_eval_time = compute_crif_aadc(
            group_trades, market, num_threads
        )
        crif_time = crif_record_time + crif_eval_time

        if group_crif.empty:
            group_results.append(make_empty_group_result(group, num_group_trades, crif_time))
            group_ims[group] = 0.0
            continue

        # Collect CRIF for logging
        crif_with_group = group_crif.copy()
        crif_with_group["GroupID"] = group
        all_crifs.append(crif_with_group)

        # Store for reallocation
        group_crifs[group] = group_crif.copy()

        # 2. Run SIMM
        portfolio, base_im, simm_time = run_simm(group_crif)
        group_ims[group] = base_im

        # 3. Compute IM gradient and per-trade contributions via AADC
        if base_im > 0.0:
            gradient, im_aadc, simm_record_time, simm_eval_time = \
                compute_im_gradient_aadc(group_crif, num_threads)
            num_sens = len(gradient)
            grad_time = simm_record_time + simm_eval_time

            # Store gradient for reallocation
            group_gradients[group] = gradient

            # Euler decomposition: per-trade contribution
            contributions = compute_trade_contributions(group_crif, gradient)
            group_contributions[group] = contributions
            print_trade_contributions(contributions, group, base_im)

            # Collect for CSV output
            for trade_id, contrib in contributions.items():
                all_contrib_rows.append({
                    "model_name": MODEL_NAME,
                    "group_id": group,
                    "trade_id": trade_id,
                    "contribution": contrib,
                    "im_total": base_im,
                    "pct_of_im": (contrib / base_im * 100) if base_im > 0 else 0.0,
                })
        else:
            grad_time = 0.0
            num_sens = 0

        group_results.append({
            "group_id": group,
            "num_group_trades": num_group_trades,
            "im_result": base_im,
            "crif_time_sec": crif_time,
            "simm_time_sec": simm_time,
            "im_sens_time_sec": grad_time,
            "num_im_sensitivities": num_sens,
        })

    # Save logs
    save_crif_log(all_crifs)
    save_trade_contributions_log(all_contrib_rows)
    print()

    # Reallocation step (if requested)
    realloc_result = None
    realloc_time = 0.0
    if n_reallocate is not None and n_reallocate > 0:
        refresh_gradients = not getattr(args, 'no_refresh_gradients', False)
        print(f"Reallocation: moving up to {n_reallocate} trades using gradient info...")
        if refresh_gradients:
            print("  (v2.6.0: iterative gradient refresh enabled)")
        else:
            print("  (gradient refresh disabled - using stale gradients)")
        realloc_result, realloc_time = reallocate_trades(
            n_reallocate,
            trades, group_ids,
            group_crifs, group_gradients, group_ims,
            group_contributions,
            market, num_threads, num_portfolios,
            refresh_gradients=refresh_gradients,
        )

    # Full optimization (if requested)
    opt_result = None
    if args.optimize:
        from model.simm_allocation_optimizer import reallocate_trades_optimal

        # Convert group_ids to allocation matrix
        T = len(trades)
        initial_allocation = np.zeros((T, num_portfolios))
        for t, g in enumerate(group_ids):
            initial_allocation[t, g] = 1.0

        print(f"\nFull Optimization (method={args.method})...")
        opt_result = reallocate_trades_optimal(
            trades, market, num_portfolios,
            initial_allocation=initial_allocation,
            num_threads=num_threads,
            allow_partial=args.allow_partial,
            method=args.method,
            max_iters=args.max_iters,
            lr=args.lr,
            tol=args.tol,
            verbose=True,
        )

        # Print optimization summary
        print()
        print(f"Optimization Summary:")
        print(f"  Method:              {args.method}")
        print(f"  Iterations:          {opt_result['num_iterations']} ({'converged' if opt_result['converged'] else 'max iters'})")
        print(f"  Initial IM:          ${opt_result['initial_im']:>18,.2f}")
        print(f"  Optimized IM:        ${opt_result['final_im']:>18,.2f}")
        reduction = (1.0 - opt_result['final_im'] / opt_result['initial_im']) * 100 if opt_result['initial_im'] > 0 else 0.0
        print(f"  Reduction:           {reduction:.1f}%")
        print(f"  Trades moved:        {opt_result['trades_moved']} of {len(trades)}")
        print(f"  Optimization time:   {opt_result['elapsed_time']:.3f} s")
        print()

    # Print results table
    totals = print_results_table(group_results, num_trades_actual)

    # Add reallocation info to totals
    if realloc_result is not None:
        totals["reallocate_n"] = n_reallocate
        totals["reallocate_time_sec"] = realloc_time
        totals["im_after_realloc"] = realloc_result.get("im_after", "")
        totals["im_realloc_estimate"] = realloc_result.get("im_estimate", "")
        totals["realloc_estimate_matches"] = realloc_result.get("matches", "")

        # Also add to each group result for per-group log rows
        for res in group_results:
            res["reallocate_n"] = n_reallocate
            res["reallocate_time_sec"] = realloc_time
            res["im_after_realloc"] = realloc_result.get("im_after", "")
            res["im_realloc_estimate"] = realloc_result.get("im_estimate", "")
            res["realloc_estimate_matches"] = realloc_result.get("matches", "")

    # Add optimization info to totals (if used)
    if opt_result is not None:
        totals["optimize_method"] = args.method
        totals["optimize_time_sec"] = opt_result['elapsed_time']
        totals["optimize_initial_im"] = opt_result['initial_im']
        totals["optimize_final_im"] = opt_result['final_im']
        totals["optimize_trades_moved"] = opt_result['trades_moved']
        totals["optimize_converged"] = opt_result['converged']
        totals["optimize_iterations"] = opt_result['num_iterations']

        # Also add to each group result for per-group log rows
        for res in group_results:
            res["optimize_method"] = args.method
            res["optimize_time_sec"] = opt_result['elapsed_time']
            res["optimize_initial_im"] = opt_result['initial_im']
            res["optimize_final_im"] = opt_result['final_im']
            res["optimize_trades_moved"] = opt_result['trades_moved']
            res["optimize_converged"] = opt_result['converged']

    # Timing summary
    total_time = totals["total_crif_time"] + totals["total_simm_time"] + totals["total_grad_time"]
    print(f"Timing Summary (AADC):")
    print(f"  CRIF (AADC):       {totals['total_crif_time']:.3f} s ({num_portfolios} groups)")
    print(f"  SIMM aggregation:  {totals['total_simm_time']:.6f} s ({num_portfolios} groups)")
    print(f"  IM gradient (AADC): {totals['total_grad_time']:.3f} s ({totals['total_num_sens']} sensitivities, single adjoint)")
    if realloc_result is not None:
        print(f"  Reallocation:      {realloc_time:.3f} s ({realloc_result.get('n_moves', 0)} trades moved)")
        total_time += realloc_time
    if opt_result is not None:
        opt_time = opt_result['elapsed_time']
        print(f"  Optimization:      {opt_time:.3f} s ({opt_result['num_iterations']} iterations, {opt_result['trades_moved']} trades moved)")
        total_time += opt_time
    print(f"  Total:             {total_time:.3f} s")
    print()

    # Write execution log
    log_rows = build_log_rows(
        group_results, totals,
        MODEL_NAME, MODEL_VERSION, trade_types_str,
        num_trades_actual, num_simm_buckets, num_portfolios,
        num_threads,
    )
    write_log(log_rows)
    print(f"Execution logged to {LOG_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
