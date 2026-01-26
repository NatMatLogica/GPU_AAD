"""
Python bindings for all C++ trade models.

Mirrors the C++ implementations in model/cpp/ with identical pricing logic:
- vanilla_irs.h    → IRSwapTrade + price_vanilla_irs
- equity_option.h  → EquityOptionTrade + price_equity_option
- fx_option.h      → FXOptionTrade + price_fx_option
- inflation_swap.h → InflationSwapTrade + price_inflation_swap
- xccy_swap.h      → XCCYSwapTrade + price_xccy_swap

Each trade type provides:
- Trade dataclass matching C++ struct
- Pricing function matching C++ template instantiation (T=double)
- CRIF generation via bump-and-revalue
- Random trade generation for benchmarking
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from scipy.special import erf as scipy_erf

# =============================================================================
# Constants matching simm_config.h
# =============================================================================

NUM_IR_TENORS = 12
IR_TENORS = np.array([2/52, 1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0])
IR_TENOR_LABELS = ["2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "7y", "10y", "15y", "30y"]

NUM_VEGA_EXPIRIES = 6
VEGA_EXPIRIES = np.array([0.5, 1.0, 3.0, 5.0, 10.0, 30.0])

BUMP_SIZE = 0.0001  # 1bp for IR/inflation
SPOT_BUMP = 1.0     # 1 unit for spot/FX
VOL_BUMP = 0.01     # 1% for vol


# =============================================================================
# Market Data (matching market_data.h)
# =============================================================================

@dataclass
class YieldCurve:
    """Yield curve: zero rates at SIMM tenors, linear interpolation."""
    zero_rates: np.ndarray = field(default_factory=lambda: np.zeros(NUM_IR_TENORS))

    def zero_rate(self, t: float) -> float:
        return float(np.interp(t, IR_TENORS, self.zero_rates))

    def discount(self, t: float) -> float:
        if t <= 0:
            return 1.0
        r = self.zero_rate(t)
        return np.exp(-r * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            return self.zero_rate(t1)
        df1 = self.discount(t1)
        df2 = self.discount(t2)
        return np.log(df1 / df2) / (t2 - t1)

    def bumped(self, tenor_idx: int, amount: float = BUMP_SIZE) -> 'YieldCurve':
        new_rates = self.zero_rates.copy()
        new_rates[tenor_idx] += amount
        return YieldCurve(zero_rates=new_rates)


@dataclass
class VolSurface:
    """Vol surface: flat or term-structure by expiry."""
    vols: np.ndarray = field(default_factory=lambda: np.full(NUM_VEGA_EXPIRIES, 0.2))

    def vol(self, expiry: float) -> float:
        return float(np.interp(expiry, VEGA_EXPIRIES, self.vols))

    def bumped(self, expiry_idx: int, amount: float = VOL_BUMP) -> 'VolSurface':
        new_vols = self.vols.copy()
        new_vols[expiry_idx] += amount
        return VolSurface(vols=new_vols)


@dataclass
class InflationCurve:
    """Inflation curve: CPI levels and zero-coupon inflation rates."""
    base_cpi: float = 100.0
    inflation_rates: np.ndarray = field(default_factory=lambda: np.full(NUM_IR_TENORS, 0.025))

    def inflation_rate(self, t: float) -> float:
        return float(np.interp(t, IR_TENORS, self.inflation_rates))

    def projected_cpi(self, t: float) -> float:
        rate = self.inflation_rate(t)
        return self.base_cpi * np.exp(rate * t)

    def bumped(self, tenor_idx: int, amount: float = BUMP_SIZE) -> 'InflationCurve':
        new_rates = self.inflation_rates.copy()
        new_rates[tenor_idx] += amount
        return InflationCurve(base_cpi=self.base_cpi, inflation_rates=new_rates)


@dataclass
class MarketEnvironment:
    """Complete market data for all trade types."""
    curves: Dict[str, YieldCurve] = field(default_factory=dict)
    vol_surfaces: Dict[str, VolSurface] = field(default_factory=dict)
    inflation: Optional[InflationCurve] = None
    fx_spots: Dict[str, float] = field(default_factory=dict)  # e.g. "EURUSD" → 1.08
    equity_spots: Dict[str, float] = field(default_factory=dict)  # e.g. "SPX" → 4500


# =============================================================================
# Helper
# =============================================================================

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + float(scipy_erf(x / np.sqrt(2.0))))


# =============================================================================
# Trade Types (matching C++ structs)
# =============================================================================

@dataclass
class IRSwapTrade:
    trade_id: str = ""
    trade_type: str = "ir_swap"
    currency: str = "USD"
    notional: float = 1e6
    fixed_rate: float = 0.03
    maturity: float = 5.0
    frequency: int = 2
    payer: bool = True


@dataclass
class EquityOptionTrade:
    trade_id: str = ""
    trade_type: str = "equity_option"
    currency: str = "USD"
    notional: float = 1e6
    strike: float = 100.0
    maturity: float = 1.0
    dividend_yield: float = 0.02
    is_call: bool = True
    equity_bucket: int = 0
    underlying: str = "SPX"


@dataclass
class FXOptionTrade:
    trade_id: str = ""
    trade_type: str = "fx_option"
    currency: str = "USD"
    notional: float = 1e6
    strike: float = 1.08
    maturity: float = 1.0
    is_call: bool = True
    domestic_ccy: str = "USD"
    foreign_ccy: str = "EUR"


@dataclass
class InflationSwapTrade:
    trade_id: str = ""
    trade_type: str = "inflation_swap"
    currency: str = "USD"
    notional: float = 1e6
    fixed_rate: float = 0.025
    maturity: float = 5.0


@dataclass
class XCCYSwapTrade:
    trade_id: str = ""
    trade_type: str = "xccy_swap"
    currency: str = "USD"
    dom_notional: float = 1e6
    fgn_notional: float = 800000.0
    dom_fixed_rate: float = 0.03
    fgn_fixed_rate: float = 0.02
    maturity: float = 5.0
    frequency: int = 2
    exchange_notional: bool = True
    domestic_ccy: str = "USD"
    foreign_ccy: str = "EUR"


# =============================================================================
# Pricing Functions (matching C++ templates with T=double)
# =============================================================================

def price_vanilla_irs(trade: IRSwapTrade, curve: YieldCurve) -> float:
    dt = 1.0 / trade.frequency
    num_periods = int(trade.maturity * trade.frequency)

    fixed_leg = 0.0
    floating_leg = 0.0

    for i in range(1, num_periods + 1):
        t = i * dt
        df = curve.discount(t)
        fixed_leg += trade.notional * trade.fixed_rate * dt * df

        t_prev = (i - 1) * dt
        fwd = curve.forward_rate(t_prev, t)
        floating_leg += trade.notional * dt * fwd * df

    npv = floating_leg - fixed_leg
    if not trade.payer:
        npv = fixed_leg - floating_leg
    return npv


def price_equity_option(trade: EquityOptionTrade, curve: YieldCurve,
                         spot: float, vol: float) -> float:
    r = curve.zero_rate(trade.maturity)
    q = trade.dividend_yield
    tau = trade.maturity
    sqrt_tau = np.sqrt(tau)

    d1 = (np.log(spot / trade.strike) + (r - q + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau

    df = np.exp(-r * tau)
    dq = np.exp(-q * tau)

    num_contracts = trade.notional / trade.strike

    if trade.is_call:
        price = num_contracts * (spot * dq * normal_cdf(d1) - trade.strike * df * normal_cdf(d2))
    else:
        price = num_contracts * (trade.strike * df * normal_cdf(-d2) - spot * dq * normal_cdf(-d1))

    return price


def price_fx_option(trade: FXOptionTrade, spot: float, vol: float,
                    dom_curve: YieldCurve, fgn_curve: YieldCurve) -> float:
    rd = dom_curve.zero_rate(trade.maturity)
    rf = fgn_curve.zero_rate(trade.maturity)
    tau = trade.maturity
    sqrt_tau = np.sqrt(tau)

    d1 = (np.log(spot / trade.strike) + (rd - rf + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau

    df_dom = np.exp(-rd * tau)
    df_fgn = np.exp(-rf * tau)

    fgn_notional = trade.notional / trade.strike

    if trade.is_call:
        price = fgn_notional * (spot * df_fgn * normal_cdf(d1) - trade.strike * df_dom * normal_cdf(d2))
    else:
        price = fgn_notional * (trade.strike * df_dom * normal_cdf(-d2) - spot * df_fgn * normal_cdf(-d1))

    return price


def price_inflation_swap(trade: InflationSwapTrade, curve: YieldCurve,
                          inflation: InflationCurve) -> float:
    tau = trade.maturity
    df = curve.discount(tau)

    fixed_leg = trade.notional * (np.exp(trade.fixed_rate * tau) - 1.0) * df
    cpi_ratio = inflation.projected_cpi(tau) / inflation.base_cpi
    inflation_leg = trade.notional * (cpi_ratio - 1.0) * df

    return inflation_leg - fixed_leg


def price_xccy_swap(trade: XCCYSwapTrade, dom_curve: YieldCurve,
                    fgn_curve: YieldCurve, fx_spot: float) -> float:
    dt = 1.0 / trade.frequency
    num_periods = int(trade.maturity * trade.frequency)

    dom_leg = 0.0
    for i in range(1, num_periods + 1):
        t = i * dt
        df = dom_curve.discount(t)
        dom_leg += trade.dom_notional * trade.dom_fixed_rate * dt * df

    fgn_leg = 0.0
    for i in range(1, num_periods + 1):
        t = i * dt
        df = fgn_curve.discount(t)
        fgn_leg += trade.fgn_notional * trade.fgn_fixed_rate * dt * df

    if trade.exchange_notional:
        dom_df_mat = dom_curve.discount(trade.maturity)
        fgn_df_mat = fgn_curve.discount(trade.maturity)
        dom_leg += trade.dom_notional * dom_df_mat
        fgn_leg += trade.fgn_notional * fgn_df_mat
        dom_leg -= trade.dom_notional
        fgn_leg -= trade.fgn_notional

    return dom_leg - fgn_leg * fx_spot


# =============================================================================
# CRIF Generation (bump-and-revalue for each trade type)
# =============================================================================

def _crif_row(trade_id, risk_type, qualifier, bucket, label1, label2, amount, ccy):
    return {
        "TradeID": trade_id,
        "ProductClass": "RatesFX" if risk_type in ("Risk_IRCurve", "Risk_FX", "Risk_Inflation", "Risk_XCcyBasis") else "Equity",
        "RiskType": risk_type,
        "Qualifier": qualifier,
        "Bucket": str(bucket),
        "Label1": label1,
        "Label2": label2,
        "Amount": amount,
        "AmountCurrency": ccy,
        "AmountUSD": amount,  # 1:1 FX assumption
    }


def compute_crif_irs(trade: IRSwapTrade, market: MarketEnvironment) -> List[dict]:
    curve = market.curves[trade.currency]
    base_pv = price_vanilla_irs(trade, curve)
    rows = []
    for i in range(NUM_IR_TENORS):
        bumped_curve = curve.bumped(i, BUMP_SIZE)
        bumped_pv = price_vanilla_irs(trade, bumped_curve)
        delta = (bumped_pv - base_pv) / BUMP_SIZE
        if abs(delta) > 1e-10:
            rows.append(_crif_row(
                trade.trade_id, "Risk_IRCurve", trade.currency,
                i + 1, IR_TENOR_LABELS[i], "OIS", delta, trade.currency
            ))
    return rows


def compute_crif_equity_option(trade: EquityOptionTrade, market: MarketEnvironment) -> List[dict]:
    curve = market.curves[trade.currency]
    spot = market.equity_spots.get(trade.underlying, 100.0)
    vol_surface = market.vol_surfaces.get(trade.underlying, VolSurface())
    vol = vol_surface.vol(trade.maturity)
    base_pv = price_equity_option(trade, curve, spot, vol)
    rows = []

    # IR Delta
    for i in range(NUM_IR_TENORS):
        bumped_curve = curve.bumped(i, BUMP_SIZE)
        bumped_pv = price_equity_option(trade, bumped_curve, spot, vol)
        delta = (bumped_pv - base_pv) / BUMP_SIZE
        if abs(delta) > 1e-10:
            rows.append(_crif_row(
                trade.trade_id, "Risk_IRCurve", trade.currency,
                i + 1, IR_TENOR_LABELS[i], "OIS", delta, trade.currency
            ))

    # Equity Delta (spot bump)
    bumped_pv = price_equity_option(trade, curve, spot + SPOT_BUMP, vol)
    eq_delta = (bumped_pv - base_pv) / SPOT_BUMP * spot  # sensitivity in notional terms
    if abs(eq_delta) > 1e-10:
        rows.append(_crif_row(
            trade.trade_id, "Risk_Equity", trade.underlying,
            str(trade.equity_bucket + 1), "", "spot", eq_delta, trade.currency
        ))

    # Equity Vega (vol bump at each expiry)
    for i in range(NUM_VEGA_EXPIRIES):
        bumped_vol_surface = vol_surface.bumped(i, VOL_BUMP)
        bumped_vol = bumped_vol_surface.vol(trade.maturity)
        bumped_pv = price_equity_option(trade, curve, spot, bumped_vol)
        vega = (bumped_pv - base_pv) / VOL_BUMP
        if abs(vega) > 1e-10:
            rows.append(_crif_row(
                trade.trade_id, "Risk_EquityVol", trade.underlying,
                str(trade.equity_bucket + 1), f"{VEGA_EXPIRIES[i]:.1f}y", "", vega, trade.currency
            ))

    return rows


def compute_crif_fx_option(trade: FXOptionTrade, market: MarketEnvironment) -> List[dict]:
    dom_curve = market.curves[trade.domestic_ccy]
    fgn_curve = market.curves.get(trade.foreign_ccy, YieldCurve(zero_rates=np.full(NUM_IR_TENORS, 0.02)))
    pair = f"{trade.foreign_ccy}{trade.domestic_ccy}"
    spot = market.fx_spots.get(pair, trade.strike)
    vol_surface = market.vol_surfaces.get(pair, VolSurface())
    vol = vol_surface.vol(trade.maturity)
    base_pv = price_fx_option(trade, spot, vol, dom_curve, fgn_curve)
    rows = []

    # Domestic IR Delta
    for i in range(NUM_IR_TENORS):
        bumped = dom_curve.bumped(i, BUMP_SIZE)
        bumped_pv = price_fx_option(trade, spot, vol, bumped, fgn_curve)
        delta = (bumped_pv - base_pv) / BUMP_SIZE
        if abs(delta) > 1e-10:
            rows.append(_crif_row(
                trade.trade_id, "Risk_IRCurve", trade.domestic_ccy,
                i + 1, IR_TENOR_LABELS[i], "OIS", delta, trade.domestic_ccy
            ))

    # Foreign IR Delta
    for i in range(NUM_IR_TENORS):
        bumped = fgn_curve.bumped(i, BUMP_SIZE)
        bumped_pv = price_fx_option(trade, spot, vol, dom_curve, bumped)
        delta = (bumped_pv - base_pv) / BUMP_SIZE
        if abs(delta) > 1e-10:
            rows.append(_crif_row(
                trade.trade_id, "Risk_IRCurve", trade.foreign_ccy,
                i + 1, IR_TENOR_LABELS[i], "OIS", delta, trade.domestic_ccy
            ))

    # FX Delta (spot bump)
    bumped_pv = price_fx_option(trade, spot + SPOT_BUMP * 0.01, vol, dom_curve, fgn_curve)
    fx_delta = (bumped_pv - base_pv) / (SPOT_BUMP * 0.01) * spot
    if abs(fx_delta) > 1e-10:
        rows.append(_crif_row(
            trade.trade_id, "Risk_FX", pair,
            "", "", "", fx_delta, trade.domestic_ccy
        ))

    # FX Vega
    for i in range(NUM_VEGA_EXPIRIES):
        bumped_vol_surface = vol_surface.bumped(i, VOL_BUMP)
        bumped_vol = bumped_vol_surface.vol(trade.maturity)
        bumped_pv = price_fx_option(trade, spot, bumped_vol, dom_curve, fgn_curve)
        vega = (bumped_pv - base_pv) / VOL_BUMP
        if abs(vega) > 1e-10:
            rows.append(_crif_row(
                trade.trade_id, "Risk_FXVol", pair,
                "", f"{VEGA_EXPIRIES[i]:.1f}y", "", vega, trade.domestic_ccy
            ))

    return rows


def compute_crif_inflation_swap(trade: InflationSwapTrade, market: MarketEnvironment) -> List[dict]:
    curve = market.curves[trade.currency]
    inflation = market.inflation or InflationCurve()
    base_pv = price_inflation_swap(trade, curve, inflation)
    rows = []

    # IR Delta
    for i in range(NUM_IR_TENORS):
        bumped_curve = curve.bumped(i, BUMP_SIZE)
        bumped_pv = price_inflation_swap(trade, bumped_curve, inflation)
        delta = (bumped_pv - base_pv) / BUMP_SIZE
        if abs(delta) > 1e-10:
            rows.append(_crif_row(
                trade.trade_id, "Risk_IRCurve", trade.currency,
                i + 1, IR_TENOR_LABELS[i], "OIS", delta, trade.currency
            ))

    # Inflation Delta
    for i in range(NUM_IR_TENORS):
        bumped_inflation = inflation.bumped(i, BUMP_SIZE)
        bumped_pv = price_inflation_swap(trade, curve, bumped_inflation)
        delta = (bumped_pv - base_pv) / BUMP_SIZE
        if abs(delta) > 1e-10:
            rows.append(_crif_row(
                trade.trade_id, "Risk_Inflation", trade.currency,
                i + 1, IR_TENOR_LABELS[i], "", delta, trade.currency
            ))

    return rows


def compute_crif_xccy_swap(trade: XCCYSwapTrade, market: MarketEnvironment) -> List[dict]:
    dom_curve = market.curves[trade.domestic_ccy]
    fgn_curve = market.curves.get(trade.foreign_ccy, YieldCurve(zero_rates=np.full(NUM_IR_TENORS, 0.02)))
    pair = f"{trade.foreign_ccy}{trade.domestic_ccy}"
    fx_spot = market.fx_spots.get(pair, trade.dom_notional / trade.fgn_notional)
    base_pv = price_xccy_swap(trade, dom_curve, fgn_curve, fx_spot)
    rows = []

    # Domestic IR Delta
    for i in range(NUM_IR_TENORS):
        bumped = dom_curve.bumped(i, BUMP_SIZE)
        bumped_pv = price_xccy_swap(trade, bumped, fgn_curve, fx_spot)
        delta = (bumped_pv - base_pv) / BUMP_SIZE
        if abs(delta) > 1e-10:
            rows.append(_crif_row(
                trade.trade_id, "Risk_IRCurve", trade.domestic_ccy,
                i + 1, IR_TENOR_LABELS[i], "OIS", delta, trade.domestic_ccy
            ))

    # Foreign IR Delta
    for i in range(NUM_IR_TENORS):
        bumped = fgn_curve.bumped(i, BUMP_SIZE)
        bumped_pv = price_xccy_swap(trade, dom_curve, bumped, fx_spot)
        delta = (bumped_pv - base_pv) / BUMP_SIZE
        if abs(delta) > 1e-10:
            rows.append(_crif_row(
                trade.trade_id, "Risk_IRCurve", trade.foreign_ccy,
                i + 1, IR_TENOR_LABELS[i], "OIS", delta, trade.domestic_ccy
            ))

    # FX Delta
    bumped_pv = price_xccy_swap(trade, dom_curve, fgn_curve, fx_spot + SPOT_BUMP * 0.01)
    fx_delta = (bumped_pv - base_pv) / (SPOT_BUMP * 0.01) * fx_spot
    if abs(fx_delta) > 1e-10:
        rows.append(_crif_row(
            trade.trade_id, "Risk_FX", pair,
            "", "", "", fx_delta, trade.domestic_ccy
        ))

    return rows


# =============================================================================
# Unified CRIF dispatch
# =============================================================================

CRIF_DISPATCH = {
    "ir_swap": compute_crif_irs,
    "equity_option": compute_crif_equity_option,
    "fx_option": compute_crif_fx_option,
    "inflation_swap": compute_crif_inflation_swap,
    "xccy_swap": compute_crif_xccy_swap,
}


def compute_crif_for_trade(trade, market: MarketEnvironment) -> List[dict]:
    """Compute CRIF rows for a single trade via bump-and-revalue."""
    fn = CRIF_DISPATCH.get(trade.trade_type)
    if fn is None:
        raise ValueError(f"Unknown trade type: {trade.trade_type}")
    return fn(trade, market)


def compute_crif_for_trades(trades: list, market: MarketEnvironment) -> pd.DataFrame:
    """Compute CRIF for a list of trades, returning a DataFrame."""
    all_rows = []
    for trade in trades:
        all_rows.extend(compute_crif_for_trade(trade, market))
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# =============================================================================
# Trade Generation
# =============================================================================

def generate_market_environment(currencies: List[str], seed: int = 42) -> MarketEnvironment:
    """Generate a complete market environment for all trade types."""
    rng = np.random.default_rng(seed)
    market = MarketEnvironment()

    # Yield curves
    for i, ccy in enumerate(currencies):
        base_rate = 0.03 + i * 0.005
        rates = base_rate + np.linspace(-0.005, 0.01, NUM_IR_TENORS) + rng.normal(0, 0.001, NUM_IR_TENORS)
        market.curves[ccy] = YieldCurve(zero_rates=rates)

    # Vol surfaces (one per currency pair and equity)
    market.vol_surfaces["SPX"] = VolSurface(vols=0.18 + rng.normal(0, 0.02, NUM_VEGA_EXPIRIES))
    if len(currencies) >= 2:
        pair = f"{currencies[1]}{currencies[0]}"
        market.vol_surfaces[pair] = VolSurface(vols=0.10 + rng.normal(0, 0.01, NUM_VEGA_EXPIRIES))

    # Inflation curve
    market.inflation = InflationCurve(
        base_cpi=100.0,
        inflation_rates=0.025 + rng.normal(0, 0.002, NUM_IR_TENORS)
    )

    # FX spots
    if len(currencies) >= 2:
        pair = f"{currencies[1]}{currencies[0]}"
        market.fx_spots[pair] = 1.08 + rng.normal(0, 0.02)

    # Equity spots
    market.equity_spots["SPX"] = 4500.0 + rng.normal(0, 100)

    return market


def generate_trades_by_type(trade_type: str, num_trades: int,
                            currencies: List[str], seed: int = 42,
                            avg_maturity: float = None, maturity_spread: float = None) -> list:
    """Generate random trades of a given type.

    Args:
        trade_type: Type of trade (ir_swap, equity_option, etc.)
        num_trades: Number of trades to generate
        currencies: List of currencies to use
        seed: Random seed
        avg_maturity: Average maturity in years (None = use default random selection)
        maturity_spread: Spread around average maturity (None = use default)
    """
    rng = np.random.default_rng(seed)
    trades = []

    # Default maturity choices by trade type
    default_maturities = {
        'ir_swap': [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
        'equity_option': [0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
        'fx_option': [0.25, 0.5, 1.0, 2.0, 3.0],
        'inflation_swap': [2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
        'xccy_swap': [2.0, 3.0, 5.0, 7.0, 10.0],
    }

    def get_maturity(trade_type_key: str) -> float:
        """Get maturity based on parameters or defaults."""
        if avg_maturity is not None:
            # Use specified average and spread
            spread = maturity_spread if maturity_spread is not None else 1.0
            min_mat = max(0.25, avg_maturity - spread)
            max_mat = min(30.0, avg_maturity + spread)
            return min_mat + rng.random() * (max_mat - min_mat)
        else:
            # Use default random selection from predefined values
            return rng.choice(default_maturities.get(trade_type_key, [5.0]))

    for i in range(num_trades):
        trade_id = f"{trade_type.upper()}_{i:06d}"
        ccy = currencies[rng.integers(len(currencies))]

        if trade_type == "ir_swap":
            trades.append(IRSwapTrade(
                trade_id=trade_id,
                currency=ccy,
                notional=rng.choice([1e6, 5e6, 10e6, 50e6, 100e6]),
                fixed_rate=0.01 + rng.random() * 0.04,
                maturity=get_maturity('ir_swap'),
                frequency=rng.choice([1, 2, 4]),
                payer=bool(rng.integers(2)),
            ))

        elif trade_type == "equity_option":
            trades.append(EquityOptionTrade(
                trade_id=trade_id,
                currency=ccy,
                notional=rng.choice([1e6, 5e6, 10e6]),
                strike=100.0 * (0.8 + rng.random() * 0.4),
                maturity=get_maturity('equity_option'),
                dividend_yield=0.01 + rng.random() * 0.03,
                is_call=bool(rng.integers(2)),
                equity_bucket=int(rng.integers(12)),
                underlying="SPX",
            ))

        elif trade_type == "fx_option":
            if len(currencies) < 2:
                dom, fgn = currencies[0], "EUR"
            else:
                idx = rng.choice(len(currencies), 2, replace=False)
                dom, fgn = currencies[idx[0]], currencies[idx[1]]
            trades.append(FXOptionTrade(
                trade_id=trade_id,
                currency=dom,
                notional=rng.choice([1e6, 5e6, 10e6]),
                strike=1.08 * (0.9 + rng.random() * 0.2),
                maturity=get_maturity('fx_option'),
                is_call=bool(rng.integers(2)),
                domestic_ccy=dom,
                foreign_ccy=fgn,
            ))

        elif trade_type == "inflation_swap":
            trades.append(InflationSwapTrade(
                trade_id=trade_id,
                currency=ccy,
                notional=rng.choice([1e6, 5e6, 10e6, 50e6]),
                fixed_rate=0.015 + rng.random() * 0.02,
                maturity=get_maturity('inflation_swap'),
            ))

        elif trade_type == "xccy_swap":
            if len(currencies) < 2:
                dom, fgn = currencies[0], "EUR"
            else:
                idx = rng.choice(len(currencies), 2, replace=False)
                dom, fgn = currencies[idx[0]], currencies[idx[1]]
            dom_not = rng.choice([1e6, 5e6, 10e6, 50e6])
            fx = 1.08 + rng.normal(0, 0.05)
            trades.append(XCCYSwapTrade(
                trade_id=trade_id,
                currency=dom,
                dom_notional=dom_not,
                fgn_notional=dom_not / fx,
                dom_fixed_rate=0.02 + rng.random() * 0.02,
                fgn_fixed_rate=0.01 + rng.random() * 0.02,
                maturity=get_maturity('xccy_swap'),
                frequency=rng.choice([1, 2, 4]),
                exchange_notional=True,
                domestic_ccy=dom,
                foreign_ccy=fgn,
            ))

        else:
            raise ValueError(f"Unknown trade type: {trade_type}")

    return trades
