"""
IR Swap Common - Shared classes and utilities for IR swap pricing.

This module contains data structures and utilities shared between
the baseline (bump & revalue) and AADC (AAD) implementations.

Version: 1.0.0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# =============================================================================
# Constants
# =============================================================================

# ISDA SIMM Tenor Buckets (in years)
TENOR_BUCKETS = {
    "2w": 2/52,
    "1m": 1/12,
    "3m": 3/12,
    "6m": 6/12,
    "1y": 1.0,
    "2y": 2.0,
    "3y": 3.0,
    "5y": 5.0,
    "10y": 10.0,
    "15y": 15.0,
    "20y": 20.0,
    "30y": 30.0,
}

TENOR_LABELS = list(TENOR_BUCKETS.keys())
TENOR_YEARS = np.array(list(TENOR_BUCKETS.values()))
NUM_TENORS = len(TENOR_LABELS)

# Bump size for finite difference (1 basis point)
BUMP_SIZE = 0.0001

# Day count convention (ACT/365)
DAYS_PER_YEAR = 365.0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MarketData:
    """Market data for IR swap pricing."""
    currency: str
    valuation_date: float  # As year fraction from epoch (simplified)

    # Discount curve: zero rates at tenor points
    discount_tenors: np.ndarray  # Years
    discount_rates: np.ndarray   # Continuous zero rates

    # Forward curve: zero rates for forward rate calculation
    forward_tenors: np.ndarray   # Years
    forward_rates: np.ndarray    # Continuous zero rates

    def discount_factor(self, t: float) -> float:
        """Calculate discount factor at time t."""
        if t <= 0:
            return 1.0
        rate = np.interp(t, self.discount_tenors, self.discount_rates)
        return np.exp(-rate * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Calculate forward rate between t1 and t2."""
        if t2 <= t1:
            return 0.0
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        return (df1 / df2 - 1) / (t2 - t1)

    def bump_curve(self, tenor_idx: int, bump: float = BUMP_SIZE) -> 'MarketData':
        """Return new MarketData with bumped rate at tenor_idx."""
        new_discount = self.discount_rates.copy()
        new_forward = self.forward_rates.copy()

        # Bump both discount and forward curves at this tenor
        if tenor_idx < len(new_discount):
            new_discount[tenor_idx] += bump
        if tenor_idx < len(new_forward):
            new_forward[tenor_idx] += bump

        return MarketData(
            currency=self.currency,
            valuation_date=self.valuation_date,
            discount_tenors=self.discount_tenors.copy(),
            discount_rates=new_discount,
            forward_tenors=self.forward_tenors.copy(),
            forward_rates=new_forward,
        )


@dataclass
class IRSwap:
    """Interest Rate Swap definition."""
    trade_id: str
    currency: str
    notional: float
    fixed_rate: float
    maturity: float  # Years
    pay_frequency: float  # Years (e.g., 0.5 for semi-annual)
    is_payer: bool  # True if paying fixed

    def payment_dates(self) -> np.ndarray:
        """Generate payment dates from now until maturity."""
        dates = []
        t = self.pay_frequency
        while t <= self.maturity + 1e-9:
            dates.append(t)
            t += self.pay_frequency
        return np.array(dates)

    def num_payments(self) -> int:
        """Return number of payment dates."""
        return int(np.ceil(self.maturity / self.pay_frequency))


@dataclass
class PricingResult:
    """Result from pricing."""
    prices: np.ndarray
    eval_time: float
    num_evals: int


@dataclass
class GreeksResult:
    """Result from pricing with Greeks."""
    prices: np.ndarray
    # IR Delta by currency and tenor bucket (num_trades x num_currencies x num_tenors)
    ir_delta: np.ndarray
    currencies: List[str]
    tenor_labels: List[str]
    eval_time: float
    first_run_time: float
    steady_state_time: float
    recording_time: float  # AADC kernel recording time
    num_evals: int
    num_sensitivities: int
    num_bumps: int  # Total number of bump scenarios (0 for AAD)


@dataclass
class ExtendedGreeksResult:
    """
    Extended Greeks result including gamma, cross-gamma, and theta.

    Required for SIMM curvature risk and P&L explain.
    """
    prices: np.ndarray

    # IR Delta: d(PV)/d(rate) - (num_trades x num_currencies x num_tenors)
    ir_delta: np.ndarray

    # IR Gamma (diagonal): d²(PV)/d(rate)² - (num_trades x num_currencies x num_tenors)
    # Used for SIMM curvature risk
    ir_gamma: np.ndarray

    # IR Cross-Gamma: d²(PV)/d(rate_i)d(rate_j) - (num_trades x num_currencies x num_tenors x num_tenors)
    # Off-diagonal second derivatives
    ir_cross_gamma: np.ndarray

    # Theta: d(PV)/dt - time decay per day (num_trades)
    theta: np.ndarray

    currencies: List[str]
    tenor_labels: List[str]
    eval_time: float
    num_evals: int


# =============================================================================
# Data Generation
# =============================================================================

def generate_market_data(
    currency: str = "USD",
    base_rate: float = 0.05,
    seed: int = 42
) -> MarketData:
    """Generate synthetic market data for a currency."""
    np.random.seed(seed)

    # Use SIMM tenor buckets
    tenors = TENOR_YEARS.copy()

    # Generate a realistic yield curve (upward sloping with some noise)
    # Base rate + term premium + small noise
    term_premium = 0.002 * np.sqrt(tenors)  # ~20bp for 10y
    noise = np.random.normal(0, 0.001, len(tenors))
    rates = base_rate + term_premium + noise

    # Ensure rates are positive
    rates = np.maximum(rates, 0.001)

    return MarketData(
        currency=currency,
        valuation_date=0.0,
        discount_tenors=tenors,
        discount_rates=rates,
        forward_tenors=tenors,
        forward_rates=rates,
    )


def generate_trades(
    num_trades: int,
    currencies: List[str] = None,
    seed: int = 42
) -> List[IRSwap]:
    """Generate synthetic IR swap trades."""
    if currencies is None:
        currencies = ["USD", "EUR", "GBP", "JPY", "CHF"]

    np.random.seed(seed)
    trades = []

    for i in range(num_trades):
        currency = currencies[i % len(currencies)]

        # Random swap parameters
        notional = np.random.choice([1e6, 5e6, 10e6, 50e6, 100e6])
        maturity = np.random.choice([1, 2, 3, 5, 7, 10, 15, 20, 30])
        fixed_rate = np.random.uniform(0.02, 0.06)
        pay_frequency = np.random.choice([0.25, 0.5, 1.0])  # Quarterly, semi, annual
        is_payer = np.random.choice([True, False])

        trades.append(IRSwap(
            trade_id=f"SWAP_{i:06d}",
            currency=currency,
            notional=notional,
            fixed_rate=fixed_rate,
            maturity=float(maturity),
            pay_frequency=pay_frequency,
            is_payer=is_payer,
        ))

    return trades


# =============================================================================
# CRIF Generation
# =============================================================================

def generate_crif(
    trades: List[IRSwap],
    greeks_result: GreeksResult,
) -> pd.DataFrame:
    """
    Convert Greeks to CRIF format for SIMM aggregation.

    Args:
        trades: List of IR swaps
        greeks_result: Result from price_with_greeks

    Returns:
        DataFrame in CRIF format
    """
    rows = []
    currencies = greeks_result.currencies
    ccy_to_idx = {ccy: i for i, ccy in enumerate(currencies)}

    for i, swap in enumerate(trades):
        ccy_idx = ccy_to_idx.get(swap.currency)
        if ccy_idx is None:
            continue

        for j, tenor_label in enumerate(greeks_result.tenor_labels):
            delta = greeks_result.ir_delta[i, ccy_idx, j]

            # Skip zero sensitivities
            if abs(delta) < 1e-10:
                continue

            # Map tenor label to bucket number (1-12)
            bucket = j + 1  # SIMM uses 1-indexed buckets

            rows.append({
                "ProductClass": "RatesFX",
                "RiskType": "Risk_IRCurve",
                "Qualifier": swap.currency,
                "Bucket": str(bucket),
                "Label1": tenor_label,
                "Label2": "OIS",  # Simplified: assume OIS curve
                "Amount": delta,
                "AmountCurrency": swap.currency,
                "AmountUSD": delta,  # Simplified: assume 1:1 FX
            })

    return pd.DataFrame(rows)


# =============================================================================
# Performance Utilities
# =============================================================================

def measure_memory() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        import tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024)
