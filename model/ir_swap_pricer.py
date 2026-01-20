"""
IR Swap Pricer - Baseline Implementation (No AADC)

Prices Interest Rate Swaps and computes sensitivities via bump-and-revalue.
Outputs CRIF format for SIMM aggregation.

Version: 1.1.0

Model Definition:
- Vanilla IR Swap: fixed leg vs floating leg
- Fixed Leg: sum of discounted fixed coupons
- Floating Leg: sum of discounted forward rates
- PV = Fixed Leg - Floating Leg (from receiver perspective)

Greeks (bump-and-revalue, bump = 1bp = 0.0001):
- IR Delta: sensitivity to each tenor bucket on the discount/forward curve
- IR Vega: sensitivity to swaption volatility (for future extension)

Tenor Buckets (ISDA SIMM):
- 2w, 1m, 3m, 6m, 1y, 2y, 3y, 5y, 10y, 15y, 20y, 30y

Usage:
    python -m model.ir_swap_pricer --trades 1000 --mode greeks
"""

import numpy as np
import pandas as pd
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Version
MODEL_VERSION = "1.1.0"
MODEL_NAME = "ir_swap_baseline_py"

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
    num_evals: int
    num_sensitivities: int
    num_bumps: int  # Total number of bump scenarios


@dataclass
class PerformanceMetrics:
    """Performance metrics for logging."""
    eval_time_sec: float
    first_run_time_sec: float
    steady_state_time_sec: float
    recording_time_sec: float = 0.0
    num_evals: int = 0
    memory_mb: float = 0.0
    data_memory_mb: float = 0.0
    kernel_memory_mb: float = 0.0


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
# Pricing Functions
# =============================================================================

def price_swap(swap: IRSwap, market: MarketData) -> float:
    """
    Price a single IR swap.

    Fixed Leg PV = sum(fixed_rate * notional * dt * DF(t))
    Float Leg PV = sum(forward_rate(t-dt, t) * notional * dt * DF(t))

    Returns PV from receiver's perspective (receives fixed).
    """
    payment_dates = swap.payment_dates()
    if len(payment_dates) == 0:
        return 0.0

    fixed_leg_pv = 0.0
    float_leg_pv = 0.0

    prev_t = 0.0
    for t in payment_dates:
        dt = t - prev_t
        df = market.discount_factor(t)

        # Fixed leg
        fixed_leg_pv += swap.fixed_rate * swap.notional * dt * df

        # Floating leg
        fwd_rate = market.forward_rate(prev_t, t)
        float_leg_pv += fwd_rate * swap.notional * dt * df

        prev_t = t

    # Receiver perspective: receives fixed, pays floating
    pv = fixed_leg_pv - float_leg_pv

    # Flip sign if payer swap
    if swap.is_payer:
        pv = -pv

    return pv


def price_only(
    trades: List[IRSwap],
    market_data: Dict[str, MarketData],
) -> PricingResult:
    """
    Price all trades without computing Greeks.

    Args:
        trades: List of IR swaps to price
        market_data: Dict of currency -> MarketData

    Returns:
        PricingResult with prices and timing
    """
    num_trades = len(trades)
    prices = np.zeros(num_trades)

    start_time = time.perf_counter()

    for i, swap in enumerate(trades):
        market = market_data.get(swap.currency)
        if market is None:
            prices[i] = 0.0
        else:
            prices[i] = price_swap(swap, market)

    eval_time = time.perf_counter() - start_time

    return PricingResult(
        prices=prices,
        eval_time=eval_time,
        num_evals=num_trades,
    )


def price_with_greeks(
    trades: List[IRSwap],
    market_data: Dict[str, MarketData],
) -> GreeksResult:
    """
    Price all trades and compute IR Delta via bump-and-revalue.

    For each (currency, tenor) pair, bump the curve by 1bp and reprice.
    Only trades in that currency are affected by the bump.
    Delta = (PV_bumped - PV_base) / bump_size

    This is the realistic SIMM sensitivity calculation where each
    currency's curve is bumped independently.

    Args:
        trades: List of IR swaps to price
        market_data: Dict of currency -> MarketData

    Returns:
        GreeksResult with prices, deltas, and timing
    """
    num_trades = len(trades)
    num_tenors = len(TENOR_LABELS)
    currencies = list(market_data.keys())
    num_currencies = len(currencies)

    # Create currency index lookup
    ccy_to_idx = {ccy: i for i, ccy in enumerate(currencies)}

    prices = np.zeros(num_trades)
    # Delta array: (num_trades, num_currencies, num_tenors)
    ir_delta = np.zeros((num_trades, num_currencies, num_tenors))

    # Track timing
    first_run_time = 0.0
    num_evals = 0
    num_bumps = num_currencies * num_tenors

    start_time = time.perf_counter()

    # First: compute base prices
    first_trade_start = time.perf_counter()
    for i, swap in enumerate(trades):
        market = market_data.get(swap.currency)
        if market is None:
            prices[i] = 0.0
        else:
            prices[i] = price_swap(swap, market)

        if i == 0:
            first_run_time = time.perf_counter() - first_trade_start

    num_evals += num_trades

    # Then: compute bumped prices for each (currency, tenor) pair
    # This is the realistic SIMM calculation - bump each curve separately
    for ccy_idx, ccy in enumerate(currencies):
        base_market = market_data[ccy]

        # Get indices of trades in this currency
        trades_in_ccy = [(i, swap) for i, swap in enumerate(trades) if swap.currency == ccy]

        if not trades_in_ccy:
            continue

        for tenor_idx in range(num_tenors):
            # Bump only this currency's curve
            bumped_market = base_market.bump_curve(tenor_idx, BUMP_SIZE)

            # Reprice only trades in this currency
            for i, swap in trades_in_ccy:
                bumped_price = price_swap(swap, bumped_market)
                ir_delta[i, ccy_idx, tenor_idx] = (bumped_price - prices[i]) / BUMP_SIZE
                num_evals += 1

    total_time = time.perf_counter() - start_time
    steady_state_time = (total_time - first_run_time) / max(1, num_evals - 1) * num_evals

    # Total sensitivities = trades × currencies × tenors (but only non-zero for matching currency)
    num_sensitivities = sum(1 for i, swap in enumerate(trades)
                           for ccy_idx in range(num_currencies)
                           for tenor_idx in range(num_tenors)
                           if swap.currency == currencies[ccy_idx])

    return GreeksResult(
        prices=prices,
        ir_delta=ir_delta,
        currencies=currencies,
        tenor_labels=TENOR_LABELS,
        eval_time=total_time,
        first_run_time=first_run_time,
        steady_state_time=steady_state_time,
        num_evals=num_evals,
        num_sensitivities=num_sensitivities,
        num_bumps=num_bumps,
    )


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
# Performance Measurement
# =============================================================================

def measure_memory() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback using tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024)


def count_operations(num_trades: int, num_currencies: int, num_tenors: int, num_evals: int):
    """
    Count mathematical operations for performance logging.

    Per swap pricing:
    - For each payment date (~maturity/frequency payments):
      - discount_factor: 1 interp + 1 mult + 1 exp
      - forward_rate: 2 df + 1 div + 1 sub + 1 div
      - fixed_leg: 3 mult + 1 add
      - float_leg: 3 mult + 1 add

    Returns:
        OperationCounts object
    """
    # Import here to avoid circular imports
    from common.logger import OperationCounts

    avg_payments = 10  # Assume average 10 payments per swap

    # Per payment operations
    # discount_factor: 1 interp (~3 ops) + 1 mult + 1 exp = 5 ops
    # forward_rate: 2 df calls + 1 div + 1 sub + 1 div = ~4 ops
    # fixed_leg: 3 mult + 1 add = 4 generic ops
    # float_leg: 3 mult + 1 add = 4 generic ops
    generic_per_payment = 15  # mult, div, add, sub
    exp_per_payment = 2       # 2 exp for discount factors

    return OperationCounts(
        total_generic_ops=num_evals * avg_payments * generic_per_payment,
        total_exp_ops=num_evals * avg_payments * exp_per_payment,
        total_sqrt_ops=0,
        total_log_ops=0,
        total_comparisons=num_evals * 2,  # is_payer check + edge cases
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for IR swap pricer benchmark."""
    import argparse
    import sys

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from common.logger import get_logger, SIMMExecutionRecord, OperationCounts
    from common.utils import count_code_lines

    parser = argparse.ArgumentParser(description="IR Swap Pricer Benchmark")
    parser.add_argument("--trades", type=int, default=1000, help="Number of trades")
    parser.add_argument("--mode", choices=["price", "greeks"], default="greeks",
                        help="Calculation mode")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--currencies", type=str, default="USD,EUR,GBP,JPY,CHF",
                        help="Comma-separated list of currencies")
    parser.add_argument("--output-crif", type=str, default=None,
                        help="Output CRIF file path")
    parser.add_argument("--run-simm", action="store_true",
                        help="Run SIMM calculation on generated CRIF")
    args = parser.parse_args()

    currencies = args.currencies.split(",")
    num_trades = args.trades
    mode = args.mode

    print("=" * 80)
    print("                    IR Swap Pricer - Baseline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model:          {MODEL_NAME} v{MODEL_VERSION}")
    print(f"  Trades:         {num_trades:,}")
    print(f"  Currencies:     {currencies}")
    print(f"  Tenor Buckets:  {len(TENOR_LABELS)}")
    print(f"  Risk Factors:   {len(TENOR_LABELS) * len(currencies)} (currencies × tenors)")
    print(f"  Mode:           {mode}")
    print(f"  Threads:        {args.threads}")
    print()

    # Count code lines
    model_file = Path(__file__)
    model_total_lines, model_math_lines = count_code_lines(str(model_file))
    print(f"Code Metrics:")
    print(f"  Total Lines:    {model_total_lines}")
    print(f"  Math Lines:     {model_math_lines}")
    print()

    # Start memory tracking
    tracemalloc.start()
    mem_before = measure_memory()

    # Generate market data
    print("Generating market data...")
    market_data = {}
    for i, ccy in enumerate(currencies):
        market_data[ccy] = generate_market_data(ccy, base_rate=0.03 + i * 0.005, seed=42 + i)

    # Generate trades
    print(f"Generating {num_trades:,} trades...")
    trades = generate_trades(num_trades, currencies, seed=42)

    data_mem = measure_memory() - mem_before

    # Show sample trades
    print(f"\nSample Trades:")
    print("-" * 80)
    print(f"{'ID':<12} {'CCY':<5} {'Notional':>15} {'Fixed':>8} {'Mat':>5} {'Type':<8}")
    print("-" * 80)
    for swap in trades[:3] + [trades[-1]]:
        swap_type = "Payer" if swap.is_payer else "Receiver"
        print(f"{swap.trade_id:<12} {swap.currency:<5} {swap.notional:>15,.0f} {swap.fixed_rate:>7.2%} {swap.maturity:>5.0f}y {swap_type:<8}")
    print()

    # Initialize variables for logging
    simm_total = 0.0
    crif_rows = 0
    first_run_time = 0.0
    steady_state_time = 0.0
    num_evals = 0
    num_sensitivities = 0
    ir_delta = None
    avg_delta = 0.0
    crif = None

    # Run pricing
    if mode == "price":
        print("Running price_only...")
        result = price_only(trades, market_data)

        num_evals = result.num_evals
        first_run_time = result.eval_time / num_trades  # Approximation

        print(f"\nResults:")
        print(f"  Portfolio PV:   ${result.prices.sum():,.2f}")
        print(f"  Avg Trade PV:   ${result.prices.mean():,.2f}")
        print(f"  Min Trade PV:   ${result.prices.min():,.2f}")
        print(f"  Max Trade PV:   ${result.prices.max():,.2f}")
        print(f"  Eval Time:      {result.eval_time*1000:.2f} ms")
        print(f"  Trades/sec:     {result.num_evals / result.eval_time:,.0f}")

    else:  # greeks
        print("Running price_with_greeks (bump-and-revalue)...")
        print(f"  Bump scenarios: {len(currencies) * len(TENOR_LABELS)} (currencies × tenors)")

        result = price_with_greeks(trades, market_data)

        num_evals = result.num_evals
        num_sensitivities = result.num_sensitivities
        first_run_time = result.first_run_time
        steady_state_time = result.steady_state_time
        ir_delta = result.ir_delta

        # Calculate average delta (sum of absolute deltas per trade)
        avg_delta = np.mean(np.abs(ir_delta).sum(axis=(1, 2)))

        print(f"\nResults:")
        print(f"  Portfolio PV:   ${result.prices.sum():,.2f}")
        print(f"  Avg Trade PV:   ${result.prices.mean():,.2f}")
        print(f"  Min Trade PV:   ${result.prices.min():,.2f}")
        print(f"  Max Trade PV:   ${result.prices.max():,.2f}")
        print(f"  Avg |Delta|:    ${avg_delta:,.2f}")
        print(f"  Eval Time:      {result.eval_time:.3f} s")
        print(f"  First Run:      {result.first_run_time*1000:.2f} ms")
        print(f"  Sensitivities:  {result.num_sensitivities:,}")
        print(f"  Num Evals:      {result.num_evals:,}")
        print(f"  Evals/sec:      {result.num_evals / result.eval_time:,.0f}")

        # Show sample deltas
        print(f"\nSample IR Deltas (DV01 per bp):")
        print("-" * 80)
        header = f"{'Trade':<12} " + " ".join(f"{t:>8}" for t in TENOR_LABELS[:6]) + " ..."
        print(header)
        print("-" * 80)
        for i in [0, 1, 2, -1]:
            row = f"{trades[i].trade_id:<12} "
            # Get currency index for this trade
            ccy_idx = currencies.index(trades[i].currency)
            row += " ".join(f"{result.ir_delta[i, ccy_idx, j]:>8.0f}" for j in range(6))
            row += " ..."
            print(row)

    # Memory measurement
    mem_after = measure_memory()
    total_mem = mem_after - mem_before + data_mem

    print(f"\nMemory:")
    print(f"  Data Memory:    {data_mem:.1f} MB")
    print(f"  Total Memory:   {total_mem:.1f} MB")

    # Generate CRIF if greeks mode
    if mode == "greeks":
        print(f"\nGenerating CRIF...")
        crif = generate_crif(trades, result)
        crif_rows = len(crif)
        print(f"  CRIF rows:      {crif_rows:,}")

        if args.output_crif:
            crif.to_csv(args.output_crif, index=False)
            print(f"  Saved to:       {args.output_crif}")

        # Run SIMM if requested
        if args.run_simm:
            print(f"\nRunning SIMM aggregation...")
            try:
                from src.agg_margins import SIMM
                simm_start = time.perf_counter()
                portfolio = SIMM(crif, "USD", 1)
                simm_time = time.perf_counter() - simm_start
                simm_total = portfolio.simm
                print(f"  SIMM Total:     ${simm_total:,.2f}")
                print(f"  SIMM Time:      {simm_time*1000:.2f} ms")
            except Exception as e:
                print(f"  SIMM Error:     {e}")

    # Count operations
    ops = count_operations(
        num_trades=num_trades,
        num_currencies=len(currencies),
        num_tenors=len(TENOR_LABELS),
        num_evals=num_evals,
    )

    # Print operation counts
    print(f"\nOperation Counts:")
    print(f"  Generic Ops:    {ops.total_generic_ops:,}")
    print(f"  Exp Ops:        {ops.total_exp_ops:,}")
    print(f"  Total Math Ops: {ops.total_math_ops:,}")

    # Calculate expected AADC performance (for reference)
    aadc_expected = result.eval_time / (num_evals / num_trades) * 4  # ~4x overhead for AAD
    if mode == "greeks":
        potential_speedup = result.eval_time / aadc_expected
        print(f"\nAADC Projection:")
        print(f"  Expected Time:  ~{aadc_expected*1000:.0f} ms (4x price_only)")
        print(f"  Potential Speedup: ~{potential_speedup:.1f}x")

    # Log execution
    logger = get_logger()

    record = SIMMExecutionRecord(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        mode="price_only" if mode == "price" else "price_with_greeks",
        num_trades=num_trades,
        num_risk_factors=len(TENOR_LABELS) * len(currencies),
        num_sensitivities=num_sensitivities,
        num_threads=args.threads,
        # Pricing results
        portfolio_value=float(result.prices.sum()),
        avg_trade_value=float(result.prices.mean()),
        min_trade_value=float(result.prices.min()),
        max_trade_value=float(result.prices.max()),
        # Greeks
        avg_delta=avg_delta,
        # SIMM specific
        simm_total=simm_total,
        crif_rows=crif_rows,
        # Timing
        eval_time_sec=result.eval_time,
        first_run_time_sec=first_run_time,
        steady_state_time_sec=steady_state_time,
        recording_time_sec=0.0,
        kernel_execution_time_sec=result.eval_time / num_evals if num_evals > 0 else 0.0,
        # Performance
        num_evals=num_evals,
        threads_used=args.threads,
        memory_mb=total_mem,
        data_memory_mb=data_mem,
        # Operation counts
        operation_counts=ops,
        # Code metrics
        model_total_lines=model_total_lines,
        model_math_lines=model_math_lines,
        # Metadata
        language="Python",
        uses_aadc=False,
        status="success",
    )
    logger.log(record)

    print()
    print("=" * 80)
    print(f"Execution logged to {logger.log_path}")
    print("=" * 80)

    tracemalloc.stop()
    return result


if __name__ == "__main__":
    main()
