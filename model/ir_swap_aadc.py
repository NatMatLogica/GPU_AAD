"""
IR Swap Pricer - AADC Implementation (AAD)

Prices Interest Rate Swaps and computes sensitivities via Adjoint Algorithmic
Differentiation (AAD) using MatLogica AADC.

Key difference from baseline:
- Baseline: O(N) pricing passes where N = number of risk factors (bump & revalue)
- AADC: O(1) pricing passes with ~2-4x overhead (single forward + adjoint pass)

Version: 1.0.0

Usage:
    python -m model.ir_swap_aadc --trades 1000 --mode greeks
"""

import numpy as np
import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Import shared classes
from model.ir_swap_common import (
    MarketData, IRSwap, PricingResult, GreeksResult,
    TENOR_LABELS, TENOR_YEARS, NUM_TENORS, BUMP_SIZE,
    generate_market_data, generate_trades, generate_crif, measure_memory
)

# Version
MODEL_VERSION = "1.2.0"  # Added kernel caching for production-level performance
MODEL_NAME = "ir_swap_aadc_py"

# Try to import AADC
try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False
    print("Warning: AADC not available. Install MatLogica AADC for AAD support.")


# =============================================================================
# Kernel Cache - Key optimization for IR swaps
# =============================================================================

class KernelCache:
    """
    Cache recorded AADC kernels by trade structure.

    IR swaps with same (maturity, pay_frequency, is_payer, currency) can
    share the same kernel, avoiding expensive re-recording.
    """

    def __init__(self):
        self._cache = {}  # (maturity, freq, is_payer, ccy) -> (funcs, args, rate_handles)
        self._hits = 0
        self._misses = 0

    def get_key(self, swap: 'IRSwap', currency: str) -> tuple:
        """Generate cache key from swap structure."""
        return (swap.maturity, swap.pay_frequency, swap.is_payer, currency)

    def get(self, swap: 'IRSwap', market: 'MarketData'):
        """Get cached kernel or record new one."""
        key = self.get_key(swap, market.currency)

        if key in self._cache:
            self._hits += 1
            return self._cache[key]

        self._misses += 1

        # Record new kernel
        funcs, args, rate_handles = self._record_kernel(swap, market)
        self._cache[key] = (funcs, args, rate_handles)
        return funcs, args, rate_handles

    def _record_kernel(self, swap: 'IRSwap', market: 'MarketData'):
        """Record a new kernel for this swap structure."""
        funcs = aadc.Functions()
        funcs.start_recording()

        # Setup AADC market data with tracked inputs (differentiable)
        aadc_market = AADCMarketData(market)
        rate_handles = aadc_market.setup_aadc_inputs()

        # Trade parameters as non-differentiable inputs
        notional = aadc.idouble(swap.notional)
        fixed_rate = aadc.idouble(swap.fixed_rate)
        notional_arg = notional.mark_as_input_no_diff()
        fixed_rate_arg = fixed_rate.mark_as_input_no_diff()

        # Record the swap pricing computation
        pv = price_swap_aadc_with_params(
            notional, fixed_rate, swap.maturity,
            swap.pay_frequency, swap.is_payer,
            aadc_market
        )
        pv_output = pv.mark_as_output()

        funcs.stop_recording()

        args = {
            'notional_arg': notional_arg,
            'fixed_rate_arg': fixed_rate_arg,
            'pv_output': pv_output,
            'rate_handles': rate_handles,
        }

        return funcs, args, rate_handles

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'cached_kernels': len(self._cache),
        }

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# Global kernel cache
_kernel_cache = KernelCache()


# =============================================================================
# AADC-Enabled Market Data
# =============================================================================

class AADCMarketData:
    """
    Market data wrapper with AADC-tracked rates.

    Wraps a base MarketData and provides AADC-tracked versions of the
    discount and forward rates for automatic differentiation.
    """

    def __init__(self, base: MarketData):
        self.base = base
        self.currency = base.currency

        # AADC-tracked rates (populated during kernel recording)
        self._discount_rates = None
        self._forward_rates = None
        self._rate_handles = {}  # tenor_idx -> input handle

    def setup_aadc_inputs(self) -> Dict[int, any]:
        """
        Mark discount rates as differentiable inputs.

        Returns:
            Dict mapping tenor_idx to AADC input handle
        """
        self._discount_rates = []
        self._rate_handles = {}

        for i, rate in enumerate(self.base.discount_rates):
            aadc_rate = aadc.idouble(float(rate))
            handle = aadc_rate.mark_as_input()  # Differentiable for AAD
            self._discount_rates.append(aadc_rate)
            self._rate_handles[i] = handle

        # Forward rates same as discount for simplicity
        self._forward_rates = self._discount_rates

        return self._rate_handles

    def discount_factor(self, t: float):
        """
        Calculate discount factor at time t using AADC-tracked rates.

        Uses linear interpolation on the AADC-tracked rate array.
        """
        if t <= 0:
            return aadc.idouble(1.0)

        # Linear interpolation to find rate at time t
        tenors = self.base.discount_tenors

        # Find bracketing indices
        if t <= tenors[0]:
            rate = self._discount_rates[0]
        elif t >= tenors[-1]:
            rate = self._discount_rates[-1]
        else:
            # Find i such that tenors[i] <= t < tenors[i+1]
            i = 0
            while i < len(tenors) - 1 and tenors[i + 1] <= t:
                i += 1

            # Linear interpolation
            t1, t2 = tenors[i], tenors[i + 1]
            r1, r2 = self._discount_rates[i], self._discount_rates[i + 1]
            weight = (t - t1) / (t2 - t1)
            rate = r1 + (r2 - r1) * weight

        # DF = exp(-r * t) - use np.exp for AADC compatibility
        return np.exp(-rate * t)

    def forward_rate(self, t1: float, t2: float):
        """
        Calculate forward rate between t1 and t2 using AADC-tracked rates.

        Forward rate = (DF(t1) / DF(t2) - 1) / (t2 - t1)
        """
        if t2 <= t1:
            return aadc.idouble(0.0)

        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)

        return (df1 / df2 - 1.0) / (t2 - t1)


# =============================================================================
# AADC Pricing Functions
# =============================================================================

def price_swap_aadc(swap: IRSwap, market: AADCMarketData):
    """
    Price a single IR swap using AADC-tracked market data.

    Same logic as baseline price_swap(), but all operations are recorded
    in the AADC computation graph for automatic differentiation.

    Args:
        swap: IR swap to price
        market: AADC-enabled market data

    Returns:
        AADC-tracked PV (idouble)
    """
    payment_dates = swap.payment_dates()
    if len(payment_dates) == 0:
        return aadc.idouble(0.0)

    fixed_leg_pv = aadc.idouble(0.0)
    float_leg_pv = aadc.idouble(0.0)

    prev_t = 0.0
    for t in payment_dates:
        dt = t - prev_t
        df = market.discount_factor(t)

        # Fixed leg: fixed_rate * notional * dt * DF(t)
        fixed_leg_pv = fixed_leg_pv + swap.fixed_rate * swap.notional * dt * df

        # Floating leg: forward_rate(t-dt, t) * notional * dt * DF(t)
        fwd_rate = market.forward_rate(prev_t, t)
        float_leg_pv = float_leg_pv + fwd_rate * swap.notional * dt * df

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
    Price all trades without computing Greeks (no AADC needed).

    For price-only, we use the baseline pricing since AADC overhead
    is not justified without differentiation.
    """
    from model.ir_swap_common import MarketData

    num_trades = len(trades)
    prices = np.zeros(num_trades)

    start_time = time.perf_counter()

    for i, swap in enumerate(trades):
        market = market_data.get(swap.currency)
        if market is None:
            prices[i] = 0.0
        else:
            # Use base market data pricing
            prices[i] = _price_swap_base(swap, market)

    eval_time = time.perf_counter() - start_time

    return PricingResult(
        prices=prices,
        eval_time=eval_time,
        num_evals=num_trades,
    )


def _price_swap_base(swap: IRSwap, market: MarketData) -> float:
    """Price a swap using base (non-AADC) market data."""
    payment_dates = swap.payment_dates()
    if len(payment_dates) == 0:
        return 0.0

    fixed_leg_pv = 0.0
    float_leg_pv = 0.0

    prev_t = 0.0
    for t in payment_dates:
        dt = t - prev_t
        df = market.discount_factor(t)
        fixed_leg_pv += swap.fixed_rate * swap.notional * dt * df
        fwd_rate = market.forward_rate(prev_t, t)
        float_leg_pv += fwd_rate * swap.notional * dt * df
        prev_t = t

    pv = fixed_leg_pv - float_leg_pv
    if swap.is_payer:
        pv = -pv

    return pv


def price_with_greeks(
    trades: List[IRSwap],
    market_data: Dict[str, MarketData],
    use_cache: bool = True,
) -> GreeksResult:
    """
    Price all trades and compute IR Delta via AAD (AADC).

    This is the key function that demonstrates the AAD approach:
    - Single forward pass: price all trades
    - Single adjoint pass: compute all sensitivities

    Optimization: Kernel caching by trade structure.
    Trades with same (maturity, frequency, is_payer, currency) share a kernel.

    Args:
        trades: List of IR swaps to price
        market_data: Dict of currency -> MarketData
        use_cache: Whether to use kernel caching (default True)

    Returns:
        GreeksResult with prices, deltas, and timing
    """
    if not AADC_AVAILABLE:
        raise RuntimeError("AADC not available. Cannot compute Greeks via AAD.")

    global _kernel_cache
    if not use_cache:
        _kernel_cache.clear()

    num_trades = len(trades)
    currencies = list(market_data.keys())
    num_currencies = len(currencies)
    ccy_to_idx = {ccy: i for i, ccy in enumerate(currencies)}

    # Results arrays
    prices = np.zeros(num_trades)
    ir_delta = np.zeros((num_trades, num_currencies, NUM_TENORS))

    # Timing
    total_recording_time = 0.0
    total_eval_time = 0.0
    first_run_time = 0.0

    # Create AADC workers for parallel evaluation
    workers = aadc.ThreadPool(1)  # Single-threaded for now

    # Process each trade - with kernel caching
    for trade_idx, swap in enumerate(trades):
        ccy = swap.currency
        ccy_idx = ccy_to_idx.get(ccy)
        if ccy_idx is None:
            continue

        market = market_data[ccy]

        # === GET OR RECORD KERNEL ===
        recording_start = time.perf_counter()
        funcs, args, rate_handles = _kernel_cache.get(swap, market)
        recording_time = time.perf_counter() - recording_start
        total_recording_time += recording_time

        if trade_idx == 0:
            first_run_time = recording_time

        # === EVALUATION PHASE ===
        eval_start = time.perf_counter()

        # Build request: PV and derivatives w.r.t. all rates
        pv_output = args['pv_output']
        request = {pv_output: list(rate_handles.values())}

        # Set input values
        inputs = {
            args['notional_arg']: swap.notional,
            args['fixed_rate_arg']: swap.fixed_rate,
        }
        for tenor_idx, handle in rate_handles.items():
            inputs[handle] = float(market.discount_rates[tenor_idx])

        # Evaluate: forward pass + adjoint pass
        results = aadc.evaluate(funcs, request, inputs, workers)

        eval_time = time.perf_counter() - eval_start
        total_eval_time += eval_time

        # Extract price
        prices[trade_idx] = float(results[0][pv_output])

        # Extract sensitivities (derivatives of PV w.r.t. each rate)
        for tenor_idx, rate_handle in rate_handles.items():
            # d(PV) / d(rate) gives rate sensitivity
            # This matches baseline which computes: (pv_bumped - pv_base) / BUMP_SIZE
            sensitivity = float(results[1][pv_output][rate_handle])
            ir_delta[trade_idx, ccy_idx, tenor_idx] = sensitivity

    # Get cache stats
    cache_stats = _kernel_cache.stats()

    # Calculate sensitivities count
    num_sensitivities = num_trades * NUM_TENORS

    return GreeksResult(
        prices=prices,
        ir_delta=ir_delta,
        currencies=currencies,
        tenor_labels=TENOR_LABELS,
        eval_time=total_eval_time,
        first_run_time=first_run_time,
        steady_state_time=total_eval_time,
        recording_time=total_recording_time,
        num_evals=num_trades,  # One eval per trade
        num_sensitivities=num_sensitivities,
        num_bumps=0,  # No bumps needed with AAD!
    )


def price_swap_aadc_with_params(
    notional, fixed_rate, maturity: float, pay_frequency: float,
    is_payer: bool, market: AADCMarketData
):
    """
    Price a swap with AADC-tracked parameters.

    This version takes notional and fixed_rate as AADC idoubles
    for use in kernel recording.
    """
    # Generate payment dates
    dates = []
    t = pay_frequency
    while t <= maturity + 1e-9:
        dates.append(t)
        t += pay_frequency

    if len(dates) == 0:
        return aadc.idouble(0.0)

    fixed_leg_pv = aadc.idouble(0.0)
    float_leg_pv = aadc.idouble(0.0)

    prev_t = 0.0
    for t in dates:
        dt = t - prev_t
        df = market.discount_factor(t)

        # Fixed leg
        fixed_leg_pv = fixed_leg_pv + fixed_rate * notional * dt * df

        # Floating leg
        fwd_rate = market.forward_rate(prev_t, t)
        float_leg_pv = float_leg_pv + fwd_rate * notional * dt * df

        prev_t = t

    pv = fixed_leg_pv - float_leg_pv

    if is_payer:
        pv = aadc.idouble(0.0) - pv

    return pv


# =============================================================================
# Operation Counting
# =============================================================================

def count_operations(num_trades: int, num_currencies: int, num_tenors: int, num_evals: int):
    """
    Count mathematical operations for performance logging.

    For AADC, we count forward pass operations only.
    The adjoint pass has similar cost but is automatic.
    """
    from common.logger import OperationCounts

    avg_payments = 10  # Assume average 10 payments per swap

    # Per payment operations (same as baseline, but only one pass)
    generic_per_payment = 15  # mult, div, add, sub
    exp_per_payment = 2       # 2 exp for discount factors

    # With AAD: only count forward pass (adjoint is ~same cost)
    # num_evals = num_currencies (one kernel per currency)
    total_trade_evals = num_trades  # Each trade priced once

    return OperationCounts(
        total_generic_ops=total_trade_evals * avg_payments * generic_per_payment,
        total_exp_ops=total_trade_evals * avg_payments * exp_per_payment,
        total_sqrt_ops=0,
        total_log_ops=0,
        total_comparisons=total_trade_evals * 2,
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for IR swap AADC pricer benchmark."""
    import argparse
    import sys

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from common.logger import get_logger, SIMMExecutionRecord
    from common.utils import count_code_lines

    parser = argparse.ArgumentParser(description="IR Swap Pricer - AADC (AAD)")
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
    parser.add_argument("--validate", action="store_true",
                        help="Validate against baseline implementation")
    args = parser.parse_args()

    currencies = args.currencies.split(",")
    num_trades = args.trades
    mode = args.mode

    print("=" * 80)
    print("                    IR Swap Pricer - AADC (AAD)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model:          {MODEL_NAME} v{MODEL_VERSION}")
    print(f"  AADC Available: {AADC_AVAILABLE}")
    print(f"  Trades:         {num_trades:,}")
    print(f"  Currencies:     {currencies}")
    print(f"  Tenor Buckets:  {NUM_TENORS}")
    print(f"  Risk Factors:   {NUM_TENORS * len(currencies)} (currencies × tenors)")
    print(f"  Mode:           {mode}")
    print(f"  Threads:        {args.threads}")
    print()

    if not AADC_AVAILABLE and mode == "greeks":
        print("ERROR: AADC not available. Cannot compute Greeks.")
        print("Install MatLogica AADC or use baseline implementation.")
        return None

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

    # Initialize variables
    simm_total = 0.0
    crif_rows = 0
    recording_time = 0.0
    num_evals = 0
    num_sensitivities = 0
    avg_delta = 0.0
    first_run_total = 0.0
    steady_state_total = 0.0

    # Run pricing
    if mode == "price":
        print("Running price_only...")
        result = price_only(trades, market_data)

        num_evals = result.num_evals
        first_run_total = result.eval_time
        steady_state_total = result.eval_time

        print(f"\nResults:")
        print(f"  Portfolio PV:   ${result.prices.sum():,.2f}")
        print(f"  Avg Trade PV:   ${result.prices.mean():,.2f}")
        print(f"  Eval Time:      {result.eval_time*1000:.2f} ms")
        print(f"  Trades/sec:     {result.num_evals / result.eval_time:,.0f}")

    else:  # greeks
        print("Running price_with_greeks (AAD via AADC)...")
        print(f"  Method:         Adjoint Algorithmic Differentiation")
        print(f"  Kernel Caching: Enabled (by trade structure)")
        print()

        # Clear cache for fair benchmark
        _kernel_cache.clear()

        # First run - includes kernel recording
        first_run_start = time.perf_counter()
        result = price_with_greeks(trades, market_data)
        first_run_total = time.perf_counter() - first_run_start

        cache_stats = _kernel_cache.stats()

        # Second run - uses cached kernels (steady state)
        steady_state_start = time.perf_counter()
        result2 = price_with_greeks(trades, market_data)
        steady_state_total = time.perf_counter() - steady_state_start

        num_evals = result.num_evals
        num_sensitivities = result.num_sensitivities
        recording_time = result.recording_time

        # Calculate average delta
        avg_delta = np.mean(np.abs(result.ir_delta).sum(axis=(1, 2)))

        print(f"\nResults:")
        print(f"  Portfolio PV:   ${result.prices.sum():,.2f}")
        print(f"  Avg Trade PV:   ${result.prices.mean():,.2f}")
        print(f"  Min Trade PV:   ${result.prices.min():,.2f}")
        print(f"  Max Trade PV:   ${result.prices.max():,.2f}")
        print(f"  Avg |Delta|:    ${avg_delta:,.2f}")
        print()
        print(f"Timing:")
        print(f"  First Run (recording + eval): {first_run_total*1000:.2f} ms")
        print(f"  Steady State (cached):        {steady_state_total*1000:.2f} ms")
        print(f"  Speedup (cached vs first):    {first_run_total/steady_state_total:.1f}x")
        print()
        print(f"  Recording Time: {result.recording_time*1000:.2f} ms")
        print(f"  Eval Time:      {result.eval_time*1000:.2f} ms")
        print(f"  Sensitivities:  {result.num_sensitivities:,}")
        print(f"  Kernel Evals:   {result.num_evals}")
        print(f"  Bump Scenarios: {result.num_bumps} (not needed with AAD)")
        print()
        print(f"Kernel Cache:")
        print(f"  Unique Kernels: {cache_stats['cached_kernels']}")
        print(f"  Cache Hits:     {cache_stats['hits']}")
        print(f"  Cache Misses:   {cache_stats['misses']}")
        print(f"  Hit Rate:       {cache_stats['hit_rate']:.1%}")

        # Show sample deltas
        print(f"\nSample IR Deltas (DV01 per bp):")
        print("-" * 80)
        header = f"{'Trade':<12} " + " ".join(f"{t:>8}" for t in TENOR_LABELS[:6]) + " ..."
        print(header)
        print("-" * 80)
        for i in [0, 1, 2, -1]:
            row = f"{trades[i].trade_id:<12} "
            ccy_idx = currencies.index(trades[i].currency)
            row += " ".join(f"{result.ir_delta[i, ccy_idx, j]:>8.0f}" for j in range(6))
            row += " ..."
            print(row)

        # Validation against baseline
        if args.validate:
            print(f"\n" + "-" * 80)
            print(f"VALIDATION vs Baseline (bump & revalue)")
            print("-" * 80)
            from model.ir_swap_pricer import price_with_greeks as baseline_greeks
            baseline = baseline_greeks(trades, market_data)

            price_diff = np.max(np.abs(result.prices - baseline.prices))
            delta_diff = np.max(np.abs(result.ir_delta - baseline.ir_delta))

            # Relative error for non-zero deltas
            nonzero = np.abs(baseline.ir_delta) > 1e-10
            if nonzero.any():
                rel_diff = np.max(np.abs(result.ir_delta[nonzero] - baseline.ir_delta[nonzero]) / np.abs(baseline.ir_delta[nonzero]))
            else:
                rel_diff = 0.0

            print(f"\nAccuracy:")
            print(f"  Max Price Diff:    ${price_diff:.6f}")
            print(f"  Max Delta Diff:    ${delta_diff:.2f}")
            print(f"  Max Rel Delta Err: {rel_diff:.4%}")
            print()
            print(f"Performance:")
            print(f"  Baseline Time:     {baseline.eval_time*1000:.2f} ms")
            print(f"  AADC First Run:    {first_run_total*1000:.2f} ms")
            print(f"  AADC Steady State: {steady_state_total*1000:.2f} ms")
            print()
            print(f"  Speedup (first):   {baseline.eval_time/first_run_total:.2f}x (includes recording)")
            print(f"  Speedup (steady):  {baseline.eval_time/steady_state_total:.1f}x (production scenario)")

            if rel_diff < 0.01:
                print(f"\n  Status: PASSED (delta rel error < 1%)")
            else:
                print(f"\n  Status: REVIEW (delta rel error {rel_diff:.2%})")

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
        num_tenors=NUM_TENORS,
        num_evals=num_evals,
    )

    print(f"\nOperation Counts (forward pass only):")
    print(f"  Generic Ops:    {ops.total_generic_ops:,}")
    print(f"  Exp Ops:        {ops.total_exp_ops:,}")
    print(f"  Total Math Ops: {ops.total_math_ops:,}")

    # Log execution
    logger = get_logger()

    # Use steady state time for greeks mode, eval_time for price mode
    if mode == "greeks":
        log_eval_time = steady_state_total
        log_first_run = first_run_total
        log_steady_state = steady_state_total
    else:
        log_eval_time = result.eval_time
        log_first_run = result.eval_time
        log_steady_state = result.eval_time

    record = SIMMExecutionRecord(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        mode="price_only" if mode == "price" else "price_with_greeks",
        num_trades=num_trades,
        num_risk_factors=NUM_TENORS * len(currencies),
        num_sensitivities=num_sensitivities,
        num_threads=args.threads,
        portfolio_value=float(result.prices.sum()),
        avg_trade_value=float(result.prices.mean()),
        min_trade_value=float(result.prices.min()),
        max_trade_value=float(result.prices.max()),
        avg_delta=avg_delta,
        simm_total=simm_total,
        crif_rows=crif_rows,
        eval_time_sec=log_eval_time,
        first_run_time_sec=log_first_run,
        steady_state_time_sec=log_steady_state,
        recording_time_sec=recording_time,
        kernel_execution_time_sec=log_eval_time / max(1, num_evals),
        num_evals=num_evals,
        threads_used=args.threads,
        memory_mb=total_mem,
        data_memory_mb=data_mem,
        operation_counts=ops,
        model_total_lines=model_total_lines,
        model_math_lines=model_math_lines,
        language="Python",
        uses_aadc=True,
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
