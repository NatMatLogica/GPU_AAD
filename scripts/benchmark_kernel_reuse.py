#!/usr/bin/env python
"""
Benchmark: Hard-coded maturity kernels vs generic maturity-as-input kernels.

Theory: Hard-coding maturity into kernels means:
- More kernels recorded (one per unique maturity)
- But each kernel is smaller/faster (loop unrolled, constants folded)

Test: Which approach has better total time (recording + evaluation)?
"""

import sys
import time
import numpy as np
sys.path.insert(0, '.')

import aadc
from model.trade_types import IRSwapTrade, generate_market_environment

# Constants
IR_TENORS = np.array([0.04, 0.08, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30])
NUM_IR_TENORS = 12
MAX_PERIODS = 120  # Max for 30Y annual


def interp_aadc(times, tenors, rates_aadc):
    """Interpolate rates at given times using AADC operations."""
    results = []
    for t in times:
        if t <= tenors[0]:
            results.append(rates_aadc[0])
        elif t >= tenors[-1]:
            results.append(rates_aadc[-1])
        else:
            for i in range(len(tenors) - 1):
                if tenors[i] <= t <= tenors[i + 1]:
                    w = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
                    results.append(rates_aadc[i] * (1 - w) + rates_aadc[i + 1] * w)
                    break
    return results


def discount_factors_aadc(times, rates_aadc):
    """Compute discount factors at given times."""
    interp_rates = interp_aadc(times, IR_TENORS, rates_aadc)
    dfs = []
    for i, t in enumerate(times):
        df = np.exp(-interp_rates[i] * t)
        dfs.append(df)
    return dfs


def forward_rates_aadc(start_times, end_times, rates_aadc):
    """Compute forward rates between start and end times."""
    df_starts = discount_factors_aadc(start_times, rates_aadc)
    df_ends = discount_factors_aadc(end_times, rates_aadc)
    fwd_rates = []
    for i in range(len(start_times)):
        dt = end_times[i] - start_times[i]
        if dt > 0:
            fwd = (df_starts[i] / df_ends[i] - 1.0) / dt
        else:
            fwd = df_starts[i] * 0.0  # Keep AADC tracking
        fwd_rates.append(fwd)
    return fwd_rates


# =============================================================================
# APPROACH 1: Hard-coded maturity (current implementation)
# =============================================================================

def price_irs_hardcoded(notional, fixed_rate, maturity, frequency, is_payer, rates_aadc):
    """Price IR swap with hard-coded maturity (loop unrolled at recording)."""
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


def record_irs_kernel_hardcoded(maturity, frequency, is_payer, currency, market):
    """Record kernel with hard-coded maturity."""
    curve = market.curves[currency]
    nodiff_inputs = {}

    with aadc.record_kernel() as funcs:
        rates_aadc = []
        rate_handles = []
        for i in range(NUM_IR_TENORS):
            r = aadc.idouble(float(curve.zero_rates[i]))
            handle = r.mark_as_input()
            rates_aadc.append(r)
            rate_handles.append(handle)

        notional = aadc.idouble(1e6)
        notional_h = notional.mark_as_input_no_diff()
        nodiff_inputs[notional_h] = 1e6
        fixed_rate = aadc.idouble(0.03)
        fixed_rate_h = fixed_rate.mark_as_input_no_diff()
        nodiff_inputs[fixed_rate_h] = 0.03

        pv = price_irs_hardcoded(
            notional, fixed_rate, maturity, frequency, is_payer, rates_aadc
        )
        pv_output = pv.mark_as_output()

    return funcs, rate_handles, pv_output, nodiff_inputs


# =============================================================================
# APPROACH 2: Maturity as nodiff input (generic kernel)
# =============================================================================

def price_irs_generic(notional, fixed_rate, maturity_val, frequency, is_payer,
                       rates_aadc, max_periods=MAX_PERIODS):
    """
    Price IR swap with maturity as runtime value.

    Uses fixed max_periods loop with conditional accumulation.
    """
    dt = 1.0 / frequency
    num_periods = int(maturity_val * frequency)

    # Precompute all possible payment/start times up to max
    all_payment_times = np.array([(i + 1) * dt for i in range(max_periods)])
    all_start_times = np.array([i * dt for i in range(max_periods)])

    # Compute all discount factors and forward rates
    all_dfs = discount_factors_aadc(all_payment_times, rates_aadc)
    all_fwd_rates = forward_rates_aadc(all_start_times, all_payment_times, rates_aadc)

    # Accumulate only up to actual num_periods
    fixed_leg = notional * fixed_rate * dt * all_dfs[0]
    floating_leg = notional * all_fwd_rates[0] * dt * all_dfs[0]

    for i in range(1, num_periods):
        fixed_leg = fixed_leg + notional * fixed_rate * dt * all_dfs[i]
        floating_leg = floating_leg + notional * all_fwd_rates[i] * dt * all_dfs[i]

    npv = floating_leg - fixed_leg
    if not is_payer:
        npv = fixed_leg - floating_leg
    return npv


def record_irs_kernel_generic(frequency, is_payer, currency, market, max_periods=30):
    """
    Record generic kernel (one per frequency/payer/currency).

    Note: We still need to limit max_periods to avoid huge kernels.
    """
    curve = market.curves[currency]
    nodiff_inputs = {}

    # Use a representative maturity for kernel structure
    # The kernel will work for any maturity up to max_periods/frequency
    representative_maturity = max_periods / frequency

    with aadc.record_kernel() as funcs:
        rates_aadc = []
        rate_handles = []
        for i in range(NUM_IR_TENORS):
            r = aadc.idouble(float(curve.zero_rates[i]))
            handle = r.mark_as_input()
            rates_aadc.append(r)
            rate_handles.append(handle)

        notional = aadc.idouble(1e6)
        notional_h = notional.mark_as_input_no_diff()
        nodiff_inputs[notional_h] = 1e6
        fixed_rate = aadc.idouble(0.03)
        fixed_rate_h = fixed_rate.mark_as_input_no_diff()
        nodiff_inputs[fixed_rate_h] = 0.03

        pv = price_irs_generic(
            notional, fixed_rate, representative_maturity, frequency,
            is_payer, rates_aadc, max_periods
        )
        pv_output = pv.mark_as_output()

    return funcs, rate_handles, pv_output, nodiff_inputs


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_hardcoded(trades, market, num_threads):
    """Benchmark hard-coded maturity approach."""
    workers = aadc.ThreadPool(num_threads)
    kernel_cache = {}

    total_record_time = 0.0
    total_eval_time = 0.0

    for trade in trades:
        # Cache key includes maturity
        key = (trade.currency, trade.maturity, trade.frequency, trade.payer)

        if key not in kernel_cache:
            record_start = time.perf_counter()
            result = record_irs_kernel_hardcoded(
                trade.maturity, trade.frequency, trade.payer, trade.currency, market
            )
            total_record_time += time.perf_counter() - record_start
            kernel_cache[key] = result

        funcs, rate_handles, pv_output, nodiff_inputs = kernel_cache[key]

        # Evaluate
        curve = market.curves[trade.currency]
        inputs = {rate_handles[i]: np.array([float(curve.zero_rates[i])])
                  for i in range(NUM_IR_TENORS)}
        inputs.update({h: np.array([v]) for h, v in nodiff_inputs.items()})

        eval_start = time.perf_counter()
        request = {pv_output: rate_handles}
        results = aadc.evaluate(funcs, request, inputs, workers)
        total_eval_time += time.perf_counter() - eval_start

    return total_record_time, total_eval_time, len(kernel_cache)


def benchmark_generic(trades, market, num_threads, max_periods=30):
    """Benchmark generic maturity-as-input approach."""
    workers = aadc.ThreadPool(num_threads)
    kernel_cache = {}

    total_record_time = 0.0
    total_eval_time = 0.0

    for trade in trades:
        # Cache key excludes maturity - just structural params
        key = (trade.currency, trade.frequency, trade.payer)

        if key not in kernel_cache:
            record_start = time.perf_counter()
            result = record_irs_kernel_generic(
                trade.frequency, trade.payer, trade.currency, market, max_periods
            )
            total_record_time += time.perf_counter() - record_start
            kernel_cache[key] = result

        funcs, rate_handles, pv_output, nodiff_inputs = kernel_cache[key]

        # Evaluate
        curve = market.curves[trade.currency]
        inputs = {rate_handles[i]: np.array([float(curve.zero_rates[i])])
                  for i in range(NUM_IR_TENORS)}
        inputs.update({h: np.array([v]) for h, v in nodiff_inputs.items()})

        eval_start = time.perf_counter()
        request = {pv_output: rate_handles}
        results = aadc.evaluate(funcs, request, inputs, workers)
        total_eval_time += time.perf_counter() - eval_start

    return total_record_time, total_eval_time, len(kernel_cache)


def main():
    print("=" * 70)
    print("KERNEL REUSE BENCHMARK: Hard-coded vs Generic Maturity")
    print("=" * 70)
    print()

    # Setup
    currencies = ['USD', 'EUR', 'GBP', 'JPY']
    market = generate_market_environment(currencies, seed=42)
    num_threads = 4

    # Test scenarios
    scenarios = [
        ("Few unique maturities (5)", 1000, [1, 2, 3, 5, 10]),
        ("Many unique maturities (20)", 1000, list(range(1, 21))),
        ("All unique maturities", 500, list(np.linspace(1, 30, 500))),
    ]

    for scenario_name, num_trades, maturities in scenarios:
        print(f"\nScenario: {scenario_name}")
        print(f"  Trades: {num_trades}, Unique maturities: {len(set(maturities))}")
        print("-" * 50)

        # Generate trades
        np.random.seed(42)
        trades = []
        for i in range(num_trades):
            mat = maturities[i % len(maturities)]
            trade = IRSwapTrade(
                trade_id=f"IRS_{i}",
                notional=1e6,
                currency=currencies[i % len(currencies)],
                maturity=mat,
                fixed_rate=0.03,
                frequency=2,  # Semi-annual
                payer=i % 2 == 0,
            )
            trades.append(trade)

        # Benchmark hard-coded approach
        rec1, eval1, kernels1 = benchmark_hardcoded(trades, market, num_threads)
        total1 = rec1 + eval1

        # Benchmark generic approach (max 30 periods = 15Y for semi-annual)
        rec2, eval2, kernels2 = benchmark_generic(trades, market, num_threads, max_periods=30)
        total2 = rec2 + eval2

        print(f"  {'Approach':<25} {'Record':<12} {'Eval':<12} {'Total':<12} {'Kernels':<10}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        print(f"  {'Hard-coded maturity':<25} {rec1*1000:>8.1f} ms  {eval1*1000:>8.1f} ms  {total1*1000:>8.1f} ms  {kernels1:>6}")
        print(f"  {'Generic (mat as input)':<25} {rec2*1000:>8.1f} ms  {eval2*1000:>8.1f} ms  {total2*1000:>8.1f} ms  {kernels2:>6}")

        speedup = total2 / total1 if total1 > 0 else 0
        winner = "Hard-coded" if total1 < total2 else "Generic"
        print(f"  Winner: {winner} ({speedup:.2f}x ratio)")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == '__main__':
    main()
