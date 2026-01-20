#!/usr/bin/env python
"""
Portfolio Benchmark for $40B AUM Client

Realistic portfolio configurations:
- Total notional: $40B - $100B (swaps have leverage)
- Trade count: 500 - 5,000
- Currencies: USD, EUR, GBP, JPY, CHF
- Maturities: 1Y - 30Y
"""

import sys
import time
import numpy as np
sys.path.insert(0, '.')

from model.ir_swap_common import (
    MarketData, IRSwap, TENOR_LABELS, NUM_TENORS,
    generate_market_data, generate_crif
)


def generate_realistic_portfolio(
    num_trades: int,
    total_notional: float,  # Total portfolio notional in USD
    currencies: list = None,
    seed: int = 42
) -> list:
    """
    Generate a realistic IR swap portfolio.

    Notional distribution: Lognormal (most trades small, few large)
    Maturity distribution: Weighted toward 2Y, 5Y, 10Y (liquid points)
    """
    if currencies is None:
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']

    np.random.seed(seed)
    trades = []

    # Currency weights (USD dominated for US manager)
    ccy_weights = {'USD': 0.50, 'EUR': 0.25, 'GBP': 0.10, 'JPY': 0.10, 'CHF': 0.05}
    ccy_list = list(ccy_weights.keys())
    ccy_probs = [ccy_weights.get(c, 0.1) for c in currencies]
    ccy_probs = np.array(ccy_probs) / sum(ccy_probs)

    # Maturity weights (liquid points)
    maturity_weights = {1: 0.05, 2: 0.15, 3: 0.10, 5: 0.25, 7: 0.10, 10: 0.20, 15: 0.05, 20: 0.05, 30: 0.05}
    maturities = list(maturity_weights.keys())
    mat_probs = list(maturity_weights.values())

    # Generate notionals (lognormal distribution)
    # Mean notional = total_notional / num_trades
    mean_notional = total_notional / num_trades
    notionals = np.random.lognormal(
        mean=np.log(mean_notional) - 0.5,  # Adjust for lognormal mean
        sigma=1.0,
        size=num_trades
    )
    # Scale to match total notional
    notionals = notionals * (total_notional / notionals.sum())

    for i in range(num_trades):
        currency = np.random.choice(currencies, p=ccy_probs)
        maturity = np.random.choice(maturities, p=mat_probs)

        # Fixed rate around par (slight spread)
        base_rate = 0.03 + np.random.normal(0, 0.01)
        fixed_rate = max(0.01, min(0.10, base_rate))

        # Frequency based on maturity
        if maturity <= 2:
            frequency = 0.25  # Quarterly for short
        elif maturity <= 5:
            frequency = 0.5   # Semi for medium
        else:
            frequency = 0.5   # Semi for long (could be annual)

        trades.append(IRSwap(
            trade_id=f"SWAP_{i:06d}",
            currency=currency,
            notional=notionals[i],
            fixed_rate=fixed_rate,
            maturity=float(maturity),
            pay_frequency=frequency,
            is_payer=np.random.choice([True, False]),
        ))

    return trades


def run_benchmark(num_trades: int, total_notional: float, currencies: list):
    """Run benchmark for a specific configuration."""
    from model.ir_swap_aadc import price_with_greeks as aadc_greeks, _kernel_cache, AADC_AVAILABLE
    from model.ir_swap_pricer import price_with_greeks as baseline_greeks

    print(f"\n{'='*70}")
    print(f"Portfolio: {num_trades:,} trades, ${total_notional/1e9:.0f}B notional, {len(currencies)} currencies")
    print(f"{'='*70}")

    # Generate market data
    market_data = {}
    for i, ccy in enumerate(currencies):
        market_data[ccy] = generate_market_data(ccy, base_rate=0.03 + i*0.005, seed=42+i)

    # Generate trades
    trades = generate_realistic_portfolio(num_trades, total_notional, currencies)

    # Portfolio statistics
    total_not = sum(t.notional for t in trades)
    avg_not = total_not / len(trades)
    max_not = max(t.notional for t in trades)
    min_not = min(t.notional for t in trades)

    print(f"\nPortfolio Statistics:")
    print(f"  Total Notional:  ${total_not/1e9:.2f}B")
    print(f"  Avg Notional:    ${avg_not/1e6:.1f}M")
    print(f"  Max Notional:    ${max_not/1e6:.1f}M")
    print(f"  Min Notional:    ${min_not/1e6:.1f}M")

    # Currency breakdown
    ccy_counts = {}
    ccy_notional = {}
    for t in trades:
        ccy_counts[t.currency] = ccy_counts.get(t.currency, 0) + 1
        ccy_notional[t.currency] = ccy_notional.get(t.currency, 0) + t.notional

    print(f"\n  Currency Breakdown:")
    for ccy in currencies:
        cnt = ccy_counts.get(ccy, 0)
        not_b = ccy_notional.get(ccy, 0) / 1e9
        print(f"    {ccy}: {cnt:>4} trades, ${not_b:.1f}B")

    # Run baseline
    print(f"\nRunning Baseline (bump & revalue)...")
    baseline_start = time.perf_counter()
    baseline_result = baseline_greeks(trades, market_data)
    baseline_time = time.perf_counter() - baseline_start

    print(f"  Time:       {baseline_time*1000:.1f} ms")
    print(f"  Bumps:      {baseline_result.num_bumps}")
    print(f"  Portfolio:  ${baseline_result.prices.sum()/1e6:.1f}M PV")

    # Run AADC
    if AADC_AVAILABLE:
        print(f"\nRunning AADC (AAD)...")

        # First run (cold cache)
        _kernel_cache.clear()
        aadc_start = time.perf_counter()
        aadc_result = aadc_greeks(trades, market_data)
        aadc_first_time = time.perf_counter() - aadc_start
        cache_stats = _kernel_cache.stats()

        print(f"  First Run:  {aadc_first_time*1000:.1f} ms (recording {cache_stats['cached_kernels']} kernels)")

        # Second run (warm cache)
        aadc_start = time.perf_counter()
        aadc_result2 = aadc_greeks(trades, market_data)
        aadc_steady_time = time.perf_counter() - aadc_start

        print(f"  Steady:     {aadc_steady_time*1000:.1f} ms (100% cache hit)")
        print(f"  Portfolio:  ${aadc_result.prices.sum()/1e6:.1f}M PV")

        # Validation
        price_diff = np.max(np.abs(aadc_result.prices - baseline_result.prices))
        delta_diff = np.max(np.abs(aadc_result.ir_delta - baseline_result.ir_delta))

        # Speedup
        speedup_first = baseline_time / aadc_first_time if aadc_first_time > 0 else 0
        speedup_steady = baseline_time / aadc_steady_time if aadc_steady_time > 0 else 0

        print(f"\nComparison:")
        print(f"  Price Diff:      ${price_diff:.2f}")
        print(f"  Speedup (first): {speedup_first:.2f}x")
        print(f"  Speedup (steady): {speedup_steady:.1f}x")

        return {
            'num_trades': num_trades,
            'total_notional': total_notional,
            'baseline_ms': baseline_time * 1000,
            'aadc_first_ms': aadc_first_time * 1000,
            'aadc_steady_ms': aadc_steady_time * 1000,
            'speedup_steady': speedup_steady,
            'kernels': cache_stats['cached_kernels'],
            'pv': baseline_result.prices.sum(),
        }
    else:
        return {
            'num_trades': num_trades,
            'baseline_ms': baseline_time * 1000,
        }


def main():
    print("="*70)
    print("   IR Swap Portfolio Benchmark - $40B AUM Client Scenarios")
    print("="*70)

    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']

    # Test configurations
    # For $40B AUM, typical swap notional is 2-5x AUM due to leverage
    configs = [
        # (num_trades, total_notional, description)
        (100, 10e9, "Small book"),
        (500, 40e9, "Medium book ($40B notional)"),
        (1000, 80e9, "Large book ($80B notional)"),
        (2000, 100e9, "Very large book ($100B notional)"),
    ]

    results = []

    for num_trades, total_notional, desc in configs:
        print(f"\n\n{'#'*70}")
        print(f"# Configuration: {desc}")
        print(f"{'#'*70}")

        result = run_benchmark(num_trades, total_notional, currencies)
        result['description'] = desc
        results.append(result)

    # Summary table
    print(f"\n\n{'='*70}")
    print("                    BENCHMARK SUMMARY")
    print("="*70)
    print(f"\n{'Config':<25} {'Trades':>8} {'Baseline':>12} {'AADC Steady':>12} {'Speedup':>10}")
    print("-"*70)

    for r in results:
        print(f"{r['description']:<25} {r['num_trades']:>8,} {r['baseline_ms']:>10.1f}ms {r.get('aadc_steady_ms', 0):>10.1f}ms {r.get('speedup_steady', 0):>9.1f}x")

    print("="*70)

    # Extrapolation
    print(f"\nPerformance Characteristics:")
    if len(results) > 1 and 'aadc_steady_ms' in results[-1]:
        # Calculate per-trade times
        baseline_per_trade = results[-1]['baseline_ms'] / results[-1]['num_trades']
        aadc_per_trade = results[-1]['aadc_steady_ms'] / results[-1]['num_trades']

        print(f"  Baseline per trade:    {baseline_per_trade:.3f} ms")
        print(f"  AADC per trade:        {aadc_per_trade:.4f} ms")
        print(f"\n  Projected for 5,000 trades:")
        print(f"    Baseline:  {5000 * baseline_per_trade / 1000:.1f} seconds")
        print(f"    AADC:      {5000 * aadc_per_trade:.0f} ms ({5000 * aadc_per_trade / 1000:.2f} seconds)")

    return results


if __name__ == "__main__":
    main()
