#!/usr/bin/env python3
"""Quick test script for IR swap pricer."""
import sys
import time
sys.path.insert(0, '/home/natashamanito/ISDA-SIMM')

from model.ir_swap_pricer import (
    generate_market_data, generate_trades,
    price_only, price_with_greeks, generate_crif
)

print("=" * 70)
print("IR Swap Pricer Test - Per-Currency Bumping")
print("=" * 70)

# Setup
currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']
market_data = {c: generate_market_data(c, seed=42+i) for i, c in enumerate(currencies)}

print(f"\nCurrencies: {currencies}")
print(f"Tenor buckets: 12 (2w, 1m, 3m, ... 30y)")
print(f"Total bump scenarios per currency: 12")
print(f"Total bump scenarios: {len(currencies)} × 12 = {len(currencies) * 12}")

for num_trades in [100, 1000]:
    print(f"\n{'='*70}")
    print(f"--- {num_trades} trades ---")
    trades = generate_trades(num_trades, currencies)

    # Count trades per currency
    trades_per_ccy = {}
    for t in trades:
        trades_per_ccy[t.currency] = trades_per_ccy.get(t.currency, 0) + 1
    print(f"Trades per currency: {trades_per_ccy}")

    # Price only
    t0 = time.perf_counter()
    r1 = price_only(trades, market_data)
    price_time = time.perf_counter() - t0
    print(f"\nprice_only:        {price_time*1000:>8.1f} ms")

    # Price with Greeks (bump-and-revalue)
    t0 = time.perf_counter()
    r2 = price_with_greeks(trades, market_data)
    greeks_time = time.perf_counter() - t0

    print(f"price_with_greeks: {greeks_time*1000:>8.1f} ms")
    print(f"  Sensitivities:     {r2.num_sensitivities:,}")
    print(f"  Bump scenarios:    {r2.num_bumps} (currencies × tenors)")
    print(f"  Total reprices:    {r2.num_evals:,}")
    print(f"  Overhead vs price: {greeks_time / price_time:.1f}x")

    # Expected AADC performance
    aadc_expected = price_time * 4  # ~4x overhead for AAD
    potential_speedup = greeks_time / aadc_expected
    print(f"\n  AADC expected:     ~{aadc_expected*1000:.0f} ms (4x price_only)")
    print(f"  Potential speedup: ~{potential_speedup:.1f}x")

    # CRIF generation
    crif = generate_crif(trades, r2)
    print(f"\n  CRIF rows: {len(crif):,}")

# Test SIMM integration
print(f"\n{'='*70}")
print("--- SIMM Integration (100 trades) ---")
trades = generate_trades(100, currencies)
result = price_with_greeks(trades, market_data)
crif = generate_crif(trades, result)

try:
    from src.agg_margins import SIMM
    t0 = time.perf_counter()
    portfolio = SIMM(crif, 'USD', 1)
    simm_time = time.perf_counter() - t0
    print(f"SIMM Total: ${portfolio.simm:,.2f}")
    print(f"SIMM Time:  {simm_time*1000:.1f} ms")
except Exception as e:
    print(f"SIMM Error: {e}")

print("\n" + "=" * 70)
print("Test complete!")
