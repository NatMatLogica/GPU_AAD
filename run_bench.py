#!/usr/bin/env python
import time
from benchmark_portfolio import generate_realistic_portfolio
from model.ir_swap_common import generate_market_data
from model.ir_swap_aadc import price_with_greeks, _kernel_cache
from model.ir_swap_pricer import price_with_greeks as baseline

ccys = ['USD','EUR','GBP']
mkt = {c: generate_market_data(c, seed=42+i) for i,c in enumerate(ccys)}

results = []
for n in [100, 200, 300, 500]:
    trades = generate_realistic_portfolio(n, n*100e6, ccys)

    # Baseline
    t0 = time.perf_counter()
    baseline(trades, mkt)
    base_ms = (time.perf_counter()-t0)*1000

    # AADC
    _kernel_cache.clear()
    t0 = time.perf_counter()
    price_with_greeks(trades, mkt)
    first_ms = (time.perf_counter()-t0)*1000

    t0 = time.perf_counter()
    price_with_greeks(trades, mkt)
    steady_ms = (time.perf_counter()-t0)*1000

    kernels = _kernel_cache.stats()['cached_kernels']
    speedup = base_ms / steady_ms if steady_ms > 0 else 0

    print(f"{n} trades: base={base_ms:.0f}ms, first={first_ms:.0f}ms, steady={steady_ms:.1f}ms, speedup={speedup:.0f}x, kernels={kernels}")
