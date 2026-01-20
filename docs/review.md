# IR Swap Pricer with AADC Integration - Technical Review

## Overview

This document summarizes the implementation of an IR swap pricer with AADC (AAD) integration for generating ISDA-SIMM sensitivities. It is intended for technical review by quants familiar with both ISDA-SIMM and AADC.

---

## 1. Starting Point

- **Base repository**: `meenmo/ISDA_SIMM` - Python SIMM aggregation engine
- **Existing functionality**: SIMM margin calculation from CRIF input, weights/correlations for v2.6/v2.7
- **What we added**: IR swap pricing layer to generate sensitivities (CRIF format) as input to SIMM

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              IR Swap Pricer                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────┐         ┌──────────────────────────────────────┐  │
│  │   ir_swap_common.py  │         │         ir_swap_pricer.py            │  │
│  │                      │         │         (Baseline)                    │  │
│  │  • MarketData        │         │                                      │  │
│  │  • IRSwap            │────────▶│  Bump & Revalue:                     │  │
│  │  • GreeksResult      │         │  • Base price                        │  │
│  │  • CRIF generation   │         │  • For each currency:                │  │
│  │                      │         │    • For each tenor (12):            │  │
│  └──────────────────────┘         │      • Bump rate +1bp                │  │
│           │                       │      • Reprice portfolio             │  │
│           │                       │      • Delta = (PV_up - PV) / bump   │  │
│           │                       └──────────────────────────────────────┘  │
│           │                                                                  │
│           │                       ┌──────────────────────────────────────┐  │
│           │                       │         ir_swap_aadc.py              │  │
│           │                       │         (AADC/AAD)                   │  │
│           │                       │                                      │  │
│           └──────────────────────▶│  AAD with Kernel Caching:            │  │
│                                   │  • Record kernel per trade structure │  │
│                                   │  • Evaluate: forward + adjoint       │  │
│                                   │  • Extract d(PV)/d(rate) directly    │  │
│                                   └──────────────────────────────────────┘  │
│                                              │                               │
│                                              ▼                               │
│                                   ┌──────────────────────────────────────┐  │
│                                   │         CRIF Output                  │  │
│                                   │  • RiskType: Risk_IRCurve            │  │
│                                   │  • Qualifier: Currency               │  │
│                                   │  • Label1: Tenor (2w..30y)           │  │
│                                   │  • Amount: Delta sensitivity         │  │
│                                   └──────────────────────────────────────┘  │
│                                              │                               │
│                                              ▼                               │
│                                   ┌──────────────────────────────────────┐  │
│                                   │      SIMM Aggregation (existing)     │  │
│                                   └──────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. AADC Integration Details

### 3.1 Recording Level

**One kernel per unique trade structure**, where structure is defined by:

```python
cache_key = (maturity, pay_frequency, is_payer, currency)
```

This allows trades with identical payment schedules to share a kernel while accommodating different notionals and fixed rates at evaluation time.

### 3.2 Kernel Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      AADC Kernel                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUTS (differentiable):                                       │
│  ├── rate[0]  (2w tenor)   ──┐                                  │
│  ├── rate[1]  (1m tenor)     │                                  │
│  ├── rate[2]  (3m tenor)     │    12 SIMM tenor buckets         │
│  ├── ...                     │                                  │
│  └── rate[11] (30y tenor)  ──┘                                  │
│                                                                 │
│  INPUTS (non-differentiable):                                   │
│  ├── notional                                                   │
│  └── fixed_rate                                                 │
│                                                                 │
│  COMPUTATION GRAPH:                                             │
│  ├── Interpolate rates to payment dates                         │
│  ├── DF(t) = exp(-r(t) * t)                                     │
│  ├── Fixed leg = Σ fixed_rate × notional × dt × DF(t)           │
│  ├── Float leg = Σ fwd_rate × notional × dt × DF(t)             │
│  └── PV = Fixed - Float (flip sign if payer)                    │
│                                                                 │
│  OUTPUT:                                                        │
│  └── PV                                                         │
│                                                                 │
│  DERIVATIVES (via adjoint):                                     │
│  └── d(PV)/d(rate[i]) for i = 0..11                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Kernel Caching Implementation

```python
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
```

### 3.4 Kernel Recording

```python
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

    return funcs, args, rate_handles
```

### 3.5 Kernel Evaluation

```python
# Build request: PV and derivatives w.r.t. all rates
pv_output = args['pv_output']
request = {pv_output: list(rate_handles.values())}

# Set input values for this specific trade
inputs = {
    args['notional_arg']: swap.notional,
    args['fixed_rate_arg']: swap.fixed_rate,
}
for tenor_idx, handle in rate_handles.items():
    inputs[handle] = float(market.discount_rates[tenor_idx])

# Evaluate: forward pass + adjoint pass
results = aadc.evaluate(funcs, request, inputs, workers)

# Extract price
price = float(results[0][pv_output])

# Extract sensitivities (derivatives of PV w.r.t. each rate)
for tenor_idx, rate_handle in rate_handles.items():
    sensitivity = float(results[1][pv_output][rate_handle])
    ir_delta[trade_idx, ccy_idx, tenor_idx] = sensitivity
```

### 3.6 Discount Factor Calculation (AADC-compatible)

```python
def discount_factor(self, t: float):
    """Calculate discount factor using AADC-tracked rates."""
    # Linear interpolation to find rate at time t
    tenors = self.base.discount_tenors

    if t <= tenors[0]:
        rate = self._discount_rates[0]
    elif t >= tenors[-1]:
        rate = self._discount_rates[-1]
    else:
        # Find bracketing indices and interpolate
        i = 0
        while i < len(tenors) - 1 and tenors[i + 1] <= t:
            i += 1
        t1, t2 = tenors[i], tenors[i + 1]
        r1, r2 = self._discount_rates[i], self._discount_rates[i + 1]
        weight = (t - t1) / (t2 - t1)
        rate = r1 + (r2 - r1) * weight

    # DF = exp(-r * t) - use np.exp for AADC compatibility
    return np.exp(-rate * t)
```

**Note**: Using `np.exp()` instead of `aadc.exp()` - AADC overloads NumPy functions.

---

## 4. Performance Results

### 4.1 Benchmark Data

| Configuration | Baseline (B&R) | AADC First Run | AADC Steady State | Speedup (Steady) |
|---------------|----------------|----------------|-------------------|------------------|
| 50 trades, 1 currency | 180 ms | 1,485 ms | **3.5 ms** | **52x** |
| 30 trades, 2 currencies | 98 ms | 1,192 ms | **2.2 ms** | **45x** |
| 20 trades, 1 currency | ~75 ms | 716 ms | **~1.5 ms** | **~50x** |

### 4.2 Timing Breakdown

```
First Run (Cold Cache):
├── Kernel recording:     ~40 ms per unique structure
├── Kernel evaluation:    ~0.07 ms per trade
└── Total:                recording dominates

Steady State (Warm Cache):
├── Cache lookup:         ~0.001 ms per trade
├── Kernel evaluation:    ~0.07 ms per trade
└── Total:                ~0.07 ms per trade
```

### 4.3 Cache Statistics (50 trades, 1 currency)

```
Unique kernels:    34  (out of 50 trades)
Cache hit rate:    32% (first run), 100% (subsequent runs)
```

The 34 unique kernels arise from combinations of:
- Maturities: 1, 2, 3, 5, 7, 10, 15, 20, 30 years
- Frequencies: quarterly (0.25), semi-annual (0.5), annual (1.0)
- Direction: payer/receiver

---

## 5. Accuracy

### 5.1 Price Comparison

| Metric | Value |
|--------|-------|
| Max absolute difference | < $0.01 |
| Mean absolute difference | < $0.0001 |

### 5.2 Delta Comparison

| Metric | Value |
|--------|-------|
| Max relative error | 0.14% |
| Mean relative error | ~0.05% |

**Expected**: AAD gives exact analytical derivatives. Finite difference has truncation error O(bump_size) and numerical noise. The 0.14% difference is consistent with FD error at 1bp bump.

### 5.3 Sensitivity Convention

Both methods compute the same quantity:

```
Baseline:  delta = (PV_bumped - PV_base) / bump_size  →  d(PV)/d(rate)
AADC:      delta = adjoint derivative                 →  d(PV)/d(rate)
```

To convert to DV01 (dollar value of 1bp move): multiply by 0.0001.

---

## 6. Important Considerations

### 6.1 Kernel Diversity Impact

- **More unique structures** = more recording overhead on first run
- **Real portfolios** with standardized tenors (1Y, 2Y, 5Y, 10Y) would have fewer unique kernels
- **Optimization opportunity**: Normalize maturities to standard tenors to increase cache hits

### 6.2 Production Deployment Pattern

```
Application Startup:
├── Load portfolio
├── Warm kernel cache (first price_with_greeks call)
└── Cache remains warm for session

Intraday Risk Recalculation:
├── Market data update
├── price_with_greeks() - uses cached kernels
└── ~50x faster than baseline
```

### 6.3 Not Yet Vectorized

Current implementation: **one kernel call per trade**

Potential optimization: Batch trades with identical structure into single vectorized kernel call, similar to Asian Options benchmark approach.

### 6.4 Single Currency Sensitivity

Each trade only shows sensitivity to its own currency's curve. No cross-currency basis risk is modeled (simplification).

### 6.5 Curve Construction

Current implementation uses identical discount and forward curves. Production would need:
- Separate OIS discount curve
- IBOR forward curves (if still relevant)
- Proper curve bootstrapping

---

## 7. Files for Review

| File | Lines | Description |
|------|-------|-------------|
| `model/ir_swap_common.py` | 285 | Shared data structures, CRIF generation |
| `model/ir_swap_pricer.py` | 786 | Baseline bump-and-revalue implementation |
| `model/ir_swap_aadc.py` | 842 | AADC implementation with kernel caching |
| `common/logger.py` | 342 | Execution logging infrastructure |
| `docs/aadc_integration_plan.md` | 320 | Original design document |

### Key Code Locations

- **KernelCache class**: `ir_swap_aadc.py:48-130`
- **AADCMarketData class**: `ir_swap_aadc.py:140-223`
- **Kernel recording**: `ir_swap_aadc.py:80-112`
- **Kernel evaluation**: `ir_swap_aadc.py:398-427`
- **Baseline bump loop**: `ir_swap_pricer.py:280-350`

---

## 8. Summary

| Aspect | Baseline | AADC |
|--------|----------|------|
| Method | Bump & revalue | AAD (adjoint) |
| Pricing passes | 1 + (currencies × 12) | 1 per trade |
| Derivative computation | Finite difference | Automatic (exact) |
| First run overhead | None | ~40ms per unique kernel |
| Steady state speedup | 1x | **45-52x** |
| Accuracy | Reference | 0.14% relative error |

**Recommendation**: The implementation is suitable for production use where:
1. Portfolio structure is relatively stable (kernel cache benefit)
2. Multiple risk recalculations per day (amortize recording cost)
3. Delta accuracy within 0.2% is acceptable (or use as fast approximation)
