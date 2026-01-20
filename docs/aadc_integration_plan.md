# AADC Integration Plan for IR Swap Pricer

## Overview

Convert `ir_swap_baseline_py` to `ir_swap_aadc_py` using AAD (Adjoint Algorithmic Differentiation) instead of bump-and-revalue.

## Key Difference: AAD vs Bump & Revalue

### Current (Bump & Revalue)
```python
# Baseline approach
base_prices = price_all_trades(trades, market_data)

for currency in currencies:
    for tenor_idx in range(12):
        bumped_market = market_data[currency].bump_curve(tenor_idx)
        bumped_prices = price_trades_in_currency(trades, bumped_market)
        delta[:, currency, tenor_idx] = (bumped_prices - base_prices) / bump_size
```
**Cost**: 1 + (5 currencies × 12 tenors) = 61 pricing passes

### AADC (AAD)
```python
# AAD approach - sensitivities computed automatically
with aadc.record_kernel():
    # Mark curve rates as differentiable inputs
    for currency in currencies:
        for tenor_idx in range(12):
            rate = aadc.idouble(market_data[currency].rates[tenor_idx])
            rate_handle = rate.mark_as_input()
            input_handles[currency, tenor_idx] = rate_handle

    # Forward pass: price all trades
    total_pv = price_all_trades_aadc(trades, market_data_aadc)

    # Mark output
    pv_handle = total_pv.mark_as_output()

# Single evaluation gives prices AND all derivatives
results = kernel.evaluate(input_values)
price = results[pv_handle]
deltas = {handle: results.derivative(handle) for handle in input_handles}
```
**Cost**: 1 forward + 1 adjoint ≈ 2-4x single pricing pass

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ir_swap_aadc.py                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐│
│  │  AADC Setup     │    │  Kernel Recording                   ││
│  │  - Import aadc  │───▶│  - Mark curve rates as inputs       ││
│  │  - Check avail  │    │  - Price trades inside kernel       ││
│  └─────────────────┘    │  - Mark PV as output                ││
│                         │  - Get derivatives automatically     ││
│                         └─────────────────────────────────────┘│
│                                    │                            │
│                                    ▼                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Kernel Evaluation                                          ││
│  │  - Set input values (curve rates)                           ││
│  │  - Evaluate kernel (forward + adjoint)                      ││
│  │  - Extract price from output                                ││
│  │  - Extract deltas from derivatives                          ││
│  └─────────────────────────────────────────────────────────────┘│
│                                    │                            │
│                                    ▼                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Output                                                      ││
│  │  - Same GreeksResult format as baseline                     ││
│  │  - Same CRIF generation                                     ││
│  │  - Performance comparison logged                            ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Step 1: Create AADC-enabled Market Data

```python
class AADCMarketData:
    """Market data with AADC-tracked rates."""

    def __init__(self, currency: str, base_market: MarketData):
        self.currency = currency
        self.base = base_market

        # AADC-tracked rates
        self.discount_rates = []
        self.rate_handles = []

    def setup_aadc_inputs(self):
        """Mark rates as differentiable inputs."""
        for i, rate in enumerate(self.base.discount_rates):
            aadc_rate = aadc.idouble(rate)
            handle = aadc_rate.mark_as_input()
            self.discount_rates.append(aadc_rate)
            self.rate_handles.append(handle)

        return self.rate_handles

    def discount_factor(self, t: float):
        """Calculate DF using AADC-tracked rates."""
        # Use np.interp with aadc arrays
        rate = np.interp(t, self.base.discount_tenors,
                        aadc.array(self.discount_rates))
        return aadc.exp(-rate * t)
```

### Step 2: AADC-enabled Pricing Function

```python
def price_swap_aadc(swap: IRSwap, market: AADCMarketData):
    """
    Price a swap using AADC-tracked market data.

    Same logic as price_swap(), but operations are recorded
    in the AADC computation graph.
    """
    payment_dates = swap.payment_dates()

    fixed_leg_pv = aadc.idouble(0.0)
    float_leg_pv = aadc.idouble(0.0)

    prev_t = 0.0
    for t in payment_dates:
        dt = t - prev_t
        df = market.discount_factor(t)

        # Fixed leg
        fixed_leg_pv = fixed_leg_pv + swap.fixed_rate * swap.notional * dt * df

        # Floating leg
        fwd_rate = market.forward_rate(prev_t, t)
        float_leg_pv = float_leg_pv + fwd_rate * swap.notional * dt * df

        prev_t = t

    pv = fixed_leg_pv - float_leg_pv

    if swap.is_payer:
        pv = -pv

    return pv
```

### Step 3: Record and Evaluate Kernel

```python
def price_with_greeks_aadc(
    trades: List[IRSwap],
    market_data: Dict[str, MarketData],
) -> GreeksResult:
    """
    Price trades and compute Greeks using AADC.

    Single forward + adjoint pass gives all sensitivities.
    """
    currencies = list(market_data.keys())
    num_tenors = len(TENOR_LABELS)

    # Recording phase
    recording_start = time.perf_counter()

    with aadc.record_kernel() as kernel:
        # Setup AADC market data with tracked inputs
        aadc_markets = {}
        all_handles = {}  # (currency, tenor_idx) -> handle

        for ccy in currencies:
            aadc_market = AADCMarketData(ccy, market_data[ccy])
            handles = aadc_market.setup_aadc_inputs()
            aadc_markets[ccy] = aadc_market

            for tenor_idx, handle in enumerate(handles):
                all_handles[(ccy, tenor_idx)] = handle

        # Price all trades (forward pass recorded)
        total_pv = aadc.idouble(0.0)
        trade_pvs = []

        for swap in trades:
            market = aadc_markets[swap.currency]
            pv = price_swap_aadc(swap, market)
            trade_pvs.append(pv)
            total_pv = total_pv + pv

        # Mark outputs
        pv_handles = [pv.mark_as_output() for pv in trade_pvs]
        total_handle = total_pv.mark_as_output()

    recording_time = time.perf_counter() - recording_start

    # Evaluation phase
    eval_start = time.perf_counter()

    # Set input values
    input_values = {}
    for ccy in currencies:
        for tenor_idx, rate in enumerate(market_data[ccy].discount_rates):
            handle = all_handles[(ccy, tenor_idx)]
            input_values[handle] = rate

    # Single evaluation (forward + adjoint)
    results = kernel.evaluate(input_values)

    eval_time = time.perf_counter() - eval_start

    # Extract results
    prices = np.array([results[h] for h in pv_handles])

    # Extract deltas from derivatives
    # d(PV_i) / d(rate_ccy_tenor) = sensitivity
    ir_delta = np.zeros((len(trades), len(currencies), num_tenors))

    for i, pv_handle in enumerate(pv_handles):
        for ccy_idx, ccy in enumerate(currencies):
            if trades[i].currency != ccy:
                continue  # Trade only sensitive to its own currency

            for tenor_idx in range(num_tenors):
                rate_handle = all_handles[(ccy, tenor_idx)]
                # Derivative of trade PV w.r.t. rate
                # Multiply by bump_size to get DV01
                ir_delta[i, ccy_idx, tenor_idx] = (
                    results.derivative(pv_handle, rate_handle) * BUMP_SIZE
                )

    return GreeksResult(
        prices=prices,
        ir_delta=ir_delta,
        currencies=currencies,
        tenor_labels=TENOR_LABELS,
        eval_time=eval_time,
        first_run_time=recording_time,  # Recording is "first run"
        steady_state_time=eval_time,
        num_evals=1,  # Single kernel evaluation!
        num_sensitivities=len(trades) * num_tenors,
        num_bumps=0,  # No bumps needed with AAD
    )
```

### Step 4: Optimizations

#### 4.1 Batch by Currency (if needed)
```python
# If kernel memory is an issue, process one currency at a time
for ccy in currencies:
    trades_in_ccy = [t for t in trades if t.currency == ccy]
    result = price_with_greeks_aadc_single_ccy(trades_in_ccy, market_data[ccy])
```

#### 4.2 Vectorized Trade Pricing
```python
# Price multiple trades in single kernel call
# Use aadc.array for vectorized operations
notionals = aadc.array([t.notional for t in trades])
fixed_rates = aadc.array([t.fixed_rate for t in trades])
# ... vectorized pricing
```

#### 4.3 Kernel Caching
```python
# Record kernel once, reuse for different market data
class CachedAADCPricer:
    def __init__(self, trades: List[IRSwap]):
        self.kernel = self._record_kernel(trades)

    def evaluate(self, market_data: Dict[str, MarketData]):
        # Just set inputs and evaluate, no re-recording
        return self.kernel.evaluate(market_data)
```

## File Structure

```
model/
├── ir_swap_pricer.py       # Existing baseline (bump & revalue)
├── ir_swap_aadc.py         # NEW: AADC implementation (AAD)
└── ir_swap_common.py       # Shared: IRSwap, MarketData, CRIF generation
```

## Expected Results

| Metric | Baseline | AADC |
|--------|----------|------|
| 100 trades, 60 risk factors | | |
| Evaluations | 1,300 | 1 |
| Time | ~1.8s | ~0.1-0.2s |
| Speedup | 1x | ~10-15x |
| | | |
| 1000 trades, 60 risk factors | | |
| Evaluations | 13,000 | 1 |
| Time | ~18s | ~0.5-1s |
| Speedup | 1x | ~20-30x |

## Validation

1. **Numerical accuracy**: AADC deltas must match baseline within 1e-8
2. **Sign convention**: Verify delta signs match for payer/receiver
3. **Currency isolation**: Confirm trades only show sensitivity to own currency
4. **Edge cases**: Zero-length swaps, at-the-money, deep ITM/OTM

## Implementation Checklist

- [ ] Create `ir_swap_common.py` with shared classes
- [ ] Create `ir_swap_aadc.py` with AAD implementation
- [ ] Add AADC availability check (graceful fallback)
- [ ] Implement `AADCMarketData` class
- [ ] Implement `price_swap_aadc()` function
- [ ] Implement `price_with_greeks_aadc()` function
- [ ] Add kernel caching for repeated evaluations
- [ ] Validation tests comparing to baseline
- [ ] Performance benchmarks
- [ ] Update logging for AADC metrics
