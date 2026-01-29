# ISDA SIMM - Architecture & Technical Documentation

## Overview

This project implements ISDA Standard Initial Margin Model (SIMM) v2.3-2.6 in Python, extended with MatLogica's AADC (Automatic Adjoint Differentiation Compiler) for gradient-based margin optimization. It computes initial margin requirements for non-cleared OTC derivatives and provides tools for trade allocation optimization, stress testing, and pre-trade analytics.

---

## Calculation Architecture

```
                          SIMM Total
                              |
              +---------------+---------------+
              v               v               v
          RatesFX          Credit         Equity/Commodity
         (Product)       (Product)         (Product)
              |               |               |
    +---------+---------+     |     +---------+---------+
    v                   v     v     v                   v
  Rates                FX  CreditQ  Equity          Commodity
(RiskClass)     (RiskClass)  ...  (RiskClass)      (RiskClass)
    |                   |           |                   |
+---+---+           +---+---+   +---+---+           +---+---+
v   v   v           v   v   v   v   v   v           v   v   v
D   V   C           D   V   C   D   V   C           D   V   C
```

**Legend:** D = Delta, V = Vega, C = Curvature

## SIMM Formula Structure

### 1. Total SIMM

```
SIMM = Sum_pc SIMM_pc
Where pc in {RatesFX, Credit, Equity, Commodity}
```

### 2. Product Class Level

```
SIMM_pc = sqrt(Sum_r Sum_s psi_rs * K_r * K_s)

Where:
- r, s are risk classes within product class pc
- psi_rs is cross-risk-class correlation
- K_r is the margin for risk class r
```

### 3. Delta Margin (per risk class)

```
K_Delta = sqrt(Sum_b Sum_c rho_bc * K_b * K_c)

Bucket-level:
K_b = sqrt(Sum_k (WS_k)^2 + Sum_k Sum_l!=k rho_kl * WS_k * WS_l)

Where:
- WS_k = RW_k * s_k * CR_k (weighted sensitivity with concentration)
- RW_k = risk weight
- rho_kl = intra-bucket correlation
```

### 4. Vega and Curvature

Vega follows the same aggregation structure with volatility-specific risk weights. Curvature captures second-order risk via stressed scenario analysis.

## Risk Classes and Buckets

### Interest Rates
- **Tenor Buckets:** 2w, 1m, 3m, 6m, 1y, 2y, 3y, 5y, 10y, 15y, 20y, 30y
- **Sub-Curves:** OIS, Libor1M/3M/6M/12M, Prime, Inflation
- **Currency Groups:** Regular (USD, EUR, GBP, ...), Low Vol (JPY), High Vol (others)

### Other Risk Classes
- **Credit Qualifying:** Buckets 1-12 (IG sectors) + Residual
- **Credit Non-Qualifying:** Buckets 1-2 (HY/EM) + Residual
- **Equity:** Buckets 1-11 (market cap/region) + Bucket 12 (vol indices)
- **Commodity:** Buckets 1-17 (energy, agriculture, metals, etc.)
- **FX:** Single bucket with concentration thresholds

## Supported Trade Types

| Trade Type | Greeks Generated |
|------------|------------------|
| IR Swap | IR Delta (12 tenors) |
| Equity Option | IR Delta, Equity Delta, Equity Vega (6 expiries) |
| FX Option | IR Delta (both ccys), FX Delta, FX Vega (6 expiries) |
| Inflation Swap | IR Delta, Inflation Delta |
| XCCY Swap | IR Delta (both ccys), FX Delta |

---

## AADC Integration

### How It Works

AADC records the SIMM computation as a differentiable kernel, producing exact gradients (dIM/dSensitivity) in a single adjoint pass. This enables use cases that are intractable with bump-and-revalue:

| Task | Traditional | AADC |
|------|------------|------|
| Greek computation | O(K) pricing passes | 1 forward + 1 adjoint |
| Allocation gradient | O(T*P) evaluations | Single pass |
| Accuracy | Approximate (FD truncation) | Exact (machine precision) |

### Kernel Design

```
AADC Kernel (recorded once, evaluated many times):

  INPUTS (differentiable):
    rate[0..11] - 12 SIMM tenor buckets

  INPUTS (non-differentiable):
    notional, fixed_rate

  COMPUTATION GRAPH:
    Interpolate rates -> Discount factors -> Leg PVs -> Net PV

  OUTPUT: PV
  DERIVATIVES (via adjoint): d(PV)/d(rate[i]) for i = 0..11
```

### Key Optimization: Single evaluate() for Multiple Portfolios

```python
# Stack all P portfolios' sensitivities, evaluate ONCE:
agg_S_all = np.column_stack([agg_sens(p) for p in range(P)])  # Shape: (K, P)
inputs = {sens_handles[k]: agg_S_all[k, :] for k in range(K)}
results = aadc.evaluate(funcs, request, inputs, workers)       # ONE call!
```

This eliminates P-1 Python->C++ dispatch calls, yielding 10-200x speedup.

### Performance Results

| Configuration | Baseline (Python) | AADC | Speedup |
|---------------|-------------------|------|---------|
| 50 trades, 2 ccy | 117 ms | 3.3 ms | 35x |
| 100 trades, 3 ccy | 177 ms | 6.8 ms | 26x |
| 500 trades, 3 ccy | 1,047 ms | 35 ms | 30x |
| 1000 trades, 3 ccy | ~2,100 ms | ~70 ms | 30x |

### Kernel Cache Efficiency

| Portfolio Size | Unique Kernels | Cache Reuse | Trades/Kernel |
|----------------|----------------|-------------|---------------|
| 100 trades | 37 | 63% | 2.7 |
| 500 trades | 53 | 89% | 9.4 |
| 1,000 trades | 53 | 95% | 18.9 |

### Accuracy

| Metric | Value |
|--------|-------|
| Price difference (AADC vs baseline) | < $0.01 |
| Delta relative error | 0.048% (FD truncation in baseline) |
| Par swap validation | All 10 tests passed (PV = $0.00) |

The small delta difference is FD truncation error in the baseline, not AADC inaccuracy. Error scales linearly with bump size, confirming FD origin.

---

## Margin Optimization

### Trade Allocation Problem

Given T trades across P portfolios, minimize total SIMM:

```
minimize Sum_p IM_p(x)
subject to: Sum_p x[t,p] = 1 for all t  (each trade in one portfolio)
            x[t,p] in {0,1}              (whole-trade assignment)
```

### Why Allocation Matters

Random allocation splits natural hedges across portfolios. Optimal allocation groups offsetting positions for maximum netting benefit. Typical reduction: 10-30% (random initial), 30-50% (large datasets with pairing).

### Gradient Computation via Chain Rule

```
d(total_IM)/d(x[t,p]) = Sum_k (dIM_p/dS_p[k]) * S[t,k]

Where:
- S_p[k] = Sum_t x[t,p] * S[t,k]  (aggregated sensitivity)
- dIM_p/dS_p[k] from AADC kernel
- S[t,k] = raw trade sensitivity
```

Small kernel with K inputs (~100 risk factors) vs T*P inputs (~25,000), giving 100-1000x faster recording.

---

## Analytics Capabilities

### Stress Margin Analysis

Shock SIMM sensitivities and recalculate margin under 7 predefined scenarios:

| Scenario | Description |
|----------|-------------|
| `parallel_up_100bp` | All rates +100bp |
| `parallel_down_100bp` | All rates -100bp |
| `steepener_50bp` | 2Y -25bp, 30Y +50bp |
| `flattener_50bp` | Inverse of steepener |
| `vol_up_25pct` | Volatility +25% |
| `credit_widen_50pct` | Credit spreads +50% |
| `crisis_scenario` | Rates +200bp, vol +50%, credit +100% |

### Margin Attribution

Identify which trades contribute most to portfolio margin. AADC enables O(N*K) computation vs naive O(N^2):

| Method | 100 trades | 1000 trades | 10000 trades |
|--------|-----------|-------------|--------------|
| Naive (O(N^2)) | 10 sec | 17 min | 28 hours |
| AADC (O(N*K)) | 0.1 sec | 1 sec | 10 sec |

### Pre-Trade Analytics

- **Marginal IM Calculator**: Margin impact of a new trade at each counterparty
- **Bilateral vs Cleared**: Compare ISDA SIMM vs CCP margin (LCH, CME, Eurex)
- **Trade Routing**: Recommend optimal counterparty for new trades

### What-If Analysis

- Unwind top margin contributors
- Add offsetting hedges
- Stress scenarios on portfolio

---

## Benchmark: AADC Python vs Acadia Java

Using Acadia's official SIMM v2.5 unit test sensitivities:

### Accuracy

| Test | AADC Python | Acadia Java | Diff |
|------|-------------|-------------|------|
| All_IR (46 sens) | $11,128,134,753 | $11,126,437,227 | +0.02% |
| All_FX (12 sens) | $42,158,541,376 | $45,609,126,471 | -7.57% |

FX discrepancy likely due to differences in high-volatility currency categorization.

### Where AADC Adds Value

AADC's value is in computing **gradients of the SIMM function**, not the function itself. For a portfolio of T trades across P portfolios, the gradient dIM/d(allocation) has T*P dimensions. Computing this via finite differences requires T*P*6.3ms in Acadia. AADC computes the full gradient in a single pass.

---

## Known Limitations

### Validation Gaps
- No independent price validation against QuantLib/Bloomberg
- ISDA SIMM unit tests not fully validated
- Limited regression test suite

### Model Simplifications
- Single curve (discount = forward) - production needs OIS/IBOR separation
- Linear interpolation - production needs cubic spline
- No holiday calendar or business day conventions

### Product Coverage Gaps
- Swaptions, caps/floors not implemented
- Cross-currency basis risk limited
- Inflation swap support basic

### Scale
- Tested up to 1,000 trades (production needs 10K+)
- Single-threaded Python baseline

---

## References

- [ISDA SIMM Methodology v2.6](https://www.isda.org/a/CeggE/ISDA-SIMM-v2.6.pdf)
- [ISDA SIMM Unit Tests](https://www.isda.org/2021/04/08/isda-simm-unit-tests/)
- [CRIF Specification](https://www.isda.org/a/I4jEE/Risk-Data-Standards-v1-36-Public.pdf)
