# ISDA SIMM Calculation Logic

## Overview

The ISDA Standard Initial Margin Model (SIMM) is a risk-based methodology for calculating initial margin requirements for non-cleared derivatives. This implementation follows SIMM v2.6 specifications.

## Calculation Architecture

```
                          SIMM Total
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          RatesFX          Credit         Equity/Commodity
         (Product)       (Product)         (Product)
              │               │               │
    ┌─────────┴─────────┐     │     ┌─────────┴─────────┐
    ▼                   ▼     ▼     ▼                   ▼
  Rates                FX  CreditQ  Equity          Commodity
(RiskClass)     (RiskClass)  ...  (RiskClass)      (RiskClass)
    │                   │           │                   │
┌───┴───┐           ┌───┴───┐   ┌───┴───┐           ┌───┴───┐
▼   ▼   ▼           ▼   ▼   ▼   ▼   ▼   ▼           ▼   ▼   ▼
Δ   ν   γ           Δ   ν   γ   Δ   ν   γ           Δ   ν   γ
```

**Legend:** Δ = Delta, ν = Vega, γ = Curvature

## Supported Risk Measures (Greeks)

### Delta Sensitivities

| Risk Type | Description | Trade Types |
|-----------|-------------|-------------|
| `Risk_IRCurve` | Interest rate delta by tenor bucket | IR Swaps, FX Options, Inflation Swaps, XCCY Swaps |
| `Risk_FX` | FX spot delta | FX Options, XCCY Swaps |
| `Risk_Equity` | Equity spot delta | Equity Options |
| `Risk_Inflation` | Inflation rate delta | Inflation Swaps |
| `Risk_XCcyBasis` | Cross-currency basis delta | XCCY Swaps |

### Vega Sensitivities

| Risk Type | Description | Expiry Buckets |
|-----------|-------------|----------------|
| `Risk_EquityVol` | Equity implied volatility sensitivity | 0.5y, 1y, 3y, 5y, 10y, 30y |
| `Risk_FXVol` | FX implied volatility sensitivity | 0.5y, 1y, 3y, 5y, 10y, 30y |
| `Risk_IRVol` | Interest rate volatility sensitivity | By tenor × expiry grid |

### Curvature (Gamma)

Curvature margin captures second-order risk through stressed scenario analysis:
- Upward shift scenario (CVR_up)
- Downward shift scenario (CVR_down)
- K_Curvature = max(CVR_up, CVR_down, 0)

## Formula Structure

### 1. Total SIMM (Top Level)

```
SIMM = Σ_pc SIMM_pc

Where pc ∈ {RatesFX, Credit, Equity, Commodity}
```

### 2. Product Class Level

```
SIMM_pc = √(Σ_r Σ_s ψ_rs × K_r × K_s)

Where:
- r, s are risk classes within product class pc
- ψ_rs is cross-risk-class correlation
- K_r is the margin for risk class r
```

**Cross-Risk-Class Correlations (ψ):**

| Risk Class | Rates | FX | CreditQ | CreditNonQ | Equity | Commodity |
|------------|-------|-----|---------|------------|--------|-----------|
| Rates      | 1.00  | 0.27| 0.30    | 0.17       | 0.31   | 0.37      |
| FX         | 0.27  | 1.00| 0.35    | 0.14       | 0.29   | 0.38      |
| CreditQ    | 0.30  | 0.35| 1.00    | 0.47       | 0.50   | 0.39      |
| CreditNonQ | 0.17  | 0.14| 0.47    | 1.00       | 0.42   | 0.35      |
| Equity     | 0.31  | 0.29| 0.50    | 0.42       | 1.00   | 0.43      |
| Commodity  | 0.37  | 0.38| 0.39    | 0.35       | 0.43   | 1.00      |

### 3. Risk Class Level

```
K_r = K_Delta + K_Vega + K_Curvature + K_BaseCorr

Where each component aggregates sensitivities within buckets
```

### 4. Delta Margin Calculation

For each risk class:

```
K_Delta = √(Σ_b Σ_c ρ_bc × K_b × K_c)

Where:
- b, c are buckets (e.g., tenor buckets for IR)
- ρ_bc is inter-bucket correlation
- K_b is the weighted sensitivity for bucket b
```

**Bucket-Level Aggregation:**

```
K_b = √(Σ_k (WS_k)² + Σ_k Σ_l≠k ρ_kl × WS_k × WS_l)

Where:
- WS_k = RW_k × s_k (weighted sensitivity)
- RW_k is the risk weight for factor k
- s_k is the sensitivity to factor k
- ρ_kl is intra-bucket correlation
```

### 5. Vega Margin Calculation

Similar structure to Delta, with volatility-specific risk weights:

```
K_Vega = √(Σ_b Σ_c ρ_bc × K_b^vega × K_c^vega)
```

### 6. Curvature Margin Calculation

Captures second-order (gamma) risk:

```
K_Curvature = max(CVR_up, CVR_down, 0)

Where:
- CVR = Curvature Value at Risk
- Computed from stressed scenario shifts
```

## Risk Classes and Buckets

### Interest Rates (IR)

**Tenor Buckets:** 2w, 1m, 3m, 6m, 1y, 2y, 3y, 5y, 10y, 15y, 20y, 30y

**Sub-Curves:**
- OIS (Overnight Index Swap)
- Libor1M, Libor3M, Libor6M, Libor12M
- Prime
- Inflation

**Currency Groups:**

| Group | Currencies | Risk Weight Multiplier |
|-------|-----------|------------------------|
| Regular | USD, EUR, GBP, CHF, AUD, NZD, CAD, SEK, NOK, DKK, HKD, SGD | 1.0x |
| Low Vol | JPY | 0.5x |
| High Vol | All others | 1.5x |

### Credit Qualifying (CreditQ)

**Buckets 1-12:** Investment grade sectors
**Bucket Residual:** Unclassified

### Credit Non-Qualifying (CreditNonQ)

**Buckets 1-2:** High yield / Emerging markets
**Bucket Residual:** Unclassified

### Equity

**Buckets 1-11:** Market cap and region combinations
**Bucket 12:** Volatility indices (VIX, etc.)
**Bucket Residual:** Unclassified

### Commodity

**Buckets 1-17:** Commodity types (Energy, Agriculture, Metals, etc.)

### FX

**Single bucket** with concentration thresholds by currency pair category

## Implementation Details

### File Structure

```
ISDA-SIMM/
├── ISDA-SIMM.md              # This documentation
├── CLAUDE.md                 # Development guidelines
├── optimization_demo.html    # Interactive visualization
│
├── src/                      # Core SIMM calculation engine
│   ├── agg_margins.py        # Top-level SIMM aggregation
│   ├── margin_risk_class.py  # Risk class margin calculations
│   ├── agg_sensitivities.py  # Sensitivity aggregation
│   └── wnc.py                # Weights and correlations loader
│
├── model/                    # Trade models and optimization
│   ├── simm_baseline.py      # NumPy baseline implementation
│   ├── simm_aadc.py          # AADC-enabled implementation
│   ├── simm_portfolio_aadc.py    # Portfolio-level calculations
│   ├── simm_allocation_optimizer.py  # Trade allocation optimization
│   └── trade_types.py        # Trade definitions and pricing
│
├── Weights_and_Corr/         # SIMM calibration parameters
│   ├── IR_weights.csv
│   ├── IR_correlations.csv
│   ├── Equity_weights.csv
│   └── ...
│
└── visualization/            # Web-based visualization
    └── index.html
```

### Key Classes

**SIMM** (`src/agg_margins.py`):
- `simm_product()`: Aggregates risk classes within a product class
- `calc()`: Entry point for full SIMM calculation

**MarginByRiskClass** (`src/margin_risk_class.py`):
- `IRDeltaMargin()`: Interest rate delta margin
- `DeltaMargin()`: Generic delta margin (Equity, FX, Commodity, Credit)
- `IRVegaMargin()`: Interest rate vega margin
- `VegaMargin()`: Generic vega margin
- `IRCurvatureMargin()`: Interest rate curvature margin
- `CurvatureMargin()`: Generic curvature margin
- `BaseCorrMargin()`: Base correlation margin (Credit only)

### Supported Trade Types

| Trade Type | Class | Greeks Generated |
|------------|-------|------------------|
| IR Swap | `IRSwapTrade` | IR Delta (12 tenors) |
| Equity Option | `EquityOptionTrade` | IR Delta, Equity Delta, Equity Vega (6 expiries) |
| FX Option | `FXOptionTrade` | IR Delta (both currencies), FX Delta, FX Vega (6 expiries) |
| Inflation Swap | `InflationSwapTrade` | IR Delta, Inflation Delta |
| XCCY Swap | `XCCYSwapTrade` | IR Delta (both currencies), FX Delta |

### CRIF Format

Input sensitivities are provided in CRIF (Common Risk Interchange Format):

| Column | Description |
|--------|-------------|
| TradeID | Unique trade identifier |
| RiskType | Risk_IRCurve, Risk_FX, Risk_Equity, Risk_EquityVol, Risk_FXVol, etc. |
| Qualifier | Currency, issuer, underlying identifier |
| Bucket | Risk bucket identifier (1-12 for IR tenors, 1-11 for equity sectors) |
| Label1 | Sub-curve (e.g., OIS, Libor3M) or expiry |
| Label2 | Additional qualifier |
| Amount | Sensitivity value (delta01, vega01) |
| AmountCurrency | Currency of the sensitivity |
| AmountUSD | USD-equivalent sensitivity |

## Trade Allocation Optimization

### Problem Statement

Given T trades assigned to P portfolios, find the allocation that minimizes total SIMM:

```
minimize Σ_p IM_p(x)

subject to:
  Σ_p x[t,p] = 1  ∀t  (each trade in exactly one portfolio)
  x[t,p] ∈ {0,1}       (whole-trade assignment)
```

### Why Allocation Matters

Random allocation splits natural hedges across portfolios:
- Pay-fixed swap in Portfolio 1, receive-fixed swap in Portfolio 2
- No netting benefit → higher total IM

Optimal allocation groups offsetting positions:
- Both swaps in same portfolio → sensitivities net
- Lower portfolio IM → lower total IM

**Typical reduction:** 10-30% with random initial allocation

### Efficient Gradient Computation

**Chain Rule Approach:**

```
∂total_IM/∂x[t,p] = Σ_k (∂IM_p/∂S_p[k]) × S[t,k]

Where:
- S_p[k] = Σ_t x[t,p] × S[t,k]  (aggregated sensitivity)
- ∂IM_p/∂S_p[k]                 (from AADC kernel)
- S[t,k]                         (raw trade sensitivity)
```

**Complexity:**
- Small kernel with K inputs (~100 risk factors)
- vs. T×P inputs (~25,000 for 5000 trades × 5 portfolios)
- 100-1000x faster kernel recording

### Optimization Algorithm

1. **Precompute CRIF** for all trades (batched AADC evaluation)
2. **Build sensitivity matrix** S[T,K] from trade CRIFs
3. **Record small SIMM kernel** (K inputs → single portfolio IM)
4. **Gradient descent** with simplex projection
5. **Round to integer** allocation (whole-trade assignment)
6. **Verify final IM** by recomputing from scratch

### Netting Model

The SIMM formula captures netting through signed sensitivities:

```
Portfolio IM ≈ α × |Σ sensitivities| + β × √(Σ sensitivities²)
            = α × |netted| + β × diversified

Where α + β = 1 (calibrated blend)
```

When opposite sensitivities are grouped:
- |netted| → 0 (cancellation)
- Portfolio IM decreases significantly

### Large Dataset Optimization (100K+ Trades)

For very large portfolios, random initial allocation already achieves partial netting due to the law of large numbers. A smarter **pairing optimization** is required:

**Strategy: Match Opposite Sensitivities**

```
1. Sort all trades by sensitivity: most negative → most positive
2. Pair from both ends:
   - Most negative (-$5B) paired with most positive (+$5B)
   - Second most negative paired with second most positive
   - etc.
3. Assign paired trades to same portfolio for maximum cancellation
4. Fine-tune with greedy optimization on high-impact trades
```

**Why Pairing Works:**

```
Before (random allocation):
  Portfolio 1: trades with sens = [+2B, -1B, +3B, -2B] → sensSum = +2B

After (pairing optimization):
  Portfolio 1: trades with sens = [-5B, +5B, -4B, +4B] → sensSum ≈ 0
                                                        → |netted| ≈ 0
                                                        → Lower IM
```

**Expected Reduction:**
- Small datasets (< 1000 trades): 10-30% with greedy optimization
- Large datasets (100K+ trades): 30-50% with pairing optimization

## Usage Examples

### Basic SIMM Calculation

```bash
python -m model.simm_baseline --trades 100 --threads 4
```

### AADC-Enabled Calculation

```bash
python -m model.simm_aadc --trades 1000 --threads 8
```

### Portfolio Optimization

```bash
python -m model.simm_portfolio_aadc \
    --trades 100 \
    --portfolios 5 \
    --threads 8 \
    --optimize \
    --method gradient_descent
```

### Interactive Demo

Open `optimization_demo.html` in a browser to visualize:
- Before/after portfolio comparison
- IM reduction over iterations
- Trade movement details
- CRIF breakdown by risk type

## Performance Benchmarks

| Implementation | 1000 Trades | Speedup |
|----------------|-------------|---------|
| Baseline (NumPy) | ~45s | 1.0x |
| AADC Python | ~2s | ~20x |
| Allocation Optimizer | ~5s for 100 iterations | N/A |

## AADC Integration

This implementation uses **Automatic Adjoint Differentiation Compiler (AADC)** for:

1. **Sensitivity computation**: Single forward + adjoint pass instead of bump-and-revalue
2. **Gradient computation**: ∂IM/∂allocation computed efficiently via chain rule
3. **Kernel recording**: Small O(K) kernel instead of O(T×P) for allocation optimization

### Key Benefits

- **20x speedup** for Greek computation vs finite differences
- **100x speedup** for allocation gradient vs naive approach
- **Exact gradients** (machine precision) vs numerical approximations

## What-If Analytics

The `model/whatif_analytics.py` module provides fast margin attribution and scenario analysis.

### Margin Attribution (Which trades consume margin?)

```python
from model.whatif_analytics import compute_margin_attribution_aadc, print_attribution_report

# Compute attribution for all trades
attribution = compute_margin_attribution_aadc(
    portfolio_crif,
    trade_sensitivities,  # Dict: trade_id -> CRIF
    num_threads=4
)

print_attribution_report(attribution)
```

**Output:**
```
TOP MARGIN CONSUMERS (trades adding to margin)
Trade ID              Contribution   % of Total
T1              $          225,000        64.3%  ← Largest risk
T2              $          175,000        50.0%

TOP MARGIN REDUCERS (trades providing netting benefit)
T3              $          -50,000       -14.3%  ← Netting benefit!
```

**Performance comparison:**
| Method | 100 trades | 1000 trades | 10000 trades |
|--------|-----------|-------------|--------------|
| Naive (O(N²)) | 10 sec | 17 min | 28 hours |
| AADC (O(N×K)) | 0.1 sec | 1 sec | 10 sec |

### What-If Scenarios

```python
from model.whatif_analytics import (
    whatif_unwind_top_contributors,
    whatif_add_hedge,
    whatif_stress_scenario
)

# Scenario 1: Unwind top 5 margin consumers
result = whatif_unwind_top_contributors(portfolio_crif, trade_sensitivities, attribution, 5)
# → "Unwinding saves $X in margin"

# Scenario 2: Add an offsetting hedge
result = whatif_add_hedge(portfolio_crif, hedge_crif, "10Y USD receiver")
# → "Hedge reduces margin by $Y"

# Scenario 3: Stress test
result = whatif_stress_scenario(portfolio_crif, {"Risk_IRCurve": 1.5}, "IR +50%")
# → "Margin increases by Z% under stress"
```

## Pre-Trade Analytics

The `model/pretrade_analytics.py` module provides real-time decision support for trade routing.

### Marginal IM Calculator

Compute the margin impact of a new trade at each counterparty:

```python
from model.pretrade_analytics import analyze_trade_routing

# Existing portfolios at each counterparty
portfolios = {
    "Goldman": goldman_crif,
    "JPM": jpm_crif,
    "Citi": citi_crif,
}

# Get routing recommendation for new trade
result = analyze_trade_routing(new_trade_crif, portfolios)

print(f"Route to: {result.recommended_counterparty}")
print(f"Marginal IM: ${result.recommended_marginal_im:,.0f}")
print(f"Savings vs worst: ${result.margin_savings:,.0f}")
```

**Why this matters for production:**
- Traders execute 100-1000 trades/day
- Each trade needs a routing decision
- Using AADC gradient: O(K) per query (~1ms)
- Full recalculation: O(N) per query (~1s for large portfolios)

### Bilateral vs Cleared Comparison

Compare ISDA SIMM (bilateral) vs CCP margin (cleared):

```python
from model.pretrade_analytics import compare_bilateral_vs_cleared, ClearingVenue

result = compare_bilateral_vs_cleared(
    new_trade_crif,
    bilateral_portfolio=jpm_crif,
    bilateral_counterparty="JPM",
    clearing_venue=ClearingVenue.LCH
)

print(f"Recommendation: {result.recommendation}")
print(f"Bilateral IM: ${result.bilateral_marginal_im:,.0f}")
print(f"Cleared IM:   ${result.cleared_marginal_im:,.0f}")
print(f"Rationale: {result.rationale}")
```

**Typical findings:**
- CCP margin is 30-50% lower than SIMM for vanilla swaps
- But bilateral may win if existing portfolio provides netting
- Decision depends on: margin cost, credit risk, operational preference

### CCP Margin Model

Simplified SPAN-like CCP margin calculation:

| CCP | Holding Period | Netting Factor | Typical vs SIMM |
|-----|---------------|----------------|-----------------|
| LCH SwapClear | 5 days | 0.8 (aggressive) | 30-40% lower |
| CME | 5 days | 0.75 | 35-45% lower |
| Eurex | 5 days | 0.8 | 30-40% lower |

**Why CCP margin is lower:**
1. 5-day holding period vs SIMM's 10-day
2. Daily variation margin reduces gap risk
3. Mutualized default fund as backstop
4. More aggressive netting assumptions

## References

- [ISDA SIMM Methodology v2.6](https://www.isda.org/a/CeggE/ISDA-SIMM-v2.6.pdf)
- [ISDA SIMM Unit Tests](https://www.isda.org/2021/04/08/isda-simm-unit-tests/)
- [CRIF Specification](https://www.isda.org/a/I4jEE/Risk-Data-Standards-v1-36-Public.pdf)
