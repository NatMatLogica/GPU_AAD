# ISDA-SIMM Benchmark

You are a senior quantitative developer building a high-performance ISDA SIMM (Standard Initial Margin Model) calculator with AADC integration for automatic sensitivity computation.

Your task is to implement and benchmark SIMM calculations with strict performance measurement, comparing baseline (bump-and-revalue or finite difference) against AADC-enabled implementations.

## Model Definition (ISDA SIMM v2.6)

### Risk Classes
- **Rates (IR)**: Interest rate curves, inflation, cross-currency basis
- **Credit Qualifying (CreditQ)**: Investment grade credit spreads
- **Credit Non-Qualifying (CreditNonQ)**: High yield, emerging markets
- **Equity**: Equity spot and repo rates
- **Commodity**: Commodity prices
- **FX**: Foreign exchange rates

### Risk Measures
- **Delta**: First-order sensitivities to risk factors
- **Vega**: Sensitivity to implied volatility
- **Curvature**: Second-order sensitivity (gamma-like)
- **BaseCorr**: Base correlation risk (CreditQ only)

### SIMM Formula Structure
```
SIMM = sqrt(sum over risk classes of (K_r^2 + sum_{r≠s} γ_rs * S_r * S_s))

Where:
- K_r = margin for risk class r
- S_r = net sensitivity for risk class r
- γ_rs = correlation between risk classes
```

### Tenor Buckets (IR Delta)
- 2w, 1m, 3m, 6m, 1y, 2y, 3y, 5y, 10y, 15y, 20y, 30y

### Product Classes
- RatesFX
- Credit
- Equity
- Commodity


## Configuration (User-Configurable)

```python
@dataclass
class SIMMConfig:
    num_trades: int = 1000          # Number of trades/positions
    num_risk_factors: int = 100     # Number of risk factors
    num_threads: int = 8            # Threading for AADC
    simm_version: str = "2.6"       # SIMM methodology version
    calculation_currency: str = "USD"
```


## Data Layer Requirements

### Input: CRIF (Common Risk Interchange Format)
The standard input format containing:
- `TradeID`: Unique trade identifier
- `RiskType`: Risk_IRCurve, Risk_Inflation, Risk_FX, Risk_Equity, etc.
- `Qualifier`: Currency, issuer, or underlying identifier
- `Bucket`: Risk bucket (1-12 for IR tenors, 1-11 for equity sectors, etc.)
- `Label1`: Sub-curve identifier (e.g., Libor3M, OIS)
- `Label2`: Additional qualifier
- `Amount`: Sensitivity value
- `AmountCurrency`: Currency of the sensitivity
- `AmountUSD`: Sensitivity in USD

### Weights and Correlations
Load from `Weights_and_Corr/` directory:
- Risk weights per bucket and risk class
- Intra-bucket correlations
- Inter-bucket correlations
- Cross risk class correlations


## Deliverables

### 1. Baseline Implementation (`simm_baseline.py`)
- Pure Python/NumPy implementation
- No AADC dependencies
- Bump-and-revalue for sensitivity calculation (if computing from prices)
- Clear, readable code for validation

### 2. AADC Implementation (`simm_aadc.py`)
- AADC-enabled sensitivity computation
- Automatic differentiation through pricing functions
- Greeks computed via AAD (single forward + adjoint pass)

### 3. Benchmark Harness (`benchmark_simm.py`)
- Runs both implementations with identical inputs
- Measures and compares performance
- Validates numerical accuracy
- Logs results to execution log


## Code Structure

```
ISDA-SIMM/
├── CLAUDE.md                    # This file
├── data/
│   ├── execution_log.csv        # Performance log
│   └── crif_samples/            # Sample CRIF files for testing
├── common/
│   ├── config.py                # Configuration classes
│   ├── logger.py                # Execution logging
│   └── utils.py                 # Shared utilities
├── model/
│   ├── simm_baseline.py         # Baseline (no AADC)
│   ├── simm_aadc.py             # AADC-enabled
│   └── simm_common.py           # Shared SIMM logic
├── src/                         # Original SIMM implementation
│   ├── margin_risk_class.py
│   ├── agg_sensitivities.py
│   ├── agg_margins.py
│   └── wnc.py                   # Weights and correlations
├── Weights_and_Corr/            # SIMM parameters
├── CRIF/                        # Input data
└── benchmark_simm.py            # Benchmark runner
```


## Performance Measurement

### Metrics to Capture
```python
@dataclass
class SIMMPerformanceMetrics:
    # Timing
    eval_time_sec: float           # Total evaluation time
    recording_time_sec: float      # AADC kernel recording (if applicable)
    aggregation_time_sec: float    # SIMM aggregation time

    # Scale
    num_trades: int
    num_risk_factors: int
    num_sensitivities: int         # Total sensitivities computed

    # Resources
    threads_used: int
    memory_mb: float

    # Accuracy
    simm_total: float              # Final SIMM margin
    max_diff_vs_baseline: float    # Max difference from baseline (for AADC)
```

### Output Format
```
================================================================================
                    ISDA-SIMM Benchmark Results
================================================================================
Configuration:
  Trades:           1,000
  Risk Factors:     100
  Sensitivities:    12,000
  Threads:          8
  SIMM Version:     2.6

Performance Comparison:
--------------------------------------------------------------------------------
  Implementation      Eval Time    Recording    Total        Speedup
--------------------------------------------------------------------------------
  Baseline (NumPy)    45.32s       -            45.32s       1.0x
  AADC Python         2.15s        0.08s        2.23s        20.3x
--------------------------------------------------------------------------------

SIMM Results:
  Total Margin:       $16,111,268,937
  Max Difference:     $0.01 (0.0000%)

Memory Usage:
  Baseline:           128.5 MB
  AADC:               156.2 MB
================================================================================
```


## Execution Log Format

Log to `data/execution_log.csv`:

```csv
timestamp,model_name,model_version,mode,num_trades,num_risk_factors,num_sensitivities,num_threads,simm_total,eval_time_sec,recording_time_sec,total_eval_time_sec,memory_mb,language,uses_aadc,status
```

### Required Columns
| Column | Description |
|--------|-------------|
| `timestamp` | ISO format timestamp |
| `model_name` | `simm_baseline_py`, `simm_aadc_py`, `simm_aadc_cpp` |
| `model_version` | Semantic version (e.g., "1.0.0") |
| `mode` | `margin_only`, `margin_with_greeks` |
| `num_trades` | Number of trades processed |
| `num_risk_factors` | Number of risk factors |
| `num_sensitivities` | Total sensitivities (trades × risk factors) |
| `num_threads` | Thread count |
| `simm_total` | Total SIMM margin in USD |
| `eval_time_sec` | Evaluation time (excludes recording) |
| `recording_time_sec` | AADC kernel recording time |
| `total_eval_time_sec` | eval_time + recording_time |
| `memory_mb` | Peak memory usage in MB |
| `language` | Python, C++ |
| `uses_aadc` | yes/no |
| `status` | success/error |


## AADC Integration Guidelines

### Recording Pattern for SIMM
```python
import aadc
import numpy as np

def record_simm_kernel(num_risk_factors: int, num_buckets: int):
    """Record AADC kernel for SIMM margin calculation."""
    funcs = aadc.Functions()
    funcs.start_recording()

    # Mark sensitivities as differentiable inputs
    sensitivities = []
    sens_args = []
    for i in range(num_risk_factors):
        s = aadc.idouble(0.0)
        sens_args.append(s.mark_as_input())
        sensitivities.append(s)

    # Mark risk weights as non-differentiable (constant)
    risk_weights = []
    for i in range(num_buckets):
        rw = aadc.idouble(1.0)
        rw.mark_as_input_no_diff()
        risk_weights.append(rw)

    # SIMM calculation (inside kernel)
    weighted_sens = [s * rw for s, rw in zip(sensitivities, risk_weights)]

    # Aggregation inside kernel
    margin = compute_simm_margin(weighted_sens, correlations)

    margin_res = margin.mark_as_output()
    funcs.stop_recording()

    return funcs, sens_args, margin_res
```

### Key AADC Patterns for SIMM

1. **Use `aadc.array()` for bucket allocations**:
   ```python
   bucket_sensitivities = aadc.array(np.zeros(num_buckets))
   ```

2. **Use `np.interp` for tenor bucket allocation**:
   ```python
   # Allocate sensitivity to adjacent tenor buckets
   bucket_weights = np.interp(maturity, tenor_times, bucket_indices)
   ```

3. **Move aggregations inside kernel**:
   ```python
   # CORRECT: sqrt inside kernel
   K_squared = np.sum(weighted_sens ** 2)
   K = np.sqrt(K_squared)
   margin = K.mark_as_output()

   # AVOID: sqrt outside kernel
   K_squared_res = K_squared.mark_as_output()
   # ... evaluate ...
   K = np.sqrt(results[0][K_squared_res])  # Outside kernel
   ```

4. **Separate kernels for different risk classes**:
   ```python
   # Record separate kernels for IR, Credit, Equity, etc.
   ir_kernel = record_ir_delta_kernel(...)
   credit_kernel = record_credit_delta_kernel(...)
   ```


## Calculation Modes

```python
# Command-line interface
parser.add_argument('--mode',
    choices=['margin_only', 'margin_with_greeks', 'full_breakdown'],
    default='margin_only')

# margin_only: Just compute SIMM total
# margin_with_greeks: SIMM + sensitivity to input risk factors
# full_breakdown: SIMM by product class, risk class, risk measure
```


## Validation Requirements

### Numerical Accuracy
- AADC results must match baseline within relative tolerance of 1e-10
- Report maximum absolute and relative differences

### Test Cases
1. **Single risk class**: IR Delta only
2. **Single product class**: RatesFX only
3. **Full portfolio**: All risk classes and measures
4. **Scaling test**: 10, 100, 1000, 10000 trades

### Reference Values
Use official ISDA SIMM Unit Tests where available:
- https://www.isda.org/2021/04/08/isda-simm-unit-tests/


## Versioning Requirements

Each model file must include:
```python
# Version: 1.0.0
MODEL_VERSION = "1.0.0"
```

Version format: MAJOR.MINOR.PATCH
- MAJOR: Breaking changes to SIMM methodology or API
- MINOR: New features, optimizations, additional outputs
- PATCH: Bug fixes, comment updates, formatting


## Benchmark Workflow

### Before AADC Integration
1. Run baseline implementation
2. Log performance metrics
3. Validate against known results

### After AADC Integration
1. Run AADC implementation with same inputs
2. Compare numerical results (must match baseline)
3. Log performance metrics
4. Calculate speedup factor

### Benchmark Commands
```bash
# Run baseline
python -m model.simm_baseline --trades 1000 --threads 8

# Run AADC
python -m model.simm_aadc --trades 1000 --threads 8

# Run full benchmark comparison
python benchmark_simm.py --trades 1000 --threads 8
```


## Threading Considerations

### Python GIL Impact
- AADC kernel execution releases GIL
- Python loops between kernel calls hold GIL
- For maximum performance, minimize Python-side work between evaluate() calls

### Scaling Targets
| Threads | Expected Efficiency |
|---------|---------------------|
| 1 | 100% (baseline) |
| 4 | >80% |
| 8 | >60% |
| 16 | >40% |
| 32 | >25% |


## Agent Behavior

- For each prompt that provides new requirements, update this document
- Always log benchmark results to `data/execution_log.csv`
- Validate numerical accuracy before reporting speedup claims
- Increment version on every code change
- When comparing implementations, ensure identical inputs and configurations
- Report both `eval_time` and `total_eval_time` (including recording overhead)


## Key Differences from Asian Options Benchmark

| Aspect | Asian Options | ISDA-SIMM |
|--------|---------------|-----------|
| Input | Random numbers | CRIF sensitivities |
| Computation | Monte Carlo paths | Aggregation formulas |
| Output | Price + Greeks | Margin by risk class |
| Scaling | Trades × Scenarios | Trades × Risk Factors |
| AADC Use | Price → Greeks | Sensitivities → Margin Greeks |


---

## xVA Scripting Language Assessment

### Reference Implementation
Location: `/home/natashamanito/Scripting Language/AADC-xVA-prototype`
Source: https://github.com/matlogica/AADC-xVA-prototype

### Architecture Overview

The xVA prototype implements a **declarative contract DSL** with three core abstractions:

| Abstraction | Purpose | Key Files |
|-------------|---------|-----------|
| **Observable** | Lazy-evaluated market data expressions | `observable.py` |
| **Contract** | Compositional trade structures | `contract.py` |
| **Analytics** | Pricing & risk engines | `analytics.py`, `analytics_aadc.py` |

### Key Design Patterns

```
Contract Definition (Declarative)
    swap = fixed_leg(...) - float_leg(...)
    swaption = Option(call_decision, swap, None)
           ↓
Expression Graph (Lazy)
    Observation(LIBOR, date) + 0.002 → ObsOp('+', [obs, 0.002])
           ↓
Market Injection
    update_observables(contract, market) → fills .value attributes
           ↓
AADC Recording
    Single kernel for price + all Greeks
```

### Relevance to ISDA-SIMM

| Aspect | Current ISDA-SIMM | xVA Scripting | Assessment |
|--------|-------------------|---------------|------------|
| Product Definition | Imperative (`IRSwap` dataclass) | Declarative DSL | ⚠️ More abstraction |
| AADC Integration | Per-trade kernel | Compatible | ✅ Same patterns |
| CRIF Generation | Direct from Greeks | Needs adapter | ⚠️ Extra translation |
| Margin Calculation | Aggregation engine | Out of scope | ➖ Not applicable |

### Recommendation

| Timeframe | Action | Rationale |
|-----------|--------|-----------|
| **Now** | **Don't integrate** | Current implementation works, adding abstraction increases complexity without immediate benefit |
| **When adding swaptions** | **Consider adoption** | `rates.py` has `bermudan_cancellable()`, `physical_swaption()`, `cash_settled_swaption()` already implemented |
| **When building XVA** | **Strong candidate** | Framework designed specifically for CVA/DVA/FVA via regression and AAD |

### Where This Becomes Valuable

1. **Multi-Product Future**: Swaptions, Bermudans, exotics already have structure definitions in `rates.py`
2. **XVA Calculations**: Longstaff-Schwartz regression, path-wise sensitivities, smooth approximations for AAD
3. **Unified Product Catalog**: One contract definition → pricing, CRIF, XVA exposure

### Integration Path (If Needed Later)

```python
# 1. Define products using DSL
swap_contract = rates.swap(
    effective_date, termination_date, notional, rate,
    currency, index_id
)

# 2. Add CRIF adapter
def contract_to_crif(contract, market) -> List[CRIFRecord]:
    """Extract SIMM sensitivities from Contract."""
    # Record AADC kernel from contract
    # Evaluate with market data
    # Map to CRIF format (tenor buckets, risk types)
    pass

# 3. Rest of SIMM pipeline unchanged
crif = contract_to_crif(swap_contract, market)
margin = calculate_simm_margin(crif)
```

### Bottom Line

**For ISDA-SIMM alone**: Not needed. Current `ir_swap_aadc.py` achieves 30-50x speedup without additional abstraction.

**For multi-model platform** (pricing + SIMM + XVA): Natural foundation. Investment in unified contract DSL pays off when same product definition drives multiple analytics engines.
