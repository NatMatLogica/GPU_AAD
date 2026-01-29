# Fair AADC vs CUDA Benchmark Results

## Initial Run (2026-01-27 11:51) — Pre-Fix

Configuration:
- Trades per type: 50
- Trade types: ir_swap, equity_option
- Portfolios: 3

### Validation: FAILED

| Backend | vs Reference | IM rel_err | Grad rel_err | Status |
|---------|-------------|------------|-------------|--------|
| numpy   | (reference) | —          | —           | OK     |
| aadc    | numpy       | 9.96e-01   | 2.61e+02    | FAIL   |

IM values:
- numpy: [4.67e+12, 9.95e+12, 2.82e+12]
- aadc:  [6.02e+10, 4.34e+10, 4.84e+10]

NumPy produced ~100x larger IM than AADC.

### Timing (10 runs, before fix)

| Backend | Mean (ms) | Std (ms) | Total IM | Speedup |
|---------|-----------|----------|----------|---------|
| numpy   | 1.349     | 0.027    | $17.4T   | 1.00x   |
| aadc    | 1.861     | 0.047    | $152.0B  | 0.72x   |

## Root Cause Analysis

Two bugs in the initial benchmark implementation:

### Bug 1: Concentration factors not passed to AADC kernel

The NumPy formula computed: `WS_k = s_k * rw_k * cr_k` (with concentration risk `cr_k`).

The AADC kernel received only `factor_weights = rw_k` and computed: `WS_k = s_k * rw_k`.

Missing concentration factors caused the NumPy IM to be ~100x larger
(CR values can be 50-100+ for concentrated positions).

**Fix**: Pre-multiply `effective_weights = risk_weights * concentration_factors`
before passing to `record_single_portfolio_simm_kernel()`.

### Bug 2: Wrong `_get_intra_correlation` function used in data generation

`data_gen.py` imported `_get_intra_correlation` from `model/simm_portfolio_aadc.py`
which has signature `(risk_class, rt1, rt2, label1, label2, bucket, calc_currency)`.

The code passed `qualifiers[i]` (e.g., "AAPL", "EURUSD") as `calc_currency`,
causing incorrect FX/equity correlation lookups.

The AADC kernel uses `_get_intra_correlation` from `model/simm_allocation_optimizer.py`
which has signature `(risk_class, rt1, rt2, label1, label2, bucket)` — no `calc_currency`.

**Fix**: Import and call the optimizer's version (no qualifier argument) to match
what the AADC kernel actually uses.

## Post-Fix Results

*To be filled after re-running with fixes applied.*
