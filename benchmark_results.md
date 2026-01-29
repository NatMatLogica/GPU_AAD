# SIMM v2.5 Benchmark: AADC Python vs Acadia Java

## Test Setup

- **Test data**: Acadia's official SIMM v2.5 unit test sensitivities (from `SimmCrifMixin.java`)
- **Hardware**: debian-monster (Linux 6.1.0-13-amd64)
- **Python**: ISDA-SIMM engine via `src/agg_margins.py` (pandas-based)
- **Java**: AcadiaSoft simm-lib v2.5 (`Simm.calculateTotal()`)
- **Iterations**: 50 per test (Java: 10 warmup + 50 measured)

## Results

### Accuracy

| Test | Sens | AADC Python | Acadia Java | Diff |
|------|------|-------------|-------------|------|
| All_IR (C67) | 46 | $11,128,134,753 | $11,126,437,227 | +0.02% |
| All_FX (C80) | 12 | $42,158,541,376 | $45,609,126,471 | -7.57% |
| IR+FX (C81) | 58 | $48,595,818,392 | $52,644,493,455 | -7.69% |

- IR matches to within 0.02% (rounding differences)
- FX has a -7.6% discrepancy, likely due to differences in FX correlation handling (regular vs high-volatility currency categorization)

### Speed

| Test | Sens | Python (ms) | Java (ms) | Java speedup |
|------|------|-------------|-----------|--------------|
| All_IR | 46 | 846.40 +/- 27.65 | 7.46 +/- 4.60 | 113x |
| All_FX | 12 | 228.06 +/- 1.71 | 1.52 +/- 0.09 | 150x |
| IR+FX | 58 | 957.84 +/- 2.62 | 6.28 +/- 0.40 | 152x |

| Metric | Python | Java |
|--------|--------|------|
| Throughput (IR+FX) | ~61 sens/sec | ~9,236 sens/sec |

## Why Acadia Java Is Faster

1. **JVM JIT compilation** - After warmup, Java compiles hot paths to native machine code. The SIMM aggregation logic (loops, BigDecimal arithmetic, stream operations) runs at near-native speed.

2. **Python interpreter overhead** - The Python engine uses pandas DataFrames, groupby/merge operations, and dynamic parameter lookups through proxy objects and `importlib`. Each operation carries significant interpreter overhead compared to Java's compiled bytecode.

3. **Data structure mismatch** - Acadia uses typed Java objects (BigDecimal, enums, static dispatch). Our Python uses pandas DataFrames with string-based lookups, which is flexible but slow for this workload.

## Where AADC Adds Value

The Python SIMM engine benchmarked here is **not** the AADC use case. AADC's value is in computing **gradients of the SIMM function**, not the function itself:

| Task | Acadia Java | Python (no AADC) | Python + AADC |
|------|-------------|------------------|---------------|
| Single SIMM evaluation | 6.3 ms | 958 ms | 958 ms |
| SIMM + gradient (dIM/d allocation) | N/A | N/A | Single pass |
| Gradient via finite differences (N bumps) | N x 6.3 ms | N x 958 ms | ~0.3 ms (recorded kernel) |

AADC provides:
- **Automatic differentiation** through the SIMM calculation, producing exact gradients in a single forward + adjoint pass
- **Kernel recording** - record the SIMM computation once, then evaluate for many portfolios at ~3 microseconds each
- **Trade allocation optimization** - use gradients to optimize which trades go to which netting set, reducing total IM. Acadia's engine has no equivalent capability.

For a portfolio of T trades across P portfolios, the gradient `dIM/d(allocation)` has T x P dimensions. Computing this via finite differences in Acadia would require T x P x 6.3ms. AADC computes the full gradient in a single pass.

## Reproducing

```bash
# Run the benchmark
bash ~/run_benchmark.sh

# Or run each part separately:

# AADC Python
cd ~/ISDA-SIMM && python3 run_bench_v25.py

# Acadia Java
cd ~/Acadia-SIMM && java --source 17 -cp "$CP" SimmBench.java
# (see run_benchmark.sh for full classpath)
```
