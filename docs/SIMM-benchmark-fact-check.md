# SIMM Benchmark Document — Fact Check & Change Plan

## Status Summary

| Section | Accuracy | Action Required |
|---------|----------|-----------------|
| Headline Results | ✅ Verified | Minor updates needed |
| Pre-Trade (1.1) | ⚠️ Needs clarification | 0.16ms valid for T=50; T=5000 kernel=0.029ms |
| What-If (1.2) | ⚠️ Partially correct | cpp_aadc shows 14,815/sec not 15,000 |
| EOD Optimization (1.3) | ✅ Verified | "0 iters" bug now fixed |
| Continuous Monitoring (1.4) | ✅ Correct | Data exists in Step 4 What-If |
| Implementation (2.x) | ⚠️ Needs additions | Missing model descriptions |
| Costs (3.x) | ⚠️ Incomplete | Missing memory data |

---

## Detailed Fact Check

### Headline Benchmark Results — ✅ VERIFIED

| Claim | Actual Data | Status |
|-------|-------------|--------|
| cpp_aadc: 34,144 evals/sec, 0.27s | 34,144 evals/sec, 0.27s | ✅ Match |
| aadc_full: 1,311 evals/sec, 6.95s | 1,311 evals/sec, 6.95s | ✅ Match |
| gpu_full: 727 evals/sec, 12.54s | 727 evals/sec, 12.54s | ✅ Match |
| bf_gpu: 33 evals/sec, 3.10s, 101 evals | 33 evals/sec, 3.10s, 101 evals | ✅ Match |

### Pre-Trade Margin Checks (1.1) — ⚠️ NEEDS CLARIFICATION

The document claims 0.16ms at "5,000 trades" but this number is from a different configuration:

**Actual measured data by trade count:**

| Trades | Portfolios | Pre-Trade Workflow | Pure Kernel (throughput) |
|--------|------------|-------------------|--------------------------|
| 50 | 3 | **0.16 ms** ← source | 0.018 ms (55K evals/sec) |
| 100 | 3 | 0.15 ms | 0.015 ms (68K evals/sec) |
| 500 | 5 | 2.08 ms | 0.058 ms (17K evals/sec) |
| 1000 | 5 | 1.74 ms | 0.053 ms (19K evals/sec) |
| **5000** | **15** | **2.44 ms** ← at T=5000 | 0.029 ms (34K evals/sec) |

**Key distinction:**
- **Pre-Trade Workflow**: Includes aggregation, routing logic, Python overhead (2.44ms at T=5000)
- **Pure Kernel Re-evaluation**: SIMM calculation only, no recording (0.029ms at T=5000)

**Document comparison (at T=5000):**

| Claim | Actual Data | Status |
|-------|-------------|--------|
| cpp_aadc: 0.16ms | 2.44ms workflow / 0.029ms kernel | ⚠️ See clarification below |
| aadc_full: 5.7ms, 5 evals | 5.69ms, 5 evals | ✅ Match |
| gpu_full: 10.7ms, 5 evals | 10.75ms, 5 evals | ✅ Match |
| bf_gpu: 136ms, 100 evals | 136.15ms, 100 evals | ✅ Match |

**Recommendation:** The document mixes trade counts. Either:
1. **Option A**: Use T=50-100 throughout → 0.16ms is correct
2. **Option B**: Use T=5000 throughout → show 2.44ms workflow or 0.029ms pure kernel
3. **Option C**: Show scaling table with multiple trade counts

**Counterparty routing at T=5000:**
- Using workflow time: 20 × 2.44ms = **48.8ms**
- Using pure kernel: 20 × 0.029ms = **0.58ms** (sub-millisecond!)

### What-If Scenarios (1.2) — ⚠️ MINOR UPDATE

| Claim | Actual Data | Status |
|-------|-------------|--------|
| cpp_aadc: 15,000 evals/sec | **14,815 evals/sec** | ⚠️ Close |
| gpu_full: 230 evals/sec | 230 evals/sec | ✅ Match |
| 65× ratio | 64× actual | ⚠️ Close |

### EOD Optimization (1.3) — ✅ VERIFIED (with fix)

| Claim | Actual Data | Status |
|-------|-------------|--------|
| cpp_aadc: 0 iterations | **Now shows 1-2 iterations** | ✅ Fixed |
| bf_gpu never converges | converged=False | ✅ Match |
| IM reduction ~7% | Verified | ✅ Match |

**Note:** The "cpp_aadc zero-iteration mystery" is **RESOLVED**. It was a use-after-move bug in allocation_optimizer.h that has been fixed. cpp_aadc now correctly reports 1-2 iterations.

### Continuous Monitoring (1.4) — ✅ DATA EXISTS

The document says "Not Yet Benchmarked" but **Step 4 What-If already measures this**:
- Applies market shocks to aggregated sensitivities
- Recomputes IM for all portfolios
- This IS the continuous monitoring workflow

**Update needed:** Reference existing What-If data, don't claim it's missing.

---

## Missing Sections

### 1. Model Descriptions (REQUIRED)

The document doesn't explain what each backend actually does:

| Backend | Description Needed |
|---------|-------------------|
| **cpp_aadc** | C++ AADC SDK, OpenMP threading, batched evaluation |
| **aadc_full** | Python wrapper around AADC, NumPy aggregation |
| **gpu_full** | Numba CUDA, analytical gradients, GPU memory |
| **bf_gpu** | Brute-force enumeration, forward-only, no gradients |
| **pure_gpu_ir** | IR-only CUDA kernel, K=12 tenors only |

### 2. Methodology Section (REQUIRED)

Missing:
- ISDA SIMM v2.6 formula explanation
- What constitutes one "evaluation"
- Trade generation methodology
- Allocation optimization algorithm (Adam + greedy)

### 3. Memory Requirements (REQUIRED)

Current benchmarks track memory but it's not in the document:

```
Expected additions:
- CPU memory per backend (tracemalloc)
- GPU memory per backend (nvidia-smi)
- Memory scaling with T, P, K
```

### 4. Hardware Specifications (INCOMPLETE)

Need:
- H100 GPU specs (80GB HBM3, 132 SMs)
- CPU specs (cores, frequency)
- System memory

---

## Planned Changes Summary

### Numbers to Update

| Location | Current | Correct | Change |
|----------|---------|---------|--------|
| Pre-trade cpp_aadc | 0.16ms | **Depends on metric** — see below | Clarify context |
| Counterparty routing | 3.2ms total | 0.58ms (kernel) or 48.8ms (workflow) | Clarify metric |
| What-if cpp_aadc | 15,000/sec | 14,815/sec | Minor |
| cpp_aadc iterations | 0 | 1-2 | Bug fixed |

**Pre-trade timing clarification needed:**
- 0.16ms is valid for T=50-100 trades workflow time
- At T=5000: workflow=2.44ms, pure kernel=0.029ms
- For apples-to-apples comparison, use pure kernel re-eval time (no recording overhead)

### Sections to Add

1. **Model Architecture** — Explain each backend's design
2. **SIMM Formula** — K_ir calculation, correlations, concentration
3. **Memory Requirements** — CPU/GPU memory by configuration
4. **Benchmark Methodology** — How tests were run
5. **Pure GPU IR Comparison** — Currently missing from draft

### Claims to Revise

1. "cpp_aadc zero-iteration mystery" → Explain it now shows 1-2 iterations (bug fixed)
2. "Continuous monitoring not benchmarked" → Reference What-If Step 4
3. "Sub-ms interactive" for pre-trade → **Actually correct!** Pure kernel at T=5000: 0.029ms × 20 = 0.58ms
4. Clarify workflow time vs pure kernel time throughout document

### Sections to Expand

1. **Scaling Analysis** — Add trade scaling (100→50K) data
2. **Portfolio Scaling** — Add P=3→100 data
3. **Risk Factor Scaling** — Add K=20→170 data

---

## When Benchmarks Complete

After `run_aadc_scaling.sh` finishes:

1. Update all throughput tables with fresh data
2. Add scaling curves (T vs evals/sec)
3. Add memory usage section
4. Verify all speedup ratios
5. Add Pure GPU IR comparison where applicable
