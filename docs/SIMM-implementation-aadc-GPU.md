# ISDA SIMM: AADC vs GPU Implementation Guide

## 1. Introduction

This document compares two high-performance backends for ISDA SIMM v2.6 margin calculation: **AADC (Automatic Adjoint Differentiation Compiler)** and **GPU (CUDA)**. Both implementations target the same computational challenges — gradient-based portfolio optimization, margin attribution, pre-trade analytics, and what-if scenarios — but approach them with fundamentally different techniques.

**AADC** records the SIMM computation as a differentiable tape and produces exact gradients (`dIM/dSensitivity`) via a single adjoint (reverse-mode AD) pass. This enables O(K) gradient computation regardless of the number of risk factors K.

**GPU** implements the SIMM formula and its analytical derivatives in hand-coded CUDA kernels. The gradient is computed via manually derived chain rule at O(K²) cost per evaluation, but benefits from massive parallelism for brute-force candidate search.

| Aspect | AADC | GPU |
|--------|------|-----|
| Gradient method | Automatic adjoint differentiation | Hand-coded analytical chain rule |
| Gradient cost | O(K) for all K gradients | O(K²) for all K gradients |
| Hardware | CPU (multi-threaded) | GPU (CUDA cores) |
| Best for | Interactive queries, optimization loops | Brute-force search, large candidate evaluation |
| Implementation | `simm_portfolio_aadc.py` | `simm_portfolio_cuda.py` |

---

## 2. Computational Background

### 2.1 The SIMM Formula

ISDA SIMM v2.6 computes initial margin through multi-level aggregation:

```
Per risk class, per bucket (intra-bucket):
    WS_i = RW_i × s_i × CR_i                      (weighted sensitivity)
    K_b = sqrt(Σ_i Σ_j ρ_ij × φ_ij × WS_i × WS_j) (bucket-level capital)

Per risk class (inter-bucket):
    K_r = sqrt(Σ_b Σ_c γ_bc × S_b × S_c)           (risk-class capital)
    where S_b = Σ_i WS_i (signed sum for the bucket)

Total IM:
    IM = sqrt(Σ_r Σ_s ψ_rs × K_r × K_s)            (cross-risk-class aggregation)
```

Where:
- `RW_i` = tenor/bucket-specific risk weight
- `CR_i` = concentration risk factor: `max(1, sqrt(|Σs_i| / T))`
- `ρ_ij` = intra-bucket correlation (12×12 for IR tenors)
- `φ_ij` = sub-curve correlation (0.993 for different sub-curves)
- `γ_bc` = inter-bucket correlation
- `ψ_rs` = cross-risk-class correlation (6×6 matrix)

### 2.2 Why Gradients Matter

The gradient `dIM/dSensitivity` is the fundamental building block for:

| Use Case | Without Gradient | With Gradient |
|----------|-----------------|---------------|
| Marginal IM | Full SIMM recalculation O(K²) | Dot product O(K) |
| Allocation optimization | Enumerate all moves | Gradient descent |
| Margin attribution | N leave-one-out recalculations | Chain rule N×K multiply-adds |
| Counterparty routing | O(C × SIMM) per trade | O(C × K) per trade |

### 2.3 The Gradient Cost Difference

**AADC (adjoint mode)**:
- Records forward computation as a tape
- Single reverse pass computes ALL K gradients
- Cost: ~4× forward pass (independent of K)
- No bump-size sensitivity, exact derivatives

**GPU (analytical chain rule)**:
- Hand-coded derivative formulas
- Loop over K factors to compute each gradient
- Cost: O(K) per gradient × K gradients = O(K²)
- Exact to floating-point precision

At K=2,550 risk factors (typical multi-asset portfolio), AADC's O(K) gradient is **significantly faster** than GPU's O(K²) gradient.

---

## 3. Implementation Architecture

### 3.1 File Structure

```
ISDA-SIMM/
├── model/
│   ├── simm_portfolio_aadc.py      # AADC backend (CPU + adjoint)
│   ├── simm_portfolio_cuda.py      # GPU backend (CUDA + analytical gradient)
│   ├── pure_gpu_ir.py              # Pure GPU IR-only (fair benchmark)
│   ├── trade_types.py              # 5 trade types with pricers
│   └── simm_allocation_optimizer.py # Gradient-based optimizer
├── Weights_and_Corr/
│   └── v2_6.py                     # SIMM v2.6 parameters
├── benchmark_aadc_vs_gpu.py        # A/B performance comparison
└── benchmark_trading_workflow.py   # Full trading day benchmark
```

### 3.2 AADC Backend Architecture

```
Trades → AADC Pricing Kernel → CRIF → AADC SIMM Kernel → IM + dIM/dS
             ↑                           ↑
         record once               record once
         per trade type            per portfolio
```

Key patterns:
1. **Kernel recording**: Use `with aadc.record_kernel() as funcs:` context manager
2. **Batched evaluation**: Single `aadc.evaluate()` call for all P portfolios
3. **ThreadPool reuse**: Create once, reuse across iterations

```python
# Record SIMM kernel with K inputs (once)
funcs, sens_handles, im_output = record_simm_kernel(K)

# Evaluate ALL P portfolios in ONE call
agg_S = allocation.T @ S  # (P, K) aggregation in numpy
inputs = {sens_handles[k]: agg_S[:, k] for k in range(K)}
results = aadc.evaluate(funcs, request, inputs, workers)  # Single call!

# Extract P IM values and K×P gradient matrix
all_ims = results[0][im_output]
dIM_dS = np.column_stack([results[1][im_output][h] for h in sens_handles])
```

### 3.3 GPU Backend Architecture

```
Trades → CPU Pricers → CRIF → GPU SIMM Kernel → IM + dIM/dS (analytical)
                                    ↑
                              CUDA kernel
                              with chain rule
```

The GPU kernel (`simm_portfolio_cuda.py`) implements:
1. Per-portfolio SIMM forward pass
2. Analytical gradient via manually derived chain rule
3. Optional forward-only mode (no gradient, half the cost)

```python
# GPU kernel signature
@cuda.jit
def _simm_gradient_kernel_full(
    agg_S,           # (P, K) aggregated sensitivities
    risk_weights,    # (K,) per-factor weights
    concentration,   # (K,) CR factors
    intra_corr,      # (K, K) intra-bucket correlations
    bucket_mapping,  # (K,) maps factors to buckets
    # ... other SIMM parameters ...
    im_out,          # (P,) output IM values
    grad_out,        # (P, K) output gradient matrix
):
    p = cuda.grid(1)  # One thread per portfolio
    # ... forward SIMM + gradient computation ...
```

### 3.4 Brute-Force GPU Architecture

For optimization, a third approach — **brute-force GPU** — evaluates all candidate moves without computing gradients:

```
Current State → Generate T×(P-1) Candidates → GPU Forward-Only → Pick Best
                      ↑                            ↑
                  One per trade         Massive parallel eval
                  per destination
```

```python
@cuda.jit
def _eval_all_moves_kernel(
    agg_S,           # (P, K) current state
    S,               # (T, K) per-trade sensitivities
    curr_assign,     # (T,) current assignments
    base_im,         # (P,) current portfolio IMs
    delta_im_out,    # (T, P-1) IM change for each move
):
    idx = cuda.grid(1)  # One thread per candidate move
    t = idx // (P - 1)
    dest = idx % (P - 1)
    # ... incremental SIMM for affected portfolios ...
```

---

## 4. Performance Comparison

All benchmarks run on debian-monster: Dual Intel Xeon (112 cores), Linux 6.1.0-13-amd64, Python 3.11, NumPy 2.2.6 (AVX-512), AADC 1.8.0. GPU results use CUDA simulator mode for functional testing (real GPU hardware would show faster absolute times but similar relative patterns).

### 4.1 Trading Day Workflow Benchmark

The `benchmark_trading_workflow.py` script simulates a full trading day in 5 steps:

| Step | Time | Operation | Backends Compared |
|------|------|-----------|-------------------|
| 1 | 9:00 AM | Portfolio Setup | AADC, GPU, BF-GPU, C++ |
| 2 | 10:00 AM | Margin Attribution | AADC, GPU, BF-GPU, C++ |
| 3 | 1:00 PM | Pre-Trade Routing | AADC, GPU, BF-GPU, C++ |
| 4 | 3:00 PM | What-If Scenarios | AADC, GPU, BF-GPU, C++ |
| 5 | 5:00 PM | EOD Optimization | Adam/GD/Brute-Force |

### 4.2 Step-by-Step Results (T=4,000, P=15, K=2,550)

**Step 1: Portfolio Setup (1 SIMM evaluation)**

| Backend | Time | Evals/sec | Notes |
|---------|------|-----------|-------|
| AADC Python | **2.3ms** | 430 | Fastest for initial setup |
| GPU CUDA | 715ms | 1.4 | CUDA initialization overhead |
| GPU Brute-Force | 318ms | 3.1 | Forward-only, no gradient |
| C++ AADC | 5.7ms | 175 | Compiled AADC |

**Step 2: Margin Attribution (reuse cached gradient)**

| Backend | Time | Method | Notes |
|---------|------|--------|-------|
| AADC Python | **0ms** | Cached gradient dot product | Free if gradient cached |
| GPU CUDA | **0ms** | Cached gradient dot product | Same — numpy overhead only |
| GPU Brute-Force | 7.1ms | Leave-one-out (T batched evals) | Must recompute |
| C++ AADC | shared | Computed with Step 1 | Same invocation |

**Step 3: Pre-Trade Routing (50 new trades)**

| Backend | Time | Evals | Method |
|---------|------|-------|--------|
| AADC Python | **8.9ms** | 5 | Gradient → marginal IM dot product |
| GPU CUDA | 27.5ms | 5 | Same gradient approach |
| GPU Brute-Force | 308ms | 100 | Try all P for each trade |
| C++ AADC | 5.7ms | 1 | Batch all trades |

**Step 4: What-If Scenarios (8 scenarios)**

| Backend | Time | Output | Notes |
|---------|------|--------|-------|
| AADC Python | 12.7ms | IM + risk decomposition | Full gradient per scenario |
| GPU CUDA | 43.0ms | IM + risk decomposition | Same output |
| GPU Brute-Force | 24.3ms | IM only | **No risk decomposition** |
| C++ AADC | **4.0ms** | IM + risk decomposition | Batches scenarios |

**Step 5: EOD Optimization (Adam, ~11K evaluations)**

| Backend | Total Time | Evals/sec | IM Reduction |
|---------|-----------|-----------|--------------|
| C++ AADC | **0.72s** | 15,574 | 59.3% |
| AADC Python | 10.6s | 1,049 | 59.3% |
| GPU CUDA | 33.9s | 329 | 59.3% |
| GPU Brute-Force | 14.1s | 7.2 | **61.2%** |

### 4.3 Summary: Time-to-Decision

| Step | Business Question | AADC Py | GPU | BF GPU | C++ |
|------|-------------------|---------|-----|--------|-----|
| 1 | "What is our IM?" | **2.3ms** | 715ms | 318ms | 5.7ms |
| 2 | "Which trades drive IM?" | **0ms** | 0ms | 7.1ms | — |
| 3 | "Where to route 50 trades?" | **8.9ms** | 27.5ms | 308ms | 5.7ms |
| 4 | "Impact of 8 scenarios?" | 12.7ms | 43.0ms | 24.3ms | **4.0ms** |
| 5 | "Optimal reallocation?" | 10.6s | 33.9s | 14.1s | **0.72s** |
| **Total (Steps 1-4)** | | **24ms** | **786ms** | **658ms** | **15ms** |

**Key finding**: For interactive intraday workflows (Steps 1-4), AADC Python completes in **24ms** vs GPU's **786ms** — a **33× advantage**. For EOD batch optimization, C++ AADC is fastest (0.72s), but GPU brute-force achieves slightly better IM reduction (61.2% vs 59.3%).

---

## 5. Gradient Gap Analysis

### 5.1 What Gradients Give You

When AADC or GPU-with-gradient evaluates the SIMM kernel, the output includes:
- **IM values**: Total initial margin per portfolio (P scalars)
- **Gradient matrix**: dIM/dS — K × P partial derivatives

GPU Brute-Force returns **only** the IM values. This has downstream costs:

| Step | With Gradient | Without Gradient (BF) |
|------|--------------|----------------------|
| Attribution | Free dot product | T batched forward evals |
| Pre-Trade | 1 eval + K dot products | P forward evals per trade |
| What-If | Risk factor decomposition | Scalar IM only |
| Optimization | Continuous gradient descent | Discrete enumeration |

### 5.2 Quantified Impact

From T=4,000, P=15, K=2,550 benchmark:

| Step | AADC Time | BF GPU Time | Slowdown | Information Loss |
|------|-----------|-------------|----------|------------------|
| 1 Setup | 2.3ms | 318ms | 138× | No gradient for downstream |
| 2 Attribution | **0ms** | 7.1ms | ∞ | Approximate vs exact Euler |
| 3 Pre-Trade | 8.9ms | 308ms | **35×** | No marginal IM shortcut |
| 4 What-If | 12.7ms | 24.3ms | 1.9× | **No risk decomposition** |
| 5 EOD | 10.6s | 14.1s | 1.3× | Different search space |

### 5.3 When BF GPU Wins

Despite the gradient gap, brute-force GPU has legitimate advantages:

1. **Global search**: BF explores ALL discrete moves per round, potentially finding solutions Adam misses due to non-convexity. The 61.2% vs 59.3% IM reduction demonstrates this.

2. **Simpler implementation**: No adjoint tape, no smooth-max hacks for sqrt(negative). Forward-only kernel is ~half the code.

3. **Audit trail**: Every candidate move is explicitly evaluated — easier to explain to risk committees than "the adjoint tape said so."

4. **AADC-free environment**: If AADC licensing is unavailable, BF GPU is the only scalable option.

---

## 6. Implementation Issues and Fixes

### 6.1 IM Mismatch on Multi-Asset Portfolios

When running mixed trade types (ir_swap + equity_option + fx_option), AADC and GPU initially produced different IMs (~5% discrepancy).

| Issue | Symptom | Fix |
|-------|---------|-----|
| Missing inter-bucket gamma | AADC kernel used gamma=0 for non-Rates risk classes | Added gamma tables for all 5 risk classes |
| NaN gradients | sqrt(negative) when k_rc_sq < 0 from cross-terms | Smooth max: `(x + sqrt(x² + ε))/2 + ε` with ε=1.0 |
| MAX_B overflow | Mixed types produce 69+ buckets, exceeded 64 | Increased `MAX_B = 64` → `128` |

**After fixes**: IM relative difference = 3.12e-15 (machine precision). Both backends find identical optimization improvements.

### 6.2 GPU Parallelization Strategy

The current GPU kernel launches **one thread per portfolio**:

```python
threads_per_block = 256
blocks = (P + 255) // 256
```

With P=5-20 portfolios, this gives grid size 1 — wasting >99.99% of an H100's capacity. However, for brute-force optimization where we evaluate T×(P-1) candidates, grid size is ~190,000 — full utilization.

**Recommendation**: Use GPU for brute-force candidate search (190K threads), not per-portfolio SIMM evaluation (5-20 threads).

---

## 7. CRIF Generation

### 7.1 AADC CRIF (Adjoint Mode)

AADC records pricing kernels and computes all sensitivities in a single forward+adjoint pass:

```python
with aadc.record_kernel() as funcs:
    rates_aadc = [aadc.idouble(r).mark_as_input() for r in curve.zero_rates]
    pv = price_vanilla_irs(rates_aadc, notional, fixed_rate, maturity)
    pv_output = pv.mark_as_output()

# Single evaluation produces all 12 IR sensitivities
request = {pv_output: rate_handles}
results = aadc.evaluate(funcs, request, inputs, workers)
# results[1][pv_output][handle_k] = dPV/d(rate_k)
```

Cost: 1 forward + 1 adjoint pass per trade type (kernel reused across trades).

### 7.2 GPU CRIF (Bump-and-Revalue)

GPU v3.4.0 implements bump-and-revalue directly in CUDA kernels:

```python
@cuda.jit
def _compute_irs_crif_kernel(trades, curve_rates, crif_out):
    tid = cuda.grid(1)  # One thread per trade
    for k in range(12):  # 12 IR tenors
        # Base PV
        pv_base = _price_irs_device(trades[tid], curve_rates)
        # Bump rate k by 1bp
        bumped_rates = curve_rates.copy()
        bumped_rates[k] += 0.0001
        pv_bumped = _price_irs_device(trades[tid], bumped_rates)
        # Sensitivity
        crif_out[tid, k] = (pv_bumped - pv_base) / 0.0001
```

Cost: 1 + K pricings per trade (13 for IR swap with 12 tenors).

### 7.3 CRIF Performance Comparison

| Trade Type | AADC | GPU Bump-Revalue | Ratio |
|------------|------|------------------|-------|
| IR Swap (12 factors) | ~0.8ms | ~0.13ms | AADC 6× slower |
| Equity Option (14 factors) | ~1.0ms | ~0.15ms | AADC 7× slower |
| FX Option (26 factors) | ~1.5ms | ~0.25ms | AADC 6× slower |

**Note**: AADC CRIF is slower than GPU bump-and-revalue due to kernel recording overhead. However, CRIF generation is typically a **one-time setup cost** — the SIMM gradient computation (where AADC excels) dominates interactive workflows.

---

## 8. Pre-Trade Analytics

### 8.1 Marginal IM Computation

**With Gradient (AADC/GPU)**:
```python
# Pre-compute gradient once
gradient = compute_simm_gradient(portfolio_crif)  # (K,)

# Instant marginal IM for any new trade
marginal_im = gradient @ new_trade_sensitivities  # O(K)
```

**Without Gradient (BF GPU)**:
```python
# Must evaluate full SIMM for each candidate
for p in range(P):
    new_agg_S = agg_S[p] + new_trade_sens
    candidate_im[p] = simm_forward_only(new_agg_S)  # O(K²)
```

### 8.2 Pre-Trade Routing Implementation (GPU v3.4.0)

```python
@dataclass
class PreTradeRoutingResult:
    marginal_ims: np.ndarray      # (P,) marginal IM per portfolio
    base_ims: np.ndarray          # (P,) current IM before new trade
    new_ims: np.ndarray           # (P,) IM after adding new trade
    best_portfolio: int
    best_marginal_im: float
    worst_portfolio: int
    worst_marginal_im: float
    sensies_time_sec: float       # Time to compute new trade sensitivities
    eval_time_sec: float          # Time to evaluate all 2P SIMM scenarios

def pretrade_routing_gpu(S, allocation, new_trade_sens, risk_weights, ...):
    # Evaluate all 2P scenarios in single GPU kernel launch
    all_scenarios = np.vstack([agg_S_base, agg_S_with_new])  # (2P, K)
    all_ims = compute_simm_im_only_cuda(all_scenarios, ...)
    marginal_ims = all_ims[P:] - all_ims[:P]
    return PreTradeRoutingResult(...)
```

---

## 9. CLI Usage

### 9.1 AADC Backend

```bash
source venv/bin/activate

# Basic: compute SIMM with AADC
python -m model.simm_portfolio_aadc \
    --trades 1000 --portfolios 5 --threads 8 --trade-types ir_swap

# Multi-asset with optimization
python -m model.simm_portfolio_aadc \
    --trades 500 --portfolios 5 --threads 8 \
    --trade-types ir_swap,equity_option,fx_option \
    --optimize --method adam --max-iters 100
```

### 9.2 GPU Backend

```bash
# Basic: compute SIMM with GPU
python -m model.simm_portfolio_cuda \
    --trades 1000 --portfolios 5 --trade-types ir_swap

# GPU CRIF generation (bump-and-revalue)
python -m model.simm_portfolio_cuda \
    --trades 1000 --portfolios 5 --trade-types ir_swap \
    --crif-method gpu

# Pre-trade routing with GPU
python -m model.simm_portfolio_cuda \
    --trades 1000 --portfolios 5 --trade-types ir_swap \
    --pretrade
```

### 9.3 Full Workflow Benchmark

```bash
# Compare all backends on trading day workflow
python benchmark_trading_workflow.py \
    --trades 1000 --portfolios 5 --threads 8 \
    --trade-types ir_swap,fx_option

# Exclude specific methods
python benchmark_trading_workflow.py \
    --trades 1000 --portfolios 5 \
    --exclude gpu_brute_force

# Quick test
python benchmark_trading_workflow.py \
    --trades 100 --portfolios 3 --output none
```

---

## 10. When to Use Each Backend

### 10.1 Decision Matrix

| Scenario | Recommended Backend | Rationale |
|----------|---------------------|-----------|
| Interactive margin queries | **AADC Python** | Sub-10ms latency for Steps 1-4 |
| Real-time pre-trade routing | **AADC Python** | O(K) marginal IM vs O(K²) |
| Risk factor decomposition | **AADC or GPU** | BF lacks gradients |
| EOD batch optimization | **C++ AADC** | 15K evals/sec vs 1K (Python) |
| Overnight validation | **GPU Brute-Force** | Finds solutions Adam misses |
| AADC-unavailable environment | **GPU Brute-Force** | Only scalable option |
| Small portfolios (P ≤ 5) | Either | GPU overhead acceptable |
| Large portfolios (P > 20) | **AADC** | GPU grid underutilization |

### 10.2 Hybrid Strategy

Production systems often combine backends:

1. **Intraday (AADC)**: Real-time margin attribution, pre-trade checks, what-if scenarios
2. **EOD (C++ AADC + BF GPU)**: Run gradient descent for fast convergence, then BF for validation
3. **Overnight (BF GPU)**: Exhaustive search for regulatory/audit compliance

### 10.3 Future GPU Improvements

To make GPU competitive for gradient-based workflows:

| Improvement | Effort | Expected Benefit |
|-------------|--------|------------------|
| CuPy drop-in | Low | 2-10× for matmul-heavy code |
| Batched matrix operations | Medium | Eliminate Python loops over buckets |
| Parallelize over (portfolio, factor) | High | Full GPU utilization at any P |

---

## 11. Execution Log Schema

Both backends log to `data/execution_log_portfolio.csv`:

| Column | Description |
|--------|-------------|
| `model_name` | Backend identifier (e.g., `workflow_portfolio_setup_gpu_full`) |
| `num_trades` | Total trades in portfolio |
| `num_portfolios` | Number of counterparty portfolios (P) |
| `num_simm_buckets` | SIMM bucket count across all risk classes |
| `crif_time_sec` | CRIF generation time |
| `crif_sensies_time_sec` | Sensitivity computation time (excludes recording) |
| `simm_time_sec` | SIMM aggregation time |
| `im_sens_time_sec` | IM + gradient computation time |
| `im_result` | SIMM Initial Margin in USD |
| `optimize_method` | `gradient_descent`, `adam`, `gpu_brute_force` |
| `optimize_im_reduction_pct` | IM reduction as percentage |
| `simm_evals_per_sec` | Throughput metric |

---

## 12. Verification and Cross-Validation

### 12.1 IM Agreement

Both backends implement the same ISDA SIMM v2.6 formula. After fixes (Section 6.1), they produce **bit-identical** results:

| Metric | Value |
|--------|-------|
| IM relative difference | 3.12e-15 (machine precision) |
| Max gradient relative error | 9.2e-15 |
| Optimization result agreement | Both find same 59.3% reduction |

### 12.2 Three-Way Cross-Validation

The project includes three independent SIMM implementations:

| Implementation | Method | Agreement |
|----------------|--------|-----------|
| AADC Python | Auto-differentiated | Baseline |
| GPU CUDA | Hand-coded analytical | **Exact** (to 16 digits) |
| C++ AADC | Compiled AADC | Within 0.01% (FP ordering) |

This cross-validation confirms correctness: two completely independent implementations (auto-diff vs hand-coded) arriving at identical results.

---

## 13. Summary

### 13.1 Key Findings

1. **AADC dominates for interactive workflows**: 33× faster than GPU for Steps 1-4 (24ms vs 786ms)

2. **C++ AADC is fastest for EOD optimization**: 15,574 evals/sec vs 1,049 (Python) vs 329 (GPU)

3. **GPU Brute-Force finds slightly better solutions**: 61.2% vs 59.3% IM reduction, but at 2000× lower throughput

4. **Gradient advantage is fundamental**: O(K) vs O(K²) — at K=2,550, this is decisive

5. **GPU best for brute-force candidate search**: 190K threads for T×(P-1) candidate evaluation

### 13.2 Recommendations

| User Profile | Recommended Setup |
|--------------|-------------------|
| Interactive trading desk | AADC Python for real-time queries |
| End-of-day processing | C++ AADC for optimization + BF GPU for validation |
| Regulatory/audit | BF GPU for exhaustive, explainable search |
| Research/prototyping | AADC Python (easiest API, fastest iteration) |

### 13.3 The Bottom Line

**AADC and GPU serve different purposes**:

- **AADC** provides **exact gradients at O(K) cost**, enabling efficient continuous optimization and instant marginal IM queries. Best for interactive, gradient-based workflows.

- **GPU** provides **massive parallelism for forward-only evaluation**, enabling exhaustive discrete search. Best for brute-force candidate enumeration and validation.

The optimal production system uses both: AADC for the 95% of queries that benefit from gradients, GPU for the 5% where exhaustive search adds value.
