# Trading Day Workflow Benchmark

This benchmark simulates a full trading day for an ISDA SIMM margin desk, based on
the workflow described at https://dev.matlogica.com/matlogica/SIMM-Demo/trading_workflow.html.

It compares three backends running the **full ISDA SIMM v2.6** formula (intra-bucket
correlations, concentration risk factors, inter-bucket gamma, Delta + Vega margins,
cross-risk-class PSI aggregation):

| Backend | Gradient Method | Description |
|---------|----------------|-------------|
| **AADC Python** | Automatic Adjoint Differentiation | Record kernel once, replay for every evaluation |
| **C++ AADC** | Same AADC library, compiled C++ | Eliminates Python-C++ dispatch overhead |
| **GPU CUDA** | Hand-coded analytical chain rule | Numba CUDA kernel, no AADC dependency |

## Core Concept: Record Once, Evaluate Many Times

The SIMM formula maps K aggregated risk-factor sensitivities to a single IM number.
AADC records this formula as a differentiable computational graph (kernel) **once** at
start of day. Every subsequent margin computation — attribution, pre-trade routing,
stress scenarios, optimization — reuses the same recorded kernel. The kernel size is
O(K) where K ~ 50-100 risk factors, independent of trade count. A 1M-trade portfolio
uses the same kernel as a 100-trade portfolio.

The GPU backend computes the same formula with an analytical gradient derived by hand
via chain rule through the SIMM aggregation tree. No recording step is needed, but the
gradient implementation must be manually maintained.

## Five Trading Day Stages

### Stage 1: 7:00 AM — Start of Day: Portfolio Setup

Generate trades, compute CRIF sensitivities, build the sensitivity matrix S (T x K),
and record the AADC kernel. The kernel captures the complete SIMM calculation:

```
WS_k = S_k x RW_k x CR_k                          (weighted sensitivities)
K_b  = sqrt(sum_ij rho_ij x WS_i x WS_j)          (intra-bucket aggregation)
M_rc = sqrt(sum_bc K_b^2 + gamma_bc x S_b x S_c)  (inter-bucket with g_bc)
IM   = sqrt(sum_rs psi_rs x M_r x M_s)             (cross-risk-class)
```

AADC recording is a one-time cost (~5-40 ms depending on K). GPU pre-allocates
constant arrays on device. Both backends then compute the initial portfolio IM.

### Stage 2: 8:00 AM — Morning Risk Report: Margin Attribution

Euler decomposition assigns margin contribution to each trade:

```
contribution[t] = S[t,:] . grad_S[portfolio(t),:]
```

This reuses the gradient from Stage 1 — zero new kernel evaluations needed. The
Euler property guarantees sum(contributions) = total IM (verified to machine precision).

### Stage 3: 9 AM-4 PM — Intraday Trading: Pre-Trade Checks

Route N new trade inquiries to the optimal counterparty portfolio. For each new trade,
compute marginal IM = grad_S @ s_new for each portfolio and pick the minimum. The
gradient is refreshed periodically (every `refresh_interval` trades) by re-evaluating
the kernel — not re-recording it.

### Stage 4: 2:00 PM — What-If Scenarios

Evaluate stress tests and scenario analysis by replaying the kernel with modified inputs:

- **Stress test**: Shock IR sensitivities +50% and re-evaluate
- **Unwind**: Remove top-5 margin consumers and re-evaluate
- **Hedge**: Add offsetting position for largest contributor
- **IM ladder**: Evaluate at 0.5x, 0.75x, 1.0x, 1.25x, 1.5x shock levels

Each scenario is one kernel evaluation (no re-recording).

### Stage 5: 5:00 PM — End of Day: Portfolio Optimization

Gradient-descent (or Adam) allocation optimization over `max_iters` iterations.
Each iteration:

1. Evaluate kernel to get IM and dIM/dS for all P portfolios (one call)
2. Compute allocation gradient: `grad_alloc = S @ grad_S.T`
3. Update allocation with projected gradient step (simplex constraint)
4. Backtracking line search for step acceptance

All iterations reuse the same recorded AADC kernel. The GPU backend uses the same
optimizer logic with its analytical gradient kernel.

## Kernel Economics

The benchmark tracks amortized kernel cost:

```
amortized_cost = recording_time / total_evaluations
```

A typical trading day performs 20-200+ kernel evaluations (attribution + pre-trade +
scenarios + optimization). With a 40 ms recording cost and 100 evaluations, the
amortized overhead is 0.4 ms/eval — negligible compared to a full SIMM recalculation.

## Usage

```bash
source venv/bin/activate

# Default: 1000 trades, 5 portfolios, gradient descent
python benchmark_trading_workflow.py --trades 1000 --portfolios 5 --threads 8

# With Adam optimizer
python benchmark_trading_workflow.py --trades 5000 --portfolios 10 --method adam

# Full options
python benchmark_trading_workflow.py --trades 10000 --portfolios 20 --threads 48 \
    --new-trades 50 --optimize-iters 100 --method adam --trade-types ir_swap
```

---

## Benchmark Results

Auto-generated results from each run are appended below.

---

## Run: 2026-02-02 12:38:58

### Configuration

| Parameter | Value |
|-----------|-------|
| Trades | 50 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| New trades (intraday) | 10 |
| Optimize iterations | 20 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | AADC Evals | GPU Evals |
|------|-----------|----------|------------|-----------|
| 7:00 AM Portfolio Setup | 380 us | 705.67 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 32 us | 10 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 792 us | 3.39 ms | 2 | 2 |
| 2:00 PM What-If Scenarios | 2.83 ms | 13.67 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 14.55 ms | 6.27 ms | 21 | 3 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Recording cost (1-time) | 19.23 ms |
| Total AADC evals | 32 |
| Total GPU evals | 14 |
| Amortized recording/eval | 0.60 ms |
| Cumulative AADC time | 18.59 ms |
| Cumulative GPU time | 729.01 ms |
| AADC total (rec + eval) | 37.82 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $171,673,642,171 | - |
| Rates +50bp | $257,510,463,256 | +50.0% |
| Unwind top 5 | $109,820,983,173 | -36.0% |
| Add hedge | $139,384,606,919 | -18.8% |

**IM Ladder:** 0.5x: $85,836,821,085, 0.75x: $128,755,231,628, 1.0x: $171,673,642,171, 1.25x: $214,592,052,714, 1.5x: $257,510,463,256

### 5:00 PM EOD Optimization

- Initial IM: $171,673,642,171
- Final IM: $158,153,675,904 (reduction: 7.9%)
- Trades moved: 1, Iterations: 20

---

## Run: 2026-02-02 12:39:47

### Configuration

| Parameter | Value |
|-----------|-------|
| Trades | 1,000 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | AADC Evals | GPU Evals |
|------|-----------|----------|------------|-----------|
| 7:00 AM Portfolio Setup | 464 us | 574.16 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 124 us | 53 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.02 ms | 8.38 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.90 ms | 13.61 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 810.46 ms | 40.69 ms | 101 | 5 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Recording cost (1-time) | 17.40 ms |
| Total AADC evals | 115 |
| Total GPU evals | 19 |
| Amortized recording/eval | 0.15 ms |
| Cumulative AADC time | 817.97 ms |
| Cumulative GPU time | 636.90 ms |
| AADC total (rec + eval) | 835.36 ms |
| GPU speedup (eval only) | 1.3x |
| GPU speedup (inc. recording) | 1.3x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $1,429,302,257,478 | - |
| Rates +50bp | $2,143,953,386,216 | +50.0% |
| Unwind top 5 | $1,339,331,372,113 | -6.3% |
| Add hedge | $1,394,435,505,397 | -2.4% |

**IM Ladder:** 0.5x: $714,651,128,739, 0.75x: $1,071,976,693,108, 1.0x: $1,429,302,257,478, 1.25x: $1,786,627,821,847, 1.5x: $2,143,953,386,216

### 5:00 PM EOD Optimization

- Initial IM: $1,429,302,257,478
- Final IM: $1,429,302,257,478 (reduction: 0.0%)
- Trades moved: 0, Iterations: 100

---

## Run: 2026-02-02 12:40:03

### Configuration

| Parameter | Value |
|-----------|-------|
| Trades | 5,000 |
| Portfolios | 10 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | AADC Evals | GPU Evals |
|------|-----------|----------|------------|-----------|
| 7:00 AM Portfolio Setup | 791 us | 596.76 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 531 us | 310 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.94 ms | 8.82 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 4.31 ms | 14.22 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 3.823 s | 165.03 ms | 101 | 5 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Recording cost (1-time) | 18.50 ms |
| Total AADC evals | 115 |
| Total GPU evals | 19 |
| Amortized recording/eval | 0.16 ms |
| Cumulative AADC time | 3831.33 ms |
| Cumulative GPU time | 785.14 ms |
| AADC total (rec + eval) | 3849.84 ms |
| GPU speedup (eval only) | 4.9x |
| GPU speedup (inc. recording) | 4.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $7,227,592,583,088 | - |
| Rates +50bp | $10,841,388,874,633 | +50.0% |
| Unwind top 5 | $7,044,173,709,635 | -2.5% |
| Add hedge | $7,184,887,979,430 | -0.6% |

**IM Ladder:** 0.5x: $3,613,796,291,544, 0.75x: $5,420,694,437,316, 1.0x: $7,227,592,583,088, 1.25x: $9,034,490,728,861, 1.5x: $10,841,388,874,633

### 5:00 PM EOD Optimization

- Initial IM: $7,227,592,583,088
- Final IM: $15,254,211,571,340 (reduction: -111.1%)
- Trades moved: 179, Iterations: 100

---

## Run: 2026-02-02 12:42:27

### Configuration

| Parameter | Value |
|-----------|-------|
| Trades | 1,000 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | AADC Evals | GPU Evals |
|------|-----------|----------|------------|-----------|
| 7:00 AM Portfolio Setup | 643 us | 580.19 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 107 us | 52 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.80 ms | 8.49 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.99 ms | 13.56 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 801.05 ms | 39.67 ms | 101 | 5 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Recording cost (1-time) | 18.48 ms |
| Total AADC evals | 115 |
| Total GPU evals | 19 |
| Amortized recording/eval | 0.16 ms |
| Cumulative AADC time | 808.59 ms |
| Cumulative GPU time | 641.96 ms |
| AADC total (rec + eval) | 827.08 ms |
| GPU speedup (eval only) | 1.3x |
| GPU speedup (inc. recording) | 1.3x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $1,429,302,257,478 | - |
| Rates +50bp | $2,143,953,386,216 | +50.0% |
| Unwind top 5 | $1,339,331,372,113 | -6.3% |
| Add hedge | $1,394,435,505,397 | -2.4% |

**IM Ladder:** 0.5x: $714,651,128,739, 0.75x: $1,071,976,693,108, 1.0x: $1,429,302,257,478, 1.25x: $1,786,627,821,847, 1.5x: $2,143,953,386,216

### 5:00 PM EOD Optimization

- Initial IM: $1,429,302,257,478
- Final IM: $1,429,302,257,478 (reduction: 0.0%)
- Trades moved: 0, Iterations: 100

---

## Run: 2026-02-02 12:50:12

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 50 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 10 |
| Optimize iterations | 20 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 340 us | 429.52 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 22 us | 9 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 385 us | 1.10 ms | 1 | 1 |
| 2:00 PM What-If Scenarios | 2.85 ms | 7.52 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 14.51 ms | 28.32 ms | 21 | 21 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 37.92 ms |
| Total AADC evals | 31 |
| Total kernel reuses | 31 |
| Total GPU evals | 31 |
| Amortized recording/eval | 1.22 ms |
| Cumulative AADC time | 18.11 ms |
| Cumulative GPU time | 466.48 ms |
| AADC total (rec + eval) | 56.03 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $399,527,208,153,231 | - |
| Rates +50bp | $599,290,812,229,847 | +50.0% |
| Unwind top 5 | $219,331,272,299,183 | -45.1% |
| Add hedge | $348,280,189,327,689 | -12.8% |

**IM Ladder:** 0.5x: $199,763,604,076,616, 0.75x: $299,645,406,114,924, 1.0x: $399,527,208,153,231, 1.25x: $499,409,010,191,539, 1.5x: $599,290,812,229,847

### 5:00 PM EOD Optimization

- Initial IM: $399,527,208,153,231
- Final IM: $299,930,884,549,796 (reduction: 24.9%)
- Trades moved: 1, Iterations: 20

---

## Run: 2026-02-02 12:53:04

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 50 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 435 pairs |
| New trades (intraday) | 10 |
| Optimize iterations | 20 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 361 us | 441.51 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 25 us | 9 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 400 us | 1.14 ms | 1 | 1 |
| 2:00 PM What-If Scenarios | 2.70 ms | 7.56 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 14.92 ms | 28.93 ms | 21 | 21 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 37.46 ms |
| Total AADC evals | 31 |
| Total kernel reuses | 31 |
| Total GPU evals | 31 |
| Amortized recording/eval | 1.21 ms |
| Cumulative AADC time | 18.40 ms |
| Cumulative GPU time | 479.15 ms |
| AADC total (rec + eval) | 55.87 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $399,527,208,153,231 | - |
| Rates +50bp | $599,290,812,229,847 | +50.0% |
| Unwind top 5 | $219,331,272,299,183 | -45.1% |
| Add hedge | $348,280,189,327,689 | -12.8% |

**IM Ladder:** 0.5x: $199,763,604,076,616, 0.75x: $299,645,406,114,924, 1.0x: $399,527,208,153,231, 1.25x: $499,409,010,191,539, 1.5x: $599,290,812,229,847

### 5:00 PM EOD Optimization

- Initial IM: $399,527,208,153,231
- Final IM: $299,930,884,549,796 (reduction: 24.9%)
- Trades moved: 1, Iterations: 20

---

## Run: 2026-02-02 12:55:00

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 50 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 435 pairs |
| New trades (intraday) | 10 |
| Optimize iterations | 20 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 490 us | 442.18 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 27 us | 11 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 598 us | 1.13 ms | 1 | 1 |
| 2:00 PM What-If Scenarios | 2.76 ms | 7.58 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 15.17 ms | 29.31 ms | 21 | 21 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 37.90 ms |
| Total AADC evals | 31 |
| Total kernel reuses | 31 |
| Total GPU evals | 31 |
| Amortized recording/eval | 1.22 ms |
| Cumulative AADC time | 19.04 ms |
| Cumulative GPU time | 480.22 ms |
| AADC total (rec + eval) | 56.94 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $399,527,208,153,231 | - |
| Rates +50bp | $599,290,812,229,847 | +50.0% |
| Unwind top 5 | $219,331,272,299,183 | -45.1% |
| Add hedge | $348,280,189,327,689 | -12.8% |

**IM Ladder:** 0.5x: $199,763,604,076,616, 0.75x: $299,645,406,114,924, 1.0x: $399,527,208,153,231, 1.25x: $499,409,010,191,539, 1.5x: $599,290,812,229,847

### 5:00 PM EOD Optimization

- Initial IM: $399,527,208,153,231
- Final IM: $299,930,884,549,796 (reduction: 24.9%)
- Trades moved: 1, Iterations: 20

---

## Run: 2026-02-02 12:55:14

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 1,000 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 435 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 551 us | 413.74 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 101 us | 52 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.02 ms | 4.94 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.74 ms | 7.36 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 811.90 ms | 853.20 ms | 101 | 101 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 37.70 ms |
| Total AADC evals | 115 |
| Total kernel reuses | 115 |
| Total GPU evals | 115 |
| Amortized recording/eval | 0.33 ms |
| Cumulative AADC time | 819.32 ms |
| Cumulative GPU time | 1.279 s |
| AADC total (rec + eval) | 857.02 ms |
| GPU speedup (eval only) | 0.6x |
| GPU speedup (inc. recording) | 0.7x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $9,581,698,819,279,568 | - |
| Rates +50bp | $14,372,548,228,919,356 | +50.0% |
| Unwind top 5 | $7,436,184,419,175,739 | -22.4% |
| Add hedge | $9,055,211,339,514,372 | -5.5% |

**IM Ladder:** 0.5x: $4,790,849,409,639,784, 0.75x: $7,186,274,114,459,678, 1.0x: $9,581,698,819,279,568, 1.25x: $11,977,123,524,099,462, 1.5x: $14,372,548,228,919,356

### 5:00 PM EOD Optimization

- Initial IM: $9,581,698,819,279,568
- Final IM: $9,581,698,819,279,568 (reduction: 0.0%)
- Trades moved: 0, Iterations: 100

---

## Run: 2026-02-02 12:57:04

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 5,000 |
| Portfolios | 10 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 435 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 861 us | 458.19 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 615 us | 321 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.26 ms | 5.28 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 4.99 ms | 7.61 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 3.823 s | 3.879 s | 101 | 101 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 39.03 ms |
| Total AADC evals | 115 |
| Total kernel reuses | 115 |
| Total GPU evals | 115 |
| Amortized recording/eval | 0.34 ms |
| Cumulative AADC time | 3.833 s |
| Cumulative GPU time | 4.350 s |
| AADC total (rec + eval) | 3.872 s |
| GPU speedup (eval only) | 0.9x |
| GPU speedup (inc. recording) | 0.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $105,959,746,066,461,936 | - |
| Rates +50bp | $158,939,619,099,692,928 | +50.0% |
| Unwind top 5 | $99,862,008,924,674,672 | -5.8% |
| Add hedge | $104,727,051,154,147,104 | -1.2% |

**IM Ladder:** 0.5x: $52,979,873,033,230,968, 0.75x: $79,469,809,549,846,464, 1.0x: $105,959,746,066,461,936, 1.25x: $132,449,682,583,077,424, 1.5x: $158,939,619,099,692,928

### 5:00 PM EOD Optimization

- Initial IM: $105,959,746,066,461,936
- Final IM: $105,959,746,066,461,936 (reduction: 0.0%)
- Trades moved: 0, Iterations: 100

---

## Run: 2026-02-02 13:06:32

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 5,000 |
| Portfolios | 10 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 435 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 581 us | 434.02 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 821 us | 310 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.92 ms | 5.19 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 4.21 ms | 7.59 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 3.752 s | 3.805 s | 101 | 101 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 38.82 ms |
| Total AADC evals | 115 |
| Total kernel reuses | 115 |
| Total GPU evals | 115 |
| Amortized recording/eval | 0.34 ms |
| Cumulative AADC time | 3.761 s |
| Cumulative GPU time | 4.252 s |
| AADC total (rec + eval) | 3.800 s |
| GPU speedup (eval only) | 0.9x |
| GPU speedup (inc. recording) | 0.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $105,959,746,066,461,936 | - |
| Rates +50bp | $158,939,619,099,692,928 | +50.0% |
| Unwind top 5 | $99,862,008,924,674,672 | -5.8% |
| Add hedge | $104,727,051,154,147,104 | -1.2% |

**IM Ladder:** 0.5x: $52,979,873,033,230,968, 0.75x: $79,469,809,549,846,464, 1.0x: $105,959,746,066,461,936, 1.25x: $132,449,682,583,077,424, 1.5x: $158,939,619,099,692,928

### 5:00 PM EOD Optimization

- Initial IM: $105,959,746,066,461,936
- Final IM: $105,959,746,066,461,936 (reduction: 0.0%)
- Trades moved: 0, Iterations: 100

---

## Run: 2026-02-02 13:09:46

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 50 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 435 pairs |
| New trades (intraday) | 10 |
| Optimize iterations | 20 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 355 us | 426.41 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 25 us | 9 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 344 us | 1.09 ms | 1 | 1 |
| 2:00 PM What-If Scenarios | 2.58 ms | 7.72 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 14.83 ms | 28.69 ms | 21 | 21 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 37.41 ms |
| Total AADC evals | 31 |
| Total kernel reuses | 31 |
| Total GPU evals | 31 |
| Amortized recording/eval | 1.21 ms |
| Cumulative AADC time | 18.12 ms |
| Cumulative GPU time | 463.92 ms |
| AADC total (rec + eval) | 55.54 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $399,527,208,153,231 | - |
| Rates +50bp | $599,290,812,229,847 | +50.0% |
| Unwind top 5 | $219,331,272,299,183 | -45.1% |
| Add hedge | $348,280,189,327,689 | -12.8% |

**IM Ladder:** 0.5x: $199,763,604,076,616, 0.75x: $299,645,406,114,924, 1.0x: $399,527,208,153,231, 1.25x: $499,409,010,191,539, 1.5x: $599,290,812,229,847

### 5:00 PM EOD Optimization

- Initial IM: $399,527,208,153,231
- Final IM: $299,930,884,549,796 (reduction: 24.9%)
- Trades moved: 1, Iterations: 20

---

## Run: 2026-02-02 13:09:56

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 1,000 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 435 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 605 us | 405.03 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 104 us | 57 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.16 ms | 4.91 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.88 ms | 7.27 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 817.35 ms | 859.69 ms | 101 | 101 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 29.81 ms |
| Total AADC evals | 115 |
| Total kernel reuses | 115 |
| Total GPU evals | 115 |
| Amortized recording/eval | 0.26 ms |
| Cumulative AADC time | 825.10 ms |
| Cumulative GPU time | 1.277 s |
| AADC total (rec + eval) | 854.91 ms |
| GPU speedup (eval only) | 0.6x |
| GPU speedup (inc. recording) | 0.7x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $9,581,698,819,279,568 | - |
| Rates +50bp | $14,372,548,228,919,356 | +50.0% |
| Unwind top 5 | $7,436,184,419,175,739 | -22.4% |
| Add hedge | $9,055,211,339,514,372 | -5.5% |

**IM Ladder:** 0.5x: $4,790,849,409,639,784, 0.75x: $7,186,274,114,459,678, 1.0x: $9,581,698,819,279,568, 1.25x: $11,977,123,524,099,462, 1.5x: $14,372,548,228,919,356

### 5:00 PM EOD Optimization

- Initial IM: $9,581,698,819,279,568
- Final IM: $9,581,698,819,279,568 (reduction: 0.0%)
- Trades moved: 0, Iterations: 100

---

## Run: 2026-02-02 13:10:17

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 5,000 |
| Portfolios | 10 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 435 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 749 us | 435.32 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 513 us | 303 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.19 ms | 5.13 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 5.76 ms | 7.67 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 3.818 s | 3.862 s | 101 | 101 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 25.55 ms |
| Total AADC evals | 115 |
| Total kernel reuses | 115 |
| Total GPU evals | 115 |
| Amortized recording/eval | 0.22 ms |
| Cumulative AADC time | 3.828 s |
| Cumulative GPU time | 4.311 s |
| AADC total (rec + eval) | 3.854 s |
| GPU speedup (eval only) | 0.9x |
| GPU speedup (inc. recording) | 0.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $105,959,746,066,461,936 | - |
| Rates +50bp | $158,939,619,099,692,928 | +50.0% |
| Unwind top 5 | $99,862,008,924,674,672 | -5.8% |
| Add hedge | $104,727,051,154,147,104 | -1.2% |

**IM Ladder:** 0.5x: $52,979,873,033,230,968, 0.75x: $79,469,809,549,846,464, 1.0x: $105,959,746,066,461,936, 1.25x: $132,449,682,583,077,424, 1.5x: $158,939,619,099,692,928

### 5:00 PM EOD Optimization

- Initial IM: $105,959,746,066,461,936
- Final IM: $105,959,746,066,461,936 (reduction: 0.0%)
- Trades moved: 0, Iterations: 100

---

## Run: 2026-02-02 13:11:16

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 10,000 |
| Portfolios | 50 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 435 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 761 us | 498.97 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 1.19 ms | 674 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.35 ms | 5.46 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 4.36 ms | 7.58 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 8.484 s | 8.648 s | 101 | 101 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 30.77 ms |
| Total AADC evals | 115 |
| Total kernel reuses | 115 |
| Total GPU evals | 115 |
| Amortized recording/eval | 0.27 ms |
| Cumulative AADC time | 8.494 s |
| Cumulative GPU time | 9.161 s |
| AADC total (rec + eval) | 8.525 s |
| GPU speedup (eval only) | 0.9x |
| GPU speedup (inc. recording) | 0.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $328,653,156,849,201,408 | - |
| Rates +50bp | $492,979,735,273,802,240 | +50.0% |
| Unwind top 5 | $319,907,633,607,072,576 | -2.7% |
| Add hedge | $326,888,028,107,727,040 | -0.5% |

**IM Ladder:** 0.5x: $164,326,578,424,600,704, 0.75x: $246,489,867,636,901,120, 1.0x: $328,653,156,849,201,408, 1.25x: $410,816,446,061,501,824, 1.5x: $492,979,735,273,802,240

### 5:00 PM EOD Optimization

- Initial IM: $328,653,156,849,201,408
- Final IM: $964,636,807,370,259,072 (reduction: -193.5%)
- Trades moved: 1243, Iterations: 100

---

## Run: 2026-02-02 13:14:04

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 50,000 |
| Portfolios | 100 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 435 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 661 us | 377.90 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 5.52 ms | 3.84 ms | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.03 ms | 5.78 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 4.66 ms | 7.91 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 46.508 s | 46.623 s | 101 | 101 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 38.59 ms |
| Total AADC evals | 115 |
| Total kernel reuses | 115 |
| Total GPU evals | 115 |
| Amortized recording/eval | 0.34 ms |
| Cumulative AADC time | 46.521 s |
| Cumulative GPU time | 47.018 s |
| AADC total (rec + eval) | 46.560 s |
| GPU speedup (eval only) | 1.0x |
| GPU speedup (inc. recording) | 1.0x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $2,831,979,561,557,022,208 | - |
| Rates +50bp | $4,247,969,342,335,533,056 | +50.0% |
| Unwind top 5 | $2,814,254,583,425,629,696 | -0.6% |
| Add hedge | $2,828,408,421,928,954,880 | -0.1% |

**IM Ladder:** 0.5x: $1,415,989,780,778,511,104, 0.75x: $2,123,984,671,167,766,528, 1.0x: $2,831,979,561,557,022,208, 1.25x: $3,539,974,451,946,277,888, 1.5x: $4,247,969,342,335,533,056

### 5:00 PM EOD Optimization

- Initial IM: $2,831,979,561,557,022,208
- Final IM: $15,581,728,096,112,656,384 (reduction: -450.2%)
- Trades moved: 11512, Iterations: 100

---

## Run: 2026-02-02 13:35:11

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 10 |
| Optimize iterations | 20 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 673 us | 860.65 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 29 us | 12 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 916 us | 1.11 ms | 1 | 1 |
| 2:00 PM What-If Scenarios | 4.31 ms | 7.29 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 25.18 ms | 35.17 ms | 21 | 21 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 39.68 ms |
| Total AADC evals | 31 |
| Total kernel reuses | 31 |
| Total GPU evals | 31 |
| Amortized recording/eval | 1.28 ms |
| Cumulative AADC time | 31.11 ms |
| Cumulative GPU time | 904.25 ms |
| AADC total (rec + eval) | 70.79 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $284,409,381,844,208 | - |
| Rates +50bp | $426,614,072,766,312 | +50.0% |
| Unwind top 5 | $289,583,247,749,459 | +1.8% |
| Add hedge | $250,624,559,774,010 | -11.9% |

**IM Ladder:** 0.5x: $142,204,690,922,104, 0.75x: $213,307,036,383,156, 1.0x: $284,409,381,844,208, 1.25x: $355,511,727,305,260, 1.5x: $426,614,072,766,312

### 5:00 PM EOD Optimization

- Initial IM: $284,409,381,844,208
- Final IM: $284,409,381,844,208 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 13:36:56

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 10,000 |
| Portfolios | 20 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 691 us | 739.77 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 1.45 ms | 660 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.17 ms | 5.28 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 4.68 ms | 7.43 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 7.768 s | 7.620 s | 101 | 101 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| Recording cost (1-time) | 40.54 ms |
| Total AADC evals | 115 |
| Total kernel reuses | 115 |
| Total GPU evals | 115 |
| Amortized recording/eval | 0.35 ms |
| Cumulative AADC time | 7.778 s |
| Cumulative GPU time | 8.373 s |
| AADC total (rec + eval) | 7.819 s |
| GPU speedup (eval only) | 0.9x |
| GPU speedup (inc. recording) | 0.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $296,933,058,957,277,248 | - |
| Rates +50bp | $445,399,588,435,915,840 | +50.0% |
| Unwind top 5 | $288,254,831,038,525,504 | -2.9% |
| Add hedge | $295,176,445,769,667,648 | -0.6% |

**IM Ladder:** 0.5x: $148,466,529,478,638,624, 0.75x: $222,699,794,217,957,920, 1.0x: $296,933,058,957,277,248, 1.25x: $371,166,323,696,596,608, 1.5x: $445,399,588,435,915,840

### 5:00 PM EOD Optimization

- Initial IM: $296,933,058,957,277,248
- Final IM: $296,933,058,957,277,248 (reduction: 0.0%)
- Trades moved: 0, Iterations: 100

---

## Run: 2026-02-02 13:46:48

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 5 |
| Optimize iterations | 10 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 421 us | 662.71 ms | 35.74 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 28 us | 12 us | 150 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 384 us | 1.09 ms | 190 us | 1 | 1 |
| 2:00 PM What-If Scenarios | 2.88 ms | 7.41 ms | 0 us | 8 | 8 |
| 5:00 PM EOD Optimization | 11.60 ms | 18.55 ms | 770 us | 11 | 11 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.76 ms |
| Total AADC Py evals | 21 |
| Total kernel reuses | 21 |
| Total GPU evals | 21 |
| Amortized recording/eval | 1.80 ms |
| Cumulative AADC Py time | 15.32 ms |
| Cumulative GPU time | 689.78 ms |
| AADC Py total (rec + eval) | 53.07 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |
| C++ AADC recording (1-time) | 35.74 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 36.85 ms |
| C++ AADC total (rec + eval) | 72.59 ms |
| C++/Py AADC speedup (eval) | 0.4x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $284,409,381,844,208 | - |
| Rates +50bp | $426,614,072,766,312 | +50.0% |
| Unwind top 5 | $289,583,247,749,459 | +1.8% |
| Add hedge | $250,624,559,774,010 | -11.9% |

**IM Ladder:** 0.5x: $142,204,690,922,104, 0.75x: $213,307,036,383,156, 1.0x: $284,409,381,844,208, 1.25x: $355,511,727,305,260, 1.5x: $426,614,072,766,312

### 5:00 PM EOD Optimization

- Initial IM: $284,409,381,844,208
- Final IM: $284,409,381,844,208 (reduction: 0.0%)
- Trades moved: 0, Iterations: 10

---

## Run: 2026-02-02 13:46:56

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 5 |
| Optimize iterations | 10 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 336 us | 658.49 ms | 34.43 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 30 us | 13 us | 160 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 398 us | 1.05 ms | 210 us | 1 | 1 |
| 2:00 PM What-If Scenarios | 3.00 ms | 7.22 ms | 0 us | 8 | 8 |
| 5:00 PM EOD Optimization | 12.23 ms | 18.79 ms | 850 us | 11 | 11 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.76 ms |
| Total AADC Py evals | 21 |
| Total kernel reuses | 21 |
| Total GPU evals | 21 |
| Amortized recording/eval | 1.80 ms |
| Cumulative AADC Py time | 16.00 ms |
| Cumulative GPU time | 685.57 ms |
| AADC Py total (rec + eval) | 53.76 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |
| C++ AADC recording (1-time) | 34.43 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 35.65 ms |
| C++ AADC total (rec + eval) | 70.08 ms |
| C++/Py AADC speedup (eval) | 0.4x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $284,409,381,844,208 | - |
| Rates +50bp | $426,614,072,766,312 | +50.0% |
| Unwind top 5 | $289,583,247,749,459 | +1.8% |
| Add hedge | $250,624,559,774,010 | -11.9% |

**IM Ladder:** 0.5x: $142,204,690,922,104, 0.75x: $213,307,036,383,156, 1.0x: $284,409,381,844,208, 1.25x: $355,511,727,305,260, 1.5x: $426,614,072,766,312

### 5:00 PM EOD Optimization

- Initial IM: $284,409,381,844,208
- Final IM: $284,409,381,844,208 (reduction: 0.0%)
- Trades moved: 0, Iterations: 10

---

## Run: 2026-02-02 13:54:09

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 5 |
| Optimize iterations | 10 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 372 us | 666.63 ms | 7.10 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 29 us | 12 us | 40 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 399 us | 1.05 ms | 70 us | 1 | 1 |
| 2:00 PM What-If Scenarios | 2.49 ms | 7.21 ms | 250 us | 8 | 8 |
| 5:00 PM EOD Optimization | 11.59 ms | 18.25 ms | 90 us | 11 | 11 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.62 ms |
| Total AADC Py evals | 21 |
| Total kernel reuses | 21 |
| Total GPU evals | 21 |
| Amortized recording/eval | 1.84 ms |
| Cumulative AADC Py time | 14.88 ms |
| Cumulative GPU time | 693.16 ms |
| AADC Py total (rec + eval) | 53.50 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |
| C++ AADC recording (1-time) | 7.10 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 7.55 ms |
| C++ AADC total (rec + eval) | 14.65 ms |
| C++/Py AADC speedup (eval) | 2.0x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $284,409,381,844,208 | - |
| Rates +50bp | $426,614,072,766,312 | +50.0% |
| Unwind top 5 | $289,583,247,749,459 | +1.8% |
| Add hedge | $250,624,559,774,010 | -11.9% |

**IM Ladder:** 0.5x: $142,204,690,922,104, 0.75x: $213,307,036,383,156, 1.0x: $284,409,381,844,208, 1.25x: $355,511,727,305,260, 1.5x: $426,614,072,766,312

### 5:00 PM EOD Optimization

- Initial IM: $284,409,381,844,208
- Final IM: $284,409,381,844,208 (reduction: 0.0%)
- Trades moved: 0, Iterations: 10

---

## Run: 2026-02-02 13:54:24

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 500 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 20 |
| Optimize iterations | 20 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 507 us | 667.27 ms | 7.20 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 56 us | 32 us | 250 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 1.23 ms | 2.10 ms | 700 us | 2 | 2 |
| 2:00 PM What-If Scenarios | 4.18 ms | 7.12 ms | 600 us | 8 | 8 |
| 5:00 PM EOD Optimization | 87.83 ms | 96.49 ms | 2.73 ms | 21 | 21 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 29.51 ms |
| Total AADC Py evals | 32 |
| Total kernel reuses | 32 |
| Total GPU evals | 32 |
| Amortized recording/eval | 0.92 ms |
| Cumulative AADC Py time | 93.80 ms |
| Cumulative GPU time | 773.01 ms |
| AADC Py total (rec + eval) | 123.31 ms |
| GPU speedup (eval only) | 0.1x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 7.20 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 11.48 ms |
| C++ AADC total (rec + eval) | 18.68 ms |
| C++/Py AADC speedup (eval) | 8.2x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $4,398,050,237,838,401 | - |
| Rates +50bp | $6,597,075,356,757,600 | +50.0% |
| Unwind top 5 | $3,244,384,055,266,466 | -26.2% |
| Add hedge | $4,189,578,842,189,735 | -4.7% |

**IM Ladder:** 0.5x: $2,199,025,118,919,200, 0.75x: $3,298,537,678,378,800, 1.0x: $4,398,050,237,838,401, 1.25x: $5,497,562,797,298,002, 1.5x: $6,597,075,356,757,600

### 5:00 PM EOD Optimization

- Initial IM: $4,398,050,237,838,401
- Final IM: $4,398,050,237,838,401 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 13:58:57

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 500 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 20 |
| Optimize iterations | 20 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 655 us | 690.55 ms | 7.48 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 58 us | 32 us | 330 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 1.08 ms | 2.14 ms | 460 us | 2 | 2 |
| 2:00 PM What-If Scenarios | 4.54 ms | 7.19 ms | 660 us | 8 | 8 |
| 5:00 PM EOD Optimization | 98.29 ms | 116.67 ms | 2.63 ms | 41 | 41 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.83 ms |
| Total AADC Py evals | 52 |
| Total kernel reuses | 52 |
| Total GPU evals | 52 |
| Amortized recording/eval | 0.73 ms |
| Cumulative AADC Py time | 104.62 ms |
| Cumulative GPU time | 816.57 ms |
| AADC Py total (rec + eval) | 142.46 ms |
| GPU speedup (eval only) | 0.1x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 7.48 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 11.56 ms |
| C++ AADC total (rec + eval) | 19.04 ms |
| C++/Py AADC speedup (eval) | 9.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $4,398,050,237,838,401 | - |
| Rates +50bp | $6,597,075,356,757,600 | +50.0% |
| Unwind top 5 | $3,244,384,055,266,466 | -26.2% |
| Add hedge | $4,189,578,842,189,735 | -4.7% |

**IM Ladder:** 0.5x: $2,199,025,118,919,200, 0.75x: $3,298,537,678,378,800, 1.0x: $4,398,050,237,838,401, 1.25x: $5,497,562,797,298,002, 1.5x: $6,597,075,356,757,600

### 5:00 PM EOD Optimization

- Initial IM: $4,398,050,237,838,401
- Final IM: $4,398,050,237,838,401 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 13:59:41

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 20 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 598 us | 667.75 ms | 4.50 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 28 us | 12 us | 180 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.84 ms | 4.86 ms | 260 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 6.32 ms | 7.25 ms | 520 us | 8 | 8 |
| 5:00 PM EOD Optimization | 4.17 ms | 6.56 ms | 200 us | 5 | 5 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.95 ms |
| Total AADC Py evals | 19 |
| Total kernel reuses | 19 |
| Total GPU evals | 19 |
| Amortized recording/eval | 2.00 ms |
| Cumulative AADC Py time | 13.95 ms |
| Cumulative GPU time | 686.42 ms |
| AADC Py total (rec + eval) | 51.90 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |
| C++ AADC recording (1-time) | 4.50 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 5.66 ms |
| C++ AADC total (rec + eval) | 10.16 ms |
| C++/Py AADC speedup (eval) | 2.5x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $284,409,381,844,208 | - |
| Rates +50bp | $426,614,072,766,312 | +50.0% |
| Unwind top 5 | $289,583,247,749,459 | +1.8% |
| Add hedge | $250,624,559,774,010 | -11.9% |

**IM Ladder:** 0.5x: $142,204,690,922,104, 0.75x: $213,307,036,383,156, 1.0x: $284,409,381,844,208, 1.25x: $355,511,727,305,260, 1.5x: $426,614,072,766,312

### 5:00 PM EOD Optimization

- Initial IM: $284,409,381,844,208
- Final IM: $284,409,381,844,208 (reduction: 0.0%)
- Trades moved: 0, Iterations: 2

---

## Run: 2026-02-02 14:03:48

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 50,000 |
| Portfolios | 20 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 48 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 2.93 ms | 1.322 s | 7.22 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 4.38 ms | 12.17 ms | 12.78 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 11.87 ms | 5.54 ms | 45.80 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 15.32 ms | 7.34 ms | 160.26 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 35.789 s | 35.034 s | 302.87 ms | 141 | 141 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 39.31 ms |
| Total AADC Py evals | 155 |
| Total kernel reuses | 155 |
| Total GPU evals | 155 |
| Amortized recording/eval | 0.25 ms |
| Cumulative AADC Py time | 35.824 s |
| Cumulative GPU time | 36.381 s |
| AADC Py total (rec + eval) | 35.863 s |
| GPU speedup (eval only) | 1.0x |
| GPU speedup (inc. recording) | 1.0x |
| C++ AADC recording (1-time) | 7.22 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 528.93 ms |
| C++ AADC total (rec + eval) | 536.15 ms |
| C++/Py AADC speedup (eval) | 67.7x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $2,599,130,598,717,363,712 | - |
| Rates +50bp | $3,898,695,898,076,045,312 | +50.0% |
| Unwind top 5 | $2,581,630,767,278,885,376 | -0.7% |
| Add hedge | $2,595,616,935,591,112,704 | -0.1% |

**IM Ladder:** 0.5x: $1,299,565,299,358,681,856, 0.75x: $1,949,347,949,038,022,656, 1.0x: $2,599,130,598,717,363,712, 1.25x: $3,248,913,248,396,704,256, 1.5x: $3,898,695,898,076,045,312

### 5:00 PM EOD Optimization

- Initial IM: $2,599,130,598,717,363,712
- Final IM: $2,599,130,598,717,363,712 (reduction: 0.0%)
- Trades moved: 0, Iterations: 49

---

## Run: 2026-02-02 14:08:15

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 20 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 750 us | 682.01 ms | 310 us | 1 | 1 |
| 8:00 AM Margin Attribution | 30 us | 12 us | 310 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.18 ms | 4.83 ms | 270 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 4.05 ms | 7.19 ms | 510 us | 8 | 8 |
| 5:00 PM EOD Optimization | 4.48 ms | 6.59 ms | 80 us | 5 | 5 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.72 ms |
| Total AADC Py evals | 19 |
| Total kernel reuses | 19 |
| Total GPU evals | 19 |
| Amortized recording/eval | 1.99 ms |
| Cumulative AADC Py time | 12.49 ms |
| Cumulative GPU time | 700.63 ms |
| AADC Py total (rec + eval) | 50.21 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |
| C++ AADC recording (1-time) | 6.96 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 1.48 ms |
| C++ AADC total (rec + eval) | 8.44 ms |
| C++/Py AADC speedup (eval) | 8.4x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $284,409,381,844,208 | - |
| Rates +50bp | $426,614,072,766,312 | +50.0% |
| Unwind top 5 | $289,583,247,749,459 | +1.8% |
| Add hedge | $250,624,559,774,010 | -11.9% |

**IM Ladder:** 0.5x: $142,204,690,922,104, 0.75x: $213,307,036,383,156, 1.0x: $284,409,381,844,208, 1.25x: $355,511,727,305,260, 1.5x: $426,614,072,766,312

### 5:00 PM EOD Optimization

- Initial IM: $284,409,381,844,208
- Final IM: $284,409,381,844,208 (reduction: 0.0%)
- Trades moved: 0, Iterations: 2

---

## Run: 2026-02-02 14:21:27

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100,000 |
| Portfolios | 100 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 20 |
| Optimize iterations | 20 |
| Threads | 48 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 6.65 ms | 766.29 ms | 78.29 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 8.88 ms | 5.67 ms | 78.29 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 8.45 ms | 2.67 ms | 242.32 ms | 2 | 2 |
| 2:00 PM What-If Scenarios | 17.61 ms | 7.68 ms | 174.29 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 70.663 s | 69.628 s | 2.625 s | 97 | 97 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.78 ms |
| Total AADC Py evals | 108 |
| Total kernel reuses | 108 |
| Total GPU evals | 108 |
| Amortized recording/eval | 0.36 ms |
| Cumulative AADC Py time | 70.705 s |
| Cumulative GPU time | 70.410 s |
| AADC Py total (rec + eval) | 70.744 s |
| GPU speedup (eval only) | 1.0x |
| GPU speedup (inc. recording) | 1.0x |
| C++ AADC recording (1-time) | 6.87 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 3.198 s |
| C++ AADC total (rec + eval) | 3.205 s |
| C++/Py AADC speedup (eval) | 22.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $7,355,184,992,969,554,944 | - |
| Rates +50bp | $11,032,777,489,454,333,952 | +50.0% |
| Unwind top 5 | $7,330,605,373,814,977,536 | -0.3% |
| Add hedge | $7,350,244,320,416,998,400 | -0.1% |

**IM Ladder:** 0.5x: $3,677,592,496,484,777,472, 0.75x: $5,516,388,744,727,166,976, 1.0x: $7,355,184,992,969,554,944, 1.25x: $9,193,981,241,211,942,912, 1.5x: $11,032,777,489,454,333,952

### 5:00 PM EOD Optimization

- Initial IM: $7,355,184,992,969,554,944
- Final IM: $7,355,184,992,969,554,944 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 14:29:58

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100,000 |
| Portfolios | 300 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 20 |
| Optimize iterations | 20 |
| Threads | 48 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 3.74 ms | 605.43 ms | 42.98 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 8.82 ms | 6.58 ms | 42.98 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 6.04 ms | 2.77 ms | 434.19 ms | 2 | 2 |
| 2:00 PM What-If Scenarios | 16.74 ms | 8.72 ms | 200.41 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 129.495 s | 129.465 s | 1.042 s | 119 | 119 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 39.03 ms |
| Total AADC Py evals | 130 |
| Total kernel reuses | 130 |
| Total GPU evals | 130 |
| Amortized recording/eval | 0.30 ms |
| Cumulative AADC Py time | 129.530 s |
| Cumulative GPU time | 130.088 s |
| AADC Py total (rec + eval) | 129.569 s |
| GPU speedup (eval only) | 1.0x |
| GPU speedup (inc. recording) | 1.0x |
| C++ AADC recording (1-time) | 6.88 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 1.763 s |
| C++ AADC total (rec + eval) | 1.770 s |
| C++/Py AADC speedup (eval) | 73.5x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $8,079,489,651,269,530,624 | - |
| Rates +50bp | $12,119,234,476,904,294,400 | +50.0% |
| Unwind top 5 | $8,055,057,130,280,869,888 | -0.3% |
| Add hedge | $8,074,526,997,021,607,936 | -0.1% |

**IM Ladder:** 0.5x: $4,039,744,825,634,765,312, 0.75x: $6,059,617,238,452,147,200, 1.0x: $8,079,489,651,269,530,624, 1.25x: $10,099,362,064,086,913,024, 1.5x: $12,119,234,476,904,294,400

### 5:00 PM EOD Optimization

- Initial IM: $8,079,489,651,269,530,624
- Final IM: $8,079,489,651,269,530,624 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 14:33:46

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 500 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 431 us | 676.22 ms | 500 us | 1 | 1 |
| 8:00 AM Margin Attribution | 60 us | 33 us | 500 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 1.95 ms | 4.80 ms | 870 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.04 ms | 7.15 ms | 1.08 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 438.07 ms | 554.28 ms | 1.41 ms | 201 | 201 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.38 ms |
| Total AADC Py evals | 215 |
| Total kernel reuses | 215 |
| Total GPU evals | 215 |
| Amortized recording/eval | 0.18 ms |
| Cumulative AADC Py time | 443.55 ms |
| Cumulative GPU time | 1.242 s |
| AADC Py total (rec + eval) | 481.94 ms |
| GPU speedup (eval only) | 0.4x |
| GPU speedup (inc. recording) | 0.4x |
| C++ AADC recording (1-time) | 25.73 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 4.36 ms |
| C++ AADC total (rec + eval) | 30.09 ms |
| C++/Py AADC speedup (eval) | 101.7x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $4,398,050,237,838,401 | - |
| Rates +50bp | $6,597,075,356,757,600 | +50.0% |
| Unwind top 5 | $3,244,384,055,266,466 | -26.2% |
| Add hedge | $4,189,578,842,189,735 | -4.7% |

**IM Ladder:** 0.5x: $2,199,025,118,919,200, 0.75x: $3,298,537,678,378,800, 1.0x: $4,398,050,237,838,401, 1.25x: $5,497,562,797,298,002, 1.5x: $6,597,075,356,757,600

### 5:00 PM EOD Optimization

- Initial IM: $4,398,050,237,838,401
- Final IM: $4,398,050,237,838,401 (reduction: 0.0%)
- Trades moved: 0, Iterations: 100

---

## Run: 2026-02-02 14:41:29

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 500 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 20 |
| Optimize iterations | 20 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 622 us | 669.27 ms | 500 us | 1 | 1 |
| 8:00 AM Margin Attribution | 52 us | 31 us | 500 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 1.43 ms | 2.16 ms | 460 us | 2 | 2 |
| 2:00 PM What-If Scenarios | 4.07 ms | 7.14 ms | 540 us | 8 | 8 |
| 5:00 PM EOD Optimization | 94.91 ms | 113.64 ms | 2.70 ms | 41 | 41 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.57 ms |
| Total AADC Py evals | 52 |
| Total kernel reuses | 52 |
| Total GPU evals | 52 |
| Amortized recording/eval | 0.72 ms |
| Cumulative AADC Py time | 101.09 ms |
| Cumulative GPU time | 792.24 ms |
| AADC Py total (rec + eval) | 138.66 ms |
| GPU speedup (eval only) | 0.1x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 22.70 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 4.70 ms |
| C++ AADC total (rec + eval) | 27.40 ms |
| C++/Py AADC speedup (eval) | 21.5x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $4,398,050,237,838,401 | - |
| Rates +50bp | $6,597,075,356,757,600 | +50.0% |
| Unwind top 5 | $3,244,384,055,266,466 | -26.2% |
| Add hedge | $4,189,578,842,189,735 | -4.7% |

**IM Ladder:** 0.5x: $2,199,025,118,919,200, 0.75x: $3,298,537,678,378,800, 1.0x: $4,398,050,237,838,401, 1.25x: $5,497,562,797,298,002, 1.5x: $6,597,075,356,757,600

### 5:00 PM EOD Optimization

- Initial IM: $4,398,050,237,838,401
- Final IM: $4,398,050,237,838,401 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 14:41:51

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 5,000 |
| Portfolios | 10 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 20 |
| Optimize iterations | 20 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 1.29 ms | 696.11 ms | 1.15 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 851 us | 335 us | 1.15 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.20 ms | 2.39 ms | 2.55 ms | 2 | 2 |
| 2:00 PM What-If Scenarios | 6.23 ms | 7.49 ms | 2.56 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 818.56 ms | 827.68 ms | 9.09 ms | 41 | 41 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.47 ms |
| Total AADC Py evals | 52 |
| Total kernel reuses | 52 |
| Total GPU evals | 52 |
| Amortized recording/eval | 0.74 ms |
| Cumulative AADC Py time | 829.12 ms |
| Cumulative GPU time | 1.534 s |
| AADC Py total (rec + eval) | 867.59 ms |
| GPU speedup (eval only) | 0.5x |
| GPU speedup (inc. recording) | 0.6x |
| C++ AADC recording (1-time) | 20.22 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 16.50 ms |
| C++ AADC total (rec + eval) | 36.72 ms |
| C++/Py AADC speedup (eval) | 50.2x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $105,959,746,066,461,936 | - |
| Rates +50bp | $158,939,619,099,692,928 | +50.0% |
| Unwind top 5 | $99,862,008,924,674,672 | -5.8% |
| Add hedge | $104,727,051,154,147,104 | -1.2% |

**IM Ladder:** 0.5x: $52,979,873,033,230,968, 0.75x: $79,469,809,549,846,464, 1.0x: $105,959,746,066,461,936, 1.25x: $132,449,682,583,077,424, 1.5x: $158,939,619,099,692,928

### 5:00 PM EOD Optimization

- Initial IM: $105,959,746,066,461,936
- Final IM: $105,959,746,066,461,936 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 14:43:08

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 10,000 |
| Portfolios | 30 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 20 |
| Optimize iterations | 20 |
| Threads | 48 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 3.32 ms | 707.03 ms | 4.07 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 1.55 ms | 662 us | 4.07 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 6.27 ms | 2.18 ms | 15.08 ms | 2 | 2 |
| 2:00 PM What-If Scenarios | 14.65 ms | 7.33 ms | 7.01 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 2.982 s | 2.917 s | 38.46 ms | 58 | 58 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.42 ms |
| Total AADC Py evals | 69 |
| Total kernel reuses | 69 |
| Total GPU evals | 69 |
| Amortized recording/eval | 0.56 ms |
| Cumulative AADC Py time | 3.008 s |
| Cumulative GPU time | 3.634 s |
| AADC Py total (rec + eval) | 3.046 s |
| GPU speedup (eval only) | 0.8x |
| GPU speedup (inc. recording) | 0.8x |
| C++ AADC recording (1-time) | 22.33 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 68.69 ms |
| C++ AADC total (rec + eval) | 91.02 ms |
| C++/Py AADC speedup (eval) | 43.8x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $311,313,243,319,469,248 | - |
| Rates +50bp | $466,969,864,979,203,904 | +50.0% |
| Unwind top 5 | $302,584,268,503,425,664 | -2.8% |
| Add hedge | $309,538,174,543,890,496 | -0.6% |

**IM Ladder:** 0.5x: $155,656,621,659,734,624, 0.75x: $233,484,932,489,601,952, 1.0x: $311,313,243,319,469,248, 1.25x: $389,141,554,149,336,512, 1.5x: $466,969,864,979,203,904

### 5:00 PM EOD Optimization

- Initial IM: $311,313,243,319,469,248
- Final IM: $311,313,243,319,469,248 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 14:46:09

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 5,000 |
| Portfolios | 10 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 20 |
| Optimize iterations | 20 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 1.04 ms | 694.65 ms | 2.85 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 732 us | 336 us | 2.85 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.53 ms | 2.26 ms | 3.16 ms | 2 | 2 |
| 2:00 PM What-If Scenarios | 6.74 ms | 7.48 ms | 5.59 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 793.45 ms | 795.48 ms | 17.54 ms | 41 | 41 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.40 ms |
| Total AADC Py evals | 52 |
| Total kernel reuses | 52 |
| Total GPU evals | 52 |
| Amortized recording/eval | 0.74 ms |
| Cumulative AADC Py time | 804.49 ms |
| Cumulative GPU time | 1.500 s |
| AADC Py total (rec + eval) | 842.89 ms |
| GPU speedup (eval only) | 0.5x |
| GPU speedup (inc. recording) | 0.6x |
| C++ AADC recording (1-time) | 24.40 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 31.99 ms |
| C++ AADC total (rec + eval) | 56.39 ms |
| C++/Py AADC speedup (eval) | 25.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $105,959,746,066,461,936 | - |
| Rates +50bp | $158,939,619,099,692,928 | +50.0% |
| Unwind top 5 | $99,862,008,924,674,672 | -5.8% |
| Add hedge | $104,727,051,154,147,104 | -1.2% |

**IM Ladder:** 0.5x: $52,979,873,033,230,968, 0.75x: $79,469,809,549,846,464, 1.0x: $105,959,746,066,461,936, 1.25x: $132,449,682,583,077,424, 1.5x: $158,939,619,099,692,928

### 5:00 PM EOD Optimization

- Initial IM: $105,959,746,066,461,936
- Final IM: $105,959,746,066,461,936 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 15:03:29

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 10 |
| Optimize iterations | 20 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 337 us | 670.83 ms | 250 us | 1 | 1 |
| 8:00 AM Margin Attribution | 27 us | 12 us | 250 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 412 us | 1.10 ms | 70 us | 1 | 1 |
| 2:00 PM What-If Scenarios | 2.67 ms | 7.73 ms | 170 us | 8 | 8 |
| 5:00 PM EOD Optimization | 26.79 ms | 54.31 ms | 240 us | 41 | 41 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 39.58 ms |
| Total AADC Py evals | 51 |
| Total kernel reuses | 51 |
| Total GPU evals | 51 |
| Amortized recording/eval | 0.78 ms |
| Cumulative AADC Py time | 30.23 ms |
| Cumulative GPU time | 733.99 ms |
| AADC Py total (rec + eval) | 69.81 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |
| C++ AADC recording (1-time) | 14.15 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 980 us |
| C++ AADC total (rec + eval) | 15.13 ms |
| C++/Py AADC speedup (eval) | 30.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $284,409,381,844,208 | - |
| Rates +50bp | $426,614,072,766,312 | +50.0% |
| Unwind top 5 | $289,583,247,749,459 | +1.8% |
| Add hedge | $250,624,559,774,010 | -11.9% |

**IM Ladder:** 0.5x: $142,204,690,922,104, 0.75x: $213,307,036,383,156, 1.0x: $284,409,381,844,208, 1.25x: $355,511,727,305,260, 1.5x: $426,614,072,766,312

### 5:00 PM EOD Optimization

- Initial IM: $284,409,381,844,208
- Final IM: $284,409,381,844,208 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 15:04:12

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 50,000 |
| Portfolios | 200 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 48 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 3.13 ms | 1.266 s | 49.40 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 6.01 ms | 3.83 ms | 49.40 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 14.34 ms | 6.16 ms | 268.42 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 17.31 ms | 8.25 ms | 140.05 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 245.835 s | 246.340 s | 2.216 s | 523 | 523 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.48 ms |
| Total AADC Py evals | 537 |
| Total kernel reuses | 537 |
| Total GPU evals | 537 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 245.876 s |
| Cumulative GPU time | 247.625 s |
| AADC Py total (rec + eval) | 245.915 s |
| GPU speedup (eval only) | 1.0x |
| GPU speedup (inc. recording) | 1.0x |
| C++ AADC recording (1-time) | 17.37 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 2.723 s |
| C++ AADC total (rec + eval) | 2.740 s |
| C++/Py AADC speedup (eval) | 90.3x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $3,184,713,831,084,513,280 | - |
| Rates +50bp | $4,777,070,746,626,769,920 | +50.0% |
| Unwind top 5 | $3,166,951,699,865,585,664 | -0.6% |
| Add hedge | $3,181,121,727,915,088,896 | -0.1% |

**IM Ladder:** 0.5x: $1,592,356,915,542,256,640, 0.75x: $2,388,535,373,313,384,960, 1.0x: $3,184,713,831,084,513,280, 1.25x: $3,980,892,288,855,642,112, 1.5x: $4,777,070,746,626,769,920

### 5:00 PM EOD Optimization

- Initial IM: $3,184,713,831,084,513,280
- Final IM: $3,184,713,831,084,513,280 (reduction: 0.0%)
- Trades moved: 0, Iterations: 83

---

## Run: 2026-02-02 15:04:23

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 10 |
| Optimize iterations | 20 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 367 us | 668.11 ms | 300 us | 1 | 1 |
| 8:00 AM Margin Attribution | 28 us | 11 us | 300 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 469 us | 1.09 ms | 110 us | 1 | 1 |
| 2:00 PM What-If Scenarios | 2.85 ms | 7.43 ms | 280 us | 8 | 8 |
| 5:00 PM EOD Optimization | 28.79 ms | 54.10 ms | 280 us | 41 | 41 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.73 ms |
| Total AADC Py evals | 51 |
| Total kernel reuses | 51 |
| Total GPU evals | 51 |
| Amortized recording/eval | 0.74 ms |
| Cumulative AADC Py time | 32.50 ms |
| Cumulative GPU time | 730.74 ms |
| AADC Py total (rec + eval) | 70.23 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |
| C++ AADC recording (1-time) | 25.78 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 1.27 ms |
| C++ AADC total (rec + eval) | 27.05 ms |
| C++/Py AADC speedup (eval) | 25.6x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $284,409,381,844,208 | - |
| Rates +50bp | $426,614,072,766,312 | +50.0% |
| Unwind top 5 | $289,583,247,749,459 | +1.8% |
| Add hedge | $250,624,559,774,010 | -11.9% |

**IM Ladder:** 0.5x: $142,204,690,922,104, 0.75x: $213,307,036,383,156, 1.0x: $284,409,381,844,208, 1.25x: $355,511,727,305,260, 1.5x: $426,614,072,766,312

### 5:00 PM EOD Optimization

- Initial IM: $284,409,381,844,208
- Final IM: $284,409,381,844,208 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 15:14:00

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 10 |
| Optimize iterations | 20 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 381 us | 669.61 ms | 170 us | 1 | 1 |
| 8:00 AM Margin Attribution | 26 us | 12 us | 170 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 428 us | 1.11 ms | 50 us | 1 | 1 |
| 2:00 PM What-If Scenarios | 2.78 ms | 7.52 ms | 140 us | 8 | 8 |
| 5:00 PM EOD Optimization | 142.62 ms | 53.89 ms | 210 us | 41 | 41 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 28.94 ms |
| Total AADC Py evals | 51 |
| Total kernel reuses | 51 |
| Total GPU evals | 51 |
| Amortized recording/eval | 0.57 ms |
| Cumulative AADC Py time | 146.24 ms |
| Cumulative GPU time | 732.13 ms |
| AADC Py total (rec + eval) | 175.18 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 14.99 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 740 us |
| C++ AADC total (rec + eval) | 15.73 ms |
| C++/Py AADC speedup (eval) | 197.6x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $284,409,381,844,208 | - |
| Rates +50bp | $426,614,072,766,312 | +50.0% |
| Unwind top 5 | $289,583,247,749,459 | +1.8% |
| Add hedge | $250,624,559,774,010 | -11.9% |

**IM Ladder:** 0.5x: $142,204,690,922,104, 0.75x: $213,307,036,383,156, 1.0x: $284,409,381,844,208, 1.25x: $355,511,727,305,260, 1.5x: $426,614,072,766,312

### 5:00 PM EOD Optimization

- Initial IM: $284,409,381,844,208
- Final IM: $284,409,381,844,208 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 15:14:33

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 10 |
| Optimize iterations | 20 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 395 us | 758.48 ms | 280 us | 1 | 1 |
| 8:00 AM Margin Attribution | 34 us | 12 us | 280 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 404 us | 1.18 ms | 90 us | 1 | 1 |
| 2:00 PM What-If Scenarios | 2.82 ms | 7.46 ms | 380 us | 8 | 8 |
| 5:00 PM EOD Optimization | 26.18 ms | 55.52 ms | 230 us | 41 | 41 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 28.66 ms |
| Total AADC Py evals | 51 |
| Total kernel reuses | 51 |
| Total GPU evals | 51 |
| Amortized recording/eval | 0.56 ms |
| Cumulative AADC Py time | 29.83 ms |
| Cumulative GPU time | 822.65 ms |
| AADC Py total (rec + eval) | 58.49 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.1x |
| C++ AADC recording (1-time) | 20.50 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 1.26 ms |
| C++ AADC total (rec + eval) | 21.76 ms |
| C++/Py AADC speedup (eval) | 23.7x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $284,409,381,844,208 | - |
| Rates +50bp | $426,614,072,766,312 | +50.0% |
| Unwind top 5 | $289,583,247,749,459 | +1.8% |
| Add hedge | $250,624,559,774,010 | -11.9% |

**IM Ladder:** 0.5x: $142,204,690,922,104, 0.75x: $213,307,036,383,156, 1.0x: $284,409,381,844,208, 1.25x: $355,511,727,305,260, 1.5x: $426,614,072,766,312

### 5:00 PM EOD Optimization

- Initial IM: $284,409,381,844,208
- Final IM: $284,409,381,844,208 (reduction: 0.0%)
- Trades moved: 0, Iterations: 20

---

## Run: 2026-02-02 15:15:01

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 500 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 13.08 ms | 699.50 ms | 500 us | 1 | 1 |
| 8:00 AM Margin Attribution | 62 us | 51 us | 500 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.78 ms | 4.90 ms | 480 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.28 ms | 7.32 ms | 790 us | 8 | 8 |
| 5:00 PM EOD Optimization | 477.80 ms | 562.26 ms | 2.15 ms | 201 | 201 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.13 ms |
| Total AADC Py evals | 215 |
| Total kernel reuses | 215 |
| Total GPU evals | 215 |
| Amortized recording/eval | 0.18 ms |
| Cumulative AADC Py time | 497.01 ms |
| Cumulative GPU time | 1.274 s |
| AADC Py total (rec + eval) | 535.14 ms |
| GPU speedup (eval only) | 0.4x |
| GPU speedup (inc. recording) | 0.4x |
| C++ AADC recording (1-time) | 26.34 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 4.42 ms |
| C++ AADC total (rec + eval) | 30.76 ms |
| C++/Py AADC speedup (eval) | 112.4x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $4,398,050,237,838,401 | - |
| Rates +50bp | $6,597,075,356,757,600 | +50.0% |
| Unwind top 5 | $3,244,384,055,266,466 | -26.2% |
| Add hedge | $4,189,578,842,189,735 | -4.7% |

**IM Ladder:** 0.5x: $2,199,025,118,919,200, 0.75x: $3,298,537,678,378,800, 1.0x: $4,398,050,237,838,401, 1.25x: $5,497,562,797,298,002, 1.5x: $6,597,075,356,757,600

### 5:00 PM EOD Optimization

- Initial IM: $4,398,050,237,838,401
- Final IM: $4,398,050,237,838,401 (reduction: 0.0%)
- Trades moved: 0, Iterations: 100

---

## Run: 2026-02-02 15:17:26

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 10,000 |
| Portfolios | 50 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 2.65 ms | 714.44 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 923 us | 697 us | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.95 ms | 5.80 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.93 ms | 7.45 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 23.448 s | 22.915 s | 320 | 320 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 29.74 ms |
| Total AADC Py evals | 334 |
| Total kernel reuses | 334 |
| Total GPU evals | 334 |
| Amortized recording/eval | 0.09 ms |
| Cumulative AADC Py time | 23.458 s |
| Cumulative GPU time | 23.643 s |
| AADC Py total (rec + eval) | 23.488 s |
| GPU speedup (eval only) | 1.0x |
| GPU speedup (inc. recording) | 1.0x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $328,653,156,849,201,408 | - |
| Rates +50bp | $492,979,735,273,802,240 | +50.0% |
| Unwind top 5 | $319,907,633,607,072,576 | -2.7% |
| Add hedge | $326,888,028,107,727,040 | -0.5% |

**IM Ladder:** 0.5x: $164,326,578,424,600,704, 0.75x: $246,489,867,636,901,120, 1.0x: $328,653,156,849,201,408, 1.25x: $410,816,446,061,501,824, 1.5x: $492,979,735,273,802,240

### 5:00 PM EOD Optimization

- Initial IM: $328,653,156,849,201,408
- Final IM: $328,653,156,849,201,408 (reduction: 0.0%)
- Trades moved: 0, Iterations: 83

---

## Run: 2026-02-02 15:17:29

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 50,000 |
| Portfolios | 200 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 48 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Time | GPU Time | Evals | Kernel Reuses |
|------|-----------|----------|-------|---------------|
| 7:00 AM Portfolio Setup | 3.09 ms | 1.256 s | 1 | 1 |
| 8:00 AM Margin Attribution | 5.93 ms | 3.27 ms | 0 | 0 |
| 9AM-4PM Intraday Pre-Trade | 13.23 ms | 5.99 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 16.67 ms | 8.07 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 249.848 s | 251.831 s | 523 | 523 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.48 ms |
| Total AADC Py evals | 537 |
| Total kernel reuses | 537 |
| Total GPU evals | 537 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 249.887 s |
| Cumulative GPU time | 253.105 s |
| AADC Py total (rec + eval) | 249.925 s |
| GPU speedup (eval only) | 1.0x |
| GPU speedup (inc. recording) | 1.0x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $3,184,713,831,084,513,280 | - |
| Rates +50bp | $4,777,070,746,626,769,920 | +50.0% |
| Unwind top 5 | $3,166,951,699,865,585,664 | -0.6% |
| Add hedge | $3,181,121,727,915,088,896 | -0.1% |

**IM Ladder:** 0.5x: $1,592,356,915,542,256,640, 0.75x: $2,388,535,373,313,384,960, 1.0x: $3,184,713,831,084,513,280, 1.25x: $3,980,892,288,855,642,112, 1.5x: $4,777,070,746,626,769,920

### 5:00 PM EOD Optimization

- Initial IM: $3,184,713,831,084,513,280
- Final IM: $3,184,713,831,084,513,280 (reduction: 0.0%)
- Trades moved: 0, Iterations: 83

---

## Run: 2026-02-02 15:21:27

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 10,000 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 611 us | 711.42 ms | 5.36 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 1.47 ms | 696 us | 5.36 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.89 ms | 5.03 ms | 4.98 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 4.23 ms | 7.19 ms | 15.75 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 7.478 s | 7.520 s | 93.59 ms | 201 | 201 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.87 ms |
| Total AADC Py evals | 215 |
| Total kernel reuses | 215 |
| Total GPU evals | 215 |
| Amortized recording/eval | 0.18 ms |
| Cumulative AADC Py time | 7.487 s |
| Cumulative GPU time | 8.245 s |
| AADC Py total (rec + eval) | 7.526 s |
| GPU speedup (eval only) | 0.9x |
| GPU speedup (inc. recording) | 0.9x |
| C++ AADC recording (1-time) | 23.21 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 125.04 ms |
| C++ AADC total (rec + eval) | 148.25 ms |
| C++/Py AADC speedup (eval) | 59.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $290,314,743,418,151,680 | - |
| Rates +50bp | $435,472,115,127,227,392 | +50.0% |
| Unwind top 5 | $281,581,699,508,929,152 | -3.0% |
| Add hedge | $288,562,870,451,485,632 | -0.6% |

**IM Ladder:** 0.5x: $145,157,371,709,075,840, 0.75x: $217,736,057,563,613,696, 1.0x: $290,314,743,418,151,680, 1.25x: $362,893,429,272,689,472, 1.5x: $435,472,115,127,227,392

### 5:00 PM EOD Optimization

- Initial IM: $290,314,743,418,151,680
- Final IM: $290,314,743,418,151,680 (reduction: 0.0%)
- Trades moved: 0, Iterations: 100

---

## Run: 2026-02-02 15:23:34

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 10,000 |
| Portfolios | 50 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 714 us | 708.14 ms | 2.44 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 1.56 ms | 704 us | 2.44 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 9.13 ms | 5.41 ms | 24.25 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 2.97 ms | 7.22 ms | 6.12 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 22.305 s | 22.403 s | 71.52 ms | 320 | 320 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 39.14 ms |
| Total AADC Py evals | 334 |
| Total kernel reuses | 334 |
| Total GPU evals | 334 |
| Amortized recording/eval | 0.12 ms |
| Cumulative AADC Py time | 22.320 s |
| Cumulative GPU time | 23.124 s |
| AADC Py total (rec + eval) | 22.359 s |
| GPU speedup (eval only) | 1.0x |
| GPU speedup (inc. recording) | 1.0x |
| C++ AADC recording (1-time) | 15.50 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 106.77 ms |
| C++ AADC total (rec + eval) | 122.27 ms |
| C++/Py AADC speedup (eval) | 209.0x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $328,653,156,849,201,408 | - |
| Rates +50bp | $492,979,735,273,802,240 | +50.0% |
| Unwind top 5 | $319,907,633,607,072,576 | -2.7% |
| Add hedge | $326,888,028,107,727,040 | -0.5% |

**IM Ladder:** 0.5x: $164,326,578,424,600,704, 0.75x: $246,489,867,636,901,120, 1.0x: $328,653,156,849,201,408, 1.25x: $410,816,446,061,501,824, 1.5x: $492,979,735,273,802,240

### 5:00 PM EOD Optimization

- Initial IM: $328,653,156,849,201,408
- Final IM: $328,653,156,849,201,408 (reduction: 0.0%)
- Trades moved: 0, Iterations: 83

---

## Run: 2026-02-02 15:25:56

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 10,000 |
| Portfolios | 50 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 501 us | 733.05 ms | 1.66 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 1.54 ms | 671 us | 1.66 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.29 ms | 5.59 ms | 15.49 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.51 ms | 7.40 ms | 4.73 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 97.71 ms | 64.21 ms | 59.20 ms | 5 | 5 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 27.87 ms |
| Total AADC Py evals | 19 |
| Total kernel reuses | 19 |
| Total GPU evals | 19 |
| Amortized recording/eval | 1.47 ms |
| Cumulative AADC Py time | 105.55 ms |
| Cumulative GPU time | 810.92 ms |
| AADC Py total (rec + eval) | 133.43 ms |
| GPU speedup (eval only) | 0.1x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 6.68 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 82.74 ms |
| C++ AADC total (rec + eval) | 89.42 ms |
| C++/Py AADC speedup (eval) | 1.3x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $328,653,156,849,201,408 | - |
| Rates +50bp | $492,979,735,273,802,240 | +50.0% |
| Unwind top 5 | $319,907,633,607,072,576 | -2.7% |
| Add hedge | $326,888,028,107,727,040 | -0.5% |

**IM Ladder:** 0.5x: $164,326,578,424,600,704, 0.75x: $246,489,867,636,901,120, 1.0x: $328,653,156,849,201,408, 1.25x: $410,816,446,061,501,824, 1.5x: $492,979,735,273,802,240

### 5:00 PM EOD Optimization

- Initial IM: $328,653,156,849,201,408
- Final IM: $328,653,156,849,201,408 (reduction: 0.0%)
- Trades moved: 0, Iterations: 2

---

## Run: 2026-02-02 15:35:57

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 500 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 381 us | 679.40 ms | 510 us | 1 | 1 |
| 8:00 AM Margin Attribution | 63 us | 33 us | 510 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.96 ms | 4.86 ms | 620 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 4.03 ms | 7.21 ms | 810 us | 8 | 8 |
| 5:00 PM EOD Optimization | 373.06 ms | 589.68 ms | 2.10 ms | 635 | 635 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.80 ms |
| Total AADC Py evals | 649 |
| Total kernel reuses | 649 |
| Total GPU evals | 649 |
| Amortized recording/eval | 0.06 ms |
| Cumulative AADC Py time | 380.49 ms |
| Cumulative GPU time | 1.281 s |
| AADC Py total (rec + eval) | 419.29 ms |
| GPU speedup (eval only) | 0.3x |
| GPU speedup (inc. recording) | 0.3x |
| C++ AADC recording (1-time) | 25.53 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 4.55 ms |
| C++ AADC total (rec + eval) | 30.08 ms |
| C++/Py AADC speedup (eval) | 83.6x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $4,398,050,237,838,401 | - |
| Rates +50bp | $6,597,075,356,757,600 | +50.0% |
| Unwind top 5 | $3,244,384,055,266,466 | -26.2% |
| Add hedge | $4,189,578,842,189,735 | -4.7% |

**IM Ladder:** 0.5x: $2,199,025,118,919,200, 0.75x: $3,298,537,678,378,800, 1.0x: $4,398,050,237,838,401, 1.25x: $5,497,562,797,298,002, 1.5x: $6,597,075,356,757,600

### 5:00 PM EOD Optimization

- Initial IM: $4,398,050,237,838,401
- Final IM: $3,506,215,886,417,054 (reduction: 20.3%)
- Trades moved: 21, Iterations: 100

---

## Run: 2026-02-02 15:36:57

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 500 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 401 us | 762.39 ms | 340 us | 1 | 1 |
| 8:00 AM Margin Attribution | 66 us | 33 us | 340 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.00 ms | 4.74 ms | 270 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.19 ms | 7.17 ms | 370 us | 8 | 8 |
| 5:00 PM EOD Optimization | 310.79 ms | 659.30 ms | 1.73 ms | 635 | 635 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 27.90 ms |
| Total AADC Py evals | 649 |
| Total kernel reuses | 649 |
| Total GPU evals | 649 |
| Amortized recording/eval | 0.04 ms |
| Cumulative AADC Py time | 316.44 ms |
| Cumulative GPU time | 1.434 s |
| AADC Py total (rec + eval) | 344.34 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 13.00 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 3.05 ms |
| C++ AADC total (rec + eval) | 16.05 ms |
| C++/Py AADC speedup (eval) | 103.8x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $4,398,050,237,838,401 | - |
| Rates +50bp | $6,597,075,356,757,600 | +50.0% |
| Unwind top 5 | $3,244,384,055,266,466 | -26.2% |
| Add hedge | $4,189,578,842,189,735 | -4.7% |

**IM Ladder:** 0.5x: $2,199,025,118,919,200, 0.75x: $3,298,537,678,378,800, 1.0x: $4,398,050,237,838,401, 1.25x: $5,497,562,797,298,002, 1.5x: $6,597,075,356,757,600

### 5:00 PM EOD Optimization

- Initial IM: $4,398,050,237,838,401
- Final IM: $3,506,215,886,417,054 (reduction: 20.3%)
- Trades moved: 21, Iterations: 100

---

## Run: 2026-02-02 15:37:25

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 500 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 513 us | 711.51 ms | 230 us | 1 | 1 |
| 8:00 AM Margin Attribution | 56 us | 30 us | 230 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.56 ms | 4.73 ms | 410 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.50 ms | 7.12 ms | 400 us | 8 | 8 |
| 5:00 PM EOD Optimization | 370.75 ms | 597.02 ms | 89.38 ms | 635 | 635 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 44.20 ms |
| Total AADC Py evals | 649 |
| Total kernel reuses | 649 |
| Total GPU evals | 649 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 377.38 ms |
| Cumulative GPU time | 1.320 s |
| AADC Py total (rec + eval) | 421.59 ms |
| GPU speedup (eval only) | 0.3x |
| GPU speedup (inc. recording) | 0.3x |
| C++ AADC recording (1-time) | 21.41 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 90.65 ms |
| C++ AADC total (rec + eval) | 112.06 ms |
| C++/Py AADC speedup (eval) | 4.2x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $4,398,050,237,838,401 | - |
| Rates +50bp | $6,597,075,356,757,600 | +50.0% |
| Unwind top 5 | $3,244,384,055,266,466 | -26.2% |
| Add hedge | $4,189,578,842,189,735 | -4.7% |

**IM Ladder:** 0.5x: $2,199,025,118,919,200, 0.75x: $3,298,537,678,378,800, 1.0x: $4,398,050,237,838,401, 1.25x: $5,497,562,797,298,002, 1.5x: $6,597,075,356,757,600

### 5:00 PM EOD Optimization

- Initial IM: $4,398,050,237,838,401
- Final IM: $3,506,215,886,417,054 (reduction: 20.3%)
- Trades moved: 21, Iterations: 100

---

## Run: 2026-02-02 15:38:53

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 500 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 391 us | 693.49 ms | 43.29 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 65 us | 34 us | 43.29 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 1.98 ms | 4.95 ms | 230 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.00 ms | 7.30 ms | 360 us | 8 | 8 |
| 5:00 PM EOD Optimization | 284.28 ms | 594.21 ms | 55.44 ms | 635 | 635 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 29.25 ms |
| Total AADC Py evals | 649 |
| Total kernel reuses | 649 |
| Total GPU evals | 649 |
| Amortized recording/eval | 0.05 ms |
| Cumulative AADC Py time | 289.71 ms |
| Cumulative GPU time | 1.300 s |
| AADC Py total (rec + eval) | 318.96 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 35.01 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 142.61 ms |
| C++ AADC total (rec + eval) | 177.62 ms |
| C++/Py AADC speedup (eval) | 2.0x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $4,398,050,237,838,401 | - |
| Rates +50bp | $6,597,075,356,757,600 | +50.0% |
| Unwind top 5 | $3,244,384,055,266,466 | -26.2% |
| Add hedge | $4,189,578,842,189,735 | -4.7% |

**IM Ladder:** 0.5x: $2,199,025,118,919,200, 0.75x: $3,298,537,678,378,800, 1.0x: $4,398,050,237,838,401, 1.25x: $5,497,562,797,298,002, 1.5x: $6,597,075,356,757,600

### 5:00 PM EOD Optimization

- Initial IM: $4,398,050,237,838,401
- Final IM: $3,506,215,886,417,054 (reduction: 20.3%)
- Trades moved: 21, Iterations: 100

---

## Run: 2026-02-02 15:39:28

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 1,000 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 462 us | 720.83 ms | 420 us | 1 | 1 |
| 8:00 AM Margin Attribution | 118 us | 59 us | 420 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.74 ms | 5.07 ms | 550 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.19 ms | 7.18 ms | 870 us | 8 | 8 |
| 5:00 PM EOD Optimization | 881.27 ms | 1.251 s | 281.65 ms | 1285 | 1285 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 29.36 ms |
| Total AADC Py evals | 1299 |
| Total kernel reuses | 1299 |
| Total GPU evals | 1299 |
| Amortized recording/eval | 0.02 ms |
| Cumulative AADC Py time | 887.77 ms |
| Cumulative GPU time | 1.984 s |
| AADC Py total (rec + eval) | 917.13 ms |
| GPU speedup (eval only) | 0.4x |
| GPU speedup (inc. recording) | 0.5x |
| C++ AADC recording (1-time) | 12.57 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 283.91 ms |
| C++ AADC total (rec + eval) | 296.48 ms |
| C++/Py AADC speedup (eval) | 3.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $9,581,698,819,279,568 | - |
| Rates +50bp | $14,372,548,228,919,356 | +50.0% |
| Unwind top 5 | $7,436,184,419,175,739 | -22.4% |
| Add hedge | $9,055,211,339,514,372 | -5.5% |

**IM Ladder:** 0.5x: $4,790,849,409,639,784, 0.75x: $7,186,274,114,459,678, 1.0x: $9,581,698,819,279,568, 1.25x: $11,977,123,524,099,462, 1.5x: $14,372,548,228,919,356

### 5:00 PM EOD Optimization

- Initial IM: $9,581,698,819,279,568
- Final IM: $8,775,354,108,099,477 (reduction: 8.4%)
- Trades moved: 29, Iterations: 100

---

## Run: 2026-02-02 16:00:42

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 10,000 |
| Portfolios | 50 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 623 us | 724.19 ms | 3.01 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 1.50 ms | 710 us | 3.01 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.49 ms | 5.38 ms | 28.24 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 4.33 ms | 7.31 ms | 7.32 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 49.153 s | 59.136 s | 123.592 s | 32948 | 32948 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 39.07 ms |
| Total AADC Py evals | 32962 |
| Total kernel reuses | 32962 |
| Total GPU evals | 32962 |
| Amortized recording/eval | 0.00 ms |
| Cumulative AADC Py time | 49.163 s |
| Cumulative GPU time | 59.874 s |
| AADC Py total (rec + eval) | 49.202 s |
| GPU speedup (eval only) | 0.8x |
| GPU speedup (inc. recording) | 0.8x |
| C++ AADC recording (1-time) | 9.89 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 123.633 s |
| C++ AADC total (rec + eval) | 123.643 s |
| C++/Py AADC speedup (eval) | 0.4x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $328,653,156,849,201,408 | - |
| Rates +50bp | $492,979,735,273,802,240 | +50.0% |
| Unwind top 5 | $319,907,633,607,072,576 | -2.7% |
| Add hedge | $326,888,028,107,727,040 | -0.5% |

**IM Ladder:** 0.5x: $164,326,578,424,600,704, 0.75x: $246,489,867,636,901,120, 1.0x: $328,653,156,849,201,408, 1.25x: $410,816,446,061,501,824, 1.5x: $492,979,735,273,802,240

### 5:00 PM EOD Optimization

- Initial IM: $328,653,156,849,201,408
- Final IM: $289,926,699,726,545,472 (reduction: 11.8%)
- Trades moved: 297, Iterations: 83

---

## Run: 2026-02-02 16:04:28

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 200 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 697 us | 659.66 ms | 650 us | 1 | 1 |
| 8:00 AM Margin Attribution | 41 us | 18 us | 650 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.00 ms | 6.56 ms | 860 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.50 ms | 10.59 ms | 370 us | 8 | 8 |
| 5:00 PM EOD Optimization | 174.49 ms | 512.75 ms | 90.38 ms | 377 | 377 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.34 ms |
| Total AADC Py evals | 391 |
| Total kernel reuses | 391 |
| Total GPU evals | 391 |
| Amortized recording/eval | 0.10 ms |
| Cumulative AADC Py time | 181.73 ms |
| Cumulative GPU time | 1.190 s |
| AADC Py total (rec + eval) | 220.06 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.26 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 92.91 ms |
| C++ AADC total (rec + eval) | 118.17 ms |
| C++/Py AADC speedup (eval) | 2.0x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $1,461,178,990,191,562 | - |
| Rates +50bp | $2,191,768,485,287,342 | +50.0% |
| Unwind top 5 | $824,735,222,717,730 | -43.6% |
| Add hedge | $1,290,658,991,257,144 | -11.7% |

**IM Ladder:** 0.5x: $730,589,495,095,781, 0.75x: $1,095,884,242,643,671, 1.0x: $1,461,178,990,191,562, 1.25x: $1,826,473,737,739,452, 1.5x: $2,191,768,485,287,342

### 5:00 PM EOD Optimization

- Initial IM: $1,461,178,990,191,562
- Final IM: $783,584,123,463,813 (reduction: 46.4%)
- Trades moved: 11, Iterations: 100

---

## Run: 2026-02-02 16:06:03

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 200 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 480 us | 681.28 ms | 1.34 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 39 us | 17 us | 1.34 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.59 ms | 4.90 ms | 750 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.96 ms | 7.30 ms | 360 us | 8 | 8 |
| 5:00 PM EOD Optimization | 176.21 ms | 342.88 ms | 113.41 ms | 377 | 377 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 39.90 ms |
| Total AADC Py evals | 391 |
| Total kernel reuses | 391 |
| Total GPU evals | 391 |
| Amortized recording/eval | 0.10 ms |
| Cumulative AADC Py time | 183.28 ms |
| Cumulative GPU time | 1.036 s |
| AADC Py total (rec + eval) | 223.18 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 29.57 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 117.20 ms |
| C++ AADC total (rec + eval) | 146.77 ms |
| C++/Py AADC speedup (eval) | 1.6x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $1,461,178,990,191,562 | - |
| Rates +50bp | $2,191,768,485,287,342 | +50.0% |
| Unwind top 5 | $824,735,222,717,730 | -43.6% |
| Add hedge | $1,290,658,991,257,144 | -11.7% |

**IM Ladder:** 0.5x: $730,589,495,095,781, 0.75x: $1,095,884,242,643,671, 1.0x: $1,461,178,990,191,562, 1.25x: $1,826,473,737,739,452, 1.5x: $2,191,768,485,287,342

### 5:00 PM EOD Optimization

- Initial IM: $1,461,178,990,191,562
- Final IM: $783,584,123,463,813 (reduction: 46.4%)
- Trades moved: 11, Iterations: 100

---

## Run: 2026-02-02 16:09:05

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 594 |
| Portfolios | 5 |
| Trade types | ir_swap,equity_option,fx_option |
| Risk factors (K) | 108 |
| Intra-bucket correlations | 223 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 890 us | 685.76 ms | 830 us | 1 | 1 |
| 8:00 AM Margin Attribution | 290 us | 71 us | 830 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 5.16 ms | 12.44 ms | 910 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 7.19 ms | 19.27 ms | 1.22 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 466.87 ms | 1.257 s | 181.42 ms | 473 | 473 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 87.63 ms |
| Total AADC Py evals | 487 |
| Total kernel reuses | 487 |
| Total GPU evals | 487 |
| Amortized recording/eval | 0.18 ms |
| Cumulative AADC Py time | 480.39 ms |
| Cumulative GPU time | 1.975 s |
| AADC Py total (rec + eval) | 568.02 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.3x |
| C++ AADC recording (1-time) | 37.63 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 185.21 ms |
| C++ AADC total (rec + eval) | 222.84 ms |
| C++/Py AADC speedup (eval) | 2.6x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $18,014,710,564,760,988 | - |
| Rates +50bp | $18,132,311,801,664,068 | +0.7% |
| Unwind top 5 | $13,048,192,324,251,282 | -27.6% |
| Add hedge | $16,896,971,293,073,774 | -6.2% |

**IM Ladder:** 0.5x: $17,925,403,076,185,306, 0.75x: $17,966,468,260,709,078, 1.0x: $18,014,710,564,760,988, 1.25x: $18,070,029,702,592,344, 1.5x: $18,132,311,801,664,068

### 5:00 PM EOD Optimization

- Initial IM: $18,014,710,564,760,988
- Final IM: $17,271,616,935,692,540 (reduction: 4.1%)
- Trades moved: 18, Iterations: 100

---

## Run: 2026-02-02 16:09:38

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 594 |
| Portfolios | 5 |
| Trade types | ir_swap,equity_option,fx_option |
| Risk factors (K) | 108 |
| Intra-bucket correlations | 223 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 868 us | 661.46 ms | 860 us | 1 | 1 |
| 8:00 AM Margin Attribution | 278 us | 71 us | 860 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 4.63 ms | 12.46 ms | 1.19 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 7.09 ms | 19.22 ms | 2.35 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 449.78 ms | 1.246 s | 175.37 ms | 473 | 473 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 86.70 ms |
| Total AADC Py evals | 487 |
| Total kernel reuses | 487 |
| Total GPU evals | 487 |
| Amortized recording/eval | 0.18 ms |
| Cumulative AADC Py time | 462.65 ms |
| Cumulative GPU time | 1.939 s |
| AADC Py total (rec + eval) | 549.35 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.3x |
| C++ AADC recording (1-time) | 32.41 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 180.63 ms |
| C++ AADC total (rec + eval) | 213.04 ms |
| C++/Py AADC speedup (eval) | 2.6x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $18,014,710,564,760,988 | - |
| Rates +50bp | $18,132,311,801,664,068 | +0.7% |
| Unwind top 5 | $13,048,192,324,251,282 | -27.6% |
| Add hedge | $16,896,971,293,073,774 | -6.2% |

**IM Ladder:** 0.5x: $17,925,403,076,185,306, 0.75x: $17,966,468,260,709,078, 1.0x: $18,014,710,564,760,988, 1.25x: $18,070,029,702,592,344, 1.5x: $18,132,311,801,664,068

### 5:00 PM EOD Optimization

- Initial IM: $18,014,710,564,760,988
- Final IM: $17,271,616,935,692,540 (reduction: 4.1%)
- Trades moved: 18, Iterations: 100

---

## Run: 2026-02-02 16:22:33

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 589 us | 666.99 ms | 630 us | 1 | 1 |
| 8:00 AM Margin Attribution | 35 us | 12 us | 630 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.06 ms | 4.74 ms | 800 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.85 ms | 7.20 ms | 290 us | 8 | 8 |
| 5:00 PM EOD Optimization | N/A | 472.14 ms | 81.63 ms | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.80 ms |
| Total AADC Py evals | 14 |
| Total kernel reuses | 14 |
| Total GPU evals | 60 |
| Amortized recording/eval | 2.77 ms |
| Cumulative AADC Py time | 7.53 ms |
| Cumulative GPU time | 1.151 s |
| AADC Py total (rec + eval) | 46.33 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.0x |
| C++ AADC recording (1-time) | 25.16 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 83.98 ms |
| C++ AADC total (rec + eval) | 109.14 ms |
| C++/Py AADC speedup (eval) | 0.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD Optimization

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 16:22:44

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 640 us | 667.61 ms | 540 us | 1 | 1 |
| 8:00 AM Margin Attribution | 35 us | 13 us | 540 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.58 ms | 4.78 ms | 780 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.71 ms | 7.18 ms | 270 us | 8 | 8 |
| 5:00 PM EOD Optimization | 148.11 ms | 320.19 ms | 84.18 ms | 361 | 361 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.74 ms |
| Total AADC Py evals | 375 |
| Total kernel reuses | 375 |
| Total GPU evals | 375 |
| Amortized recording/eval | 0.10 ms |
| Cumulative AADC Py time | 155.08 ms |
| Cumulative GPU time | 999.77 ms |
| AADC Py total (rec + eval) | 192.81 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.11 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 86.31 ms |
| C++ AADC total (rec + eval) | 111.42 ms |
| C++/Py AADC speedup (eval) | 1.8x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD Optimization

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

---

## Run: 2026-02-02 16:22:52

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 667 us | 681.60 ms | 570 us | 1 | 1 |
| 8:00 AM Margin Attribution | 36 us | 13 us | 570 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.96 ms | 4.73 ms | 830 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.88 ms | 7.10 ms | 320 us | 8 | 8 |
| 5:00 PM EOD Optimization | 163.62 ms | 320.62 ms | 79.82 ms | 361 | 361 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.10 ms |
| Total AADC Py evals | 375 |
| Total kernel reuses | 375 |
| Total GPU evals | 375 |
| Amortized recording/eval | 0.10 ms |
| Cumulative AADC Py time | 171.17 ms |
| Cumulative GPU time | 1.014 s |
| AADC Py total (rec + eval) | 209.26 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.02 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 82.11 ms |
| C++ AADC total (rec + eval) | 107.13 ms |
| C++/Py AADC speedup (eval) | 2.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD Optimization

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

---

## Run: 2026-02-02 16:23:41

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 594 |
| Portfolios | 5 |
| Trade types | ir_swap,equity_option,fx_option |
| Risk factors (K) | 108 |
| Intra-bucket correlations | 223 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 980 us | 671.27 ms | 870 us | 1 | 1 |
| 8:00 AM Margin Attribution | 275 us | 72 us | 870 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 4.88 ms | 12.57 ms | 1.32 ms | 5 | 5 |
| 2:00 PM What-If Scenarios | 7.22 ms | 19.25 ms | 2.31 ms | 8 | 8 |
| 5:00 PM EOD Optimization | 457.84 ms | 1.179 s | 226.11 ms | 473 | 473 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 87.29 ms |
| Total AADC Py evals | 487 |
| Total kernel reuses | 487 |
| Total GPU evals | 487 |
| Amortized recording/eval | 0.18 ms |
| Cumulative AADC Py time | 471.20 ms |
| Cumulative GPU time | 1.882 s |
| AADC Py total (rec + eval) | 558.49 ms |
| GPU speedup (eval only) | 0.3x |
| GPU speedup (inc. recording) | 0.3x |
| C++ AADC recording (1-time) | 65.16 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 231.48 ms |
| C++ AADC total (rec + eval) | 296.64 ms |
| C++/Py AADC speedup (eval) | 2.0x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $18,014,710,564,760,988 | - |
| Rates +50bp | $18,132,311,801,664,068 | +0.7% |
| Unwind top 5 | $13,048,192,324,251,282 | -27.6% |
| Add hedge | $16,896,971,293,073,774 | -6.2% |

**IM Ladder:** 0.5x: $17,925,403,076,185,306, 0.75x: $17,966,468,260,709,078, 1.0x: $18,014,710,564,760,988, 1.25x: $18,070,029,702,592,344, 1.5x: $18,132,311,801,664,068

### 5:00 PM EOD Optimization

- Initial IM: $18,014,710,564,760,988
- Final IM: $17,271,616,935,692,540 (reduction: 4.1%)
- Trades moved: 18, Iterations: 100

---

## Run: 2026-02-02 16:24:45

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 590 us | 664.99 ms | 560 us | 1 | 1 |
| 8:00 AM Margin Attribution | 35 us | 13 us | 560 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.88 ms | 4.56 ms | 790 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.99 ms | 7.11 ms | 270 us | 8 | 8 |
| 5:00 PM EOD Optimization | N/A | 347.35 ms | 86.08 ms | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.02 ms |
| Total AADC Py evals | 14 |
| Total kernel reuses | 14 |
| Total GPU evals | 60 |
| Amortized recording/eval | 2.72 ms |
| Cumulative AADC Py time | 7.49 ms |
| Cumulative GPU time | 1.024 s |
| AADC Py total (rec + eval) | 45.51 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.0x |
| C++ AADC recording (1-time) | 25.54 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 88.26 ms |
| C++ AADC total (rec + eval) | 113.80 ms |
| C++/Py AADC speedup (eval) | 0.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD Optimization

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 16:26:46

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 200 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 707 us | 662.59 ms | 630 us | 1 | 1 |
| 8:00 AM Margin Attribution | 44 us | 18 us | 630 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.30 ms | 4.96 ms | 520 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.76 ms | 7.26 ms | 390 us | 8 | 8 |
| 5:00 PM EOD Optimization | N/A | 447.48 ms | 112.11 ms | 98 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 39.06 ms |
| Total AADC Py evals | 14 |
| Total kernel reuses | 14 |
| Total GPU evals | 112 |
| Amortized recording/eval | 2.79 ms |
| Cumulative AADC Py time | 7.81 ms |
| Cumulative GPU time | 1.122 s |
| AADC Py total (rec + eval) | 46.87 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.0x |
| C++ AADC recording (1-time) | 25.22 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 114.28 ms |
| C++ AADC total (rec + eval) | 139.50 ms |
| C++/Py AADC speedup (eval) | 0.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $1,461,178,990,191,562 | - |
| Rates +50bp | $2,191,768,485,287,342 | +50.0% |
| Unwind top 5 | $824,735,222,717,730 | -43.6% |
| Add hedge | $1,290,658,991,257,144 | -11.7% |

**IM Ladder:** 0.5x: $730,589,495,095,781, 0.75x: $1,095,884,242,643,671, 1.0x: $1,461,178,990,191,562, 1.25x: $1,826,473,737,739,452, 1.5x: $2,191,768,485,287,342

### 5:00 PM EOD Optimization

- Initial IM: $1,461,178,990,191,562
- Final IM: $768,900,239,560,082 (reduction: 47.4%)
- Trades moved: 90, Iterations: 96

---

## Run: 2026-02-02 16:28:47

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 754 us | 668.31 ms | 540 us | 1 | 1 |
| 8:00 AM Margin Attribution | 37 us | 13 us | 540 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.96 ms | 4.93 ms | 800 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 4.06 ms | 7.64 ms | 270 us | 8 | 8 |
| 5:00 PM EOD Optimization | N/A | 355.84 ms | 81.76 ms | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.26 ms |
| Total AADC Py evals | 14 |
| Total kernel reuses | 14 |
| Total GPU evals | 60 |
| Amortized recording/eval | 2.73 ms |
| Cumulative AADC Py time | 7.81 ms |
| Cumulative GPU time | 1.037 s |
| AADC Py total (rec + eval) | 46.07 ms |
| GPU speedup (eval only) | 0.0x |
| GPU speedup (inc. recording) | 0.0x |
| C++ AADC recording (1-time) | 25.07 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 83.91 ms |
| C++ AADC total (rec + eval) | 108.98 ms |
| C++/Py AADC speedup (eval) | 0.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD Optimization

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 16:32:33

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 661 us | 681.71 ms | 540 us | 1 | 1 |
| 8:00 AM Margin Attribution | 36 us | 12 us | 540 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.21 ms | 4.87 ms | 480 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 5.43 ms | 7.36 ms | 260 us | 8 | 8 |
| 5:00 PM EOD: GD | 170.57 ms | 323.22 ms | 82.66 ms | 361 | 361 |
| 5:00 PM EOD: Adam | 71.40 ms | 137.94 ms | N/A | 160 | 160 |
| 5:00 PM EOD: Brute-Force | N/A | 354.78 ms | N/A | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.58 ms |
| Total AADC Py evals | 535 |
| Total kernel reuses | 535 |
| Total GPU evals | 581 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 251.31 ms |
| Cumulative GPU time | 1.510 s |
| AADC Py total (rec + eval) | 289.88 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 24.99 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 84.48 ms |
| C++ AADC total (rec + eval) | 109.47 ms |
| C++/Py AADC speedup (eval) | 3.0x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $431,171,465,361,081
- Final IM: $242,727,029,107,781 (reduction: 43.7%)
- Trades moved: 10, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 16:32:43

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 771 us | 663.64 ms | 310 us | 1 | 1 |
| 8:00 AM Margin Attribution | 34 us | 12 us | 310 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.13 ms | 4.82 ms | 580 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.98 ms | 7.26 ms | 170 us | 8 | 8 |
| 5:00 PM EOD: GD | 163.40 ms | 317.05 ms | 57.53 ms | 361 | 361 |
| 5:00 PM EOD: Adam | 66.66 ms | 135.47 ms | N/A | 160 | 160 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.92 ms |
| Total AADC Py evals | 535 |
| Total kernel reuses | 535 |
| Total GPU evals | 535 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 237.98 ms |
| Cumulative GPU time | 1.128 s |
| AADC Py total (rec + eval) | 275.90 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 21.09 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 58.90 ms |
| C++ AADC total (rec + eval) | 79.99 ms |
| C++/Py AADC speedup (eval) | 4.0x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $431,171,465,361,081
- Final IM: $242,727,029,107,781 (reduction: 43.7%)
- Trades moved: 10, Iterations: 2

---

## Run: 2026-02-02 16:32:51

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 650 us | 674.84 ms | 520 us | 1 | 1 |
| 8:00 AM Margin Attribution | 35 us | 12 us | 520 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.50 ms | 4.63 ms | 830 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.84 ms | 7.18 ms | 270 us | 8 | 8 |
| 5:00 PM EOD: GD | 158.32 ms | 347.27 ms | 74.38 ms | 361 | 361 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.10 ms |
| Total AADC Py evals | 375 |
| Total kernel reuses | 375 |
| Total GPU evals | 375 |
| Amortized recording/eval | 0.10 ms |
| Cumulative AADC Py time | 165.34 ms |
| Cumulative GPU time | 1.034 s |
| AADC Py total (rec + eval) | 203.45 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.36 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 76.52 ms |
| C++ AADC total (rec + eval) | 101.88 ms |
| C++/Py AADC speedup (eval) | 2.2x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

---

## Run: 2026-02-02 16:33:39

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 543 us | 662.44 ms | 540 us | 1 | 1 |
| 8:00 AM Margin Attribution | 40 us | 13 us | 540 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.82 ms | 4.61 ms | 590 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 4.63 ms | 7.54 ms | 260 us | 8 | 8 |
| 5:00 PM EOD: GD | 149.62 ms | 319.45 ms | 82.64 ms | 361 | 361 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.13 ms |
| Total AADC Py evals | 375 |
| Total kernel reuses | 375 |
| Total GPU evals | 375 |
| Amortized recording/eval | 0.10 ms |
| Cumulative AADC Py time | 157.66 ms |
| Cumulative GPU time | 994.05 ms |
| AADC Py total (rec + eval) | 195.79 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.04 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 84.57 ms |
| C++ AADC total (rec + eval) | 109.61 ms |
| C++/Py AADC speedup (eval) | 1.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

---

## Run: 2026-02-02 16:35:51

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 200 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | C++ AADC Time | Evals | Kernel Reuses |
|------|-------------|----------|---------------|-------|---------------|
| 7:00 AM Portfolio Setup | 679 us | 653.59 ms | 650 us | 1 | 1 |
| 8:00 AM Margin Attribution | 41 us | 18 us | 650 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.86 ms | 4.83 ms | 860 us | 5 | 5 |
| 2:00 PM What-If Scenarios | 3.78 ms | 7.28 ms | 380 us | 8 | 8 |
| 5:00 PM EOD: GD | 168.44 ms | 335.98 ms | 101.25 ms | 377 | 377 |
| 5:00 PM EOD: Adam | 93.70 ms | 175.10 ms | N/A | 203 | 203 |
| 5:00 PM EOD: Brute-Force | N/A | 445.51 ms | N/A | 98 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.69 ms |
| Total AADC Py evals | 594 |
| Total kernel reuses | 594 |
| Total GPU evals | 692 |
| Amortized recording/eval | 0.06 ms |
| Cumulative AADC Py time | 269.49 ms |
| Cumulative GPU time | 1.622 s |
| AADC Py total (rec + eval) | 307.18 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.34 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 103.79 ms |
| C++ AADC total (rec + eval) | 129.13 ms |
| C++/Py AADC speedup (eval) | 2.6x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $1,461,178,990,191,562 | - |
| Rates +50bp | $2,191,768,485,287,342 | +50.0% |
| Unwind top 5 | $824,735,222,717,730 | -43.6% |
| Add hedge | $1,290,658,991,257,144 | -11.7% |

**IM Ladder:** 0.5x: $730,589,495,095,781, 0.75x: $1,095,884,242,643,671, 1.0x: $1,461,178,990,191,562, 1.25x: $1,826,473,737,739,452, 1.5x: $2,191,768,485,287,342

### 5:00 PM EOD: GD

- Initial IM: $1,461,178,990,191,562
- Final IM: $783,584,123,463,813 (reduction: 46.4%)
- Trades moved: 11, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $1,461,178,990,191,562
- Final IM: $782,379,255,335,625 (reduction: 46.5%)
- Trades moved: 16, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $1,461,178,990,191,562
- Final IM: $768,900,239,560,082 (reduction: 47.4%)
- Trades moved: 90, Iterations: 96

---

## Run: 2026-02-02 16:42:41

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 667 us | 662.79 ms | 313.44 ms | 220 us | 1 | 1 |
| 8:00 AM Margin Attribution | 32 us | 12 us | 733 us | 220 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.80 ms | 4.73 ms | 60.89 ms | 550 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 4.26 ms | 7.57 ms | 4.76 ms | 110 us | 8 | 8 |
| 5:00 PM EOD: GD | 171.08 ms | 322.48 ms | N/A | 53.65 ms | 361 | 361 |
| 5:00 PM EOD: Adam | 73.42 ms | 140.15 ms | N/A | N/A | 160 | 160 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 36.11 ms | N/A | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.35 ms |
| Total AADC Py evals | 535 |
| Total kernel reuses | 535 |
| Total GPU evals | 535 |
| Total BF (forward-only) evals | 156 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 253.25 ms |
| Cumulative GPU time | 1.138 s |
| Cumulative BF time | 415.93 ms |
| AADC Py total (rec + eval) | 291.60 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.3x |
| C++ AADC recording (1-time) | 20.44 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 54.75 ms |
| C++ AADC total (rec + eval) | 75.19 ms |
| C++/Py AADC speedup (eval) | 4.6x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $431,171,465,361,081
- Final IM: $242,727,029,107,781 (reduction: 43.7%)
- Trades moved: 10, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 16:42:53

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 727 us | 681.72 ms | 319.66 ms | 520 us | 1 | 1 |
| 8:00 AM Margin Attribution | 34 us | 13 us | 766 us | 520 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.93 ms | 4.72 ms | 60.16 ms | 540 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 4.27 ms | 7.16 ms | 4.78 ms | 320 us | 8 | 8 |
| 5:00 PM EOD: GD | 167.14 ms | 321.30 ms | N/A | 88.00 ms | 361 | 361 |
| 5:00 PM EOD: Adam | 65.88 ms | 137.98 ms | N/A | N/A | 160 | 160 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.76 ms |
| Total AADC Py evals | 535 |
| Total kernel reuses | 535 |
| Total GPU evals | 535 |
| Total BF (forward-only) evals | 110 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 240.97 ms |
| Cumulative GPU time | 1.153 s |
| Cumulative BF time | 385.37 ms |
| AADC Py total (rec + eval) | 278.73 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.50 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 89.90 ms |
| C++ AADC total (rec + eval) | 115.40 ms |
| C++/Py AADC speedup (eval) | 2.7x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $431,171,465,361,081
- Final IM: $242,727,029,107,781 (reduction: 43.7%)
- Trades moved: 10, Iterations: 2

---

## Run: 2026-02-02 16:43:10

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 543 us | 666.85 ms | 315.25 ms | 600 us | 1 | 1 |
| 8:00 AM Margin Attribution | 36 us | 12 us | 737 us | 600 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.78 ms | 4.74 ms | 61.64 ms | 800 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 4.43 ms | 7.12 ms | 4.75 ms | 280 us | 8 | 8 |
| 5:00 PM EOD: GD | 161.27 ms | 321.59 ms | N/A | 81.17 ms | 361 | 361 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.49 ms |
| Total AADC Py evals | 375 |
| Total kernel reuses | 375 |
| Total GPU evals | 375 |
| Total BF (forward-only) evals | 110 |
| Amortized recording/eval | 0.10 ms |
| Cumulative AADC Py time | 169.06 ms |
| Cumulative GPU time | 1.000 s |
| Cumulative BF time | 382.38 ms |
| AADC Py total (rec + eval) | 206.56 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 26.32 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 83.45 ms |
| C++ AADC total (rec + eval) | 109.77 ms |
| C++/Py AADC speedup (eval) | 2.0x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

---

## Run: 2026-02-02 16:45:25

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 730 us | 682.42 ms | 318.41 ms | 530 us | 1 | 1 |
| 8:00 AM Margin Attribution | 34 us | 12 us | 744 us | 530 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.61 ms | 4.68 ms | 60.32 ms | 800 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 4.33 ms | 7.08 ms | 4.84 ms | 270 us | 8 | 8 |
| 5:00 PM EOD: GD | 166.92 ms | 321.92 ms | N/A | 76.39 ms | 361 | 361 |
| 5:00 PM EOD: Adam | 70.08 ms | 139.11 ms | N/A | N/A | 160 | 160 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 35.70 ms | N/A | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 42.03 ms |
| Total AADC Py evals | 535 |
| Total kernel reuses | 535 |
| Total GPU evals | 535 |
| Total BF (forward-only) evals | 156 |
| Amortized recording/eval | 0.08 ms |
| Cumulative AADC Py time | 244.71 ms |
| Cumulative GPU time | 1.155 s |
| Cumulative BF time | 420.02 ms |
| AADC Py total (rec + eval) | 286.74 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.99 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 78.52 ms |
| C++ AADC total (rec + eval) | 104.51 ms |
| C++/Py AADC speedup (eval) | 3.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $431,171,465,361,081
- Final IM: $242,727,029,107,781 (reduction: 43.7%)
- Trades moved: 10, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 16:54:24

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 10,000 |
| Portfolios | 55 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.08 ms | 705.48 ms | 313.29 ms | 3.37 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 1.59 ms | 667 us | 1.74 ms | 3.37 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.79 ms | 5.21 ms | 63.17 ms | 27.36 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 5.49 ms | 7.14 ms | 4.89 ms | 8.22 ms | 8 | 8 |
| 5:00 PM EOD: GD | 54.989 s | 59.156 s | N/A | 136.188 s | 35339 | 35339 |
| 5:00 PM EOD: Adam | 51.656 s | 57.894 s | N/A | N/A | 35042 | 35042 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 24.831 s | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 39.37 ms |
| Total AADC Py evals | 70395 |
| Total kernel reuses | 70395 |
| Total GPU evals | 70391 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.00 ms |
| Cumulative AADC Py time | 106.657 s |
| Cumulative GPU time | 117.769 s |
| Cumulative BF time | 25.214 s |
| AADC Py total (rec + eval) | 106.696 s |
| GPU speedup (eval only) | 0.9x |
| GPU speedup (inc. recording) | 0.9x |
| C++ AADC recording (1-time) | 22.76 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 136.230 s |
| C++ AADC total (rec + eval) | 136.253 s |
| C++/Py AADC speedup (eval) | 0.8x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $338,095,924,305,762,112 | - |
| Rates +50bp | $507,143,886,458,643,328 | +50.0% |
| Unwind top 5 | $329,322,738,695,503,488 | -2.6% |
| Add hedge | $336,331,369,222,568,832 | -0.5% |

**IM Ladder:** 0.5x: $169,047,962,152,881,056, 0.75x: $253,571,943,229,321,664, 1.0x: $338,095,924,305,762,112, 1.25x: $422,619,905,382,202,752, 1.5x: $507,143,886,458,643,328

### 5:00 PM EOD: GD

- Initial IM: $338,095,924,305,762,112
- Final IM: $289,972,498,532,747,712 (reduction: 14.2%)
- Trades moved: 347, Iterations: 70

### 5:00 PM EOD: Adam

- Initial IM: $338,095,924,305,762,112
- Final IM: $289,972,498,532,747,712 (reduction: 14.2%)
- Trades moved: 347, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $338,095,924,305,762,112
- Final IM: $295,503,324,384,436,800 (reduction: 12.6%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 16:59:29

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 677 us | 664.78 ms | 316.57 ms | 570 us | 1 | 1 |
| 8:00 AM Margin Attribution | 34 us | 13 us | 739 us | 570 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.09 ms | 4.71 ms | 60.86 ms | 490 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 3.78 ms | 7.16 ms | 4.83 ms | 260 us | 8 | 8 |
| 5:00 PM EOD: GD | 162.85 ms | 326.65 ms | N/A | 88.28 ms | 361 | 361 |
| 5:00 PM EOD: Adam | 69.13 ms | 140.77 ms | N/A | N/A | 160 | 160 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 35.48 ms | N/A | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 37.58 ms |
| Total AADC Py evals | 535 |
| Total kernel reuses | 535 |
| Total GPU evals | 535 |
| Total BF (forward-only) evals | 156 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 239.56 ms |
| Cumulative GPU time | 1.144 s |
| Cumulative BF time | 418.48 ms |
| AADC Py total (rec + eval) | 277.14 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.38 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 90.17 ms |
| C++ AADC total (rec + eval) | 115.55 ms |
| C++/Py AADC speedup (eval) | 2.7x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $431,171,465,361,081
- Final IM: $242,727,029,107,781 (reduction: 43.7%)
- Trades moved: 10, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 17:03:48

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 50 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 4 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 377 us | 672.31 ms | 316.55 ms | 300 us | 1 | 1 |
| 8:00 AM Margin Attribution | 32 us | 10 us | 767 us | 300 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.21 ms | 4.69 ms | 58.95 ms | 90 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 2.87 ms | 7.15 ms | 4.76 ms | 200 us | 8 | 8 |
| 5:00 PM EOD: GD | 102.38 ms | 301.89 ms | N/A | 1.98 ms | 341 | 341 |
| 5:00 PM EOD: Adam | 24.38 ms | 74.29 ms | N/A | N/A | 85 | 85 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 15.52 ms | N/A | 23 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 39.34 ms |
| Total AADC Py evals | 440 |
| Total kernel reuses | 440 |
| Total GPU evals | 440 |
| Total BF (forward-only) evals | 133 |
| Amortized recording/eval | 0.09 ms |
| Cumulative AADC Py time | 132.25 ms |
| Cumulative GPU time | 1.060 s |
| Cumulative BF time | 396.56 ms |
| AADC Py total (rec + eval) | 171.58 ms |
| GPU speedup (eval only) | 0.1x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.68 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 2.87 ms |
| C++ AADC total (rec + eval) | 28.55 ms |
| C++/Py AADC speedup (eval) | 46.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $399,527,208,153,231 | - |
| Rates +50bp | $599,290,812,229,847 | +50.0% |
| Unwind top 5 | $219,331,272,299,183 | -45.1% |
| Add hedge | $348,280,189,327,689 | -12.8% |

**IM Ladder:** 0.5x: $199,763,604,076,616, 0.75x: $299,645,406,114,924, 1.0x: $399,527,208,153,231, 1.25x: $499,409,010,191,539, 1.5x: $599,290,812,229,847

### 5:00 PM EOD: GD

- Initial IM: $399,527,208,153,231
- Final IM: $224,906,438,534,310 (reduction: 43.7%)
- Trades moved: 9, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $399,527,208,153,231
- Final IM: $226,097,595,695,469 (reduction: 43.4%)
- Trades moved: 9, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $399,527,208,153,231
- Final IM: $223,523,754,079,867 (reduction: 44.1%)
- Trades moved: 20, Iterations: 21

---

## Run: 2026-02-02 17:05:05

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 676 us | 669.00 ms | 315.17 ms | 550 us | 1 | 1 |
| 8:00 AM Margin Attribution | 34 us | 13 us | 755 us | 550 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.03 ms | 4.72 ms | 60.36 ms | 800 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 4.05 ms | 7.15 ms | 4.81 ms | 280 us | 8 | 8 |
| 5:00 PM EOD: GD | 174.54 ms | 323.01 ms | N/A | 87.56 ms | 361 | 361 |
| 5:00 PM EOD: Adam | 68.58 ms | 139.20 ms | N/A | N/A | 160 | 160 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 35.24 ms | N/A | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.86 ms |
| Total AADC Py evals | 535 |
| Total kernel reuses | 535 |
| Total GPU evals | 535 |
| Total BF (forward-only) evals | 156 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 250.90 ms |
| Cumulative GPU time | 1.143 s |
| Cumulative BF time | 416.34 ms |
| AADC Py total (rec + eval) | 289.76 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.3x |
| C++ AADC recording (1-time) | 25.44 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 89.74 ms |
| C++ AADC total (rec + eval) | 115.18 ms |
| C++/Py AADC speedup (eval) | 2.8x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $431,171,465,361,081
- Final IM: $242,727,029,107,781 (reduction: 43.7%)
- Trades moved: 10, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 17:05:58

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 666 us | 667.80 ms | 316.10 ms | 540 us | 1 | 1 |
| 8:00 AM Margin Attribution | 35 us | 13 us | 744 us | 540 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.55 ms | 4.69 ms | 60.17 ms | 780 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 3.67 ms | 7.16 ms | 4.80 ms | 280 us | 8 | 8 |
| 5:00 PM EOD: GD | 147.71 ms | 323.01 ms | N/A | 85.22 ms | 361 | 361 |
| 5:00 PM EOD: Adam | 62.28 ms | 139.46 ms | N/A | N/A | 160 | 160 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 35.44 ms | N/A | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.66 ms |
| Total AADC Py evals | 535 |
| Total kernel reuses | 535 |
| Total GPU evals | 535 |
| Total BF (forward-only) evals | 156 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 216.92 ms |
| Cumulative GPU time | 1.142 s |
| Cumulative BF time | 417.25 ms |
| AADC Py total (rec + eval) | 255.58 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.85 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 87.36 ms |
| C++ AADC total (rec + eval) | 113.21 ms |
| C++/Py AADC speedup (eval) | 2.5x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $431,171,465,361,081
- Final IM: $242,727,029,107,781 (reduction: 43.7%)
- Trades moved: 10, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 17:09:34

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 1,000 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 921 us | 680.39 ms | 375.20 ms | 1.33 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 106 us | 59 us | 933 us | 1.33 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 4.76 ms | 4.91 ms | 60.53 ms | 2.19 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 6.25 ms | 7.10 ms | 4.70 ms | 1.41 ms | 8 | 8 |
| 5:00 PM EOD: GD | 963.33 ms | 1.187 s | N/A | 1.860 s | 1285 | 1285 |
| 5:00 PM EOD: Adam | 784.63 ms | 953.59 ms | N/A | N/A | 1089 | 1089 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 177.85 ms | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 30.30 ms |
| Total AADC Py evals | 2388 |
| Total kernel reuses | 2388 |
| Total GPU evals | 2388 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.01 ms |
| Cumulative AADC Py time | 1.760 s |
| Cumulative GPU time | 2.833 s |
| Cumulative BF time | 619.21 ms |
| AADC Py total (rec + eval) | 1.790 s |
| GPU speedup (eval only) | 0.6x |
| GPU speedup (inc. recording) | 0.6x |
| C++ AADC recording (1-time) | 27.62 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 1.866 s |
| C++ AADC total (rec + eval) | 1.893 s |
| C++/Py AADC speedup (eval) | 0.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $9,581,698,819,279,568 | - |
| Rates +50bp | $14,372,548,228,919,356 | +50.0% |
| Unwind top 5 | $7,436,184,419,175,739 | -22.4% |
| Add hedge | $9,055,211,339,514,372 | -5.5% |

**IM Ladder:** 0.5x: $4,790,849,409,639,784, 0.75x: $7,186,274,114,459,678, 1.0x: $9,581,698,819,279,568, 1.25x: $11,977,123,524,099,462, 1.5x: $14,372,548,228,919,356

### 5:00 PM EOD: GD

- Initial IM: $9,581,698,819,279,568
- Final IM: $8,775,354,108,099,477 (reduction: 8.4%)
- Trades moved: 29, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $9,581,698,819,279,568
- Final IM: $8,775,354,108,099,477 (reduction: 8.4%)
- Trades moved: 29, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $9,581,698,819,279,568
- Final IM: $8,761,984,846,896,136 (reduction: 8.6%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 17:11:29

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 722 us | 688.10 ms | 320.18 ms | 590 us | 1 | 1 |
| 8:00 AM Margin Attribution | 36 us | 13 us | 774 us | 590 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.36 ms | 4.95 ms | 60.28 ms | 800 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 4.43 ms | 7.19 ms | 4.74 ms | 280 us | 8 | 8 |
| 5:00 PM EOD: GD | 166.24 ms | 322.06 ms | N/A | 85.75 ms | 361 | 361 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 35.80 ms | N/A | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 39.28 ms |
| Total AADC Py evals | 375 |
| Total kernel reuses | 375 |
| Total GPU evals | 375 |
| Total BF (forward-only) evals | 156 |
| Amortized recording/eval | 0.10 ms |
| Cumulative AADC Py time | 174.79 ms |
| Cumulative GPU time | 1.022 s |
| Cumulative BF time | 421.78 ms |
| AADC Py total (rec + eval) | 214.07 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 26.07 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 88.01 ms |
| C++ AADC total (rec + eval) | 114.08 ms |
| C++/Py AADC speedup (eval) | 2.0x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Brute-Force

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 17:12:03

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 649 us | 699.33 ms | 326.65 ms | 590 us | 1 | 1 |
| 8:00 AM Margin Attribution | 36 us | 13 us | 773 us | 590 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.01 ms | 5.03 ms | 62.14 ms | 680 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 4.01 ms | 7.37 ms | 4.99 ms | 280 us | 8 | 8 |
| 5:00 PM EOD: GD | 157.11 ms | 331.46 ms | N/A | 72.56 ms | 361 | 361 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 36.46 ms | N/A | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 45.28 ms |
| Total AADC Py evals | 375 |
| Total kernel reuses | 375 |
| Total GPU evals | 375 |
| Total BF (forward-only) evals | 156 |
| Amortized recording/eval | 0.12 ms |
| Cumulative AADC Py time | 164.81 ms |
| Cumulative GPU time | 1.043 s |
| Cumulative BF time | 431.01 ms |
| AADC Py total (rec + eval) | 210.09 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 17.51 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 74.70 ms |
| C++ AADC total (rec + eval) | 92.21 ms |
| C++/Py AADC speedup (eval) | 2.2x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Brute-Force

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 17:12:29

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 649 us | 672.66 ms | 313.57 ms | 590 us | 1 | 1 |
| 8:00 AM Margin Attribution | 34 us | 13 us | 779 us | 590 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.95 ms | 4.65 ms | 60.15 ms | 870 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 4.40 ms | 7.38 ms | 4.81 ms | 280 us | 8 | 8 |
| 5:00 PM EOD: GD | 153.66 ms | 320.26 ms | N/A | 90.91 ms | 361 | 361 |
| 5:00 PM EOD: Adam | 69.44 ms | 137.81 ms | N/A | N/A | 160 | 160 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 35.18 ms | N/A | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.48 ms |
| Total AADC Py evals | 535 |
| Total kernel reuses | 535 |
| Total GPU evals | 535 |
| Total BF (forward-only) evals | 156 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 231.14 ms |
| Cumulative GPU time | 1.143 s |
| Cumulative BF time | 414.49 ms |
| AADC Py total (rec + eval) | 269.62 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.76 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 93.24 ms |
| C++ AADC total (rec + eval) | 119.00 ms |
| C++/Py AADC speedup (eval) | 2.5x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $431,171,465,361,081
- Final IM: $242,727,029,107,781 (reduction: 43.7%)
- Trades moved: 10, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 17:15:14

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 685 us | 665.68 ms | 315.00 ms | 620 us | 1 | 1 |
| 8:00 AM Margin Attribution | 32 us | 13 us | 755 us | 620 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 2.87 ms | 4.68 ms | 59.98 ms | 630 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 3.97 ms | 7.14 ms | 4.82 ms | 250 us | 8 | 8 |
| 5:00 PM EOD: GD | 159.42 ms | 327.40 ms | N/A | 80.88 ms | 361 | 361 |
| 5:00 PM EOD: Adam | 71.47 ms | 141.46 ms | N/A | N/A | 160 | 160 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 35.01 ms | N/A | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.72 ms |
| Total AADC Py evals | 535 |
| Total kernel reuses | 535 |
| Total GPU evals | 535 |
| Total BF (forward-only) evals | 156 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 238.44 ms |
| Cumulative GPU time | 1.146 s |
| Cumulative BF time | 415.56 ms |
| AADC Py total (rec + eval) | 277.16 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 18.04 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 83.00 ms |
| C++ AADC total (rec + eval) | 101.04 ms |
| C++/Py AADC speedup (eval) | 2.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $431,171,465,361,081
- Final IM: $242,727,029,107,781 (reduction: 43.7%)
- Trades moved: 10, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 17:16:09

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 1,000 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.08 ms | 643.09 ms | 353.55 ms | 1.59 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 93 us | 57 us | 890 us | 1.59 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 4.86 ms | 4.86 ms | 61.89 ms | 1.27 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 6.16 ms | 7.08 ms | 4.70 ms | 1.42 ms | 8 | 8 |
| 5:00 PM EOD: GD | 910.61 ms | 1.159 s | N/A | 1.885 s | 1285 | 1285 |
| 5:00 PM EOD: Adam | 788.78 ms | 975.79 ms | N/A | N/A | 1089 | 1089 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 175.55 ms | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 30.03 ms |
| Total AADC Py evals | 2388 |
| Total kernel reuses | 2388 |
| Total GPU evals | 2388 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.01 ms |
| Cumulative AADC Py time | 1.712 s |
| Cumulative GPU time | 2.790 s |
| Cumulative BF time | 596.58 ms |
| AADC Py total (rec + eval) | 1.742 s |
| GPU speedup (eval only) | 0.6x |
| GPU speedup (inc. recording) | 0.6x |
| C++ AADC recording (1-time) | 26.71 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 1.891 s |
| C++ AADC total (rec + eval) | 1.918 s |
| C++/Py AADC speedup (eval) | 0.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $9,581,698,819,279,568 | - |
| Rates +50bp | $14,372,548,228,919,356 | +50.0% |
| Unwind top 5 | $7,436,184,419,175,739 | -22.4% |
| Add hedge | $9,055,211,339,514,372 | -5.5% |

**IM Ladder:** 0.5x: $4,790,849,409,639,784, 0.75x: $7,186,274,114,459,678, 1.0x: $9,581,698,819,279,568, 1.25x: $11,977,123,524,099,462, 1.5x: $14,372,548,228,919,356

### 5:00 PM EOD: GD

- Initial IM: $9,581,698,819,279,568
- Final IM: $8,775,354,108,099,477 (reduction: 8.4%)
- Trades moved: 29, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $9,581,698,819,279,568
- Final IM: $8,775,354,108,099,477 (reduction: 8.4%)
- Trades moved: 29, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $9,581,698,819,279,568
- Final IM: $8,761,984,846,896,136 (reduction: 8.6%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 17:17:35

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 5 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 628 us | 689.34 ms | 321.23 ms | 510 us | 1 | 1 |
| 8:00 AM Margin Attribution | 36 us | 13 us | 768 us | 510 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.09 ms | 4.98 ms | 60.21 ms | 560 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 4.44 ms | 7.24 ms | 4.94 ms | 340 us | 8 | 8 |
| 5:00 PM EOD: GD | 165.32 ms | 325.06 ms | N/A | 82.71 ms | 361 | 361 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 36.11 ms | N/A | 46 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.30 ms |
| Total AADC Py evals | 376 |
| Total kernel reuses | 375 |
| Total GPU evals | 376 |
| Total BF (forward-only) evals | 156 |
| Amortized recording/eval | 0.10 ms |
| Cumulative AADC Py time | 173.52 ms |
| Cumulative GPU time | 1.027 s |
| Cumulative BF time | 423.26 ms |
| AADC Py total (rec + eval) | 211.81 ms |
| GPU speedup (eval only) | 0.2x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 25.55 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 84.63 ms |
| C++ AADC total (rec + eval) | 110.18 ms |
| C++/Py AADC speedup (eval) | 2.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $431,171,465,361,081 | - |
| Rates +50bp | $646,757,198,041,622 | +50.0% |
| Unwind top 5 | $285,111,375,625,315 | -33.9% |
| Add hedge | $397,384,055,991,420 | -7.8% |

**IM Ladder:** 0.5x: $215,585,732,680,541, 0.75x: $323,378,599,020,811, 1.0x: $431,171,465,361,081, 1.25x: $538,964,331,701,352, 1.5x: $646,757,198,041,622

### 5:00 PM EOD: GD

- Initial IM: $431,171,465,361,081
- Final IM: $257,121,259,762,633 (reduction: 40.4%)
- Trades moved: 7, Iterations: 100

### 5:00 PM EOD: Brute-Force

- Initial IM: $431,171,465,361,081
- Final IM: $223,743,793,992,934 (reduction: 48.1%)
- Trades moved: 41, Iterations: 44

---

## Run: 2026-02-02 17:18:44

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 2,964 |
| Portfolios | 5 |
| Trade types | ir_swap,equity_option,fx_option |
| Risk factors (K) | 108 |
| Intra-bucket correlations | 223 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.78 ms | 672.59 ms | 380.58 ms | 2.83 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 1.13 ms | 518 us | 3.43 ms | 2.83 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 7.31 ms | 12.99 ms | 162.50 ms | 3.43 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 9.41 ms | 19.43 ms | 12.96 ms | 10.58 ms | 8 | 8 |
| 5:00 PM EOD: GD | 3.394 s | 7.058 s | N/A | 5.706 s | 2802 | 2802 |
| 5:00 PM EOD: Adam | 3.082 s | 6.560 s | N/A | N/A | 2624 | 2624 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 925.30 ms | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 88.65 ms |
| Total AADC Py evals | 5441 |
| Total kernel reuses | 5440 |
| Total GPU evals | 5441 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.02 ms |
| Cumulative AADC Py time | 6.495 s |
| Cumulative GPU time | 14.324 s |
| Cumulative BF time | 1.485 s |
| AADC Py total (rec + eval) | 6.584 s |
| GPU speedup (eval only) | 0.5x |
| GPU speedup (inc. recording) | 0.5x |
| C++ AADC recording (1-time) | 63.31 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 5.726 s |
| C++ AADC total (rec + eval) | 5.789 s |
| C++/Py AADC speedup (eval) | 1.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $212,749,734,346,840,384 | - |
| Rates +50bp | $213,316,797,033,357,760 | +0.3% |
| Unwind top 5 | $199,550,299,735,914,240 | -6.2% |
| Add hedge | $210,083,103,920,062,080 | -1.3% |

**IM Ladder:** 0.5x: $212,287,206,342,988,896, 0.75x: $212,505,328,096,443,264, 1.0x: $212,749,734,346,840,384, 1.25x: $213,020,278,526,655,808, 1.5x: $213,316,797,033,357,760

### 5:00 PM EOD: GD

- Initial IM: $212,749,734,346,840,384
- Final IM: $211,594,477,442,935,776 (reduction: 0.5%)
- Trades moved: 64, Iterations: 100

### 5:00 PM EOD: Adam

- Initial IM: $212,749,734,346,840,384
- Final IM: $211,594,477,442,935,776 (reduction: 0.5%)
- Trades moved: 64, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $212,749,734,346,840,352
- Final IM: $211,584,141,702,786,208 (reduction: 0.5%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 17:36:08

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 20 |
| Intra-bucket correlations | 90 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.06 ms | 682.64 ms | 321.75 ms | 1.24 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 36 us | 13 us | 736 us | 1.24 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 4.35 ms | 4.46 ms | 56.30 ms | 70 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 5.55 ms | 6.80 ms | 4.51 ms | 330 us | 8 | 8 |
| 5:00 PM EOD: Adam | 68.46 ms | 87.55 ms | N/A | 3.77 ms | 107 | 107 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 28.89 ms | N/A | 43 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 30.76 ms |
| Total AADC Py evals | 122 |
| Total kernel reuses | 121 |
| Total GPU evals | 122 |
| Total BF (forward-only) evals | 153 |
| Amortized recording/eval | 0.25 ms |
| Cumulative AADC Py time | 79.45 ms |
| Cumulative GPU time | 781.46 ms |
| Cumulative BF time | 412.19 ms |
| AADC Py total (rec + eval) | 110.22 ms |
| GPU speedup (eval only) | 0.1x |
| GPU speedup (inc. recording) | 0.1x |
| C++ AADC recording (1-time) | 19.11 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 6.65 ms |
| C++ AADC total (rec + eval) | 25.76 ms |
| C++/Py AADC speedup (eval) | 11.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $106,830,216,563,284 | - |
| Rates +50bp | $160,245,324,844,926 | +50.0% |
| Unwind top 5 | $188,269,064,578,634 | +76.2% |
| Add hedge | $89,767,422,336,212 | -16.0% |

**IM Ladder:** 0.5x: $53,415,108,281,642, 0.75x: $80,122,662,422,463, 1.0x: $106,830,216,563,284, 1.25x: $133,537,770,704,105, 1.5x: $160,245,324,844,926

### 5:00 PM EOD: Adam

- Initial IM: $106,830,216,563,284
- Final IM: $72,489,770,325,692 (reduction: 32.1%)
- Trades moved: 8, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $106,830,216,563,284
- Final IM: $70,629,206,338,981 (reduction: 33.9%)
- Trades moved: 33, Iterations: 41

---

## Run: 2026-02-02 17:36:20

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 600 |
| Portfolios | 5 |
| Trade types | ir_swap,fx_option |
| Risk factors (K) | 66 |
| Intra-bucket correlations | 195 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.18 ms | 686.11 ms | 319.28 ms | 1.26 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 248 us | 56 us | 1.37 ms | 1.26 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 5.75 ms | 7.10 ms | 92.26 ms | 2.44 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 7.40 ms | 10.67 ms | 7.17 ms | 1.55 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 569.84 ms | 896.29 ms | N/A | 1.126 s | 651 | 651 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 251.82 ms | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 52.58 ms |
| Total AADC Py evals | 666 |
| Total kernel reuses | 665 |
| Total GPU evals | 666 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.08 ms |
| Cumulative AADC Py time | 584.41 ms |
| Cumulative GPU time | 1.600 s |
| Cumulative BF time | 671.91 ms |
| AADC Py total (rec + eval) | 636.99 ms |
| GPU speedup (eval only) | 0.4x |
| GPU speedup (inc. recording) | 0.4x |
| C++ AADC recording (1-time) | 27.12 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 1.133 s |
| C++ AADC total (rec + eval) | 1.160 s |
| C++/Py AADC speedup (eval) | 0.5x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $2,773,021,236,664,914 | - |
| Rates +50bp | $4,159,504,765,459,268 | +50.0% |
| Unwind top 5 | $2,149,287,796,067,922 | -22.5% |
| Add hedge | $2,468,354,226,942,088 | -11.0% |

**IM Ladder:** 0.5x: $1,386,537,736,158,341, 0.75x: $2,079,779,479,339,892, 1.0x: $2,773,021,236,664,914, 1.25x: $3,466,262,999,647,626, 1.5x: $4,159,504,765,459,268

### 5:00 PM EOD: Adam

- Initial IM: $2,773,021,236,664,914
- Final IM: $2,161,803,902,862,534 (reduction: 22.0%)
- Trades moved: 24, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $2,773,021,236,664,914
- Final IM: $2,139,177,773,412,270 (reduction: 22.9%)
- Trades moved: 97, Iterations: 100

---

## Run: 2026-02-02 17:36:34

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 979 |
| Portfolios | 5 |
| Trade types | ir_swap,equity_option |
| Risk factors (K) | 72 |
| Intra-bucket correlations | 163 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.33 ms | 647.41 ms | 357.05 ms | 1.47 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 305 us | 82 us | 1.64 ms | 1.47 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 5.66 ms | 8.46 ms | 108.66 ms | 2.18 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 8.71 ms | 12.57 ms | 8.30 ms | 2.85 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 971.46 ms | 1.707 s | N/A | 1.911 s | 1060 | 1060 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 330.96 ms | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 73.36 ms |
| Total AADC Py evals | 1075 |
| Total kernel reuses | 1074 |
| Total GPU evals | 1075 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 987.47 ms |
| Cumulative GPU time | 2.376 s |
| Cumulative BF time | 806.61 ms |
| AADC Py total (rec + eval) | 1.061 s |
| GPU speedup (eval only) | 0.4x |
| GPU speedup (inc. recording) | 0.4x |
| C++ AADC recording (1-time) | 50.43 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 1.919 s |
| C++ AADC total (rec + eval) | 1.969 s |
| C++/Py AADC speedup (eval) | 0.5x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $68,560,915,637,167,264 | - |
| Rates +50bp | $68,865,573,280,689,968 | +0.4% |
| Unwind top 5 | $59,609,636,215,110,856 | -13.1% |
| Add hedge | $66,736,041,256,672,112 | -2.7% |

**IM Ladder:** 0.5x: $68,323,038,053,711,968, 0.75x: $68,433,559,624,430,544, 1.0x: $68,560,915,637,167,264, 1.25x: $68,704,970,924,525,024, 1.5x: $68,865,573,280,689,968

### 5:00 PM EOD: Adam

- Initial IM: $68,560,915,637,167,264
- Final IM: $67,384,794,321,390,376 (reduction: 1.7%)
- Trades moved: 49, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $68,560,915,637,167,248
- Final IM: $67,375,372,966,280,336 (reduction: 1.7%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 17:37:14

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 2,964 |
| Portfolios | 10 |
| Trade types | ir_swap,equity_option,fx_option |
| Risk factors (K) | 108 |
| Intra-bucket correlations | 223 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 2.03 ms | 679.00 ms | 392.63 ms | 2.97 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 1.57 ms | 591 us | 3.43 ms | 2.97 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 7.18 ms | 12.86 ms | 163.49 ms | 5.09 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 10.26 ms | 19.33 ms | 12.84 ms | 10.74 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 4.395 s | 8.588 s | N/A | 9.611 s | 3360 | 3360 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 3.563 s | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 87.59 ms |
| Total AADC Py evals | 3375 |
| Total kernel reuses | 3374 |
| Total GPU evals | 3375 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.03 ms |
| Cumulative AADC Py time | 4.416 s |
| Cumulative GPU time | 9.300 s |
| Cumulative BF time | 4.135 s |
| AADC Py total (rec + eval) | 4.503 s |
| GPU speedup (eval only) | 0.5x |
| GPU speedup (inc. recording) | 0.5x |
| C++ AADC recording (1-time) | 51.11 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 9.633 s |
| C++ AADC total (rec + eval) | 9.684 s |
| C++/Py AADC speedup (eval) | 0.5x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $216,217,517,821,877,728 | - |
| Rates +50bp | $216,888,416,859,220,352 | +0.3% |
| Unwind top 5 | $202,861,917,173,488,480 | -6.2% |
| Add hedge | $213,518,286,645,067,648 | -1.2% |

**IM Ladder:** 0.5x: $215,678,284,035,141,984, 0.75x: $215,931,336,473,427,104, 1.0x: $216,217,517,821,877,728, 1.25x: $216,536,621,624,930,944, 1.5x: $216,888,416,859,220,352

### 5:00 PM EOD: Adam

- Initial IM: $216,217,517,821,877,728
- Final IM: $211,682,955,091,146,144 (reduction: 2.1%)
- Trades moved: 89, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $216,217,517,821,877,664
- Final IM: $211,613,825,692,745,856 (reduction: 2.1%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 17:53:25

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 20 |
| Intra-bucket correlations | 90 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.15 ms | 668.88 ms | 316.90 ms | 960 us | 1 | 1 |
| 8:00 AM Margin Attribution | 34 us | 12 us | 688 us | 960 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 4.26 ms | 4.48 ms | 57.26 ms | 80 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 5.75 ms | 6.66 ms | 4.49 ms | 280 us | 8 | 8 |
| 5:00 PM EOD: Adam | 69.35 ms | 88.68 ms | N/A | 3.61 ms | 107 | 107 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 28.44 ms | N/A | 43 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 30.83 ms |
| Total AADC Py evals | 122 |
| Total kernel reuses | 121 |
| Total GPU evals | 122 |
| Total BF (forward-only) evals | 153 |
| Amortized recording/eval | 0.25 ms |
| Cumulative AADC Py time | 80.54 ms |
| Cumulative GPU time | 768.72 ms |
| Cumulative BF time | 407.78 ms |
| AADC Py total (rec + eval) | 111.38 ms |
| GPU speedup (eval only) | 0.1x |
| GPU speedup (inc. recording) | 0.1x |
| C++ AADC recording (1-time) | 19.13 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 5.89 ms |
| C++ AADC total (rec + eval) | 25.02 ms |
| C++/Py AADC speedup (eval) | 13.7x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $106,830,216,563,284 | - |
| Rates +50bp | $160,245,324,844,926 | +50.0% |
| Unwind top 5 | $188,269,064,578,634 | +76.2% |
| Add hedge | $89,767,422,336,212 | -16.0% |

**IM Ladder:** 0.5x: $53,415,108,281,642, 0.75x: $80,122,662,422,463, 1.0x: $106,830,216,563,284, 1.25x: $133,537,770,704,105, 1.5x: $160,245,324,844,926

### 5:00 PM EOD: Adam

- Initial IM: $106,830,216,563,284
- Final IM: $72,489,770,325,692 (reduction: 32.1%)
- Trades moved: 8, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $106,830,216,563,284
- Final IM: $70,629,206,338,981 (reduction: 33.9%)
- Trades moved: 33, Iterations: 41

---

## Run: 2026-02-02 17:53:37

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 600 |
| Portfolios | 5 |
| Trade types | ir_swap,fx_option |
| Risk factors (K) | 66 |
| Intra-bucket correlations | 195 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.27 ms | 671.60 ms | 317.21 ms | 1.29 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 245 us | 55 us | 1.35 ms | 1.29 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 6.02 ms | 7.08 ms | 92.10 ms | 2.10 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 7.18 ms | 10.58 ms | 7.15 ms | 1.53 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 565.86 ms | 879.81 ms | N/A | 1.163 s | 651 | 651 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 226.45 ms | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 53.14 ms |
| Total AADC Py evals | 666 |
| Total kernel reuses | 665 |
| Total GPU evals | 666 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.08 ms |
| Cumulative AADC Py time | 580.57 ms |
| Cumulative GPU time | 1.569 s |
| Cumulative BF time | 644.27 ms |
| AADC Py total (rec + eval) | 633.71 ms |
| GPU speedup (eval only) | 0.4x |
| GPU speedup (inc. recording) | 0.4x |
| C++ AADC recording (1-time) | 41.77 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 1.169 s |
| C++ AADC total (rec + eval) | 1.211 s |
| C++/Py AADC speedup (eval) | 0.5x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $2,773,021,236,664,914 | - |
| Rates +50bp | $4,159,504,765,459,268 | +50.0% |
| Unwind top 5 | $2,149,287,796,067,922 | -22.5% |
| Add hedge | $2,468,354,226,942,088 | -11.0% |

**IM Ladder:** 0.5x: $1,386,537,736,158,341, 0.75x: $2,079,779,479,339,892, 1.0x: $2,773,021,236,664,914, 1.25x: $3,466,262,999,647,626, 1.5x: $4,159,504,765,459,268

### 5:00 PM EOD: Adam

- Initial IM: $2,773,021,236,664,914
- Final IM: $2,161,803,902,862,534 (reduction: 22.0%)
- Trades moved: 24, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $2,773,021,236,664,914
- Final IM: $2,139,177,773,412,270 (reduction: 22.9%)
- Trades moved: 97, Iterations: 100

---

## Run: 2026-02-02 17:53:52

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 979 |
| Portfolios | 5 |
| Trade types | ir_swap,equity_option |
| Risk factors (K) | 72 |
| Intra-bucket correlations | 163 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.29 ms | 663.55 ms | 370.05 ms | 1.58 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 344 us | 82 us | 1.65 ms | 1.58 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 5.50 ms | 8.80 ms | 106.63 ms | 2.49 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 7.67 ms | 12.83 ms | 8.45 ms | 2.80 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 1.260 s | 1.707 s | N/A | 1.822 s | 1060 | 1060 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 324.39 ms | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 72.87 ms |
| Total AADC Py evals | 1075 |
| Total kernel reuses | 1074 |
| Total GPU evals | 1075 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 1.275 s |
| Cumulative GPU time | 2.392 s |
| Cumulative BF time | 811.16 ms |
| AADC Py total (rec + eval) | 1.348 s |
| GPU speedup (eval only) | 0.5x |
| GPU speedup (inc. recording) | 0.6x |
| C++ AADC recording (1-time) | 49.72 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 1.830 s |
| C++ AADC total (rec + eval) | 1.880 s |
| C++/Py AADC speedup (eval) | 0.7x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $68,560,915,637,167,264 | - |
| Rates +50bp | $68,865,573,280,689,968 | +0.4% |
| Unwind top 5 | $59,609,636,215,110,856 | -13.1% |
| Add hedge | $66,736,041,256,672,112 | -2.7% |

**IM Ladder:** 0.5x: $68,323,038,053,711,968, 0.75x: $68,433,559,624,430,544, 1.0x: $68,560,915,637,167,264, 1.25x: $68,704,970,924,525,024, 1.5x: $68,865,573,280,689,968

### 5:00 PM EOD: Adam

- Initial IM: $68,560,915,637,167,264
- Final IM: $67,384,794,321,390,376 (reduction: 1.7%)
- Trades moved: 49, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $68,560,915,637,167,248
- Final IM: $67,375,372,966,280,336 (reduction: 1.7%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 17:54:33

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 2,964 |
| Portfolios | 10 |
| Trade types | ir_swap,equity_option,fx_option |
| Risk factors (K) | 108 |
| Intra-bucket correlations | 223 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.93 ms | 663.14 ms | 377.41 ms | 2.93 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 1.43 ms | 522 us | 3.36 ms | 2.93 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 9.02 ms | 13.83 ms | 172.78 ms | 5.03 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 10.36 ms | 19.38 ms | 12.90 ms | 10.68 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 4.368 s | 8.582 s | N/A | 9.891 s | 3360 | 3360 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 3.586 s | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 87.61 ms |
| Total AADC Py evals | 3375 |
| Total kernel reuses | 3374 |
| Total GPU evals | 3375 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.03 ms |
| Cumulative AADC Py time | 4.391 s |
| Cumulative GPU time | 9.279 s |
| Cumulative BF time | 4.153 s |
| AADC Py total (rec + eval) | 4.478 s |
| GPU speedup (eval only) | 0.5x |
| GPU speedup (inc. recording) | 0.5x |
| C++ AADC recording (1-time) | 63.60 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 9.912 s |
| C++ AADC total (rec + eval) | 9.976 s |
| C++/Py AADC speedup (eval) | 0.4x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $216,217,517,821,877,728 | - |
| Rates +50bp | $216,888,416,859,220,352 | +0.3% |
| Unwind top 5 | $202,861,917,173,488,480 | -6.2% |
| Add hedge | $213,518,286,645,067,648 | -1.2% |

**IM Ladder:** 0.5x: $215,678,284,035,141,984, 0.75x: $215,931,336,473,427,104, 1.0x: $216,217,517,821,877,728, 1.25x: $216,536,621,624,930,944, 1.5x: $216,888,416,859,220,352

### 5:00 PM EOD: Adam

- Initial IM: $216,217,517,821,877,728
- Final IM: $211,682,955,091,146,144 (reduction: 2.1%)
- Trades moved: 89, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $216,217,517,821,877,664
- Final IM: $211,613,825,692,745,856 (reduction: 2.1%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 17:58:20

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 4,000 |
| Portfolios | 15 |
| Trade types | ir_swap,fx_option |
| Risk factors (K) | 170 |
| Intra-bucket correlations | 425 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 2.56 ms | 727.76 ms | 321.75 ms | 3.88 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 2.63 ms | 2.04 ms | 7.18 ms | 3.88 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 8.84 ms | 27.83 ms | 309.16 ms | 11.19 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 12.18 ms | 43.22 ms | 24.40 ms | 24.09 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 18.724 s | 62.755 s | N/A | 109.786 s | 11150 | 11150 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 14.893 s | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 99.88 ms |
| Total AADC Py evals | 11165 |
| Total kernel reuses | 11164 |
| Total GPU evals | 11165 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.01 ms |
| Cumulative AADC Py time | 18.750 s |
| Cumulative GPU time | 63.555 s |
| Cumulative BF time | 15.555 s |
| AADC Py total (rec + eval) | 18.850 s |
| GPU speedup (eval only) | 0.3x |
| GPU speedup (inc. recording) | 0.3x |
| C++ AADC recording (1-time) | 97.65 ms |
| Total C++ AADC evals | 9 |
| Cumulative C++ AADC time | 109.829 s |
| C++ AADC total (rec + eval) | 109.926 s |
| C++/Py AADC speedup (eval) | 0.2x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $13,932,030,453,680,308 | - |
| Rates +50bp | $20,897,747,732,650,044 | +50.0% |
| Unwind top 5 | $12,317,632,766,450,662 | -11.6% |
| Add hedge | $13,528,772,206,547,374 | -2.9% |

**IM Ladder:** 0.5x: $6,966,313,729,359,304, 0.75x: $10,449,171,952,864,270, 1.0x: $13,932,030,453,680,308, 1.25x: $17,414,889,065,430,350, 1.5x: $20,897,747,732,650,044

### 5:00 PM EOD: Adam

- Initial IM: $13,932,030,453,680,308
- Final IM: $5,664,577,599,484,880 (reduction: 59.3%)
- Trades moved: 99, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $13,932,030,453,680,308
- Final IM: $5,408,455,226,856,156 (reduction: 61.2%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 18:14:33

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 15,000 |
| Portfolios | 20 |
| Trade types | ir_swap,equity_option,fx_option |
| Risk factors (K) | 225 |
| Intra-bucket correlations | 486 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 3.04 ms | 636.41 ms | 325.92 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 11.38 ms | 9.10 ms | 14.27 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 14.29 ms | 40.05 ms | 607.96 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 16.24 ms | 60.17 ms | 33.54 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 61.293 s | 177.609 s | N/A | 22794 | 22794 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 97.973 s | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 128.66 ms |
| Total AADC Py evals | 22809 |
| Total kernel reuses | 22808 |
| Total GPU evals | 21354 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.01 ms |
| Cumulative AADC Py time | 61.338 s |
| Cumulative GPU time | 178.354 s |
| Cumulative BF time | 98.955 s |
| AADC Py total (rec + eval) | 61.467 s |
| GPU speedup (eval only) | 0.3x |
| GPU speedup (inc. recording) | 0.3x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $1,202,225,029,184,050,944 | - |
| Rates +50bp | $1,205,655,345,236,324,096 | +0.3% |
| Unwind top 5 | $1,185,464,442,061,506,560 | -1.4% |
| Add hedge | $1,198,564,034,492,908,032 | -0.3% |

**IM Ladder:** 0.5x: $1,199,447,586,911,671,040, 0.75x: $1,200,754,187,901,314,304, 1.0x: $1,202,225,029,184,050,944, 1.25x: $1,203,859,118,337,918,976, 1.5x: $1,205,655,345,236,324,096

### 5:00 PM EOD: Adam

- Initial IM: $1,202,225,029,184,050,944
- Final IM: $1,179,241,502,754,065,152 (reduction: 1.9%)
- Trades moved: 246, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $1,200,615,302,274,381,056
- Final IM: $1,179,797,037,041,046,528 (reduction: 1.7%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 18:22:19

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 30 |
| Intra-bucket correlations | 135 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 731 us | 698.58 ms | 322.05 ms | 880 us | 1 | 1 |
| 8:00 AM Margin Attribution | 35 us | 12 us | 819 us | 880 us | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 3.14 ms | 4.75 ms | 60.09 ms | 110 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 3.94 ms | 7.05 ms | 4.72 ms | 350 us | 8 | 8 |
| 5:00 PM EOD: Adam | 87.51 ms | 148.36 ms | N/A | 1.90 ms | 171 | 171 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 23.92 ms | N/A | 33 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 38.03 ms |
| Total AADC Py evals | 186 |
| Total kernel reuses | 185 |
| Total GPU evals | 186 |
| Total BF (forward-only) evals | 143 |
| Amortized recording/eval | 0.20 ms |
| Cumulative AADC Py time | 95.35 ms |
| Cumulative GPU time | 858.75 ms |
| Cumulative BF time | 411.60 ms |
| AADC Py total (rec + eval) | 133.38 ms |
| GPU speedup (eval only) | 0.1x |
| GPU speedup (inc. recording) | 0.2x |
| C++ AADC recording (1-time) | 26.62 ms |
| Total C++ AADC evals | 176 |
| Cumulative C++ AADC time | 4.12 ms |
| C++ AADC total (rec + eval) | 30.74 ms |
| C++/Py AADC speedup (eval) | 23.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $284,409,381,844,208 | - |
| Rates +50bp | $426,614,072,766,312 | +50.0% |
| Unwind top 5 | $289,583,247,749,459 | +1.8% |
| Add hedge | $250,624,559,774,010 | -11.9% |

**IM Ladder:** 0.5x: $142,204,690,922,104, 0.75x: $213,307,036,383,156, 1.0x: $284,409,381,844,208, 1.25x: $355,511,727,305,260, 1.5x: $426,614,072,766,312

### 5:00 PM EOD: Adam

- Initial IM: $284,409,381,844,208
- Final IM: $221,432,930,940,615 (reduction: 22.1%)
- Trades moved: 9, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $284,409,381,844,208
- Final IM: $214,886,123,957,701 (reduction: 24.4%)
- Trades moved: 31, Iterations: 31

---

## Run: 2026-02-02 18:24:11

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 8,000 |
| Portfolios | 15 |
| Trade types | ir_swap,fx_option |
| Risk factors (K) | 66 |
| Intra-bucket correlations | 195 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 8 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.06 ms | 637.89 ms | 319.71 ms | 6.24 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 2.47 ms | 1.42 ms | 2.76 ms | 6.24 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 4.43 ms | 7.70 ms | 103.25 ms | 3.51 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 5.46 ms | 10.84 ms | 7.31 ms | 24.98 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 28.396 s | 45.865 s | N/A | 798.99 ms | 26473 | 26473 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 10.684 s | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 59.10 ms |
| Total AADC Py evals | 26488 |
| Total kernel reuses | 26487 |
| Total GPU evals | 26488 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.00 ms |
| Cumulative AADC Py time | 28.409 s |
| Cumulative GPU time | 46.523 s |
| Cumulative BF time | 11.117 s |
| AADC Py total (rec + eval) | 28.469 s |
| GPU speedup (eval only) | 0.6x |
| GPU speedup (inc. recording) | 0.6x |
| C++ AADC recording (1-time) | 37.14 ms |
| Total C++ AADC evals | 26478 |
| Cumulative C++ AADC time | 839.96 ms |
| C++ AADC total (rec + eval) | 877.10 ms |
| C++/Py AADC speedup (eval) | 33.8x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $92,009,065,287,074,416 | - |
| Rates +50bp | $138,012,167,775,200,192 | +50.0% |
| Unwind top 5 | $86,281,133,742,006,224 | -6.2% |
| Add hedge | $90,850,188,799,661,616 | -1.3% |

**IM Ladder:** 0.5x: $46,005,964,824,934,608, 0.75x: $69,007,514,549,526,584, 1.0x: $92,009,065,287,074,416, 1.25x: $115,010,616,429,831,296, 1.5x: $138,012,167,775,200,192

### 5:00 PM EOD: Adam

- Initial IM: $92,009,065,287,074,416
- Final IM: $83,560,293,079,355,088 (reduction: 9.2%)
- Trades moved: 146, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $92,009,065,287,074,400
- Final IM: $83,533,170,591,851,680 (reduction: 9.2%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 18:27:14

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 100 |
| Portfolios | 3 |
| Trade types | ir_swap |
| Risk factors (K) | 20 |
| Intra-bucket correlations | 90 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.05 ms | 664.77 ms | 315.70 ms | 1.99 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 30 us | 12 us | 726 us | 1.99 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 4.24 ms | 4.35 ms | 55.76 ms | 110 us | 100 | 5 |
| 2:00 PM What-If Scenarios | 5.67 ms | 6.74 ms | 4.51 ms | 390 us | 8 | 8 |
| 5:00 PM EOD: Adam | 67.73 ms | 87.30 ms | N/A | 1.47 ms | 107 | 107 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 28.32 ms | N/A | 43 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 31.07 ms |
| Total AADC Py evals | 122 |
| Total kernel reuses | 121 |
| Total GPU evals | 122 |
| Total BF (forward-only) evals | 153 |
| Amortized recording/eval | 0.25 ms |
| Cumulative AADC Py time | 78.72 ms |
| Cumulative GPU time | 763.18 ms |
| Cumulative BF time | 405.01 ms |
| AADC Py total (rec + eval) | 109.79 ms |
| GPU speedup (eval only) | 0.1x |
| GPU speedup (inc. recording) | 0.1x |
| C++ AADC recording (1-time) | 19.06 ms |
| Total C++ AADC evals | 112 |
| Cumulative C++ AADC time | 5.95 ms |
| C++ AADC total (rec + eval) | 25.01 ms |
| C++/Py AADC speedup (eval) | 13.2x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $106,830,216,563,284 | - |
| Rates +50bp | $160,245,324,844,926 | +50.0% |
| Unwind top 5 | $188,269,064,578,634 | +76.2% |
| Add hedge | $89,767,422,336,212 | -16.0% |

**IM Ladder:** 0.5x: $53,415,108,281,642, 0.75x: $80,122,662,422,463, 1.0x: $106,830,216,563,284, 1.25x: $133,537,770,704,105, 1.5x: $160,245,324,844,926

### 5:00 PM EOD: Adam

- Initial IM: $106,830,216,563,284
- Final IM: $72,489,770,325,692 (reduction: 32.1%)
- Trades moved: 8, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $106,830,216,563,284
- Final IM: $70,629,206,338,981 (reduction: 33.9%)
- Trades moved: 33, Iterations: 41

---

## Run: 2026-02-02 18:27:25

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 600 |
| Portfolios | 5 |
| Trade types | ir_swap,fx_option |
| Risk factors (K) | 66 |
| Intra-bucket correlations | 195 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.12 ms | 683.08 ms | 320.30 ms | 2.29 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 250 us | 52 us | 1.41 ms | 2.29 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 5.54 ms | 7.21 ms | 91.56 ms | 2.36 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 7.74 ms | 10.73 ms | 7.18 ms | 1.84 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 540.64 ms | 880.07 ms | N/A | 57.34 ms | 651 | 651 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 248.15 ms | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 52.33 ms |
| Total AADC Py evals | 666 |
| Total kernel reuses | 665 |
| Total GPU evals | 666 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.08 ms |
| Cumulative AADC Py time | 555.29 ms |
| Cumulative GPU time | 1.581 s |
| Cumulative BF time | 668.61 ms |
| AADC Py total (rec + eval) | 607.62 ms |
| GPU speedup (eval only) | 0.4x |
| GPU speedup (inc. recording) | 0.4x |
| C++ AADC recording (1-time) | 41.55 ms |
| Total C++ AADC evals | 656 |
| Cumulative C++ AADC time | 66.12 ms |
| C++ AADC total (rec + eval) | 107.67 ms |
| C++/Py AADC speedup (eval) | 8.4x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $2,773,021,236,664,914 | - |
| Rates +50bp | $4,159,504,765,459,268 | +50.0% |
| Unwind top 5 | $2,149,287,796,067,922 | -22.5% |
| Add hedge | $2,468,354,226,942,088 | -11.0% |

**IM Ladder:** 0.5x: $1,386,537,736,158,341, 0.75x: $2,079,779,479,339,892, 1.0x: $2,773,021,236,664,914, 1.25x: $3,466,262,999,647,626, 1.5x: $4,159,504,765,459,268

### 5:00 PM EOD: Adam

- Initial IM: $2,773,021,236,664,914
- Final IM: $2,161,803,902,862,534 (reduction: 22.0%)
- Trades moved: 24, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $2,773,021,236,664,914
- Final IM: $2,139,177,773,412,270 (reduction: 22.9%)
- Trades moved: 97, Iterations: 100

---

## Run: 2026-02-02 18:27:38

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 979 |
| Portfolios | 5 |
| Trade types | ir_swap,equity_option |
| Risk factors (K) | 72 |
| Intra-bucket correlations | 163 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 1.30 ms | 654.40 ms | 350.62 ms | 2.55 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 309 us | 85 us | 1.64 ms | 2.55 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 6.15 ms | 8.34 ms | 104.69 ms | 2.69 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 8.29 ms | 12.69 ms | 8.35 ms | 3.22 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 990.81 ms | 1.714 s | N/A | 83.58 ms | 1060 | 1060 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 323.78 ms | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 72.26 ms |
| Total AADC Py evals | 1075 |
| Total kernel reuses | 1074 |
| Total GPU evals | 1075 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.07 ms |
| Cumulative AADC Py time | 1.007 s |
| Cumulative GPU time | 2.389 s |
| Cumulative BF time | 789.07 ms |
| AADC Py total (rec + eval) | 1.079 s |
| GPU speedup (eval only) | 0.4x |
| GPU speedup (inc. recording) | 0.5x |
| C++ AADC recording (1-time) | 50.12 ms |
| Total C++ AADC evals | 1047 |
| Cumulative C++ AADC time | 94.59 ms |
| C++ AADC total (rec + eval) | 144.71 ms |
| C++/Py AADC speedup (eval) | 10.6x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $68,560,915,637,167,264 | - |
| Rates +50bp | $68,865,573,280,689,968 | +0.4% |
| Unwind top 5 | $59,609,636,215,110,856 | -13.1% |
| Add hedge | $66,736,041,256,672,112 | -2.7% |

**IM Ladder:** 0.5x: $68,323,038,053,711,968, 0.75x: $68,433,559,624,430,544, 1.0x: $68,560,915,637,167,264, 1.25x: $68,704,970,924,525,024, 1.5x: $68,865,573,280,689,968

### 5:00 PM EOD: Adam

- Initial IM: $68,560,915,637,167,264
- Final IM: $67,384,794,321,390,376 (reduction: 1.7%)
- Trades moved: 49, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $68,560,915,637,167,248
- Final IM: $67,375,372,966,280,336 (reduction: 1.7%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 18:28:08

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 2,964 |
| Portfolios | 10 |
| Trade types | ir_swap,equity_option,fx_option |
| Risk factors (K) | 108 |
| Intra-bucket correlations | 223 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 16.70 ms | 654.96 ms | 366.69 ms | 4.31 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 1.44 ms | 520 us | 3.36 ms | 4.31 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 7.74 ms | 12.72 ms | 162.74 ms | 3.41 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 9.69 ms | 19.30 ms | 12.84 ms | 11.54 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 4.239 s | 8.501 s | N/A | 224.96 ms | 3360 | 3360 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 3.216 s | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 88.86 ms |
| Total AADC Py evals | 3375 |
| Total kernel reuses | 3374 |
| Total GPU evals | 3375 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.03 ms |
| Cumulative AADC Py time | 4.275 s |
| Cumulative GPU time | 9.188 s |
| Cumulative BF time | 3.762 s |
| AADC Py total (rec + eval) | 4.363 s |
| GPU speedup (eval only) | 0.5x |
| GPU speedup (inc. recording) | 0.5x |
| C++ AADC recording (1-time) | 62.94 ms |
| Total C++ AADC evals | 3347 |
| Cumulative C++ AADC time | 248.53 ms |
| C++ AADC total (rec + eval) | 311.47 ms |
| C++/Py AADC speedup (eval) | 17.2x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $216,217,517,821,877,728 | - |
| Rates +50bp | $216,888,416,859,220,352 | +0.3% |
| Unwind top 5 | $202,861,917,173,488,480 | -6.2% |
| Add hedge | $213,518,286,645,067,648 | -1.2% |

**IM Ladder:** 0.5x: $215,678,284,035,141,984, 0.75x: $215,931,336,473,427,104, 1.0x: $216,217,517,821,877,728, 1.25x: $216,536,621,624,930,944, 1.5x: $216,888,416,859,220,352

### 5:00 PM EOD: Adam

- Initial IM: $216,217,517,821,877,728
- Final IM: $211,682,955,091,146,144 (reduction: 2.1%)
- Trades moved: 89, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $216,217,517,821,877,664
- Final IM: $211,613,825,692,745,856 (reduction: 2.1%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 18:30:04

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 4,000 |
| Portfolios | 15 |
| Trade types | ir_swap,fx_option |
| Risk factors (K) | 170 |
| Intra-bucket correlations | 425 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 2.33 ms | 720.23 ms | 319.23 ms | 5.66 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 2.46 ms | 1.98 ms | 7.23 ms | 5.66 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 9.15 ms | 27.63 ms | 307.94 ms | 4.64 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 12.63 ms | 43.05 ms | 24.41 ms | 30.95 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 18.696 s | 62.363 s | N/A | 698.56 ms | 11150 | 11150 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 14.001 s | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 98.48 ms |
| Total AADC Py evals | 11165 |
| Total kernel reuses | 11164 |
| Total GPU evals | 11165 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.01 ms |
| Cumulative AADC Py time | 18.723 s |
| Cumulative GPU time | 63.156 s |
| Cumulative BF time | 14.659 s |
| AADC Py total (rec + eval) | 18.821 s |
| GPU speedup (eval only) | 0.3x |
| GPU speedup (inc. recording) | 0.3x |
| C++ AADC recording (1-time) | 97.54 ms |
| Total C++ AADC evals | 11155 |
| Cumulative C++ AADC time | 745.47 ms |
| C++ AADC total (rec + eval) | 843.01 ms |
| C++/Py AADC speedup (eval) | 25.1x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $13,932,030,453,680,308 | - |
| Rates +50bp | $20,897,747,732,650,044 | +50.0% |
| Unwind top 5 | $12,317,632,766,450,662 | -11.6% |
| Add hedge | $13,528,772,206,547,374 | -2.9% |

**IM Ladder:** 0.5x: $6,966,313,729,359,304, 0.75x: $10,449,171,952,864,270, 1.0x: $13,932,030,453,680,308, 1.25x: $17,414,889,065,430,350, 1.5x: $20,897,747,732,650,044

### 5:00 PM EOD: Adam

- Initial IM: $13,932,030,453,680,308
- Final IM: $5,664,577,599,484,880 (reduction: 59.3%)
- Trades moved: 99, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $13,932,030,453,680,308
- Final IM: $5,408,455,226,856,156 (reduction: 61.2%)
- Trades moved: 100, Iterations: 100

---

## Run: 2026-02-02 18:36:28

### Configuration

| Parameter | Value |
|-----------|-------|
| SIMM formula | Full ISDA v2.6 (correlations + concentration) |
| Trades | 15,000 |
| Portfolios | 20 |
| Trade types | ir_swap,equity_option,fx_option |
| Risk factors (K) | 225 |
| Intra-bucket correlations | 486 pairs |
| New trades (intraday) | 50 |
| Optimize iterations | 100 |
| Threads | 16 |
| AADC available | True |
| CUDA available | True |
| C++ AADC available | True |

### Per-Step Results

| Step | AADC Py Time | GPU Time | BF Time | C++ AADC Time | Evals | Kernel Reuses |
|------|------|------|------|------|------|------|
| 7:00 AM Portfolio Setup | 2.50 ms | 618.27 ms | 319.25 ms | 12.23 ms | 1 | 1 |
| 8:00 AM Margin Attribution | 11.47 ms | 8.12 ms | 13.66 ms | 12.23 ms | 1 | 0 |
| 9AM-4PM Intraday Pre-Trade | 11.12 ms | 38.89 ms | 605.42 ms | 17.85 ms | 100 | 5 |
| 2:00 PM What-If Scenarios | 15.64 ms | 60.38 ms | 33.84 ms | 91.95 ms | 8 | 8 |
| 5:00 PM EOD: Adam | 63.923 s | 177.978 s | N/A | 2.240 s | 24312 | 22794 |
| 5:00 PM EOD: Brute-Force | N/A | N/A | 102.627 s | N/A | 101 | 0 |

### Kernel Economics

| Metric | Value |
|--------|-------|
| Kernel recordings | 1 |
| AADC Py recording (1-time) | 127.63 ms |
| Total AADC Py evals | 22809 |
| Total kernel reuses | 22808 |
| Total GPU evals | 21354 |
| Total BF (forward-only) evals | 211 |
| Amortized recording/eval | 0.01 ms |
| Cumulative AADC Py time | 63.964 s |
| Cumulative GPU time | 178.704 s |
| Cumulative BF time | 103.599 s |
| AADC Py total (rec + eval) | 64.092 s |
| GPU speedup (eval only) | 0.4x |
| GPU speedup (inc. recording) | 0.4x |
| C++ AADC recording (1-time) | 111.65 ms |
| Total C++ AADC evals | 24319 |
| Cumulative C++ AADC time | 2.374 s |
| C++ AADC total (rec + eval) | 2.486 s |
| C++/Py AADC speedup (eval) | 26.9x |

### 8:00 AM Margin Attribution

- Euler decomposition error: 0.0000%

### 2:00 PM What-If Scenarios

| Scenario | IM | Change |
|----------|-------|--------|
| Baseline | $1,202,225,029,184,050,944 | - |
| Rates +50bp | $1,205,655,345,236,324,096 | +0.3% |
| Unwind top 5 | $1,185,464,442,061,506,560 | -1.4% |
| Add hedge | $1,198,564,034,492,908,032 | -0.3% |

**IM Ladder:** 0.5x: $1,199,447,586,911,671,040, 0.75x: $1,200,754,187,901,314,304, 1.0x: $1,202,225,029,184,050,944, 1.25x: $1,203,859,118,337,918,976, 1.5x: $1,205,655,345,236,324,096

### 5:00 PM EOD: Adam

- Initial IM: $1,202,225,029,184,050,944
- Final IM: $1,179,241,502,754,065,152 (reduction: 1.9%)
- Trades moved: 246, Iterations: 2

### 5:00 PM EOD: Brute-Force

- Initial IM: $1,200,615,302,274,381,056
- Final IM: $1,179,797,037,041,046,528 (reduction: 1.7%)
- Trades moved: 100, Iterations: 100

