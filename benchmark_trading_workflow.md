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

