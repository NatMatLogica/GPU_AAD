# Full ISDA SIMM v2.6 Benchmark: AADC vs GPU

Trading day workflow benchmark results using `benchmark_trading_workflow.py` v2.1.0.
Both backends use the full ISDA SIMM v2.6 formula with intra-bucket correlations,
inter-bucket gamma aggregation (IR), concentration factors, and cross-risk-class
PSI aggregation. AADC and GPU produce machine-precision identical results (max rel diff ~1e-15).

---

## Correctness

All v2.1.0 runs confirm:

- **AADC vs GPU agreement**: max relative difference ~1e-15 (machine epsilon)
- **Euler decomposition error**: 0.0000% across all runs
- **Concentration factors**: Computed from CRIF via `_precompute_concentration_factors`,
  ensuring identical CR values for both backends
- **Inter-bucket gamma**: IR cross-currency correlations encoded into the K×K correlation
  matrix as `gamma * g_bc` (gamma = 0.32 for v2.6)

---

## Performance Scaling

| Trades | Portfolios | K | AADC Total | GPU Total | GPU Speedup | Evals |
|--------|------------|---|------------|-----------|-------------|-------|
| 50 | 3 | 30 | 55 ms | 464 ms | 0.1x | 31 |
| 1,000 | 5 | 30 | 855 ms | 1.28 s | 0.7x | 115 |
| 5,000 | 10 | 30 | 3.85 s | 4.31 s | 0.9x | 115 |
| 10,000 | 50 | 30 | 8.52 s | 9.16 s | 0.9x | 115 |
| 50,000 | 100 | 30 | 46.56 s | 47.02 s | 1.0x | 115 |

AADC is faster at every scale tested. The gap narrows from 8x at 50 trades
to virtual parity at 50K trades, but the GPU never pulls ahead — even with
100 portfolios providing 100 parallel threads.

---

## Per-Step Breakdown

### 50 Trades, 3 Portfolios

| Step | AADC | GPU | Winner |
|------|------|-----|--------|
| 7:00 AM Portfolio Setup | 355 us | 426 ms | AADC 1200x |
| 8:00 AM Margin Attribution | 25 us | 9 us | GPU 2.5x |
| 9AM-4PM Intraday Pre-Trade | 344 us | 1.09 ms | AADC 3.2x |
| 2:00 PM What-If Scenarios | 2.58 ms | 7.72 ms | AADC 3.0x |
| 5:00 PM EOD Optimization | 14.83 ms | 28.69 ms | AADC 1.9x |

### 1,000 Trades, 5 Portfolios

| Step | AADC | GPU | Winner |
|------|------|-----|--------|
| 7:00 AM Portfolio Setup | 605 us | 405 ms | AADC 670x |
| 8:00 AM Margin Attribution | 104 us | 57 us | GPU 1.8x |
| 9AM-4PM Intraday Pre-Trade | 3.16 ms | 4.91 ms | AADC 1.6x |
| 2:00 PM What-If Scenarios | 3.88 ms | 7.27 ms | AADC 1.9x |
| 5:00 PM EOD Optimization | 817 ms | 860 ms | AADC 1.1x |

### 5,000 Trades, 10 Portfolios

| Step | AADC | GPU | Winner |
|------|------|-----|--------|
| 7:00 AM Portfolio Setup | 749 us | 435 ms | AADC 581x |
| 8:00 AM Margin Attribution | 513 us | 303 us | GPU 1.7x |
| 9AM-4PM Intraday Pre-Trade | 3.19 ms | 5.13 ms | AADC 1.6x |
| 2:00 PM What-If Scenarios | 5.76 ms | 7.67 ms | AADC 1.3x |
| 5:00 PM EOD Optimization | 3.818 s | 3.862 s | AADC 1.0x |

### 10,000 Trades, 50 Portfolios

| Step | AADC | GPU | Winner |
|------|------|-----|--------|
| 7:00 AM Portfolio Setup | 761 us | 499 ms | AADC 655x |
| 8:00 AM Margin Attribution | 1.19 ms | 674 us | GPU 1.8x |
| 9AM-4PM Intraday Pre-Trade | 3.35 ms | 5.46 ms | AADC 1.6x |
| 2:00 PM What-If Scenarios | 4.36 ms | 7.58 ms | AADC 1.7x |
| 5:00 PM EOD Optimization | 8.484 s | 8.648 s | AADC 1.0x |

### 50,000 Trades, 100 Portfolios

| Step | AADC | GPU | Winner |
|------|------|-----|--------|
| 7:00 AM Portfolio Setup | 661 us | 378 ms | AADC 572x |
| 8:00 AM Margin Attribution | 5.52 ms | 3.84 ms | GPU 1.4x |
| 9AM-4PM Intraday Pre-Trade | 3.03 ms | 5.78 ms | AADC 1.9x |
| 2:00 PM What-If Scenarios | 4.66 ms | 7.91 ms | AADC 1.7x |
| 5:00 PM EOD Optimization | 46.508 s | 46.623 s | AADC 1.0x |

### Step-Level Observations

- **Portfolio Setup**: AADC wins by 500-1200x. The GPU pays ~430ms kernel launch/JIT
  overhead on every first call; AADC's `evaluate()` costs <1ms after the one-time recording.
- **Attribution**: GPU 1.7-2.5x faster. Pure numpy dot product on pre-computed gradients,
  no kernel evaluation needed. GPU wins because the array operations are trivial.
- **Pre-Trade (5 evals)**: AADC 1.6-3.2x faster consistently across all scales.
- **What-If (8 evals)**: AADC 1.3-3.0x faster. Advantage narrows at larger trade counts.
- **Optimization (101 evals)**: Nearly parity at 5K+ trades. At 50K/100P the two
  backends are within 0.2% of each other (46.508s vs 46.623s).

---

## Kernel Economics

| Metric | 50 Trades | 1K Trades | 5K Trades | 10K Trades | 50K Trades |
|--------|-----------|-----------|-----------|------------|------------|
| Recording cost (1-time) | 37 ms | 30 ms | 26 ms | 31 ms | 39 ms |
| Total evaluations | 31 | 115 | 115 | 115 | 115 |
| Kernel reuses | 31 (100%) | 115 (100%) | 115 (100%) | 115 (100%) | 115 (100%) |
| Re-recordings | 0 | 0 | 0 | 0 | 0 |
| Amortized rec/eval | 1.21 ms | 0.26 ms | 0.22 ms | 0.27 ms | 0.34 ms |

Key findings:

- **Recording cost is constant** (~25-39ms) regardless of trade count. The kernel
  has K=30 inputs, not T inputs, so the tape size doesn't grow with portfolio size.
- **100% kernel reuse** throughout the entire trading day. All 115 evaluations
  (setup + pre-trade + what-if + optimization) reuse the single recorded kernel.
  Zero re-recordings needed.
- **Amortized cost drops** with more evaluations: 1.21ms/eval for 31 evals down
  to 0.22ms/eval for 115 evals.

---

## Why GPU Doesn't Win

With K=30 risk factors and P=3-50 portfolios, the GPU CUDA kernel runs one thread
per portfolio. This means:

- P=3: 3 GPU threads active (massive under-utilization)
- P=50: 50 GPU threads active (still far below typical GPU occupancy of thousands)
- P=100: 100 GPU threads — and GPU still doesn't win (46.6s vs 46.5s AADC)

The GPU's strength is massive parallelism. At these dimensions, the computation
per thread is too light to overcome launch and memory transfer overhead. The SIMM
kernel is compute-light (weighted sums + sqrt over K=30 factors) — each thread
does ~O(K^2) = ~900 FLOPs. GPU advantage would likely appear with:

- **K=200+** risk factors (more work per thread, ~40K FLOPs)
- **Monte Carlo paths** inside the kernel (the classic GPU use case)
- **Fused multi-step kernels** that avoid repeated host-device round-trips

The 50K/100P result is particularly telling: even with 100 portfolios the GPU
only reaches parity. The bottleneck is not parallelism but the numpy aggregation
`agg_S_T = S.T @ allocation` which scales with T and dominates at 50K trades.

---

## EOD Optimization Results

| Trades | Portfolios | Initial IM | Final IM | Reduction | Trades Moved |
|--------|------------|------------|----------|-----------|--------------|
| 50 | 3 | $400T | $300T | **24.9%** | 1 |
| 1,000 | 5 | $9,582T | $9,582T | 0.0% | 0 |
| 5,000 | 10 | $105,960T | $105,960T | 0.0% | 0 |
| 10,000 | 50 | $329Q | $965Q | **-193.5%** | 1,243 |
| 50,000 | 100 | $2,832Q | $15,582Q | **-450.2%** | 11,512 |

The optimizer has two failure modes:

1. **Stuck at 0% (1K, 5K trades)**: Too many equally-weighted trades dilute the
   per-trade gradient signal. No single trade move produces a meaningful IM reduction,
   so the optimizer never triggers a move.

2. **Divergence at 10K+ trades**: The gradient-descent step size is too aggressive
   for many portfolios. At 10K/50P it triples IM; at 50K/100P it increases IM by
   5.5x, moving 11,512 trades chaotically. This worsens with scale — more portfolios
   means more dimensions for the optimizer to diverge in. This is a step-size /
   convergence issue in the optimizer, not a GPU vs AADC issue (both backends
   produce identical gradients).

The 50-trade case works because the portfolio is small enough that individual trade
moves have a material impact on IM.

---

## What-If Scenario Results

Consistent across all runs:

| Scenario | Effect |
|----------|--------|
| Rates +50bp stress | +50.0% IM (linear in sensitivity scaling) |
| Unwind top 5 contributors | -2.5% to -45.1% (depends on concentration) |
| Add hedge (reverse top contributor) | -0.5% to -12.8% (depends on concentration) |
| IM ladder | Linear scaling (0.5x to 1.5x) |

Hedge and unwind effectiveness decreases with portfolio size: removing 5 trades from
50 (-45%) has more impact than from 10,000 (-2.7%).

---

## Reproducibility

All runs use the same random seed per trade count, producing identical portfolios
and deterministic results. The two 5K-trade runs (timestamps 12:57 and 13:06)
produce identical IM values and scenario results with <2% timing variance.

```
# Commands to reproduce
source venv/bin/activate
python benchmark_trading_workflow.py --trades 50 --portfolios 3 --threads 4 --new-trades 10 --optimize-iters 20
python benchmark_trading_workflow.py --trades 1000 --portfolios 5 --threads 8
python benchmark_trading_workflow.py --trades 5000 --portfolios 10 --threads 8 --optimize-iters 100
python benchmark_trading_workflow.py --trades 10000 --portfolios 50 --threads 8 --optimize-iters 100
python benchmark_trading_workflow.py --trades 50000 --portfolios 100 --threads 8 --optimize-iters 100
```
