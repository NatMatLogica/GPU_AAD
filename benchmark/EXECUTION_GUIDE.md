# ISDA-SIMM Benchmark Execution Guide

Publication-quality benchmarking of AADC CPU vs CUDA GPU for ISDA SIMM v2.6
initial margin computation on NVIDIA HGX H100 8-GPU.

---

## Target System

| Component | Spec |
|-----------|------|
| **CPU** | Dual Intel Xeon Platinum 8568Y+ (48C/96T each, 96 physical cores) |
| **GPU** | 8x NVIDIA H100 80GB HBM3 (NVSwitch + NVLink 4.0) |
| **RAM** | 2TB DDR5 |
| **OS** | Linux (Debian 12 / Ubuntu 22.04) |

---

## Phase 0: Prerequisites

```bash
cd /home/natashamanito/ISDA-SIMM
source venv/bin/activate

# Ensure pip dependencies are installed
pip install scipy   # for t-distribution CI computation
```

---

## Phase 1: Environment Verification

Confirm hardware detection and backend availability before collecting any data.

### 1.1 Print environment snapshot

```bash
python -m benchmark.environment
```

**Expected output:**
```
=== Benchmark Environment ===
OS:         Linux-6.x-x86_64-with-glibc2.xx
Python:     3.11.x
NumPy:      2.x.x
Numba:      0.63.x
CUDA:       12.x
AADC:       installed (no version)
CPU:        Intel(R) Xeon(R) Platinum 8568Y+
Cores:      96 physical / 192 logical
CPU MHz:    2700
GPU 0:      NVIDIA H100 80GB HBM3 (81920 MB, CC 9.0, driver 5xx.xx)
GPU 1:      NVIDIA H100 80GB HBM3 (81920 MB, CC 9.0, driver 5xx.xx)
...
GPU 7:      NVIDIA H100 80GB HBM3 (81920 MB, CC 9.0, driver 5xx.xx)
Git hash:   <12-char hash>
Seed:       42
```

### 1.2 Verify CUDA kernels compile and execute

```bash
python -c "
from numba import cuda
import numpy as np
@cuda.jit
def test_kernel(out):
    out[0] = 42.0
out = np.zeros(1)
d_out = cuda.to_device(out)
test_kernel[1, 1](d_out)
d_out.copy_to_host(out)
assert out[0] == 42.0
print('CUDA kernel OK')
"
```

### 1.3 Verify AADC

```bash
python -c "import aadc; print('AADC OK')"
```

---

## Phase 2: Correctness Validation

All backends must agree on IM and gradient values before timing is meaningful.

### 2.1 Cross-validate all backends

```bash
python -m benchmark.benchmark_fair \
    --trades 50 --portfolios 3 --validate-only
```

**Expected output:**
```
Step 3: Validating backend consistency...
  aadc vs numpy: IM rel_err=0.00e+00 (OK), grad rel_err=0.00e+00 (OK) [PASS]
  cuda vs numpy: IM rel_err=0.00e+00 (OK), grad rel_err=0.00e+00 (OK) [PASS]
  cuda_bumpeval vs numpy: IM rel_err=0.00e+00 (OK), grad rel_err=5.xx-03 (OK) [PASS]

  All backends produce consistent results.
```

The bump-and-revalue backend uses finite differences, so its gradient tolerance
is looser (< 1e-2) compared to the analytical backends (< 1e-4).

### 2.2 Validate SIMM formula (methodology section evidence)

```bash
python -m benchmark.validate_simm
```

**Expected output:**
```
Results: 7/7 tests passed
ALL TESTS PASSED
```

This validates hand-calculable test cases: single factor, correlated factors,
cross-risk-class aggregation, gradient vs finite difference, batch consistency,
and PSI matrix properties.

---

## Phase 3: Quick Sanity Benchmark

A fast preliminary run to verify the harness works end-to-end before
committing to the full publication runs.

```bash
python -m benchmark.benchmark_fair \
    --trades 100 --portfolios 3 --min-runs 10 --threads 96 --num-gpus 1
```

**Expected output** (approximate, on H100 system):
```
Backend         Median [P5-P95] (ms)   Mean (ms)    CI 95%              CV   Total IM        Speedup
--------------------------------------------------------------------------------------------------------------
  numpy           2.xxx [x.xx-x.xx]       2.xxx  [x.xx-x.xx]          x.x%  $x.xxB           1.00x
  aadc            0.xxx [x.xx-x.xx]       0.xxx  [x.xx-x.xx]          x.x%  $x.xxB           x.xx
  cuda            0.0xx [x.xx-x.xx]       0.0xx  [x.xx-x.xx]          x.x%  $x.xxB          xx.xx
  cuda_bumpeval   0.xxx [x.xx-x.xx]       0.xxx  [x.xx-x.xx]          x.x%  $x.xxB           x.xx
```

Verify: total IM matches across all backends (last column).

---

## Phase 4: Publication Runs

### 4.1 Trade Scaling Sweep (Core Data for Paper)

This is the primary dataset. Measures how each backend scales as portfolio
complexity (number of trades and risk factors) increases.

```bash
python -m benchmark.sweep \
    --sweep-trades 50,100,500,1000,5000,10000 \
    --portfolios 5 --threads 96 --num-gpus 1 \
    --min-runs 30 \
    --cost-per-hour 3.50 --platform hgx_h100_8gpu_onprem
```

**Runtime estimate:** The sweep runs 4 backends x 6 trade counts x 30 runs each
(720 timed evaluations per backend). Data generation dominates for large trade
counts. Expect 15-30 minutes total.

**Output files:**
```
benchmark/results/sweep_trades_YYYYMMDD_HHMMSS.csv
benchmark/results/sweep_trades_YYYYMMDD_HHMMSS.json
```

#### Predicted Results: Trade Scaling

Assumptions:
- K (risk factors) scales roughly as 13*num_currencies + extras from equity.
  For 2 trade types (ir_swap, equity_option) with 3 currencies:
  ~65 factors at 50 trades, ~70 at 100, ~90 at 500, ~110 at 1000, ~150 at 5000, ~180 at 10000.
- SIMM kernel is O(K^2) per portfolio (intra-bucket correlation double loop).
- NumPy: pure Python loops, single-threaded reference. ~1.5ms at K=65.
- AADC: SIMD-vectorized adjoint kernel, 96 threads. Amortized cost decreases
  with portfolio count; dominant cost is kernel dispatch overhead.
- CUDA analytical: One thread per portfolio, O(K^2) compute per thread.
  H2D/kernel/D2H pipeline; kernel is sub-microsecond for small P.
- CUDA bump-and-reval: K+1 SIMM evaluations per portfolio, all parallel.
  O(K^3) total FLOPs but fully parallelized.

| Backend | Trades=50 | 100 | 500 | 1,000 | 5,000 | 10,000 |
|---------|-----------|-----|-----|-------|-------|--------|
| | K~65 | K~70 | K~90 | K~110 | K~150 | K~180 |
| **NumPy (ref)** | 1.5 ms | 1.8 ms | 4.5 ms | 8.0 ms | 32 ms | 75 ms |
| **AADC CPU (96T)** | 0.4 ms | 0.5 ms | 0.8 ms | 1.2 ms | 3.5 ms | 7.0 ms |
| **CUDA 1xH100** | 0.06 ms | 0.07 ms | 0.12 ms | 0.20 ms | 0.6 ms | 1.2 ms |
| **CUDA B&R 1xH100** | 0.15 ms | 0.20 ms | 0.8 ms | 2.0 ms | 12 ms | 35 ms |

Predicted speedup vs NumPy at 1,000 trades:

| Backend | Speedup |
|---------|---------|
| AADC CPU (96T) | ~6-8x |
| CUDA 1xH100 analytical | ~35-50x |
| CUDA B&R 1xH100 | ~3-5x |

Key insight: CUDA analytical dominates because it computes IM + exact gradient
in a single O(K^2) kernel. Bump-and-reval pays O(K) penalty per portfolio,
eroding GPU parallelism advantage at large K. AADC benefits from SIMD
vectorization and multi-threading but is ultimately memory-bandwidth-limited
on CPU.

### 4.2 CPU Thread Scaling

Measures AADC parallel efficiency as thread count increases on the dual-socket
Xeon 8568Y+ (96 physical cores, 192 logical with HT).

```bash
python -m benchmark.sweep \
    --trades 1000 --portfolios 5 \
    --sweep-threads 1,4,8,16,32,64,96 \
    --min-runs 30
```

**Runtime estimate:** 7 thread configurations x ~30 runs each. ~10 minutes.

**Output files:**
```
benchmark/results/sweep_threads_YYYYMMDD_HHMMSS.csv
benchmark/results/sweep_threads_YYYYMMDD_HHMMSS.json
```

#### Predicted Results: CPU Thread Scaling

AADC generates AVX2/512 SIMD code and distributes P portfolio evaluations
across threads via its ThreadPool. With P=5 portfolios, thread scaling is
limited by the small parallel work unit. The SIMM kernel per portfolio is
compute-bound (O(K^2) correlation sums) with good cache locality.

| Threads | AADC Median (ms) | Parallel Efficiency |
|---------|-----------------|---------------------|
| 1 | 5.0 | 100% (baseline) |
| 4 | 1.5 | 83% |
| 8 | 1.2 | 52% |
| 16 | 1.1 | 28% |
| 32 | 1.05 | 15% |
| 64 | 1.0 | 8% |
| 96 | 1.0 | 5% |

Efficiency drops sharply because P=5 portfolios means only 5 independent work
units. At T>8, threads compete for the same few portfolios. Scaling improves
significantly with larger portfolio counts. At P=100, expect near-linear
scaling up to ~32 threads.

NUMA effects: with dual-socket, threads >48 cross the socket boundary. Expect
a ~10% penalty for NUMA-remote memory access at 64+ threads.

### 4.3 GPU Scaling

Measures multi-GPU scaling by partitioning portfolios across H100 devices.

```bash
python -m benchmark.sweep \
    --trades 1000 --portfolios 5 \
    --sweep-gpus 1,2,4,8 \
    --min-runs 30
```

**Runtime estimate:** 4 GPU configurations x ~30 runs each. ~5 minutes.

**Output files:**
```
benchmark/results/sweep_gpus_YYYYMMDD_HHMMSS.csv
benchmark/results/sweep_gpus_YYYYMMDD_HHMMSS.json
```

#### Predicted Results: GPU Scaling

With P=5 portfolios partitioned across N GPUs, each GPU processes ceil(5/N)
portfolios. The kernel is extremely lightweight (~microseconds), so H2D/D2H
transfer and launch overhead dominate.

| GPUs | CUDA Median (ms) | Speedup vs 1 GPU | Efficiency |
|------|------------------|-------------------|------------|
| 1 | 0.20 | 1.0x | 100% |
| 2 | 0.15 | 1.3x | 65% |
| 4 | 0.12 | 1.7x | 42% |
| 8 | 0.11 | 1.8x | 23% |

Multi-GPU scaling is poor at P=5 because:
1. Only 5 portfolios to distribute (1 GPU gets 1 portfolio, 3 GPUs idle at N=8)
2. Per-device H2D/D2H overhead is fixed (~0.05-0.1ms per GPU context switch)
3. Sequential device launch (not overlapped) adds latency

Multi-GPU becomes valuable at P >= 100, where each GPU processes a meaningful
batch. For P=1000: expect near-linear scaling up to 4 GPUs, ~3.5x at 8 GPUs.

### 4.4 Full Benchmark with Optimization (Table 2 for Paper)

```bash
python -m benchmark.benchmark_fair \
    --trades 1000 --portfolios 5 \
    --trade-types ir_swap,equity_option \
    --threads 96 --min-runs 30 \
    --optimize --opt-runs 3 --max-iters 100 \
    --cost-per-hour 3.50 --platform hgx_h100_8gpu_onprem \
    --collect-gpu-timing --num-gpus 1
```

**Runtime estimate:** 30 timed runs per backend + 3 optimization runs per backend
(each up to 100 iterations). ~5-10 minutes.

#### Predicted Results: Optimization

All backends start from the same initial allocation (seed=42) and run identical
gradient descent with simplex projection. The optimizer converges to the same
final IM (within numerical tolerance) regardless of backend.

| Backend | Eval/Iter (ms) | Total Opt (ms) | Final IM | Iterations |
|---------|---------------|----------------|----------|------------|
| NumPy | 8.0 | 1,200 | $X.XXB | ~80 |
| AADC (96T) | 1.2 | 250 | $X.XXB | ~80 |
| CUDA 1xH100 | 0.2 | 80 | $X.XXB | ~80 |
| CUDA B&R | 2.0 | 400 | $X.XXB | ~80 |

Key: all backends converge to identical final IM and iteration count because
the optimizer is deterministic. The only difference is wall-clock time per
SIMM+gradient evaluation.

### 4.5 GPU Timing Breakdown (Supplementary Data)

```bash
python -m benchmark.benchmark_fair \
    --trades 1000 --portfolios 5 \
    --threads 96 --min-runs 30 \
    --collect-gpu-timing --num-gpus 1
```

#### Predicted GPU Timing Breakdown (1000 trades, K~110)

| Phase | Time (ms) | % of Total |
|-------|-----------|------------|
| H2D transfer | 0.02-0.05 | 15-25% |
| Kernel execution | 0.05-0.10 | 40-55% |
| D2H transfer | 0.02-0.04 | 10-20% |
| **Total** | **0.10-0.20** | **100%** |

At small K, transfer overhead dominates. At large K (10,000 trades, K~180),
the kernel becomes dominant (O(K^2) per portfolio).

---

## Phase 5: Export for Publication

### 5.1 Generate Markdown tables

```bash
python -m benchmark.export \
    benchmark/results/sweep_trades_*.csv \
    --markdown \
    --title "ISDA-SIMM Benchmark: AADC CPU vs CUDA GPU (H100)" \
    --output benchmark/results/publication_tables.md
```

### 5.2 Generate CSV summary

```bash
python -m benchmark.export \
    benchmark/results/sweep_trades_*.csv \
    --csv-summary \
    --output benchmark/results/publication_summary.csv
```

### 5.3 Combined export from all sweeps

```bash
python -m benchmark.export \
    benchmark/results/sweep_*.csv \
    --markdown --csv-summary \
    --output benchmark/results/full_publication_export.md
```

**Expected Markdown output:**
```
## ISDA-SIMM Benchmark: AADC CPU vs CUDA GPU (H100)

| Backend              | Trades | Median (ms) | P95 (ms) | Speedup |  Cost ($/eval) |
|----------------------|--------|-------------|----------|---------|----------------|
| NumPy (ref)          |   1000 |       8.000 |     9.20 |    1.0x |        $0.0078 |
| AADC CPU             |   1000 |       1.200 |     1.50 |    6.7x |        $0.0012 |
| CUDA GPU             |   1000 |       0.200 |     0.28 |   40.0x |        $0.0002 |
| CUDA B&R             |   1000 |       2.000 |     2.80 |    4.0x |        $0.0019 |
```

---

## Phase 6: Supplementary Experiments (Optional)

### 6.1 Large-scale trade test

For demonstrating scaling to production portfolio sizes:

```bash
python -m benchmark.benchmark_fair \
    --trades 10000 --portfolios 10 \
    --threads 96 --min-runs 30 --num-gpus 1
```

### 6.2 Multi-GPU with large portfolio count

```bash
python -m benchmark.benchmark_fair \
    --trades 1000 --portfolios 100 \
    --threads 96 --min-runs 30 --num-gpus 8
```

At P=100, multi-GPU partitioning should show better efficiency (~5-6x at 8 GPUs).

### 6.3 Repeated optimization for convergence analysis

```bash
python -m benchmark.benchmark_fair \
    --trades 1000 --portfolios 5 \
    --threads 96 --optimize --opt-runs 10 --max-iters 200
```

Captures median convergence behavior across 10 independent runs.

---

## Summary of Predicted Key Results

### Headline Numbers (1,000 trades, 5 portfolios, K~110 factors)

| Backend | Hardware | Median (ms) | Speedup vs NumPy | Method |
|---------|----------|-------------|------------------|--------|
| NumPy | Xeon 8568Y+ (1 core) | ~8.0 | 1.0x | Python loops |
| AADC | Xeon 8568Y+ (96 cores) | ~1.2 | ~6.7x | SIMD + AAD adjoint |
| CUDA analytical | 1x H100 | ~0.20 | ~40x | GPU chain rule |
| CUDA bump-and-reval | 1x H100 | ~2.0 | ~4x | GPU finite diff |

### Scaling Characteristics

| Dimension | AADC CPU | CUDA GPU |
|-----------|----------|----------|
| **Trade scaling** (K) | O(K^2), single-digit ms up to K=200 | O(K^2) per thread, sub-ms up to K=200 |
| **Portfolio scaling** (P) | Near-linear to ~32T, then plateaus | Near-linear on single GPU (P is block count) |
| **Thread/GPU scaling** | Good to ~P threads, diminishing after | Poor at P=5, good at P>=100 |

### Cost Comparison (at $3.50/hr)

| Backend | Cost per SIMM eval | Cost for 100-iter optimization |
|---------|--------------------|--------------------------------|
| NumPy | $7.8e-6 | $0.0012 |
| AADC (96T) | $1.2e-6 | $0.00024 |
| CUDA 1xH100 | $1.9e-7 | $0.000078 |

### What Drives the Results

1. **CUDA analytical >> AADC**: GPU parallelism across portfolios + O(1) gradient
   in a single fused kernel. H100's 80 SM x 128 CUDA cores = 10,240 threads
   vs CPU's 96 cores. Each SIMM evaluation is a small, regular computation
   that maps well to GPU architecture.

2. **CUDA analytical >> CUDA B&R**: Both use the same GPU hardware. The
   analytical kernel computes IM + gradient in one O(K^2) pass. Bump-and-reval
   requires K+1 separate O(K^2) evaluations (one per bumped factor). Even
   though all K+1 are launched in parallel, the total FLOP count is K+1 times
   higher. At K=110, this is a ~100x computational penalty, partially recovered
   by GPU parallelism.

3. **AADC >> NumPy**: AADC generates AVX-512 SIMD code and distributes across
   96 cores. NumPy reference uses explicit Python loops over portfolios and
   risk classes (no vectorized kernel). AADC's adjoint mode computes the
   gradient in a single reverse pass with O(1) overhead vs the forward pass.

### Caveats for the Paper

- **Small P caveat**: P=5 portfolios underutilizes both AADC ThreadPool and
  GPU parallelism. Production SIMM systems evaluate 100-10,000 portfolios.
  GPU advantage increases with P (more threads occupied).

- **K capped at 200**: The CUDA kernel uses `cuda.local.array(200)` for
  thread-local storage. Portfolios with >200 risk factors would need a
  kernel redesign (shared memory tiling or multiple passes).

- **Numba JIT overhead**: First CUDA kernel invocation includes JIT
  compilation (~2-5 seconds). All benchmarks include warmup runs to
  exclude this cost.

- **AADC version**: AADC does not expose a version string. The benchmark
  records "installed (no version)" — note this in reproducibility section.

---

## Quick Reference: All Commands

```bash
# Phase 1: Environment
python -m benchmark.environment

# Phase 2: Validation
python -m benchmark.benchmark_fair --trades 50 --portfolios 3 --validate-only
python -m benchmark.validate_simm

# Phase 3: Sanity check
python -m benchmark.benchmark_fair --trades 100 --portfolios 3 --min-runs 10 --threads 96

# Phase 4: Publication runs
python -m benchmark.sweep --sweep-trades 50,100,500,1000,5000,10000 \
    --portfolios 5 --threads 96 --num-gpus 1 --min-runs 30 \
    --cost-per-hour 3.50 --platform hgx_h100_8gpu_onprem

python -m benchmark.sweep --trades 1000 --portfolios 5 \
    --sweep-threads 1,4,8,16,32,64,96 --min-runs 30

python -m benchmark.sweep --trades 1000 --portfolios 5 \
    --sweep-gpus 1,2,4,8 --min-runs 30

python -m benchmark.benchmark_fair --trades 1000 --portfolios 5 \
    --threads 96 --min-runs 30 --optimize --opt-runs 3 --max-iters 100 \
    --cost-per-hour 3.50 --platform hgx_h100_8gpu_onprem \
    --collect-gpu-timing --num-gpus 1

# Phase 5: Export
python -m benchmark.export benchmark/results/sweep_*.csv --markdown \
    --title "ISDA-SIMM Benchmark: AADC CPU vs CUDA GPU (H100)" \
    --output benchmark/results/publication_tables.md

python -m benchmark.export benchmark/results/sweep_*.csv --csv-summary \
    --output benchmark/results/publication_summary.csv
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA not available` | No GPU or driver mismatch | Check `nvidia-smi`; ensure numba CUDA toolkit matches driver |
| `AADC is not available` | AADC not installed in venv | `pip install aadc` or check MatLogica license |
| `CV > 5% WARNING` | Noisy measurements | Increase `--min-runs` to 50+; check for background load |
| `Validation FAILED` | Backend mismatch | Run `--validate-only` with small trades; check CRIF generation |
| `scipy not found` | CI uses t-distribution | `pip install scipy`; falls back to z-approximation |
| `K > 200 factors` | Exceeds kernel local array | Reduce trades or currencies; kernel redesign needed |
| JIT slow on first run | Numba compilation | Normal; warmup runs exclude this |
