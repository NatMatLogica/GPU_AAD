# AADC CPU vs CUDA GPU Benchmark Plan

## Target System: X13 8U GPU System

| Component | Spec |
|-----------|------|
| **GPU** | NVIDIA HGX H100 8-GPU (80GB HBM3 each) |
| **CPU** | Dual Intel Xeon Platinum 8568Y+ (5th Gen, 48C/96T each) |
| **Total CPU Cores** | 96 physical / 192 logical |
| **GPU Memory** | 640GB HBM3 total (8 x 80GB) |
| **Interconnect** | NVSwitch, NVLink 4.0 |

---

## Comparison Matrix

| Backend | Hardware | Gradient Method | Complexity | Gradients |
|---------|----------|----------------|------------|-----------|
| **AADC CPU** | Xeon 8568Y+ (96C) | AAD adjoint | O(1) passes | Exact |
| **CUDA GPU analytical** | H100 (1-8 GPUs) | Hand-coded chain rule | O(1) passes | Exact |
| **CUDA GPU bump-and-reval** | H100 (1-8 GPUs) | Finite difference | O(K+1) passes | Approx |

AADC is CPU-only (generates AVX2/512 SIMD code). No GPU AAD toolchain available.
The GPU bump-and-revalue tests whether raw GPU parallelism overcomes O(K) factor count.

---

## Publication Workflow

### 1. Environment Check

```bash
source venv/bin/activate
python -m benchmark.environment
```

### 2. Validate All Backends Match

```bash
python -m benchmark.benchmark_fair --trades 50 --portfolios 3 --validate-only
```

### 3. Validate SIMM Formula (Methodology Section)

```bash
python -m benchmark.validate_simm
```

### 4. Trade Scaling Sweep (Core Data)

```bash
python -m benchmark.sweep \
    --sweep-trades 50,100,500,1000,5000,10000 \
    --portfolios 5 --threads 96 --num-gpus 1 \
    --min-runs 30 --cost-per-hour YOUR_RATE --platform hgx_h100_onprem
```

### 5. CPU Thread Scaling

```bash
python -m benchmark.sweep \
    --trades 1000 --portfolios 5 \
    --sweep-threads 1,8,16,32,64,96 --min-runs 30
```

### 6. GPU Scaling

```bash
python -m benchmark.sweep \
    --trades 1000 --portfolios 5 \
    --sweep-gpus 1,2,4,8 --min-runs 30
```

### 7. Export for Publication

```bash
python -m benchmark.export results/sweep_*.csv --markdown --csv-summary
```

---

## Statistical Methodology

- **Minimum 30 runs** per configuration for statistical significance
- **Report format**: `median [P5-P95]` (standard for systems benchmarks)
- **95% confidence intervals** via t-distribution (or bootstrap)
- **IQR outlier exclusion**: runs > Q3 + 1.5 x IQR excluded (reported)
- **Coefficient of variation**: warn if CV > 5%

---

## Preparation Checklist

### 1. Environment Setup (do first)

```bash
# Check CUDA is available and GPU is detected
nvidia-smi
python -c "from numba import cuda; print(f'GPUs: {cuda.gpus}'); print(cuda.detect())"

# Verify AADC is installed
python -c "import aadc; print('AADC OK')"

# Verify numba CUDA (not simulator!) works
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

# Activate venv
cd /home/natashamanito/ISDA-SIMM
source venv/bin/activate
```

### 2. Validate Implementations Match (before timing)

```bash
# Cross-validate all backends produce identical IM and gradients
python -m benchmark.benchmark_fair --trades 50 --portfolios 3 --validate-only

# Expected: all backends pass with rel_err < 1e-6 for IM, < 1e-4 for gradient
# (bump-and-revalue uses looser tolerance: < 1e-2 for gradient)
```

### 3. CUDA Warm-up

Numba JIT-compiles CUDA kernels on first call. Run a small warm-up before benchmarking:

```bash
# Warm-up: small run to trigger JIT compilation
python -m benchmark.benchmark_fair --trades 10 --portfolios 2 --num-runs 3
```

---

## Benchmark Commands

### A. Fair Benchmark (recommended)

```bash
# Validate first
python -m benchmark.benchmark_fair --trades 100 --portfolios 3 --validate-only

# Publication-quality: all backends, with optimization, 30 runs
python -m benchmark.benchmark_fair \
    --trades 1000 --portfolios 5 \
    --trade-types ir_swap,equity_option \
    --threads 96 --min-runs 30 --optimize --opt-runs 3

# With cost tracking and GPU timing
python -m benchmark.benchmark_fair \
    --trades 1000 --portfolios 5 \
    --threads 96 --min-runs 30 \
    --cost-per-hour 3.50 --platform hgx_h100_onprem \
    --collect-gpu-timing --num-gpus 1
```

### B. Multi-GPU Benchmark

```bash
python -m benchmark.benchmark_fair \
    --trades 1000 --portfolios 5 \
    --threads 96 --min-runs 30 \
    --num-gpus 8
```

### C. Automated Sweeps

```bash
# Trade scaling
python -m benchmark.sweep \
    --sweep-trades 50,100,500,1000,5000,10000 \
    --portfolios 5 --threads 96 --min-runs 30

# Thread scaling
python -m benchmark.sweep \
    --trades 1000 --portfolios 5 \
    --sweep-threads 1,4,8,16,32,64,96 --min-runs 30

# GPU scaling
python -m benchmark.sweep \
    --trades 1000 --portfolios 5 \
    --sweep-gpus 1,2,4,8 --min-runs 30
```

---

## What Gets Compared

| Aspect | AADC CPU | CUDA GPU (analytical) | CUDA GPU (bump & reval) |
|--------|----------|----------------------|------------------------|
| **SIMM Evaluation** | AADC kernel + evaluate() | Numba CUDA kernel | Numba CUDA kernel |
| **Gradient** | AAD adjoint (exact, O(1)) | Chain rule on GPU (exact, O(1)) | Finite diff (approx, O(K+1)) |
| **Parallelism** | AADC ThreadPool | GPU warp/block | GPU warp/block |
| **SIMM Formula** | Full ISDA v2.6 | Full ISDA v2.6 | Full ISDA v2.6 |

### Key Metrics

1. **SIMM eval time** — median [P5-P95] with 95% CI
2. **Gradient time** — included in SIMM eval
3. **Optimization wall time** — median across repeats
4. **Total IM** — must match between implementations
5. **GPU timing breakdown** — H2D / kernel / D2H (optional)
6. **Cost per evaluation** — $/eval from $/hr rate

---

## Output Files

| File | Purpose |
|------|---------|
| `benchmark/results/benchmark_*.json` | Full results with environment |
| `benchmark/results/sweep_*.csv` | Sweep data (one row per config) |
| `benchmark/results/sweep_*.json` | Sweep data with environment |

### Export

```bash
# Markdown table for publication
python -m benchmark.export results/sweep_trades_*.csv --markdown

# CSV summary for appendix
python -m benchmark.export results/sweep_trades_*.csv --csv-summary
```

---

## Source Files

| File | Status | Purpose |
|------|--------|---------|
| `benchmark/environment.py` | NEW | HW/SW environment capture |
| `benchmark/cost.py` | NEW | Cost tracking ($/hr to $/eval) |
| `benchmark/sweep.py` | NEW | Automated parameter sweeps |
| `benchmark/export.py` | NEW | Markdown/CSV export for publication |
| `benchmark/validate_simm.py` | NEW | SIMM formula standalone validation |
| `benchmark/backends/cuda_bumpeval_backend.py` | NEW | Bump-and-revalue GPU baseline |
| `benchmark/benchmark_fair.py` | MODIFIED | Statistical rigor, opt repeats, env/cost |
| `benchmark/backends/cuda_backend.py` | MODIFIED | GPU timing breakdown, multi-GPU |
| `benchmark/simm_formula.py` | UNCHANGED | Shared SIMM formula (reference) |
| `benchmark/optimizer.py` | UNCHANGED | Shared optimizer |
| `benchmark/data_gen.py` | UNCHANGED | Deterministic data generation |
| `benchmark/backends/base.py` | UNCHANGED | Abstract backend interface |
| `benchmark/backends/aadc_backend.py` | UNCHANGED | AADC CPU backend |
| `benchmark/backends/numpy_backend.py` | UNCHANGED | NumPy reference backend |

---

## Fixes Applied Before Benchmark

| Issue | Fixed In | What Changed |
|-------|----------|-------------|
| Wrong IR tenors (7y/10y/15y -> 10y/15y/20y) | `model/trade_types.py` | ISDA-standard tenor points |
| Wrong PSI matrix in CUDA standalone | `model/simm_cuda.py` | Correct ISDA v2.6 cross-RC correlations |
| Simplified risk weights in CUDA | `model/simm_portfolio_cuda.py` | v2.6 tenor-specific IR weights |
| Different initial allocation | `model/simm_portfolio_cuda.py` | Uses generate_portfolio groups (same as AADC) |
| CUDA imports wrong PSI matrix | `model/simm_portfolio_cuda.py` | Now imports from simm_portfolio_aadc |
| Inconsistent logging schema | `model/simm_portfolio_cuda.py` | Uses common/logger.py SIMMLogger |
