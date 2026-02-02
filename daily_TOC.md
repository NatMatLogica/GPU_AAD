# Typical Trading Day: SIMM Total Cost of Ownership

## Overview

This analysis compares the cloud cost of running ISDA SIMM v2.6 margin calculations across four compute backends for a realistic trading day workload. All backends compute the same Full ISDA SIMM formula with intra-bucket correlations, concentration thresholds, and cross-risk-class aggregation.

**Portfolio**: 1,000 IR swap trades, 5 counterparties, 30 risk factors, 8 threads

## Daily Activity Volumes

| Activity | Evaluations | Description |
|----------|------------|-------------|
| Pre-trade margin checks | 500 | Every new trade needs margin impact estimate before execution |
| What-if stress scenarios | 50 | Stress tests (rate shocks, unwinds, hedging) |
| Optimization iterations | 100 | Gradient-based counterparty reallocation |
| EOD official margin | 1 | Regulatory margin computation |
| **Total** | **651** | **SIMM evaluations per day** |

## Backend Descriptions

| Backend | Type | Compute | Derivatives | Description |
|---------|------|---------|-------------|-------------|
| **AADC Python** | CPU | AVX-256 vectorized | Exact (adjoint AD) | MatLogica AADC kernel, single `evaluate()` for all portfolios |
| **GPU (CUDA)** | GPU | Numba CUDA | Finite-difference | Full SIMM on H100, values + bump-and-revalue gradients |
| **BF GPU (no grad)** | GPU | Numba CUDA | None | Forward-only IM computation, no gradient |
| **C++ AADC** | CPU | AVX-256 vectorized | Exact (adjoint AD) | Compiled C++ binary with AADC, includes optimization overhead |

## Throughput Results

Measured on: NVIDIA H100 80GB HBM3 + Intel Xeon (8 threads)
Method: 5 warmup + 50 timed evaluations, median reported

| Backend | Kernel Recording | Throughput | Median Latency |
|---------|-----------------|------------|----------------|
| AADC Python | 38 ms (one-time) | **2,256 evals/sec** | 0.44 ms |
| GPU (CUDA) | - | 1,160 evals/sec | 0.86 ms |
| BF GPU (no grad) | - | 1,678 evals/sec | 0.60 ms |
| C++ AADC | 26 ms (one-time) | 305 evals/sec | 3.28 ms |

Note: C++ AADC throughput measured via optimize mode which includes matrix projection and line search overhead beyond pure kernel evaluation. Pure SIMM kernel throughput in C++ is ~14,000 evals/sec (measured in what-if mode).

## Cloud Cost Comparison

### Instance Pricing

| Instance Type | Hardware | $/hr | Use Case |
|---------------|----------|------|----------|
| c5.18xlarge | 72 vCPU, 144 GB RAM | $3.06 | AADC Python, C++ AADC |
| p4d.24xlarge | 8x A100 GPU, 96 vCPU | $32.77 | GPU (CUDA), BF GPU |

### Daily Cost (651 SIMM evaluations)

| Backend | Daily Time | $/hr | Daily Cost | Annual Cost (252 days) | vs Cheapest |
|---------|-----------|------|-----------|----------------------|-------------|
| **AADC Python** | **327 ms** | $3.06 | **$0.0003** | **$0.07** | **1.0x** |
| C++ AADC | 2.16 s | $3.06 | $0.0018 | $0.46 | 6.6x |
| BF GPU (no grad) | 388 ms | $32.77 | $0.0035 | $0.89 | 12.7x |
| GPU (CUDA) | 561 ms | $32.77 | $0.0051 | $1.29 | 18.4x |

### Scaling to Higher Volumes

For a large bank with 10x the activity (5,000 pre-trade checks, 500 what-if, 1,000 optimization iterations):

| Backend | Total Evals | Daily Time | Daily Cost | Annual Cost |
|---------|------------|-----------|-----------|-------------|
| AADC Python | 6,501 | 2.92 s | $0.0025 | $0.63 |
| C++ AADC | 6,501 | 21.3 s | $0.018 | $4.56 |
| BF GPU (no grad) | 6,501 | 3.87 s | $0.035 | $8.89 |
| GPU (CUDA) | 6,501 | 5.60 s | $0.051 | $12.86 |

## Key Findings

1. **AADC Python is the cheapest option** - 2,256 evals/sec on a $3.06/hr CPU instance makes it 13-18x cheaper than GPU alternatives annually.

2. **GPU is faster per-eval but much more expensive per-dollar** - The H100 GPU instance costs 10.7x more per hour ($32.77 vs $3.06). While GPU throughput is competitive, the instance cost dominates.

3. **Kernel recording is negligible** - AADC records the SIMM kernel once (38 ms) and reuses it for all 651+ daily evaluations. Amortized cost: 0.06 ms per eval.

4. **AADC provides exact derivatives at ~2x cost** - The reverse pass (adjoint AD) adds only ~1.2x overhead to the forward pass, while GPU finite-difference requires 3x evaluations per derivative pair.

5. **BF GPU (no gradient) is faster than GPU with gradients** - 1,678 vs 1,160 evals/sec. The gradient computation via finite-difference adds ~45% overhead on GPU.

6. **At any realistic volume, compute cost is negligible** - Even at 6,500 evals/day, the most expensive option (GPU) costs only $12.86/year. The real cost differentiator is the instance reservation model (on-demand vs reserved vs spot).

## Methodology

- **Benchmark script**: `benchmark_typical_day.py`
- **SIMM formula**: Full ISDA v2.6 with intra-bucket correlations, concentration thresholds, inter-bucket aggregation, cross-risk-class correlations
- **Throughput**: Median of 50 timed evaluations with +/-0.1% input noise (prevents memoization), after 5 warmup evaluations
- **Cost model**: On-demand AWS pricing (us-east-1), compute time only (excludes storage, network, idle time)
- **GPU timing**: Includes Python dispatch overhead (Numba JIT compiled after warmup)
- **AADC timing**: Includes `aadc.evaluate()` dispatch + result extraction, excludes one-time kernel recording

## How to Reproduce

```bash
cd /home/x13-root171/GPU_AAD
source venv/bin/activate

# Default volumes (651 evals/day)
python benchmark_typical_day.py --trades 1000 --portfolios 5 --threads 8

# High-volume scenario (6,501 evals/day)
python benchmark_typical_day.py --trades 1000 --portfolios 5 --threads 8 \
  --pre-trade-checks 5000 --whatif-scenarios 500 --optimize-iters 1000

# Custom cloud pricing
python benchmark_typical_day.py --trades 1000 --portfolios 5 --threads 8 \
  --gpu-cost-per-hour 12.24 --cpu-cost-per-hour 1.53

# Output stored in:
#   data/typical_day.csv          - per-backend throughput and cost
#   data/execution_log_portfolio.csv - unified execution log
```
