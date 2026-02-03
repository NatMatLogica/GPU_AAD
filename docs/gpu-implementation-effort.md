# GPU Implementation Effort Analysis

This document analyzes the development effort required for different GPU implementation strategies, providing a fair basis for comparing AADC vs GPU performance claims.

## Current Implementation Status

The current GPU implementation covers **SIMM aggregation only** — it takes pre-computed sensitivities as input and computes IM + gradients on GPU. The pricers that generate those sensitivities run on CPU.

```
Current:  CPU Pricers → CRIF → [Transfer] → GPU SIMM Aggregation
                               ~~~~~~~~~~~~
                               PCIe bottleneck
```

### What Was Built

| Component | Lines | Tokens | Effort | Description |
|-----------|-------|--------|--------|-------------|
| `cuda_backend.py` | 405 | ~3,500 | 2-3 days | Full SIMM kernel with analytical gradient |
| `cuda_bumpeval_backend.py` | 209 | ~1,700 | 1 day | Finite-difference variant |
| `simm_portfolio_cuda.py` | 2,425 | ~23,000 | 5-7 days | Full portfolio wrapper (unused) |
| **Total GPU work** | ~3,000 | ~28,000 | **8-10 days** | Aggregation only |

## What a Full GPU Implementation Would Require

A truly fair comparison would have pricers running on GPU, eliminating the CPU→GPU transfer bottleneck:

```
Full GPU:  GPU Pricers → GPU Sensitivities → GPU SIMM → GPU Gradients
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           Everything stays in GPU memory (HBM)
```

### C++ Pricer Sizes (Baseline)

| Asset Class | Lines | Tokens | Complexity |
|-------------|-------|--------|------------|
| IR Swap (`vanilla_irs.h`) | 44 | ~350 | Simple loop over periods |
| Equity Option (`equity_option.h`) | 46 | ~340 | Black-Scholes formula |
| FX Option (`fx_option.h`) | 48 | ~380 | Garman-Kohlhagen |
| Inflation Swap (`inflation_swap.h`) | 34 | ~290 | Similar to IR swap |
| XCCY Swap (`xccy_swap.h`) | 62 | ~540 | Two curves + FX |
| **Total** | 234 | ~1,900 | |

---

## Option A: Brute-Force GPU Pricers (Finite Difference)

Port each C++ pricer to CUDA. Compute sensitivities via bump-and-revalue (finite difference).

### Effort Breakdown

| Component | CUDA Lines (est.) | Tokens (est.) | Effort |
|-----------|-------------------|---------------|--------|
| IR Swap kernel | ~150 | ~600 | 0.5 day |
| Equity Option kernel | ~180 | ~700 | 0.5 day |
| FX Option kernel | ~200 | ~800 | 0.5 day |
| Inflation Swap kernel | ~150 | ~600 | 0.5 day |
| XCCY Swap kernel | ~250 | ~1,000 | 1 day |
| Market data structs (GPU) | ~200 | ~800 | 0.5 day |
| Memory management | ~300 | ~1,200 | 1 day |
| Integration & testing | ~400 | ~1,600 | 2 days |
| **Total** | **~1,800** | **~7,300** | **6-7 days** |

### Pros/Cons

✅ Straightforward mechanical translation
✅ No new frameworks needed
⚠️ Finite-difference gradients (less accurate than AADC's exact gradients)
⚠️ O(K) evaluations per trade for sensitivities

---

## Option B: Full GPU with Custom AAD Engine

Implement GPU-native automatic adjoint differentiation from scratch.

### Effort Breakdown

| Component | Effort | Notes |
|-----------|--------|-------|
| CUDA pricers (same as Option A) | 6-7 days | Base pricing kernels |
| GPU tape recording | 5-8 days | Operation graph on device |
| GPU adjoint pass | 5-8 days | Reverse-mode differentiation |
| Memory pool management | 3-5 days | Efficient GPU allocation |
| Kernel fusion & optimization | 3-5 days | Performance tuning |
| Integration & testing | 3-5 days | End-to-end validation |
| **Total** | **25-40 days** | High complexity |

### Pros/Cons

✅ Exact gradients (matches AADC accuracy)
✅ Full control over implementation
❌ Significant development effort
❌ Maintenance burden (custom AD engine)
❌ Risk of bugs in AD logic

---

## Option C: Use Existing GPU AD Framework

Leverage existing frameworks that provide GPU + automatic differentiation.

### JAX Approach

```python
import jax
import jax.numpy as jnp

@jax.jit
def price_ir_swap(rates, trade_params):
    # JAX-native implementation
    ...
    return npv

# Automatic gradient on GPU
grad_fn = jax.grad(price_ir_swap)
sensitivities = grad_fn(rates, trade_params)
```

### Framework Comparison

| Framework | Effort | GPU | AD | Notes |
|-----------|--------|-----|-----|-------|
| **JAX** | 5-7 days | ✅ | ✅ | Functional style, XLA compilation |
| **PyTorch** | 5-7 days | ✅ | ✅ | Imperative style, dynamic graphs |
| **Enzyme** | 10-15 days | ✅ | ✅ | LLVM-based, keeps C++ code |
| **CuPy + custom** | 8-12 days | ✅ | Manual | NumPy API, manual gradients |

### Pros/Cons

✅ Exact gradients
✅ Maintained by large communities
✅ Reasonable development effort
⚠️ Requires rewriting pricers in new framework
⚠️ May have framework overhead

---

## Token → Man-Day Conversion Guide

Based on this project's complexity:

| Task Type | Tokens/Day | Examples |
|-----------|------------|----------|
| Straightforward port | 1,500-2,000 | C++ → CUDA mechanical translation |
| New algorithm | 800-1,200 | SIMM kernel with correlations |
| Integration/debugging | 500-800 | Testing, edge cases, validation |
| Framework code | 400-600 | AD engine, memory management |

### Complexity Multipliers

| Factor | Multiplier |
|--------|------------|
| Well-documented source | 1.0x |
| Unfamiliar codebase | 1.5x |
| Numerical precision requirements | 1.3x |
| Multi-GPU support | 1.5x |
| Production hardening | 2.0x |

---

## Summary: Implementation Options

| Option | New Code | Tokens | Effort | Gradient Type | Fair Comparison? |
|--------|----------|--------|--------|---------------|------------------|
| **Current** (aggregation only) | 3,000 lines | ~28K | 8-10 days | Analytical | ⚠️ Partial |
| **A: + Brute-force pricers** | +1,800 lines | +7K | +6-7 days | Finite diff | ✅ Fair |
| **B: + Custom GPU AAD** | +5,000 lines | +20K | +25-40 days | Exact (AAD) | ✅ Fair |
| **C: JAX/PyTorch rewrite** | ~3,000 lines | ~12K | 5-7 days | Exact (AD) | ✅ Fair |

---

## Recommendation

For a **fair benchmark comparison**, the minimum viable approach is:

1. **Option A (+6-7 days)**: Add CUDA pricers with bump-and-revalue. This eliminates the CPU→GPU transfer bottleneck. GPU would use finite-difference gradients vs AADC's exact gradients, which is a known trade-off.

2. **Option C with JAX (+5-7 days)**: If exact gradients are required for fairness, JAX provides the fastest path to GPU + automatic differentiation without building a custom AD engine.

### Important Caveat

Even with full GPU pricers, **AADC is expected to remain faster at typical portfolio scales (50-500)**. The GPU underutilization problem (filling only 1-2 of 132 SMs on H100) is fundamental at these scales. The crossover point where GPU wins is ~10,000+ portfolios.

The value of implementing full GPU pricers is **methodological fairness** — ensuring the comparison measures algorithm efficiency, not implementation completeness.

---

## Appendix: Code Metrics

### Current Codebase

```
C++ Pricers:           234 lines  (~1,900 tokens)
C++ SIMM Aggregation:  821 lines  (~8,300 tokens)
GPU SIMM Backend:      614 lines  (~5,200 tokens)
GPU Portfolio Wrapper: 2,425 lines (~23,000 tokens)
```

### Estimated Full GPU Implementation

```
GPU Pricers:           1,800 lines (~7,300 tokens)
GPU Memory Mgmt:         300 lines (~1,200 tokens)
Integration:             400 lines (~1,600 tokens)
─────────────────────────────────────────────────
Additional:            2,500 lines (~10,100 tokens)
```
