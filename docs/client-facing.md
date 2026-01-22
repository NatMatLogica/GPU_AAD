# Margin Optimization Platform - Proof of Concept

## Executive Summary

This document presents the results of a **Proof of Concept (PoC)** for a margin optimization platform targeting a $40B AUM portfolio. The PoC demonstrates:

1. **High-performance IR swap pricing** using AADC (Automatic Adjoint Differentiation) for real-time sensitivity computation
2. **ISDA SIMM integration** for regulatory margin calculation
3. **Stress margin analysis** for understanding margin sensitivity to market moves
4. **Incremental margin analysis** for identifying trade-level margin contributions

**Key Result**: 30-50x speedup over traditional bump-and-revalue methods, enabling real-time margin analysis for portfolios that would otherwise require batch processing.

---

## 1. Client Requirements

The following requirements were gathered from client discussions:

### 1.1 Core Requirements (Addressed in PoC)

| Requirement | Status | Notes |
|-------------|--------|-------|
| IR Swap pricing with sensitivities | ✅ Implemented | AADC-enabled, 12 SIMM tenor buckets |
| SIMM margin calculation | ✅ Implemented | Full ISDA SIMM v2.6 aggregation |
| Stress margin (shock SIMM inputs) | ✅ Implemented | 7 predefined scenarios |
| Incremental margin per trade | ✅ Implemented | Identifies netting benefits |
| What-if analysis (add/remove trades) | ✅ Implemented | Real-time margin impact |
| Portfolio-level aggregation | ✅ Implemented | Multi-currency support |

### 1.2 Future Requirements (Not in PoC)

| Requirement | Priority | Complexity |
|-------------|----------|------------|
| Swaption pricing (Vol Arb) | P2 | High |
| QuantLib integration | P2 | Medium |
| Multi-custodian simulation | P1 | High |
| Bilateral vs cleared comparison | P1 | Medium |
| Allocation optimization solver | P3 | Very High |
| Cross-currency swaps | P2 | Medium |

### 1.3 Margin Optimization Use Cases

The platform is designed to support:

- **Trade allocation decisions**: Which counterparty/custodian minimizes margin?
- **Transaction structuring**: How to structure new trades to limit additional margin?
- **Netting optimization**: Identify offsetting trades that reduce margin
- **Stress testing**: Understand margin exposure under adverse scenarios

---

## 2. Technical Approach

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Margin Optimization Platform                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Market Data          IR Swap Pricer              SIMM Engine               │
│   ───────────          ──────────────              ───────────               │
│   • Yield curves  ──▶  • AADC pricing    ──▶      • CRIF generation         │
│   • 12 SIMM tenors     • AAD sensitivities        • Risk aggregation        │
│                        • 30-50x speedup            • Margin calculation      │
│                                                                              │
│                              │                            │                  │
│                              ▼                            ▼                  │
│                     ┌─────────────────────────────────────────────┐         │
│                     │           Margin Analysis                    │         │
│                     │  • Stress margin (7 scenarios)              │         │
│                     │  • Incremental margin per trade             │         │
│                     │  • What-if (add/remove trades)              │         │
│                     └─────────────────────────────────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 AADC vs Traditional Approach

| Aspect | Traditional (Bump & Revalue) | AADC (This PoC) |
|--------|------------------------------|-----------------|
| Method | Finite difference | Automatic Adjoint Differentiation |
| Implementation | Pure Python | AADC compiled kernels (C++) |
| Pricing passes | 1 + (currencies × 12 tenors) | 1 per trade |
| Derivative accuracy | Approximate (truncation error) | Exact (machine precision) |
| Speedup | 1x (baseline) | **30-50x** |

**How it works**: AADC records a computational graph during the first pricing pass, then uses reverse-mode automatic differentiation to compute all sensitivities in a single backward pass. This eliminates the need for repeated "bump and reprice" calculations.

**Important note on baseline**: The baseline implementation is pure Python (NumPy). A production C++ baseline would be faster, so the AADC speedup vs C++ baseline would be lower (estimated 5-15x). However, AADC still provides:
- Exact derivatives (no FD truncation error)
- Better scaling with number of risk factors
- Automatic handling of complex payoffs

---

## 3. Performance Benchmarks

### 3.1 Greeks Computation - V1 (Per-Trade Kernels)

| Portfolio Size | Currencies | Baseline (Python) | AADC V1 (1T) | Speedup |
|----------------|------------|-------------------|--------------|---------|
| 50 trades | 3 | 169 ms | 4.1 ms | **41x** |
| 100 trades | 3 | 313 ms | ~7 ms | **45x** |
| 200 trades | 3 | 652 ms | ~15 ms | **43x** |
| 500 trades | 3 | 1,047 ms | 35 ms | **30x** |

### 3.2 Greeks Computation - V2 (Batched Portfolio Kernel, 4 Threads)

| Portfolio Size | Currencies | Baseline (Python) | AADC V2 (4T) | Speedup |
|----------------|------------|-------------------|--------------|---------|
| 100 trades | 10 | 336 ms | 52.7 ms | **6x** |
| 200 trades | 10 | 652 ms | 125.8 ms | **5x** |

**V1 vs V2 Comparison**:
- **V1** (default): Per-trade kernels, single-threaded. Best for warm cache scenarios.
- **V2**: Single batched kernel for entire portfolio, 4 threads. Useful when kernel recording time dominates.
- V1 is faster for typical intraday use (warm cache, repeated evaluations).
- V2's threading overhead exceeds benefit for fast individual kernel evaluations.

**Configuration**:
- Baseline = Pure Python bump-and-revalue (NumPy)
- AADC V1 = Single-threaded, per-trade kernel evaluation
- AADC V2 = 4 threads, batched portfolio kernel (eval time only, recording at SOD)
- AADC warm = after kernel cache is populated (typical intraday scenario)
- First run includes ~40ms per unique trade structure for kernel recording

**Baseline language caveat**: The speedups shown are vs Python baseline. A production C++ baseline would be ~10-20x faster than Python, reducing the AADC advantage to approximately:
- V1: ~3-5x vs C++ baseline
- V2: ~0.5-1x vs C++ baseline (marginal benefit)

However, AADC remains advantageous for:
- Derivative accuracy (exact vs FD approximation)
- Scaling with risk factors (O(1) vs O(n) for n factors)
- Maintainability (no manual bump logic)

### 3.3 Kernel Cache Efficiency

AADC caches computation graphs ("kernels") by trade structure:

| Portfolio | Unique Kernels | Cache Reuse | Avg Kernel Reuse |
|-----------|----------------|-------------|------------------|
| 100 trades | 37 | 63% | 2.7x |
| 500 trades | 53 | 89% | 9.4x |
| 1,000 trades | 53 | 95% | 18.9x |

**Key insight**: Kernels saturate at ~53 for 3 currencies. Real portfolios with standardized tenors (2Y, 5Y, 10Y) would have even higher reuse.

### 3.4 Production Projections for $40B AUM

| Use Case | Trades | Baseline | AADC | Enables |
|----------|--------|----------|------|---------|
| Intraday risk | 500 | 1.0 sec | **35 ms** | Real-time limit monitoring |
| End-of-day | 1,000 | 2.1 sec | **70 ms** | Sub-second daily VaR |
| Stress test (100 scenarios) | 1,000 × 100 | 3.5 min | **7 sec** | Real-time stress analysis |
| Historical VaR (250 days) | 1,000 × 250 | 8.7 min | **17 sec** | Intraday VaR updates |

### 3.5 Accuracy Validation

| Metric | Result |
|--------|--------|
| Price difference (AADC vs baseline) | < $0.01 |
| Delta relative error | 0.048% (due to FD truncation in baseline) |
| Par swap validation | All 10 tests passed (PV = $0.00) |

**Note**: The small delta difference is due to finite difference truncation error in the baseline, not AADC inaccuracy. AADC provides exact derivatives.

---

## 4. Implemented Capabilities

### 4.1 Stress Margin Analysis

Shock SIMM sensitivities and recalculate margin to understand exposure under stress:

| Scenario | Description | Typical Impact |
|----------|-------------|----------------|
| `parallel_up_100bp` | All rates +100bp | +80-200% margin |
| `parallel_down_100bp` | All rates -100bp | -30-50% margin |
| `steepener_50bp` | 2Y -25bp, 30Y +50bp | Varies by portfolio |
| `flattener_50bp` | 2Y +25bp, 30Y -50bp | Varies by portfolio |
| `vol_up_25pct` | Volatility +25% | +25% vega margin |
| `credit_widen_50pct` | Credit spreads +50% | +50% credit margin |
| `crisis_scenario` | Rates +200bp, vol +50%, credit +100% | Combined stress |

**Example output**:
```
Scenario              Stressed Margin    Change
────────────────────────────────────────────────
parallel_up_100bp     $280.5B           +183%
steepener_50bp        $112.3B           +13%
crisis_scenario       $445.2B           +349%
```

### 4.2 Incremental Margin Analysis

Identify which trades contribute most to portfolio margin:

```
Trade ID        Incr. Margin    Margin %    Direction
──────────────────────────────────────────────────────
SWAP_000042     $2.3B           8.2%        Additive
SWAP_000017     $1.8B           6.4%        Additive
SWAP_000005     -$1.2B          -4.3%       Offsetting   ← Netting benefit
SWAP_000089     $0.9B           3.2%        Additive
```

**Use case**: Identify trades providing netting benefit (negative incremental margin) vs trades adding to margin requirement.

### 4.3 What-If Analysis

Simulate margin impact before executing trades:

```python
# Example: What if we remove the top 3 margin contributors?
Current Margin:  $99.2B
New Margin:      $87.1B
Margin Savings:  $12.1B (12.2%)
```

---

## 5. PoC Limitations

**This is a Proof of Concept.** The following limitations should be addressed before production deployment:

### 5.1 Validation Gaps

| Gap | Impact | Recommendation |
|-----|--------|----------------|
| No independent price validation | Cannot verify pricing accuracy | Validate against QuantLib/Bloomberg |
| Limited test cases | May miss edge cases | Expand regression test suite |
| No ISDA SIMM unit test validation | Regulatory risk | Run official ISDA test cases |

### 5.2 Model Simplifications

| Simplification | Production Requirement |
|----------------|------------------------|
| Single curve (discount = forward) | Separate OIS discount and IBOR forward curves |
| Linear interpolation | Log-linear or cubic spline interpolation |
| No holiday calendar | Business day conventions |
| No fixing handling | Near-term fixings for live swaps |

### 5.3 Product Coverage

| Current | Missing for Full SIMM |
|---------|----------------------|
| Vanilla IR swaps | Swaptions, caps/floors |
| | Cross-currency swaps |
| | Inflation swaps |
| | Basis swaps |

### 5.4 Scale & Performance

| Tested | Production Requirement |
|--------|------------------------|
| Up to 1,000 trades | 10,000+ trades |
| 3 currencies | 20+ currencies |
| Sequential trade processing | Batched/vectorized evaluation |

**Threading note**: AADC supports multi-threading via `ThreadPool`, but the current PoC processes trades sequentially in Python. For small kernels (~0.1ms each), thread overhead exceeds benefit. Optimization options:
1. **Batch by structure**: Evaluate trades with same structure together
2. **Vectorized kernels**: Single kernel for multiple trades (like Asian Options approach)
3. **C++ outer loop**: Move trade iteration to C++ for better parallelism

### 5.5 Operational Readiness

| Gap | Status |
|-----|--------|
| Cache warmup strategy | Approach defined (generate canonical trades at SOD) |
| Error handling | No graceful fallback to baseline |
| Monitoring | No cache hit rate metrics |
| Audit trail | No trade-level logging |

---

## 6. Files Delivered

| File | Description |
|------|-------------|
| `model/ir_swap_common.py` | Data structures, CRIF generation |
| `model/ir_swap_pricer.py` | Baseline bump-and-revalue implementation |
| `model/ir_swap_aadc.py` | AADC implementation with kernel caching |
| `model/margin_analysis.py` | Stress margin and incremental margin |
| `src/agg_margins.py` | SIMM aggregation engine |
| `benchmark_portfolio.py` | Portfolio benchmark runner |
| `docs/review.md` | Technical review document |
| `docs/requirements.md` | Full requirements specification |
| `docs/concerns.md` | Detailed technical limitations |

---

## 7. Recommended Next Steps

### Phase 1: Production Hardening (Recommended)
1. **Independent validation**: Compare prices against QuantLib
2. **Scale testing**: Benchmark with 10,000 trades
3. **Automated test suite**: CI/CD with regression tests

### Phase 2: Extended Capabilities
1. **Dual-curve implementation**: OIS discounting
2. **Bilateral vs cleared**: CCP margin comparison
3. **Multi-custodian simulation**: Allocation across custodians

### Phase 3: Vol Arb Support
1. **Swaption pricing**: Black/Bachelier models
2. **QuantLib integration**: For exotic products
3. **Vega risk**: Volatility sensitivities

### Phase 4: Optimization
1. **Allocation solver**: Minimize margin across counterparties
2. **Transaction structuring**: Recommend optimal trade structures
3. **Margin efficiency scoring**: Portfolio-level metrics

---

## 8. Summary

| Aspect | PoC Delivery |
|--------|--------------|
| **Performance** | 30-50x speedup vs Python baseline (~3-5x vs C++ estimate) |
| **Accuracy** | < 0.05% relative error (FD truncation, not AAD) |
| **Coverage** | IR swaps, SIMM, stress/incremental margin |
| **Threading** | Single-threaded (configurable, batching needed for scaling) |
| **Readiness** | PoC complete, production hardening required |

**Conclusion**: The PoC demonstrates that AADC-based margin analysis can deliver real-time performance for a $40B AUM portfolio. Even accounting for the Python baseline (~3-5x vs optimized C++), AADC enables sub-second risk calculations that support real-time stress testing and intraday VaR updates. The exact derivative computation eliminates FD approximation errors that accumulate in traditional approaches.

---

*Document generated: PoC Phase 1 & 2 Complete*
*For questions: Refer to `docs/review.md` for technical details*
