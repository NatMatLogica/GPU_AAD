# Outstanding Concerns and Future Work

This document captures known limitations and areas requiring further development before production deployment.

---

## 1. Validation Gaps (Partially Addressed)

### ✅ Implemented
- Par swap validation (PV ≈ 0)
- FD truncation error analysis (varying bump sizes)
- Gamma, cross-gamma, theta computation
- SIMM end-to-end integration test
- Bucket mapping verification

### ❌ Still Missing
- **Independent price validation**: No comparison against QuantLib, Bloomberg, or vendor system
- **Known analytical results**: Need more test cases beyond par swaps (e.g., zero-coupon swap = zero-coupon bond)
- **Regression test suite**: No automated CI/CD tests

---

## 2. Curve Construction (Not Addressed)

### Current State
- Same curve used for discounting and forwarding
- Linear interpolation on zero rates

### Production Requirements
- **OIS discounting**: Separate OIS curve for discounting
- **IBOR forwards**: Separate forward curves (3M LIBOR, 6M EURIBOR, etc.) - or SOFR-based post-IBOR transition
- **Interpolation**: Log-linear on discount factors or cubic spline on zero rates
- **Turn-of-year effects**: Stub handling at year-end

### Impact
- Current sensitivities may not match production systems
- Dual-curve setup would double the number of risk factors

---

## 3. Missing Risk Metrics (Partially Addressed)

### ✅ Implemented
- IR Delta (first-order rate sensitivity)
- IR Gamma (diagonal second-order, d²PV/dr²)
- Cross-Gamma (off-diagonal, d²PV/dr_i dr_j for nearby tenors)
- Theta (time decay)

### ❌ Still Missing
- **Vega**: Sensitivity to implied volatility (not applicable for vanilla swaps, but needed for swaptions)
- **Cross-currency delta**: For cross-currency swaps
- **Inflation delta**: For inflation-linked swaps
- **Credit delta**: For credit-contingent trades

### SIMM Curvature Risk
The implemented gamma can be used for SIMM curvature risk calculation:
```
Curvature Risk = Σ max(0, -CVR_i)
where CVR_i = (PV_up - PV) + (PV_down - PV)
            = Gamma * bump² (approximately)
```

---

## 4. Product Coverage (Not Addressed)

### Current Support
- Vanilla fixed-for-floating interest rate swaps only

### Missing Products for Full SIMM Coverage
- **Swaptions**: Need Black/Bachelier model, vega risk
- **Caps/Floors**: Volatility surface, caplet stripping
- **Cross-currency swaps**: FX risk, basis risk
- **Inflation swaps**: Inflation curve construction
- **Basis swaps**: Basis spread risk
- **Amortizing swaps**: Time-varying notional schedule
- **Forward starting swaps**: Fixing risk handling

---

## 5. Edge Cases (Not Addressed)

### Swap Lifecycle
- **Expired swaps**: Maturity < 0 (should return 0)
- **First fixing set**: Near-term fixing already known
- **Broken periods**: Stubs at start/end
- **Holiday calendar**: Business day conventions

### Numerical Edge Cases
- **Very short tenors**: 2w swap interpolation at curve short end
- **Very long tenors**: 30Y+ extrapolation
- **Deep ITM/OTM**: Numerical stability for large PVs
- **Zero notional**: Degenerate case handling

---

## 6. Scale Testing (Not Addressed)

### Current Testing
- 50-100 trades
- Single-threaded execution

### Production Requirements
- **10,000+ trades**: Memory profile, cache size limits
- **100+ currencies**: Kernel cache explosion
- **Multi-threading**: AADC ThreadPool scaling
- **Memory limits**: Cache eviction policy

### Open Questions
- Does 50x speedup hold at scale?
- What's the memory footprint with 1000 cached kernels?
- How does performance degrade with cache misses?

---

## 7. Operational Concerns (Not Addressed)

### Cache Management
- **Warm-up strategy**: How to pre-warm cache on startup?
- **Eviction policy**: No LRU or memory-based eviction
- **DR/Failover**: Cache lost on restart, need re-warming
- **Cache persistence**: Could serialize kernels to disk?

### Monitoring
- **Cache hit rate metrics**: Not exposed
- **Performance degradation alerts**: No monitoring hooks
- **Memory usage tracking**: Basic only via psutil

### Error Handling
- **AADC failures**: No graceful fallback to baseline
- **NaN/Inf detection**: Not implemented
- **Audit trail**: No logging of individual trade valuations

---

## 8. Regulatory Considerations (Not Addressed)

### SIMM Compliance
- **ISDA SIMM v2.6/2.7**: Weights and correlations loaded, but not verified against ISDA unit tests
- **Concentration thresholds**: Not implemented
- **Margin period of risk**: Assumed standard, not configurable

### Model Risk
- **Model validation**: No independent validation
- **Back-testing**: No historical comparison
- **Sensitivity analysis**: No parameter stress testing

---

## 9. Code Quality (Partially Addressed)

### ✅ Done
- Type hints on main functions
- Docstrings for public API
- Modular structure (common, baseline, AADC)

### ❌ Missing
- **Unit test coverage**: No pytest suite
- **Integration tests**: Manual only
- **Code review**: Not peer-reviewed
- **Performance benchmarks**: Ad-hoc, not automated

---

## 10. Documentation (Partially Addressed)

### ✅ Done
- `review.md`: Technical overview
- `aadc_integration_plan.md`: Design document
- Code comments in key areas

### ❌ Missing
- **API documentation**: No Sphinx/autodoc
- **User guide**: How to run, configure
- **Deployment guide**: Production setup
- **Troubleshooting guide**: Common issues

---

## Priority Ranking

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| **P0** | Independent price validation | Medium | Critical for sign-off |
| **P0** | Scale testing (10K trades) | Medium | Production readiness |
| **P1** | Dual-curve implementation | High | Accuracy vs production |
| **P1** | Automated test suite | Medium | Maintainability |
| **P2** | Additional products (swaptions) | High | SIMM coverage |
| **P2** | Cache management | Medium | Operational stability |
| **P3** | Vega/inflation risk | High | Full risk coverage |
| **P3** | Documentation | Low | Usability |

---

## Recommendation

**For PoC/Demo**: Current implementation is sufficient to demonstrate AADC value proposition (45-52x speedup).

**For Production**: Address P0 items before go-live:
1. Validate against independent pricer
2. Scale test with realistic portfolio
3. Add automated regression tests

**For Full SIMM Coverage**: Address P1/P2 items:
1. Implement dual-curve
2. Add swaption support for vega
3. Implement proper cache management
