# Benchmark: Fair AADC vs CUDA SIMM Comparison

## Purpose
Fair A/B comparison where the **only variable** is how SIMM + gradient is computed:
- NumPy reference (baseline, guaranteed correct)
- AADC CPU (AAD adjoint via MatLogica AADC)
- CUDA GPU (analytical gradient via Numba CUDA)

All backends use **identical**:
- Trade data and CRIF sensitivities
- SIMM formula (ISDA v2.6 PSI matrix, risk weights, intra-bucket correlations)
- Optimizer (gradient descent with simplex projection)
- Initial allocation (seed=42)

## SIMM Formula (ISDA v2.6)

### Per risk class (with intra-bucket correlation):
```
WS_k = s_k * rw_k * cr_k
K_b = sqrt(Σ_k Σ_l ρ_kl * WS_k * WS_l)    [intra-bucket]
K_rc = sqrt(Σ_b K²_b + Σ_{b≠c} γ * S_b * S_c)  [inter-bucket]
```

### Total IM (cross-risk-class):
```
IM = sqrt(Σ_r Σ_s ψ_rs * K_r * K_s)
```

### Key Parameters
- PSI matrix: 6x6 from `model/simm_portfolio_aadc.py:99`
- IR correlations: 12x12 from `Weights_and_Corr/v2_6.py`
- Risk weights: currency-specific from v2_6
- Concentration: CR = max(1, sqrt(|sum_sens|/T))

## File Organization
```
benchmark/
├── CLAUDE.md                  # This file
├── __init__.py
├── simm_formula.py            # Shared SIMM math (NumPy reference)
├── data_gen.py                # Deterministic data generation
├── optimizer.py               # Shared optimizer
├── backends/
│   ├── __init__.py
│   ├── base.py                # Abstract backend interface
│   ├── numpy_backend.py       # NumPy reference
│   ├── aadc_backend.py        # AADC CPU
│   └── cuda_backend.py        # CUDA GPU
├── benchmark_fair.py          # Main harness
└── results/
    └── .gitkeep
```

## Running
```bash
source venv/bin/activate

# Validate only (no optimization)
python -m benchmark.benchmark_fair --trades 50 --portfolios 3 --validate-only

# Full benchmark with optimization
python -m benchmark.benchmark_fair --trades 100 --portfolios 3 --optimize

# CUDA simulator (no GPU)
NUMBA_ENABLE_CUDASIM=1 python -m benchmark.benchmark_fair --trades 100 --portfolios 3 --optimize
```

## Validation Criteria
1. All backends produce same IM values (relative tolerance < 1e-6)
2. All backends produce same gradients (relative tolerance < 1e-4)
3. Optimization reaches same final IM (relative tolerance < 1e-3)
