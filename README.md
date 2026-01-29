# ISDA SIMM

Python implementation of ISDA Standard Initial Margin Model ([ISDA SIMM](https://www.isda.org/category/margin/isda-simm/)) v2.3-2.6 for calculating initial margin of uncleared OTC derivatives, with [AADC](https://matlogica.com/) integration for automatic sensitivity computation and gradient-based margin optimization.

Based on the official [ISDA SIMM Methodology](https://www.isda.org/a/b4ugE/ISDA-SIMM_v2.6_PUBLIC.pdf).

## Features

- **SIMM Calculation**: Full ISDA SIMM v2.6 aggregation (Delta, Vega, Curvature, BaseCorr)
- **AADC Integration**: 30-50x speedup via automatic adjoint differentiation
- **Trade Allocation Optimization**: Gradient-based netting set optimization (10-30% IM reduction)
- **Stress Margin Analysis**: 7 predefined stress scenarios
- **Pre-Trade Analytics**: Marginal IM, bilateral vs cleared comparison, counterparty routing
- **What-If Analysis**: Real-time margin impact for add/remove/hedge scenarios
- **Multi-Asset**: IR swaps, FX options, equity options, inflation swaps, cross-currency swaps

## Getting Started

### Prerequisites

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For GPU benchmarks:
```bash
pip install -r requirements_gpu.txt
```

### Basic SIMM Calculation

Place your [CRIF](https://www.isda.org/a/owEDE/risk-data-standards-v1-36-public.pdf) file in the `CRIF/` directory and run:

```bash
python -m model.simm_baseline --trades 100 --threads 4
```

### AADC-Enabled Calculation

```bash
python -m model.simm_aadc --trades 1000 --threads 8
```

### Portfolio Optimization

```bash
python -m model.simm_portfolio_aadc \
    --trades 100 --portfolios 5 --threads 8 \
    --optimize --method gradient_descent
```

### Benchmark

```bash
python benchmark_simm.py --trades 1000 --threads 8
```

## Project Structure

```
ISDA-SIMM/
├── model/                  # Trade models and SIMM implementations
│   ├── simm_baseline.py    # NumPy baseline (bump-and-revalue)
│   ├── simm_aadc.py        # AADC-enabled implementation
│   ├── simm_common.py      # Shared SIMM logic
│   ├── simm_portfolio_aadc.py      # Portfolio-level AADC calculations
│   ├── simm_allocation_optimizer.py # Trade allocation optimization
│   └── trade_types.py      # Trade definitions and pricing
├── src/                    # Core SIMM calculation engine
│   ├── agg_margins.py      # Top-level SIMM aggregation
│   ├── margin_risk_class.py # Risk class margin calculations
│   ├── agg_sensitivities.py # Sensitivity aggregation
│   └── wnc.py              # Weights and correlations loader
├── common/                 # Configuration, logging, utilities
├── benchmark/              # Benchmarking framework (CPU vs GPU)
├── tests/                  # Test suite
├── scripts/                # Benchmark and utility scripts
├── Weights_and_Corr/       # SIMM calibration parameters (v2.3-v2.7)
├── CRIF/                   # Input CRIF data
├── visualization/          # Interactive optimization demo
├── docs/                   # Technical documentation
├── data/                   # Output data (execution logs, CRIF samples)
└── benchmark_simm.py       # Main benchmark runner
```

## Performance

| Implementation | 1000 Trades | Speedup |
|----------------|-------------|---------|
| Baseline (NumPy) | ~45s | 1.0x |
| AADC Python | ~2s | ~20x |

## Results Example

| SIMM Total | Product Class | Risk Class | Risk Measure | SIMM |
|:---:|:---:|:---:|:---:|---:|
| 16,111,268,937 | RatesFX | Rates | Delta | 759,108,218 |
| | | | Curvature | 1,947 |
| | | FX | Delta | 11,926,343 |
| | Credit | CreditQ | Delta | 3,922,360,448 |
| | | CreditNonQ | Delta | 11,472,297,989 |
| | Equity | Equity | Delta | 158,843,566 |
| | Commodity | Commodity | Delta | 171,187,064 |

## Documentation

- [Architecture & Technical Details](docs/architecture.md)
- [SIMM Implementation with AADC (Practitioner's Guide)](docs/SIMM-implementation-aadc.md)

## License

If you intend to implement this for any commercial purpose, reach out to [ISDA SIMM](mailto:isdalegal@isda.org) to obtain a proper license. See [ISDA SIMM Licensing FAQ](https://www.isda.org/2021/04/08/isda-simm-licensing-faq/).
