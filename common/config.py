"""
ISDA-SIMM Configuration

Configuration classes for SIMM benchmark runs.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SIMMConfig:
    """Configuration for SIMM calculations."""

    # Scale parameters
    num_trades: int = 1000
    num_risk_factors: int = 100
    num_threads: int = 8

    # SIMM parameters
    simm_version: str = "2.6"
    calculation_currency: str = "USD"

    # Risk classes to include
    risk_classes: List[str] = field(default_factory=lambda: [
        "Rates", "CreditQ", "CreditNonQ", "Equity", "Commodity", "FX"
    ])

    # Risk measures to compute
    risk_measures: List[str] = field(default_factory=lambda: [
        "Delta", "Vega", "Curvature"
    ])

    # AADC-specific
    scenario_batch_size: int = 1024
    vector_size: int = 8  # AVX512 = 8, AVX2 = 4

    @property
    def num_sensitivities(self) -> int:
        """Estimate total number of sensitivities."""
        return self.num_trades * self.num_risk_factors


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Implementations to benchmark
    run_baseline: bool = True
    run_aadc: bool = True

    # Validation
    validate_accuracy: bool = True
    tolerance: float = 1e-10

    # Output
    verbose: bool = True
    log_to_file: bool = True

    # Scaling tests
    trade_counts: List[int] = field(default_factory=lambda: [10, 100, 1000])
    thread_counts: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
