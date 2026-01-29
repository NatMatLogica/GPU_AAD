"""Abstract backend interface for SIMM benchmark."""

from abc import ABC, abstractmethod
from typing import Tuple
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FactorMetadata:
    """Metadata for each risk factor in the sensitivity matrix.

    All arrays have length K (number of risk factors).
    """
    risk_classes: list          # Risk class name per factor ('Rates', 'FX', ...)
    risk_class_idx: np.ndarray  # Risk class index (0-5) per factor
    risk_weights: np.ndarray    # Risk weight per factor
    risk_types: list            # RiskType string per factor
    labels: list                # Label1 (tenor) per factor
    buckets: list               # Bucket string per factor
    qualifiers: list            # Qualifier (currency/issuer) per factor
    concentration_factors: np.ndarray  # CR per factor (>= 1.0)

    # Pre-computed correlation data for intra-bucket aggregation
    # intra_corr_matrix[k1, k2] = rho * phi for factors k1, k2 in same bucket
    intra_corr_matrix: np.ndarray = field(default=None)


class SIMMBackend(ABC):
    """Abstract interface for SIMM computation backends.

    All backends must implement compute_im_and_gradient() which takes
    aggregated sensitivities and returns IM values and gradients.
    """

    name: str = "abstract"

    def setup(self, factor_meta: FactorMetadata) -> None:
        """One-time setup (kernel recording, GPU memory allocation).

        Args:
            factor_meta: Metadata describing each risk factor.
        """
        self.factor_meta = factor_meta

    @abstractmethod
    def compute_im_and_gradient(
        self, agg_sensitivities: np.ndarray  # (P, K)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SIMM IM and gradient for P portfolios.

        Args:
            agg_sensitivities: (P, K) aggregated sensitivities per portfolio.

        Returns:
            im_values: (P,) IM value per portfolio.
            gradients: (P, K) dIM/dS per portfolio per factor.
        """
        pass
