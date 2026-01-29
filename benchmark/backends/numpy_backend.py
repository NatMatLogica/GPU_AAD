"""
NumPy reference backend for SIMM benchmark.

Calls simm_formula.py directly. Serves as correctness reference.
Slowest but guaranteed correct.
"""

import numpy as np
from typing import Tuple

from benchmark.backends.base import SIMMBackend, FactorMetadata
from benchmark.simm_formula import compute_simm_gradient


class NumPyBackend(SIMMBackend):
    """NumPy reference SIMM backend."""

    name = "numpy"

    def setup(self, factor_meta: FactorMetadata) -> None:
        super().setup(factor_meta)

    def compute_im_and_gradient(
        self, agg_sensitivities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SIMM IM and gradient using pure NumPy reference formula."""
        return compute_simm_gradient(agg_sensitivities, self.factor_meta)
