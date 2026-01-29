"""
AADC CPU backend for SIMM benchmark.

Wraps record_single_portfolio_simm_kernel() from the allocation optimizer.
Records kernel once, evaluates for P portfolios in single aadc.evaluate() call.
Gradients computed via AAD adjoint.
"""

import numpy as np
from typing import Tuple

from benchmark.backends.base import SIMMBackend, FactorMetadata

try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False


class AADCBackend(SIMMBackend):
    """AADC CPU backend using AAD adjoint for gradients."""

    name = "aadc"

    def __init__(self, num_threads: int = 8):
        self.num_threads = num_threads
        self._funcs = None
        self._sens_handles = None
        self._im_output = None
        self._workers = None

    def setup(self, factor_meta: FactorMetadata) -> None:
        """Record AADC kernel for SIMM computation."""
        super().setup(factor_meta)

        if not AADC_AVAILABLE:
            raise RuntimeError("AADC is not available. Install MatLogica AADC.")

        from model.simm_allocation_optimizer import record_single_portfolio_simm_kernel

        K = len(factor_meta.risk_classes)

        # Pre-multiply concentration factors into weights so the kernel
        # computes WS_k = s_k * (rw_k * cr_k), matching the NumPy formula.
        effective_weights = factor_meta.risk_weights * factor_meta.concentration_factors

        self._funcs, self._sens_handles, self._im_output = (
            record_single_portfolio_simm_kernel(
                K=K,
                factor_risk_classes=factor_meta.risk_classes,
                factor_weights=effective_weights,
                factor_risk_types=factor_meta.risk_types,
                factor_labels=factor_meta.labels,
                factor_buckets=factor_meta.buckets,
                use_correlations=True,
            )
        )

        self._workers = aadc.ThreadPool(self.num_threads)

    def compute_im_and_gradient(
        self, agg_sensitivities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SIMM IM and gradient via AADC adjoint.

        Single aadc.evaluate() call for all P portfolios.
        """
        if not AADC_AVAILABLE:
            raise RuntimeError("AADC is not available.")

        P, K = agg_sensitivities.shape

        # Build inputs: each handle maps to array of P values
        inputs = {}
        for k in range(K):
            inputs[self._sens_handles[k]] = agg_sensitivities[:, k].copy()

        # Request: IM output, differentiated w.r.t. all sensitivity handles
        request = {self._im_output: self._sens_handles}

        # Single evaluate() call for all P portfolios
        results = aadc.evaluate(
            self._funcs, request, inputs, self._workers
        )

        # Extract IM values: (P,) array
        im_values = np.array(results[0][self._im_output])

        # Extract gradients: (P, K) array
        gradients = np.zeros((P, K))
        for k in range(K):
            gradients[:, k] = results[1][self._im_output][self._sens_handles[k]]

        return im_values, gradients
