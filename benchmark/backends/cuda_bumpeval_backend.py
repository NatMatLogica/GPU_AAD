"""
CUDA GPU bump-and-revalue backend for SIMM benchmark.

Computes gradients via finite difference (bump each of K factors by epsilon).
This is the GPU baseline: K+1 SIMM evaluations per portfolio.

All K+1 evaluations are launched as parallel CUDA threads, exploiting
GPU parallelism to offset the O(K) factor count.

Reports both timing and gradient accuracy vs analytical.
"""

import os
import numpy as np
from typing import Tuple

from benchmark.backends.base import SIMMBackend, FactorMetadata
from benchmark.simm_formula import PSI_MATRIX, NUM_RISK_CLASSES

CUDA_SIMULATOR = os.environ.get('NUMBA_ENABLE_CUDASIM', '0') == '1'

try:
    from numba import cuda
    import numba
    CUDA_BUMPEVAL_AVAILABLE = cuda.is_available() or CUDA_SIMULATOR
except ImportError:
    CUDA_BUMPEVAL_AVAILABLE = False
    cuda = None

# =============================================================================
# CUDA Kernel: SIMM evaluation only (no gradient)
# =============================================================================

if CUDA_BUMPEVAL_AVAILABLE:
    @cuda.jit
    def _simm_eval_kernel(
        sensitivities,      # (N, K) — N = P * (K+1) bumped scenarios
        risk_weights,       # (K,)
        concentration,      # (K,)
        risk_class_idx,     # (K,)
        intra_corr_flat,    # (K*K,)
        psi_matrix,         # (6, 6)
        im_output,          # (N,) output
    ):
        """SIMM evaluation kernel — one thread per scenario (no gradient)."""
        import math

        n = cuda.grid(1)
        if n >= sensitivities.shape[0]:
            return

        K = sensitivities.shape[1]

        # Weighted sensitivities
        ws = cuda.local.array(200, dtype=numba.float64)
        for k in range(min(K, 200)):
            ws[k] = sensitivities[n, k] * risk_weights[k] * concentration[k]

        # K_r^2 per risk class
        k_r_sq = cuda.local.array(6, dtype=numba.float64)
        for r in range(6):
            k_r_sq[r] = 0.0

        for ki in range(min(K, 200)):
            rc_i = risk_class_idx[ki]
            for kj in range(min(K, 200)):
                rc_j = risk_class_idx[kj]
                if rc_i == rc_j:
                    rho = intra_corr_flat[ki * K + kj]
                    k_r_sq[rc_i] += rho * ws[ki] * ws[kj]

        # K_r values
        k_r = cuda.local.array(6, dtype=numba.float64)
        for r in range(6):
            k_r[r] = math.sqrt(max(k_r_sq[r], 0.0))

        # Cross-RC aggregation
        im_sq = 0.0
        for r in range(6):
            for s in range(6):
                im_sq += psi_matrix[r, s] * k_r[r] * k_r[s]

        im_output[n] = math.sqrt(max(im_sq, 0.0))


class CUDABumpRevalBackend(SIMMBackend):
    """CUDA GPU bump-and-revalue backend.

    Computes gradient via finite difference: for each of K factors,
    bump sensitivity by epsilon and re-evaluate SIMM.

    grad_k = (SIMM(S + eps*e_k) - SIMM(S)) / eps

    All K+1 evaluations per portfolio run in parallel on GPU.
    """

    name = "cuda_bumpeval"

    def __init__(self, device: int = 0, epsilon: float = 1.0):
        self.device = device
        self.epsilon = epsilon
        self._d_weights = None
        self._d_concentration = None
        self._d_rc_idx = None
        self._d_corr_flat = None
        self._d_psi = None
        self._K = 0

    def setup(self, factor_meta: FactorMetadata) -> None:
        """Pre-allocate GPU memory for constant arrays."""
        super().setup(factor_meta)

        if not CUDA_BUMPEVAL_AVAILABLE:
            raise RuntimeError(
                "CUDA not available. Set NUMBA_ENABLE_CUDASIM=1 for testing."
            )

        K = len(factor_meta.risk_classes)
        self._K = K

        if factor_meta.intra_corr_matrix is not None:
            corr_flat = factor_meta.intra_corr_matrix.flatten().astype(np.float64)
        else:
            corr_flat = np.eye(K).flatten().astype(np.float64)

        self._d_weights = cuda.to_device(
            np.ascontiguousarray(factor_meta.risk_weights, dtype=np.float64)
        )
        self._d_concentration = cuda.to_device(
            np.ascontiguousarray(factor_meta.concentration_factors, dtype=np.float64)
        )
        self._d_rc_idx = cuda.to_device(
            np.ascontiguousarray(factor_meta.risk_class_idx, dtype=np.int32)
        )
        self._d_corr_flat = cuda.to_device(
            np.ascontiguousarray(corr_flat, dtype=np.float64)
        )
        self._d_psi = cuda.to_device(
            np.ascontiguousarray(PSI_MATRIX, dtype=np.float64)
        )

    def compute_im_and_gradient(
        self, agg_sensitivities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SIMM IM and gradient via bump-and-revalue on GPU.

        For P portfolios with K factors, constructs P*(K+1) scenarios:
        - P base scenarios (unbumped)
        - P*K bumped scenarios (one per factor per portfolio)

        All launched in parallel on GPU.
        """
        if not CUDA_BUMPEVAL_AVAILABLE:
            raise RuntimeError("CUDA not available.")

        P, K = agg_sensitivities.shape
        eps = self.epsilon
        N = P * (K + 1)  # Total scenarios

        # Build scenario matrix: (N, K)
        # Layout: [base_0, bump_0_0, bump_0_1, ..., bump_0_{K-1},
        #          base_1, bump_1_0, ..., bump_{P-1}_{K-1}]
        scenarios = np.empty((N, K), dtype=np.float64)
        for p in range(P):
            offset = p * (K + 1)
            # Base scenario
            scenarios[offset] = agg_sensitivities[p]
            # Bumped scenarios
            for k in range(K):
                scenarios[offset + 1 + k] = agg_sensitivities[p].copy()
                scenarios[offset + 1 + k, k] += eps

        # Transfer to GPU
        d_scenarios = cuda.to_device(np.ascontiguousarray(scenarios))
        im_all = np.zeros(N, dtype=np.float64)
        d_im = cuda.to_device(im_all)

        # Launch kernel
        threads_per_block = min(256, N)
        blocks = (N + threads_per_block - 1) // threads_per_block

        _simm_eval_kernel[blocks, threads_per_block](
            d_scenarios,
            self._d_weights,
            self._d_concentration,
            self._d_rc_idx,
            self._d_corr_flat,
            self._d_psi,
            d_im,
        )

        if not CUDA_SIMULATOR:
            cuda.synchronize()

        d_im.copy_to_host(im_all)

        # Extract results
        im_values = np.zeros(P, dtype=np.float64)
        gradients = np.zeros((P, K), dtype=np.float64)

        for p in range(P):
            offset = p * (K + 1)
            base_im = im_all[offset]
            im_values[p] = base_im
            for k in range(K):
                bumped_im = im_all[offset + 1 + k]
                gradients[p, k] = (bumped_im - base_im) / eps

        return im_values, gradients
