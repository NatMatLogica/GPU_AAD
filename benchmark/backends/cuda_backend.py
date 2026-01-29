"""
CUDA GPU backend for SIMM benchmark.

Implements the FULL ISDA v2.6 SIMM formula on GPU:
- Correct PSI matrix
- Full intra-bucket correlations
- Concentration factors applied
- Analytical gradient via chain rule

Uses Numba CUDA for GPU kernel compilation.

Supports:
- GPU timing breakdown (H2D, kernel, D2H)
- Multi-GPU portfolio partitioning
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

from benchmark.backends.base import SIMMBackend, FactorMetadata
from benchmark.simm_formula import PSI_MATRIX, NUM_RISK_CLASSES

CUDA_SIMULATOR = os.environ.get('NUMBA_ENABLE_CUDASIM', '0') == '1'

try:
    from numba import cuda
    import numba
    CUDA_AVAILABLE = cuda.is_available() or CUDA_SIMULATOR
except ImportError:
    CUDA_AVAILABLE = False
    cuda = None


@dataclass
class GPUTimingDetail:
    """Breakdown of GPU execution timing."""
    h2d_ms: float = 0.0        # Host-to-device transfer
    kernel_ms: float = 0.0     # Kernel execution
    d2h_ms: float = 0.0        # Device-to-host transfer
    total_ms: float = 0.0      # Total GPU time
    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    num_gpus: int = 1

    def to_dict(self) -> dict:
        return {
            "h2d_ms": self.h2d_ms,
            "kernel_ms": self.kernel_ms,
            "d2h_ms": self.d2h_ms,
            "total_ms": self.total_ms,
            "gpu_mem_used_mb": self.gpu_mem_used_mb,
            "gpu_mem_total_mb": self.gpu_mem_total_mb,
            "num_gpus": self.num_gpus,
        }


# =============================================================================
# CUDA Kernel: Full SIMM with correlations + analytical gradient
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def _simm_full_kernel(
        sensitivities,      # (P, K) aggregated sensitivities
        risk_weights,       # (K,)
        concentration,      # (K,) concentration factors
        risk_class_idx,     # (K,) -> 0..5
        intra_corr_flat,    # (K*K,) flattened intra-bucket correlation matrix
        psi_matrix,         # (6, 6)
        im_output,          # (P,) output
        gradients,          # (P, K) output
    ):
        """
        Full SIMM kernel: one thread per portfolio.

        Computes both IM and dIM/dS analytically:
        1. WS_k = S_k * rw_k * cr_k
        2. K_rc = sqrt(Sigma_{k,l in rc} rho_kl * WS_k * WS_l)
        3. IM = sqrt(Sigma_rs psi_rs * K_r * K_s)
        4. Gradient via chain rule through correlations
        """
        import math

        p = cuda.grid(1)
        if p >= sensitivities.shape[0]:
            return

        K = sensitivities.shape[1]

        # Step 1: Compute weighted sensitivities
        ws = cuda.local.array(200, dtype=numba.float64)
        for k in range(min(K, 200)):
            ws[k] = sensitivities[p, k] * risk_weights[k] * concentration[k]

        # Step 2: Compute K_r^2 per risk class (with correlations)
        k_r = cuda.local.array(6, dtype=numba.float64)
        k_r_sq = cuda.local.array(6, dtype=numba.float64)
        for r in range(6):
            k_r[r] = 0.0
            k_r_sq[r] = 0.0

        # For each risk class, compute K_rc^2 = sum_{k,l in rc} rho_kl * WS_k * WS_l
        for ki in range(min(K, 200)):
            rc_i = risk_class_idx[ki]
            for kj in range(min(K, 200)):
                rc_j = risk_class_idx[kj]
                if rc_i == rc_j:
                    rho = intra_corr_flat[ki * K + kj]
                    k_r_sq[rc_i] += rho * ws[ki] * ws[kj]

        for r in range(6):
            k_r[r] = math.sqrt(max(k_r_sq[r], 0.0))

        # Step 3: Cross-RC aggregation: IM^2 = sum_rs psi[r,s] * K_r * K_s
        im_sq = 0.0
        for r in range(6):
            for s in range(6):
                im_sq += psi_matrix[r, s] * k_r[r] * k_r[s]

        im_p = math.sqrt(max(im_sq, 0.0))
        im_output[p] = im_p

        # Step 4: Compute gradients
        if im_p < 1e-15:
            for k in range(K):
                gradients[p, k] = 0.0
            return

        # dIM/dK_r = (1/IM) * sum_s psi[r,s] * K_s
        dim_dk = cuda.local.array(6, dtype=numba.float64)
        for r in range(6):
            dim_dk[r] = 0.0
            for s in range(6):
                dim_dk[r] += psi_matrix[r, s] * k_r[s]
            dim_dk[r] /= im_p

        # dK_r/dWS_k = (1/K_r) * sum_{l in rc} rho_kl * WS_l
        # dIM/dS_k = dim_dk[rc] * dk_dws_k * rw_k * cr_k
        for k in range(min(K, 200)):
            rc = risk_class_idx[k]
            if k_r[rc] < 1e-15:
                gradients[p, k] = 0.0
                continue

            # dk_dws = (1/K_r) * sum_{l in rc} rho_kl * WS_l
            dk_dws = 0.0
            for l in range(min(K, 200)):
                if risk_class_idx[l] == rc:
                    rho = intra_corr_flat[k * K + l]
                    dk_dws += rho * ws[l]
            dk_dws /= k_r[rc]

            gradients[p, k] = dim_dk[rc] * dk_dws * risk_weights[k] * concentration[k]


def _get_gpu_memory() -> Tuple[float, float]:
    """Return (used_mb, total_mb) for current GPU context."""
    if CUDA_SIMULATOR:
        return 0.0, 0.0
    try:
        ctx = cuda.current_context()
        free, total = ctx.get_memory_info()
        used = total - free
        return used / (1024 * 1024), total / (1024 * 1024)
    except Exception:
        return 0.0, 0.0


class CUDABackend(SIMMBackend):
    """CUDA GPU backend with full ISDA v2.6 SIMM formula.

    Supports multi-GPU portfolio partitioning and timing breakdown.
    """

    name = "cuda"

    def __init__(self, device: int = 0, num_gpus: int = 1, collect_timing: bool = False):
        self.device = device
        self.num_gpus = num_gpus
        self.collect_timing = collect_timing
        self.last_timing: Optional[GPUTimingDetail] = None
        self._d_weights = None
        self._d_concentration = None
        self._d_rc_idx = None
        self._d_corr_flat = None
        self._d_psi = None
        self._K = 0
        # Multi-GPU: per-device constant arrays
        self._multi_gpu_data: List[dict] = []

    def setup(self, factor_meta: FactorMetadata) -> None:
        """Pre-allocate GPU memory for constant arrays."""
        super().setup(factor_meta)

        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA not available. Set NUMBA_ENABLE_CUDASIM=1 for testing."
            )

        K = len(factor_meta.risk_classes)
        self._K = K

        # Flatten intra-correlation matrix
        if factor_meta.intra_corr_matrix is not None:
            corr_flat = factor_meta.intra_corr_matrix.flatten().astype(np.float64)
        else:
            corr_flat = np.eye(K).flatten().astype(np.float64)

        weights_np = np.ascontiguousarray(factor_meta.risk_weights, dtype=np.float64)
        conc_np = np.ascontiguousarray(factor_meta.concentration_factors, dtype=np.float64)
        rc_idx_np = np.ascontiguousarray(factor_meta.risk_class_idx, dtype=np.int32)
        corr_np = np.ascontiguousarray(corr_flat, dtype=np.float64)
        psi_np = np.ascontiguousarray(PSI_MATRIX, dtype=np.float64)

        if self.num_gpus <= 1 or CUDA_SIMULATOR:
            # Single-GPU setup
            self._d_weights = cuda.to_device(weights_np)
            self._d_concentration = cuda.to_device(conc_np)
            self._d_rc_idx = cuda.to_device(rc_idx_np)
            self._d_corr_flat = cuda.to_device(corr_np)
            self._d_psi = cuda.to_device(psi_np)
        else:
            # Multi-GPU: copy constants to each device
            actual_gpus = min(self.num_gpus, len(cuda.gpus))
            self._multi_gpu_data = []
            for gpu_id in range(actual_gpus):
                with cuda.gpus[gpu_id]:
                    self._multi_gpu_data.append({
                        "weights": cuda.to_device(weights_np),
                        "concentration": cuda.to_device(conc_np),
                        "rc_idx": cuda.to_device(rc_idx_np),
                        "corr_flat": cuda.to_device(corr_np),
                        "psi": cuda.to_device(psi_np),
                    })
            self.num_gpus = actual_gpus

    def _run_single_gpu(
        self, agg_sensitivities: np.ndarray,
        d_weights=None, d_concentration=None, d_rc_idx=None,
        d_corr_flat=None, d_psi=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run SIMM kernel on a single GPU (current context)."""
        P, K = agg_sensitivities.shape

        im_output = np.zeros(P, dtype=np.float64)
        gradients_out = np.zeros((P, K), dtype=np.float64)

        # Use provided device arrays or defaults
        dw = d_weights or self._d_weights
        dc = d_concentration or self._d_concentration
        dr = d_rc_idx or self._d_rc_idx
        dcf = d_corr_flat or self._d_corr_flat
        dp = d_psi or self._d_psi

        d_sens = cuda.to_device(np.ascontiguousarray(agg_sensitivities, dtype=np.float64))
        d_im = cuda.to_device(im_output)
        d_grad = cuda.to_device(gradients_out)

        threads_per_block = min(256, P)
        blocks = (P + threads_per_block - 1) // threads_per_block

        _simm_full_kernel[blocks, threads_per_block](
            d_sens, dw, dc, dr, dcf, dp, d_im, d_grad,
        )

        if not CUDA_SIMULATOR:
            cuda.synchronize()

        d_im.copy_to_host(im_output)
        d_grad.copy_to_host(gradients_out)

        return im_output, gradients_out

    def _run_with_timing(
        self, agg_sensitivities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run on single GPU with CUDA event timing."""
        P, K = agg_sensitivities.shape

        im_output = np.zeros(P, dtype=np.float64)
        gradients_out = np.zeros((P, K), dtype=np.float64)

        sens_np = np.ascontiguousarray(agg_sensitivities, dtype=np.float64)

        if CUDA_SIMULATOR:
            # No CUDA events in simulator
            return self._run_single_gpu(agg_sensitivities)

        # Create CUDA events
        ev_start = cuda.event()
        ev_h2d_done = cuda.event()
        ev_kernel_done = cuda.event()
        ev_d2h_done = cuda.event()

        stream = cuda.stream()

        # Record start
        ev_start.record(stream=stream)

        # H2D transfer
        d_sens = cuda.to_device(sens_np, stream=stream)
        d_im = cuda.to_device(im_output, stream=stream)
        d_grad = cuda.to_device(gradients_out, stream=stream)
        ev_h2d_done.record(stream=stream)

        # Kernel launch
        threads_per_block = min(256, P)
        blocks = (P + threads_per_block - 1) // threads_per_block
        _simm_full_kernel[blocks, threads_per_block, stream](
            d_sens, self._d_weights, self._d_concentration,
            self._d_rc_idx, self._d_corr_flat, self._d_psi,
            d_im, d_grad,
        )
        ev_kernel_done.record(stream=stream)

        # D2H transfer
        d_im.copy_to_host(im_output, stream=stream)
        d_grad.copy_to_host(gradients_out, stream=stream)
        ev_d2h_done.record(stream=stream)

        # Synchronize
        ev_d2h_done.synchronize()

        # Compute timings
        h2d_ms = cuda.event_elapsed_time(ev_start, ev_h2d_done)
        kernel_ms = cuda.event_elapsed_time(ev_h2d_done, ev_kernel_done)
        d2h_ms = cuda.event_elapsed_time(ev_kernel_done, ev_d2h_done)

        mem_used, mem_total = _get_gpu_memory()

        self.last_timing = GPUTimingDetail(
            h2d_ms=h2d_ms,
            kernel_ms=kernel_ms,
            d2h_ms=d2h_ms,
            total_ms=h2d_ms + kernel_ms + d2h_ms,
            gpu_mem_used_mb=mem_used,
            gpu_mem_total_mb=mem_total,
            num_gpus=1,
        )

        return im_output, gradients_out

    def _run_multi_gpu(
        self, agg_sensitivities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Partition portfolios across GPUs and gather results."""
        P, K = agg_sensitivities.shape
        n_gpus = len(self._multi_gpu_data)

        # Partition: round-robin or contiguous chunks
        chunk_size = (P + n_gpus - 1) // n_gpus
        im_all = np.zeros(P, dtype=np.float64)
        grad_all = np.zeros((P, K), dtype=np.float64)

        for gpu_id in range(n_gpus):
            start = gpu_id * chunk_size
            end = min(start + chunk_size, P)
            if start >= P:
                break

            chunk = agg_sensitivities[start:end]
            data = self._multi_gpu_data[gpu_id]

            with cuda.gpus[gpu_id]:
                im_chunk, grad_chunk = self._run_single_gpu(
                    chunk,
                    d_weights=data["weights"],
                    d_concentration=data["concentration"],
                    d_rc_idx=data["rc_idx"],
                    d_corr_flat=data["corr_flat"],
                    d_psi=data["psi"],
                )

            im_all[start:end] = im_chunk
            grad_all[start:end] = grad_chunk

        if self.collect_timing:
            mem_used, mem_total = _get_gpu_memory()
            self.last_timing = GPUTimingDetail(
                num_gpus=n_gpus,
                gpu_mem_used_mb=mem_used,
                gpu_mem_total_mb=mem_total,
            )

        return im_all, grad_all

    def compute_im_and_gradient(
        self, agg_sensitivities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SIMM IM and gradient using CUDA kernel."""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available.")

        # Multi-GPU path
        if self.num_gpus > 1 and self._multi_gpu_data and not CUDA_SIMULATOR:
            return self._run_multi_gpu(agg_sensitivities)

        # Single-GPU with timing
        if self.collect_timing and not CUDA_SIMULATOR:
            return self._run_with_timing(agg_sensitivities)

        # Single-GPU default
        return self._run_single_gpu(agg_sensitivities)
