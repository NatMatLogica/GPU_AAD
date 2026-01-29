#!/usr/bin/env python
"""
SIMM Benchmark: CPU vs GPU Performance Comparison

Target Hardware:
- GPU: NVIDIA HGX H100 8-GPU (80GB HBM3 each)
- CPU: Dual Intel Xeon Platinum 8568Y+ (48 cores each, 96 total)

This benchmark compares:
1. CPU NumPy (MKL-optimized, multi-threaded)
2. CPU Numba Parallel (explicit parallelization)
3. GPU CUDA (single H100)
4. GPU Multi-CUDA (multiple H100s)
5. GPU CuPy (alternative GPU backend)

Usage:
    python benchmark_cpu_vs_gpu.py --portfolios 10000 --factors 100
    python benchmark_cpu_vs_gpu.py --scale large  # Pre-configured large scale
    python benchmark_cpu_vs_gpu.py --all          # Run all configurations

Version: 1.0.0
"""

import os
import sys
import time
import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Try to import GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from numba import cuda, prange, njit
    import numba
    NUMBA_AVAILABLE = True
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    NUMBA_AVAILABLE = False
    CUDA_AVAILABLE = False

# Check for MKL
try:
    import numpy as np
    # Check if NumPy is using MKL
    np_config = np.__config__
    MKL_AVAILABLE = 'mkl' in str(np_config.show()).lower() if hasattr(np_config, 'show') else False
except:
    MKL_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    num_portfolios: int = 10000      # Number of portfolios
    num_risk_factors: int = 100      # Risk factors per portfolio
    num_iterations: int = 10         # Timing iterations
    warmup_iterations: int = 3       # Warmup iterations

    # CPU settings
    cpu_threads: int = 96            # Xeon 8568Y+ has 48 cores × 2 sockets
    use_mkl: bool = True             # Use MKL if available

    # GPU settings
    gpu_device: int = 0              # Primary GPU device
    num_gpus: int = 8                # Number of H100 GPUs
    gpu_batch_size: int = 0          # 0 = auto (all portfolios)

    # Output
    output_file: str = "benchmark_results.json"
    verbose: bool = True


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    backend: str
    num_portfolios: int
    num_risk_factors: int
    total_time_ms: float
    avg_time_ms: float
    throughput_portfolios_per_sec: float
    memory_mb: float
    num_threads_or_gpus: int
    validated: bool
    notes: str = ""


# =============================================================================
# SIMM Parameters
# =============================================================================

# Cross-risk-class correlation matrix (ψ) - ISDA SIMM v2.6
PSI_MATRIX = np.array([
    [1.00, 0.28, 0.18, 0.18, 0.30, 0.22],  # IR
    [0.28, 1.00, 0.30, 0.66, 0.46, 0.27],  # CreditQ
    [0.18, 0.30, 1.00, 0.23, 0.25, 0.18],  # CreditNonQ
    [0.18, 0.66, 0.23, 1.00, 0.39, 0.24],  # Equity
    [0.30, 0.46, 0.25, 0.39, 1.00, 0.32],  # Commodity
    [0.22, 0.27, 0.18, 0.24, 0.32, 1.00],  # FX
], dtype=np.float64)


# =============================================================================
# CPU Implementations
# =============================================================================

def simm_numpy_vectorized(
    sensitivities: np.ndarray,
    risk_weights: np.ndarray,
    risk_class_idx: np.ndarray,
    psi_matrix: np.ndarray = PSI_MATRIX,
) -> np.ndarray:
    """
    Fully vectorized NumPy SIMM - optimal for MKL.

    Uses broadcasting and einsum for maximum vectorization.
    MKL will automatically parallelize across CPU cores.
    """
    P, K = sensitivities.shape

    # Weighted sensitivities: (P, K)
    ws = sensitivities * risk_weights

    # Compute K_r for each risk class using masks
    # K_r[p, r] = sqrt(sum over k in r of ws[p,k]^2)
    k_r = np.zeros((P, 6), dtype=np.float64)

    for r in range(6):
        mask = (risk_class_idx == r)
        if np.any(mask):
            # Sum of squares for this risk class
            k_r[:, r] = np.sqrt(np.sum(ws[:, mask] ** 2, axis=1))

    # Cross-RC aggregation using einsum: IM^2 = sum_r sum_s psi[r,s] * K_r * K_s
    # This is equivalent to: diag(K_r @ psi @ K_r.T)
    # But einsum is more efficient: im_sq[p] = sum_rs psi[r,s] * k_r[p,r] * k_r[p,s]
    im_sq = np.einsum('pr,rs,ps->p', k_r, psi_matrix, k_r)

    return np.sqrt(im_sq)


if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True, cache=True)
    def simm_numba_parallel(
        sensitivities: np.ndarray,
        risk_weights: np.ndarray,
        risk_class_idx: np.ndarray,
        psi_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Numba JIT-compiled SIMM with explicit parallelization.

        Uses prange for parallel loop over portfolios.
        """
        P, K = sensitivities.shape
        im_values = np.zeros(P, dtype=np.float64)

        # Parallel over portfolios
        for p in prange(P):
            # Compute weighted sensitivities
            ws = np.zeros(K, dtype=np.float64)
            for k in range(K):
                ws[k] = sensitivities[p, k] * risk_weights[k]

            # Compute K_r for each risk class
            k_r = np.zeros(6, dtype=np.float64)
            for k in range(K):
                rc = risk_class_idx[k]
                k_r[rc] += ws[k] * ws[k]

            for r in range(6):
                k_r[r] = np.sqrt(k_r[r])

            # Cross-RC aggregation
            im_sq = 0.0
            for r in range(6):
                for s in range(6):
                    im_sq += psi_matrix[r, s] * k_r[r] * k_r[s]

            im_values[p] = np.sqrt(im_sq)

        return im_values


def _process_chunk_worker(args):
    """Worker function for multiprocess SIMM (must be at module level for pickling)."""
    start_idx, end_idx, sensitivities, risk_weights, risk_class_idx = args
    chunk_sens = sensitivities[start_idx:end_idx]
    return simm_numpy_vectorized(chunk_sens, risk_weights, risk_class_idx)


def simm_cpu_multiprocess(
    sensitivities: np.ndarray,
    risk_weights: np.ndarray,
    risk_class_idx: np.ndarray,
    num_workers: int,
) -> np.ndarray:
    """
    Multi-process SIMM using ProcessPoolExecutor.

    Splits portfolios across processes for maximum CPU utilization.
    Note: For small problems, overhead may exceed gains.
    """
    P = sensitivities.shape[0]

    # For small problems, just use numpy (process overhead not worth it)
    if P < 1000:
        return simm_numpy_vectorized(sensitivities, risk_weights, risk_class_idx)

    # Limit workers to avoid excessive overhead
    actual_workers = min(num_workers, max(1, P // 100))
    chunk_size = (P + actual_workers - 1) // actual_workers

    # Prepare work items
    work_items = []
    for i in range(actual_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, P)
        if start_idx < P:
            work_items.append((start_idx, end_idx, sensitivities, risk_weights, risk_class_idx))

    try:
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            results = list(executor.map(_process_chunk_worker, work_items))
        return np.concatenate(results)
    except Exception as e:
        # Fallback to single-process
        return simm_numpy_vectorized(sensitivities, risk_weights, risk_class_idx)


# =============================================================================
# AADC Implementation (for comparison)
# =============================================================================

try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False


def simm_aadc_cpu(
    sensitivities: np.ndarray,
    risk_weights: np.ndarray,
    risk_class_idx: np.ndarray,
    num_threads: int = 8,
    compute_gradients: bool = False,
) -> np.ndarray:
    """
    SIMM calculation using AADC on CPU.

    Uses single evaluate() call for all portfolios (optimized pattern).
    """
    if not AADC_AVAILABLE:
        raise RuntimeError("AADC not available")

    P, K = sensitivities.shape

    # Record kernel once (if not cached)
    with aadc.record_kernel() as funcs:
        # Mark sensitivities as inputs
        sens_inputs = []
        sens_handles = []
        for k in range(K):
            s_k = aadc.idouble(0.0)
            handle = s_k.mark_as_input()
            sens_handles.append(handle)
            sens_inputs.append(s_k)

        # Compute weighted sensitivities
        ws_list = []
        for k in range(K):
            ws_k = sens_inputs[k] * float(risk_weights[k])
            ws_list.append(ws_k)

        # Compute K_r for each risk class (sum of squares, no correlation for simplicity)
        k_r = []
        for r in range(6):
            k_sq = aadc.idouble(0.0)
            for k in range(K):
                if risk_class_idx[k] == r:
                    k_sq = k_sq + ws_list[k] * ws_list[k]
            k_r.append(np.sqrt(k_sq))

        # Cross-RC aggregation
        im_sq = aadc.idouble(0.0)
        for r in range(6):
            for s in range(6):
                im_sq = im_sq + PSI_MATRIX[r, s] * k_r[r] * k_r[s]

        im = np.sqrt(im_sq)
        im_output = im.mark_as_output()

    # Create thread pool
    workers = aadc.ThreadPool(num_threads)

    # Single evaluate call for ALL portfolios
    inputs = {sens_handles[k]: sensitivities[:, k] for k in range(K)}

    if compute_gradients:
        request = {im_output: sens_handles}
    else:
        request = {im_output: []}

    results = aadc.evaluate(funcs, request, inputs, workers)

    im_values = np.array(results[0][im_output])

    return im_values


# =============================================================================
# GPU Implementations
# =============================================================================

if NUMBA_AVAILABLE and CUDA_AVAILABLE:
    @cuda.jit
    def _simm_cuda_kernel(
        sensitivities,      # (P, K)
        risk_weights,       # (K,)
        risk_class_idx,     # (K,)
        psi_matrix,         # (6, 6)
        im_output,          # (P,)
    ):
        """
        CUDA kernel for SIMM calculation - one thread per portfolio.

        Optimized for H100 with:
        - Coalesced memory access
        - Minimal divergence
        - Local memory for intermediate values
        """
        p = cuda.grid(1)

        if p >= sensitivities.shape[0]:
            return

        K = sensitivities.shape[1]

        # Local arrays for intermediate computation
        k_r = cuda.local.array(6, dtype=numba.float64)
        for r in range(6):
            k_r[r] = 0.0

        # Compute weighted sensitivities and accumulate K_r^2
        for k in range(K):
            ws_k = sensitivities[p, k] * risk_weights[k]
            rc = risk_class_idx[k]
            k_r[rc] += ws_k * ws_k

        # Take sqrt
        for r in range(6):
            k_r[r] = numba.float64(k_r[r]) ** 0.5

        # Cross-RC aggregation
        im_sq = 0.0
        for r in range(6):
            for s in range(6):
                im_sq += psi_matrix[r, s] * k_r[r] * k_r[s]

        im_output[p] = im_sq ** 0.5


    @cuda.jit
    def _simm_cuda_kernel_shared(
        sensitivities,      # (P, K)
        risk_weights,       # (K,)
        risk_class_idx,     # (K,)
        psi_matrix,         # (6, 6)
        im_output,          # (P,)
    ):
        """
        CUDA kernel with shared memory for weights and psi matrix.

        More efficient when K is small enough to fit in shared memory.
        """
        # Shared memory for frequently accessed data
        shared_weights = cuda.shared.array(256, dtype=numba.float64)
        shared_psi = cuda.shared.array((6, 6), dtype=numba.float64)

        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        bs = cuda.blockDim.x

        K = sensitivities.shape[1]

        # Cooperative load of shared data
        if tx < K and tx < 256:
            shared_weights[tx] = risk_weights[tx]

        if tx < 36:  # 6x6 = 36
            r, s = tx // 6, tx % 6
            shared_psi[r, s] = psi_matrix[r, s]

        cuda.syncthreads()

        # Each thread processes one portfolio
        p = bx * bs + tx
        if p >= sensitivities.shape[0]:
            return

        # Local arrays
        k_r = cuda.local.array(6, dtype=numba.float64)
        for r in range(6):
            k_r[r] = 0.0

        # Compute
        for k in range(K):
            ws_k = sensitivities[p, k] * shared_weights[k]
            rc = risk_class_idx[k]
            k_r[rc] += ws_k * ws_k

        for r in range(6):
            k_r[r] = k_r[r] ** 0.5

        im_sq = 0.0
        for r in range(6):
            for s in range(6):
                im_sq += shared_psi[r, s] * k_r[r] * k_r[s]

        im_output[p] = im_sq ** 0.5


def simm_cuda_single_gpu(
    sensitivities: np.ndarray,
    risk_weights: np.ndarray,
    risk_class_idx: np.ndarray,
    device: int = 0,
    use_shared_memory: bool = True,
) -> np.ndarray:
    """
    SIMM calculation on single H100 GPU using Numba CUDA.
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")

    cuda.select_device(device)

    P, K = sensitivities.shape

    # Ensure contiguous arrays
    sensitivities = np.ascontiguousarray(sensitivities, dtype=np.float64)
    risk_weights = np.ascontiguousarray(risk_weights, dtype=np.float64)
    risk_class_idx = np.ascontiguousarray(risk_class_idx, dtype=np.int32)

    # Allocate output
    im_output = np.zeros(P, dtype=np.float64)

    # Transfer to GPU
    d_sens = cuda.to_device(sensitivities)
    d_weights = cuda.to_device(risk_weights)
    d_rc_idx = cuda.to_device(risk_class_idx)
    d_psi = cuda.to_device(PSI_MATRIX)
    d_im = cuda.to_device(im_output)

    # Launch configuration
    # H100 has 132 SMs, each can run multiple blocks
    # Use 256 threads per block (good occupancy)
    threads_per_block = 256
    blocks = (P + threads_per_block - 1) // threads_per_block

    # Launch kernel
    if use_shared_memory and K <= 256:
        _simm_cuda_kernel_shared[blocks, threads_per_block](
            d_sens, d_weights, d_rc_idx, d_psi, d_im
        )
    else:
        _simm_cuda_kernel[blocks, threads_per_block](
            d_sens, d_weights, d_rc_idx, d_psi, d_im
        )

    # Synchronize and copy back
    cuda.synchronize()
    d_im.copy_to_host(im_output)

    return im_output


def simm_cuda_multi_gpu(
    sensitivities: np.ndarray,
    risk_weights: np.ndarray,
    risk_class_idx: np.ndarray,
    num_gpus: int = 8,
) -> np.ndarray:
    """
    SIMM calculation distributed across multiple H100 GPUs.

    Splits portfolios evenly across GPUs and runs in parallel.
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")

    P = sensitivities.shape[0]
    available_gpus = min(num_gpus, len(cuda.gpus))
    chunk_size = (P + available_gpus - 1) // available_gpus

    results = [None] * available_gpus

    def process_on_gpu(gpu_id, start_idx, end_idx):
        chunk_sens = sensitivities[start_idx:end_idx]
        results[gpu_id] = simm_cuda_single_gpu(
            chunk_sens, risk_weights, risk_class_idx, device=gpu_id
        )

    # Launch threads for each GPU
    import threading
    threads = []
    for gpu_id in range(available_gpus):
        start_idx = gpu_id * chunk_size
        end_idx = min(start_idx + chunk_size, P)
        if start_idx < P:
            t = threading.Thread(target=process_on_gpu, args=(gpu_id, start_idx, end_idx))
            threads.append(t)
            t.start()

    # Wait for all GPUs
    for t in threads:
        t.join()

    return np.concatenate([r for r in results if r is not None])


if CUPY_AVAILABLE:
    def simm_cupy(
        sensitivities: np.ndarray,
        risk_weights: np.ndarray,
        risk_class_idx: np.ndarray,
        device: int = 0,
    ) -> np.ndarray:
        """
        SIMM calculation using CuPy - NumPy-like API on GPU.

        Often faster than Numba for array operations due to cuBLAS/cuDNN.
        """
        with cp.cuda.Device(device):
            # Transfer to GPU
            sens_gpu = cp.asarray(sensitivities)
            weights_gpu = cp.asarray(risk_weights)
            psi_gpu = cp.asarray(PSI_MATRIX)

            P, K = sens_gpu.shape

            # Weighted sensitivities
            ws = sens_gpu * weights_gpu

            # Compute K_r for each risk class
            k_r = cp.zeros((P, 6), dtype=cp.float64)
            for r in range(6):
                mask = (risk_class_idx == r)
                if np.any(mask):
                    k_r[:, r] = cp.sqrt(cp.sum(ws[:, mask] ** 2, axis=1))

            # Cross-RC aggregation
            im_sq = cp.einsum('pr,rs,ps->p', k_r, psi_gpu, k_r)
            im_values = cp.sqrt(im_sq)

            # Transfer back
            return cp.asnumpy(im_values)


# =============================================================================
# Benchmark Runner
# =============================================================================

def generate_test_data(config: BenchmarkConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random test data."""
    np.random.seed(42)

    sensitivities = np.random.randn(config.num_portfolios, config.num_risk_factors) * 1e6
    risk_weights = np.random.uniform(10, 100, config.num_risk_factors)
    risk_class_idx = np.random.randint(0, 6, config.num_risk_factors).astype(np.int32)

    return sensitivities, risk_weights, risk_class_idx


def run_benchmark(
    name: str,
    func,
    sensitivities: np.ndarray,
    risk_weights: np.ndarray,
    risk_class_idx: np.ndarray,
    config: BenchmarkConfig,
    reference_result: np.ndarray = None,
    **kwargs
) -> BenchmarkResult:
    """Run a single benchmark and return results."""
    P = sensitivities.shape[0]

    # Warmup
    for _ in range(config.warmup_iterations):
        try:
            _ = func(sensitivities, risk_weights, risk_class_idx, **kwargs)
        except Exception as e:
            return BenchmarkResult(
                backend=name,
                num_portfolios=P,
                num_risk_factors=sensitivities.shape[1],
                total_time_ms=0,
                avg_time_ms=0,
                throughput_portfolios_per_sec=0,
                memory_mb=0,
                num_threads_or_gpus=0,
                validated=False,
                notes=f"Error: {str(e)}"
            )

    # Timing
    times = []
    result = None
    for _ in range(config.num_iterations):
        start = time.perf_counter()
        result = func(sensitivities, risk_weights, risk_class_idx, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    total_time = sum(times)
    avg_time = np.mean(times)

    # Validation
    validated = True
    if reference_result is not None:
        max_diff = np.max(np.abs(result - reference_result))
        rel_diff = max_diff / np.mean(reference_result)
        validated = rel_diff < 1e-6

    # Memory estimate (rough)
    memory_mb = (sensitivities.nbytes + risk_weights.nbytes + result.nbytes) / 1e6

    return BenchmarkResult(
        backend=name,
        num_portfolios=P,
        num_risk_factors=sensitivities.shape[1],
        total_time_ms=total_time * 1000,
        avg_time_ms=avg_time * 1000,
        throughput_portfolios_per_sec=P / avg_time,
        memory_mb=memory_mb,
        num_threads_or_gpus=kwargs.get('num_workers', kwargs.get('num_gpus', 1)),
        validated=validated,
    )


def run_all_benchmarks(config: BenchmarkConfig) -> Tuple[List[BenchmarkResult], float]:
    """Run all benchmark configurations.

    Returns:
        Tuple of (results_list, mean_simm_total)
    """
    results = []

    print("=" * 80)
    print("SIMM CPU vs GPU Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Portfolios:     {config.num_portfolios:,}")
    print(f"  Risk Factors:   {config.num_risk_factors}")
    print(f"  Iterations:     {config.num_iterations}")
    print(f"  CPU Threads:    {config.cpu_threads}")
    print(f"  GPUs:           {config.num_gpus}")
    print()

    # Generate data
    print("Generating test data...")
    sensitivities, risk_weights, risk_class_idx = generate_test_data(config)
    print(f"  Data size: {sensitivities.nbytes / 1e6:.1f} MB")
    print()

    # Reference result (NumPy)
    print("Computing reference result (NumPy)...")
    reference = simm_numpy_vectorized(sensitivities, risk_weights, risk_class_idx)
    mean_simm = float(np.mean(reference))
    print(f"  Mean IM: ${mean_simm:,.0f}")
    print()

    # CPU Benchmarks
    print("-" * 80)
    print("CPU Benchmarks")
    print("-" * 80)

    # 1. NumPy Vectorized (MKL)
    print("Running: NumPy Vectorized (MKL)...")
    result = run_benchmark(
        "CPU NumPy (MKL)",
        simm_numpy_vectorized,
        sensitivities, risk_weights, risk_class_idx,
        config, reference
    )
    results.append(result)
    print(f"  Avg time: {result.avg_time_ms:.2f} ms, Throughput: {result.throughput_portfolios_per_sec:,.0f} portfolios/sec")

    # 2. Numba Parallel
    if NUMBA_AVAILABLE:
        print("Running: Numba Parallel...")
        # Set thread count
        numba.set_num_threads(config.cpu_threads)
        result = run_benchmark(
            f"CPU Numba Parallel ({config.cpu_threads} threads)",
            simm_numba_parallel,
            sensitivities, risk_weights, risk_class_idx,
            config, reference,
            psi_matrix=PSI_MATRIX
        )
        results.append(result)
        print(f"  Avg time: {result.avg_time_ms:.2f} ms, Throughput: {result.throughput_portfolios_per_sec:,.0f} portfolios/sec")

    # 3. Multi-process
    print(f"Running: Multi-process ({config.cpu_threads} workers)...")
    result = run_benchmark(
        f"CPU Multi-process ({config.cpu_threads} workers)",
        simm_cpu_multiprocess,
        sensitivities, risk_weights, risk_class_idx,
        config, reference,
        num_workers=config.cpu_threads
    )
    results.append(result)
    print(f"  Avg time: {result.avg_time_ms:.2f} ms, Throughput: {result.throughput_portfolios_per_sec:,.0f} portfolios/sec")

    # 4. AADC CPU
    if AADC_AVAILABLE:
        print(f"Running: AADC CPU ({config.cpu_threads} threads)...")
        result = run_benchmark(
            f"CPU AADC ({config.cpu_threads} threads)",
            simm_aadc_cpu,
            sensitivities, risk_weights, risk_class_idx,
            config, reference,
            num_threads=config.cpu_threads
        )
        results.append(result)
        print(f"  Avg time: {result.avg_time_ms:.2f} ms, Throughput: {result.throughput_portfolios_per_sec:,.0f} portfolios/sec")

        # 4b. AADC CPU with gradients
        print(f"Running: AADC CPU + Gradients ({config.cpu_threads} threads)...")
        result = run_benchmark(
            f"CPU AADC + Greeks ({config.cpu_threads} threads)",
            simm_aadc_cpu,
            sensitivities, risk_weights, risk_class_idx,
            config, reference,
            num_threads=config.cpu_threads,
            compute_gradients=True
        )
        results.append(result)
        print(f"  Avg time: {result.avg_time_ms:.2f} ms, Throughput: {result.throughput_portfolios_per_sec:,.0f} portfolios/sec")

    # GPU Benchmarks
    if CUDA_AVAILABLE:
        print()
        print("-" * 80)
        print("GPU Benchmarks")
        print("-" * 80)

        # Get GPU info
        try:
            device = cuda.get_current_device()
            print(f"  GPU: {device.name}")
        except:
            pass

        # 4. Single GPU
        print("Running: Single H100 GPU (Numba CUDA)...")
        result = run_benchmark(
            "GPU Single H100 (Numba)",
            simm_cuda_single_gpu,
            sensitivities, risk_weights, risk_class_idx,
            config, reference,
            device=config.gpu_device
        )
        results.append(result)
        print(f"  Avg time: {result.avg_time_ms:.2f} ms, Throughput: {result.throughput_portfolios_per_sec:,.0f} portfolios/sec")

        # 5. Multi-GPU
        if config.num_gpus > 1:
            print(f"Running: Multi-GPU ({config.num_gpus}x H100)...")
            result = run_benchmark(
                f"GPU Multi-H100 ({config.num_gpus}x)",
                simm_cuda_multi_gpu,
                sensitivities, risk_weights, risk_class_idx,
                config, reference,
                num_gpus=config.num_gpus
            )
            results.append(result)
            print(f"  Avg time: {result.avg_time_ms:.2f} ms, Throughput: {result.throughput_portfolios_per_sec:,.0f} portfolios/sec")

    # CuPy
    if CUPY_AVAILABLE:
        print("Running: CuPy GPU...")
        result = run_benchmark(
            "GPU CuPy",
            simm_cupy,
            sensitivities, risk_weights, risk_class_idx,
            config, reference,
            device=config.gpu_device
        )
        results.append(result)
        print(f"  Avg time: {result.avg_time_ms:.2f} ms, Throughput: {result.throughput_portfolios_per_sec:,.0f} portfolios/sec")

    return results, mean_simm


def print_results_table(results: List[BenchmarkResult]):
    """Print formatted results table."""
    print()
    print("=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print()
    print(f"{'Backend':<40} {'Time (ms)':<12} {'Throughput':<20} {'Speedup':<10} {'Valid'}")
    print("-" * 100)

    # Use first result as baseline
    baseline_time = results[0].avg_time_ms if results else 1

    for r in results:
        speedup = baseline_time / r.avg_time_ms if r.avg_time_ms > 0 else 0
        throughput_str = f"{r.throughput_portfolios_per_sec:,.0f} port/sec"
        valid_str = "✓" if r.validated else "✗"
        print(f"{r.backend:<40} {r.avg_time_ms:<12.2f} {throughput_str:<20} {speedup:<10.1f}x {valid_str}")

    print("=" * 100)


def save_results(results: List[BenchmarkResult], filename: str):
    """Save results to JSON file."""
    # Convert numpy bools to Python bools for JSON serialization
    results_data = []
    for r in results:
        d = asdict(r)
        d['validated'] = bool(d['validated'])  # Ensure Python bool
        results_data.append(d)

    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results_data
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filename}")


def log_to_execution_log(
    results: List[BenchmarkResult],
    log_file: str = "data/execution_log.csv",
    simm_total: float = 0.0,
):
    """
    Log benchmark results to execution_log.csv for comparison with AADC.

    Format matches existing SIMM execution log for easy comparison.
    """
    import csv
    from datetime import datetime
    import os

    # Ensure data directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(log_file)

    # Extended header with GPU-specific columns
    fieldnames = [
        'timestamp',
        'model_name',
        'model_version',
        'mode',
        'num_trades',           # = num_portfolios for this benchmark
        'num_risk_factors',
        'num_sensitivities',    # = num_portfolios * num_risk_factors
        'num_threads',
        'simm_total',
        'eval_time_sec',
        'recording_time_sec',
        'total_eval_time_sec',
        'memory_mb',
        'language',
        'uses_aadc',
        'status',
        # Extended fields for GPU benchmark
        'backend',
        'throughput_per_sec',
        'speedup_vs_baseline',
        'device_info',
    ]

    # Calculate baseline for speedup
    baseline_time = results[0].avg_time_ms if results else 1.0

    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')

        # Write header if new file
        if not file_exists:
            writer.writeheader()

        timestamp = datetime.now().isoformat()

        for r in results:
            # Determine language and device info
            if 'GPU' in r.backend or 'CUDA' in r.backend or 'CuPy' in r.backend:
                language = 'CUDA'
                uses_aadc = 'no'
                device_info = f"GPU x{r.num_threads_or_gpus}"
            elif 'AADC' in r.backend:
                language = 'Python'
                uses_aadc = 'yes'
                device_info = f"CPU x{r.num_threads_or_gpus} threads"
            else:
                language = 'Python'
                uses_aadc = 'no'
                device_info = f"CPU x{r.num_threads_or_gpus} threads"

            # Calculate speedup
            speedup = baseline_time / r.avg_time_ms if r.avg_time_ms > 0 else 0

            row = {
                'timestamp': timestamp,
                'model_name': f"simm_benchmark_{r.backend.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                'model_version': '1.0.0',
                'mode': 'margin_only',
                'num_trades': r.num_portfolios,
                'num_risk_factors': r.num_risk_factors,
                'num_sensitivities': r.num_portfolios * r.num_risk_factors,
                'num_threads': r.num_threads_or_gpus,
                'simm_total': simm_total,
                'eval_time_sec': r.avg_time_ms / 1000.0,
                'recording_time_sec': 0.0,
                'total_eval_time_sec': r.avg_time_ms / 1000.0,
                'memory_mb': r.memory_mb,
                'language': language,
                'uses_aadc': uses_aadc,
                'status': 'success' if r.validated else 'validation_failed',
                # Extended fields
                'backend': r.backend,
                'throughput_per_sec': r.throughput_portfolios_per_sec,
                'speedup_vs_baseline': speedup,
                'device_info': device_info,
            }
            writer.writerow(row)

    print(f"Results logged to {log_file}")


# =============================================================================
# Predefined Configurations
# =============================================================================

SCALE_CONFIGS = {
    "small": BenchmarkConfig(
        num_portfolios=1000,
        num_risk_factors=50,
        num_iterations=20,
    ),
    "medium": BenchmarkConfig(
        num_portfolios=10000,
        num_risk_factors=100,
        num_iterations=10,
    ),
    "large": BenchmarkConfig(
        num_portfolios=100000,
        num_risk_factors=100,
        num_iterations=5,
    ),
    "xlarge": BenchmarkConfig(
        num_portfolios=1000000,
        num_risk_factors=100,
        num_iterations=3,
    ),
    "stress": BenchmarkConfig(
        num_portfolios=10000000,
        num_risk_factors=100,
        num_iterations=1,
        warmup_iterations=1,
    ),
}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SIMM CPU vs GPU Benchmark")
    parser.add_argument("--portfolios", "-p", type=int, default=10000,
                        help="Number of portfolios")
    parser.add_argument("--factors", "-k", type=int, default=100,
                        help="Number of risk factors")
    parser.add_argument("--iterations", "-n", type=int, default=10,
                        help="Number of timing iterations")
    parser.add_argument("--cpu-threads", type=int, default=96,
                        help="CPU threads (default: 96 for dual Xeon 8568Y+)")
    parser.add_argument("--gpus", type=int, default=8,
                        help="Number of GPUs (default: 8 for HGX H100)")
    parser.add_argument("--scale", choices=SCALE_CONFIGS.keys(),
                        help="Use predefined scale configuration")
    parser.add_argument("--all", action="store_true",
                        help="Run all scale configurations")
    parser.add_argument("--output", "-o", type=str, default="benchmark_results.json",
                        help="Output JSON file")
    parser.add_argument("--log", "-l", action="store_true",
                        help="Log results to data/execution_log.csv")
    parser.add_argument("--log-file", type=str, default="data/execution_log.csv",
                        help="Execution log file path")

    args = parser.parse_args()

    # Print system info
    print("=" * 80)
    print("System Information")
    print("=" * 80)
    print(f"NumPy version:     {np.__version__}")
    print(f"MKL available:     {MKL_AVAILABLE}")
    print(f"Numba available:   {NUMBA_AVAILABLE}")
    print(f"AADC available:    {AADC_AVAILABLE}")
    print(f"CUDA available:    {CUDA_AVAILABLE}")
    print(f"CuPy available:    {CUPY_AVAILABLE}")

    if CUDA_AVAILABLE:
        print(f"CUDA devices:      {len(cuda.gpus)}")
        for i, gpu in enumerate(cuda.gpus):
            try:
                with gpu:
                    device = cuda.get_current_device()
                    print(f"  GPU {i}: {device.name}")
            except:
                print(f"  GPU {i}: (info unavailable)")

    print(f"CPU cores:         {mp.cpu_count()}")
    print()

    if args.all:
        # Run all scale configurations
        all_results = []
        total_simm = 0.0
        for scale_name, config in SCALE_CONFIGS.items():
            print(f"\n{'#' * 80}")
            print(f"# Scale: {scale_name.upper()}")
            print(f"{'#' * 80}")
            config.cpu_threads = args.cpu_threads
            config.num_gpus = args.gpus
            results, mean_simm = run_all_benchmarks(config)
            print_results_table(results)
            all_results.extend(results)
            total_simm = mean_simm  # Use last one
        save_results(all_results, args.output)
        if args.log:
            log_to_execution_log(all_results, args.log_file, total_simm)
    else:
        # Single configuration
        if args.scale:
            config = SCALE_CONFIGS[args.scale]
        else:
            config = BenchmarkConfig(
                num_portfolios=args.portfolios,
                num_risk_factors=args.factors,
                num_iterations=args.iterations,
            )

        config.cpu_threads = args.cpu_threads
        config.num_gpus = args.gpus
        config.output_file = args.output

        results, mean_simm = run_all_benchmarks(config)
        print_results_table(results)
        save_results(results, args.output)
        if args.log:
            log_to_execution_log(results, args.log_file, mean_simm)


if __name__ == "__main__":
    main()
