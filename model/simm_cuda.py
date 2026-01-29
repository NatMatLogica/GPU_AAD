"""
CUDA-Simulated ISDA SIMM Calculator

This module implements SIMM margin calculation using CUDA kernels.
Can run on:
- Real NVIDIA GPU (if available)
- Numba CUDA Simulator (for testing without GPU)

Usage:
    # Enable simulator mode (no GPU required)
    import os
    os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

    from model.simm_cuda import compute_simm_cuda, benchmark_cuda_vs_numpy

Version: 1.0.0
"""

import os
import numpy as np
import time
from typing import Tuple, List, Dict
from dataclasses import dataclass

# Check if we're in simulator mode
CUDA_SIMULATOR = os.environ.get('NUMBA_ENABLE_CUDASIM', '0') == '1'

from numba import cuda
import math

# =============================================================================
# SIMM Parameters (simplified for CUDA demo)
# =============================================================================

# Cross-risk-class correlation matrix (ψ) — ISDA SIMM v2.6
# Order: Rates, CreditQ, CreditNonQ, Equity, Commodity, FX
PSI_MATRIX = np.array([
    [1.00, 0.04, 0.04, 0.07, 0.37, 0.14],  # Rates
    [0.04, 1.00, 0.54, 0.70, 0.27, 0.37],  # CreditQ
    [0.04, 0.54, 1.00, 0.46, 0.24, 0.15],  # CreditNonQ
    [0.07, 0.70, 0.46, 1.00, 0.35, 0.39],  # Equity
    [0.37, 0.27, 0.24, 0.35, 1.00, 0.35],  # Commodity
    [0.14, 0.37, 0.15, 0.39, 0.35, 1.00],  # FX
], dtype=np.float64)

# IR tenor correlation matrix (12x12) — ISDA SIMM v2.6 simplified
# Tenors: 2w, 1m, 3m, 6m, 1y, 2y, 3y, 5y, 10y, 15y, 20y, 30y
def _build_ir_correlation_matrix():
    """Build IR tenor correlation matrix using exponential decay (v2.6 approximation)."""
    tenors_years = np.array([2/52, 1/12, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30])
    n = len(tenors_years)
    corr = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            # Exponential decay correlation
            corr[i, j] = np.exp(-0.03 * abs(tenors_years[i] - tenors_years[j]))
    return corr

IR_CORRELATION_MATRIX = _build_ir_correlation_matrix()

# Default risk weights by risk class — ISDA SIMM v2.6 (approximate averages)
DEFAULT_RISK_WEIGHTS = {
    'IR': 55.0,        # basis points (v2.6 regular vol avg: 50-77 bps)
    'CreditQ': 75.0,
    'CreditNonQ': 500.0,
    'Equity': 25.0,    # percentage
    'Commodity': 19.0,
    'FX': 8.4,
}


# =============================================================================
# CUDA Kernels
# =============================================================================

@cuda.jit
def _simm_kernel_simple(
    sensitivities,      # (P, K) - sensitivities for P portfolios, K risk factors
    risk_weights,       # (K,) - risk weight per factor
    risk_class_idx,     # (K,) - which risk class each factor belongs to (0-5)
    psi_matrix,         # (6, 6) - cross-risk-class correlations
    im_output,          # (P,) - output IM for each portfolio
):
    """
    Simple SIMM kernel: one thread per portfolio.

    For each portfolio p:
    1. Compute weighted sensitivities: WS[k] = S[p,k] * w[k]
    2. Aggregate per risk class: K_r = sqrt(sum(WS[k]^2)) for k in risk_class r
    3. Cross-RC aggregation: IM = sqrt(sum_r sum_s psi[r,s] * K_r * K_s)
    """
    p = cuda.grid(1)  # Portfolio index

    if p >= sensitivities.shape[0]:
        return

    K = sensitivities.shape[1]

    # Step 1: Compute risk class margins (K_r)
    # Using local array for 6 risk classes
    k_r = cuda.local.array(6, dtype=np.float64)
    for r in range(6):
        k_r[r] = 0.0

    # Sum of squares per risk class (simplified - no intra-RC correlation)
    for k in range(K):
        ws_k = sensitivities[p, k] * risk_weights[k]
        rc = risk_class_idx[k]
        k_r[rc] += ws_k * ws_k

    # Take sqrt for each risk class
    for r in range(6):
        k_r[r] = math.sqrt(k_r[r])

    # Step 2: Cross-risk-class aggregation
    im_sq = 0.0
    for r in range(6):
        for s in range(6):
            im_sq += psi_matrix[r, s] * k_r[r] * k_r[s]

    im_output[p] = math.sqrt(im_sq)


@cuda.jit
def _simm_kernel_with_ir_correlation(
    sensitivities,      # (P, K) - sensitivities for P portfolios, K risk factors
    risk_weights,       # (K,) - risk weight per factor
    risk_class_idx,     # (K,) - which risk class each factor belongs to (0-5)
    ir_tenor_idx,       # (K,) - IR tenor index (0-11) or -1 if not IR
    ir_corr_matrix,     # (12, 12) - IR tenor correlations
    psi_matrix,         # (6, 6) - cross-risk-class correlations
    im_output,          # (P,) - output IM for each portfolio
):
    """
    SIMM kernel with IR intra-bucket correlations.

    For IR risk class, uses full correlation matrix:
    K_IR = sqrt(sum_i sum_j rho_ij * WS_i * WS_j)
    """
    p = cuda.grid(1)  # Portfolio index

    if p >= sensitivities.shape[0]:
        return

    K = sensitivities.shape[1]

    # Compute weighted sensitivities
    ws = cuda.local.array(200, dtype=np.float64)  # Max 200 risk factors
    for k in range(min(K, 200)):
        ws[k] = sensitivities[p, k] * risk_weights[k]

    # Step 1: Compute risk class margins
    k_r = cuda.local.array(6, dtype=np.float64)
    for r in range(6):
        k_r[r] = 0.0

    # IR risk class (index 0) - use correlation matrix
    ir_k_sq = 0.0
    for i in range(K):
        if risk_class_idx[i] == 0:  # IR
            for j in range(K):
                if risk_class_idx[j] == 0:  # IR
                    ti = ir_tenor_idx[i]
                    tj = ir_tenor_idx[j]
                    if ti >= 0 and tj >= 0:
                        rho_ij = ir_corr_matrix[ti, tj]
                        ir_k_sq += rho_ij * ws[i] * ws[j]
    k_r[0] = math.sqrt(max(ir_k_sq, 0.0))

    # Other risk classes - simplified (sum of squares)
    for k in range(K):
        rc = risk_class_idx[k]
        if rc > 0:  # Not IR
            k_r[rc] += ws[k] * ws[k]

    for r in range(1, 6):
        k_r[r] = math.sqrt(k_r[r])

    # Step 2: Cross-risk-class aggregation
    im_sq = 0.0
    for r in range(6):
        for s in range(6):
            im_sq += psi_matrix[r, s] * k_r[r] * k_r[s]

    im_output[p] = math.sqrt(im_sq)


@cuda.jit
def _simm_gradient_kernel(
    sensitivities,      # (P, K) - sensitivities
    risk_weights,       # (K,) - risk weights
    risk_class_idx,     # (K,) - risk class indices
    psi_matrix,         # (6, 6) - cross-RC correlations
    im_values,          # (P,) - pre-computed IM values
    gradients,          # (P, K) - output: dIM/dS for each portfolio and factor
):
    """
    Compute gradient dIM/dS[k] for each portfolio.

    Using chain rule:
    dIM/dS[k] = dIM/dK_r * dK_r/dWS[k] * dWS[k]/dS[k]
              = dIM/dK_r * (WS[k] / K_r) * w[k]

    where:
    dIM/dK_r = (1/IM) * sum_s psi[r,s] * K_s
    """
    p = cuda.grid(1)

    if p >= sensitivities.shape[0]:
        return

    K = sensitivities.shape[1]
    im_p = im_values[p]

    if im_p < 1e-10:
        for k in range(K):
            gradients[p, k] = 0.0
        return

    # Compute weighted sensitivities and K_r
    ws = cuda.local.array(200, dtype=np.float64)
    k_r = cuda.local.array(6, dtype=np.float64)

    for r in range(6):
        k_r[r] = 0.0

    for k in range(min(K, 200)):
        ws[k] = sensitivities[p, k] * risk_weights[k]
        rc = risk_class_idx[k]
        k_r[rc] += ws[k] * ws[k]

    for r in range(6):
        k_r[r] = math.sqrt(max(k_r[r], 1e-20))

    # Compute dIM/dK_r for each risk class
    dim_dk = cuda.local.array(6, dtype=np.float64)
    for r in range(6):
        dim_dk[r] = 0.0
        for s in range(6):
            dim_dk[r] += psi_matrix[r, s] * k_r[s]
        dim_dk[r] /= im_p

    # Compute gradient for each risk factor
    for k in range(K):
        rc = risk_class_idx[k]
        w_k = risk_weights[k]

        if k_r[rc] > 1e-10:
            # dK_r/dWS[k] = WS[k] / K_r (for sum-of-squares aggregation)
            dk_dws = ws[k] / k_r[rc]
            gradients[p, k] = dim_dk[rc] * dk_dws * w_k
        else:
            gradients[p, k] = 0.0


# =============================================================================
# Python Wrapper Functions
# =============================================================================

def compute_simm_cuda(
    sensitivities: np.ndarray,
    risk_weights: np.ndarray = None,
    risk_class_idx: np.ndarray = None,
    use_ir_correlation: bool = False,
    ir_tenor_idx: np.ndarray = None,
) -> np.ndarray:
    """
    Compute SIMM margin for multiple portfolios using CUDA.

    Args:
        sensitivities: (P, K) array - sensitivities for P portfolios, K risk factors
        risk_weights: (K,) array - risk weight per factor (default: 50.0)
        risk_class_idx: (K,) array - risk class index 0-5 (default: all IR)
        use_ir_correlation: Whether to use IR tenor correlations
        ir_tenor_idx: (K,) array - IR tenor index 0-11 or -1 (required if use_ir_correlation)

    Returns:
        (P,) array of IM values
    """
    P, K = sensitivities.shape

    # Default parameters
    if risk_weights is None:
        risk_weights = np.full(K, 50.0, dtype=np.float64)
    if risk_class_idx is None:
        risk_class_idx = np.zeros(K, dtype=np.int32)  # All IR

    # Ensure correct dtypes
    sensitivities = np.ascontiguousarray(sensitivities, dtype=np.float64)
    risk_weights = np.ascontiguousarray(risk_weights, dtype=np.float64)
    risk_class_idx = np.ascontiguousarray(risk_class_idx, dtype=np.int32)

    # Allocate output
    im_output = np.zeros(P, dtype=np.float64)

    # Transfer to device (or use directly in simulator mode)
    d_sens = cuda.to_device(sensitivities)
    d_weights = cuda.to_device(risk_weights)
    d_rc_idx = cuda.to_device(risk_class_idx)
    d_psi = cuda.to_device(PSI_MATRIX)
    d_im = cuda.to_device(im_output)

    # Launch kernel
    threads_per_block = 256
    blocks = (P + threads_per_block - 1) // threads_per_block

    if use_ir_correlation and ir_tenor_idx is not None:
        ir_tenor_idx = np.ascontiguousarray(ir_tenor_idx, dtype=np.int32)
        d_ir_tenor = cuda.to_device(ir_tenor_idx)
        d_ir_corr = cuda.to_device(IR_CORRELATION_MATRIX)
        _simm_kernel_with_ir_correlation[blocks, threads_per_block](
            d_sens, d_weights, d_rc_idx, d_ir_tenor, d_ir_corr, d_psi, d_im
        )
    else:
        _simm_kernel_simple[blocks, threads_per_block](
            d_sens, d_weights, d_rc_idx, d_psi, d_im
        )

    # Copy back
    d_im.copy_to_host(im_output)

    return im_output


def compute_simm_gradient_cuda(
    sensitivities: np.ndarray,
    risk_weights: np.ndarray = None,
    risk_class_idx: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SIMM margin and gradients for multiple portfolios using CUDA.

    Args:
        sensitivities: (P, K) array
        risk_weights: (K,) array
        risk_class_idx: (K,) array

    Returns:
        (im_values, gradients) where:
        - im_values: (P,) array of IM values
        - gradients: (P, K) array of dIM/dS
    """
    P, K = sensitivities.shape

    # Compute IM first
    im_values = compute_simm_cuda(sensitivities, risk_weights, risk_class_idx)

    # Default parameters
    if risk_weights is None:
        risk_weights = np.full(K, 50.0, dtype=np.float64)
    if risk_class_idx is None:
        risk_class_idx = np.zeros(K, dtype=np.int32)

    # Ensure correct dtypes
    sensitivities = np.ascontiguousarray(sensitivities, dtype=np.float64)
    risk_weights = np.ascontiguousarray(risk_weights, dtype=np.float64)
    risk_class_idx = np.ascontiguousarray(risk_class_idx, dtype=np.int32)

    # Allocate gradient output
    gradients = np.zeros((P, K), dtype=np.float64)

    # Transfer to device
    d_sens = cuda.to_device(sensitivities)
    d_weights = cuda.to_device(risk_weights)
    d_rc_idx = cuda.to_device(risk_class_idx)
    d_psi = cuda.to_device(PSI_MATRIX)
    d_im = cuda.to_device(im_values)
    d_grad = cuda.to_device(gradients)

    # Launch gradient kernel
    threads_per_block = 256
    blocks = (P + threads_per_block - 1) // threads_per_block

    _simm_gradient_kernel[blocks, threads_per_block](
        d_sens, d_weights, d_rc_idx, d_psi, d_im, d_grad
    )

    # Copy back
    d_grad.copy_to_host(gradients)

    return im_values, gradients


def compute_simm_numpy(
    sensitivities: np.ndarray,
    risk_weights: np.ndarray = None,
    risk_class_idx: np.ndarray = None,
) -> np.ndarray:
    """
    NumPy reference implementation for validation.
    """
    P, K = sensitivities.shape

    if risk_weights is None:
        risk_weights = np.full(K, 50.0)
    if risk_class_idx is None:
        risk_class_idx = np.zeros(K, dtype=np.int32)

    im_values = np.zeros(P)

    for p in range(P):
        # Weighted sensitivities
        ws = sensitivities[p, :] * risk_weights

        # Risk class margins
        k_r = np.zeros(6)
        for r in range(6):
            mask = risk_class_idx == r
            k_r[r] = np.sqrt(np.sum(ws[mask] ** 2))

        # Cross-RC aggregation
        im_sq = 0.0
        for r in range(6):
            for s in range(6):
                im_sq += PSI_MATRIX[r, s] * k_r[r] * k_r[s]

        im_values[p] = np.sqrt(im_sq)

    return im_values


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_cuda_vs_numpy(
    num_portfolios: int = 100,
    num_risk_factors: int = 50,
    num_iterations: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Benchmark CUDA vs NumPy SIMM implementations.

    Args:
        num_portfolios: Number of portfolios P
        num_risk_factors: Number of risk factors K
        num_iterations: Number of timing iterations
        verbose: Print results

    Returns:
        Dict with timing results
    """
    # Generate test data
    np.random.seed(42)
    sensitivities = np.random.randn(num_portfolios, num_risk_factors) * 1e6
    risk_weights = np.random.uniform(10, 100, num_risk_factors)
    risk_class_idx = np.random.randint(0, 6, num_risk_factors).astype(np.int32)

    results = {
        'num_portfolios': num_portfolios,
        'num_risk_factors': num_risk_factors,
        'cuda_simulator': CUDA_SIMULATOR,
    }

    # Warm-up
    _ = compute_simm_cuda(sensitivities, risk_weights, risk_class_idx)
    _ = compute_simm_numpy(sensitivities, risk_weights, risk_class_idx)

    # Benchmark NumPy
    start = time.perf_counter()
    for _ in range(num_iterations):
        im_numpy = compute_simm_numpy(sensitivities, risk_weights, risk_class_idx)
    numpy_time = (time.perf_counter() - start) / num_iterations
    results['numpy_time_ms'] = numpy_time * 1000

    # Benchmark CUDA
    start = time.perf_counter()
    for _ in range(num_iterations):
        im_cuda = compute_simm_cuda(sensitivities, risk_weights, risk_class_idx)
    cuda_time = (time.perf_counter() - start) / num_iterations
    results['cuda_time_ms'] = cuda_time * 1000

    # Validate
    max_diff = np.max(np.abs(im_cuda - im_numpy))
    rel_diff = max_diff / np.mean(im_numpy) if np.mean(im_numpy) > 0 else 0
    results['max_abs_diff'] = max_diff
    results['max_rel_diff'] = rel_diff
    results['validation_passed'] = rel_diff < 1e-6

    # Speedup (note: simulator mode is slower than numpy!)
    results['speedup'] = numpy_time / cuda_time if cuda_time > 0 else 0

    if verbose:
        print("=" * 70)
        print("CUDA vs NumPy SIMM Benchmark")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Portfolios:     {num_portfolios}")
        print(f"  Risk Factors:   {num_risk_factors}")
        print(f"  CUDA Simulator: {CUDA_SIMULATOR}")
        print()
        print(f"Timing (avg of {num_iterations} iterations):")
        print(f"  NumPy:          {numpy_time*1000:.3f} ms")
        print(f"  CUDA:           {cuda_time*1000:.3f} ms")
        print(f"  Speedup:        {results['speedup']:.2f}x")
        if CUDA_SIMULATOR:
            print(f"  (Note: Simulator mode is slower than NumPy - this tests correctness)")
        print()
        print(f"Validation:")
        print(f"  Max abs diff:   {max_diff:.6e}")
        print(f"  Max rel diff:   {rel_diff:.6e}")
        print(f"  Status:         {'PASS' if results['validation_passed'] else 'FAIL'}")
        print("=" * 70)

    return results


def benchmark_gradient_cuda(
    num_portfolios: int = 100,
    num_risk_factors: int = 50,
    verbose: bool = True,
) -> Dict:
    """
    Benchmark CUDA gradient computation with finite difference validation.
    """
    np.random.seed(42)
    sensitivities = np.random.randn(num_portfolios, num_risk_factors) * 1e6
    risk_weights = np.random.uniform(10, 100, num_risk_factors)
    risk_class_idx = np.random.randint(0, 6, num_risk_factors).astype(np.int32)

    # Compute CUDA gradient
    start = time.perf_counter()
    im_values, gradients = compute_simm_gradient_cuda(
        sensitivities, risk_weights, risk_class_idx
    )
    cuda_grad_time = time.perf_counter() - start

    # Validate with finite differences (for first portfolio only)
    eps = 1e-6
    fd_gradient = np.zeros(num_risk_factors)

    start = time.perf_counter()
    base_im = im_values[0]
    for k in range(num_risk_factors):
        sens_plus = sensitivities.copy()
        sens_plus[0, k] += eps
        im_plus = compute_simm_cuda(sens_plus, risk_weights, risk_class_idx)[0]
        fd_gradient[k] = (im_plus - base_im) / eps
    fd_time = time.perf_counter() - start

    # Compare
    max_grad_diff = np.max(np.abs(gradients[0, :] - fd_gradient))
    rel_grad_diff = max_grad_diff / (np.mean(np.abs(fd_gradient)) + 1e-10)

    results = {
        'cuda_gradient_time_ms': cuda_grad_time * 1000,
        'fd_gradient_time_ms': fd_time * 1000,
        'speedup_vs_fd': fd_time / cuda_grad_time if cuda_grad_time > 0 else 0,
        'max_grad_diff': max_grad_diff,
        'rel_grad_diff': rel_grad_diff,
        'gradient_valid': rel_grad_diff < 1e-4,
    }

    if verbose:
        print("=" * 70)
        print("CUDA Gradient Benchmark")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Portfolios:     {num_portfolios}")
        print(f"  Risk Factors:   {num_risk_factors}")
        print()
        print(f"Timing:")
        print(f"  CUDA gradient (all P):  {cuda_grad_time*1000:.3f} ms")
        print(f"  Finite diff (1 portfolio, K bumps): {fd_time*1000:.3f} ms")
        print(f"  Speedup vs FD:          {results['speedup_vs_fd']:.1f}x")
        print()
        print(f"Gradient Validation (portfolio 0):")
        print(f"  Max abs diff:   {max_grad_diff:.6e}")
        print(f"  Max rel diff:   {rel_grad_diff:.6e}")
        print(f"  Status:         {'PASS' if results['gradient_valid'] else 'FAIL'}")
        print("=" * 70)

    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run benchmarks."""
    print("\n" + "=" * 70)
    print("SIMM CUDA Implementation Test")
    print("=" * 70)

    if CUDA_SIMULATOR:
        print("\nRunning in CUDA SIMULATOR mode (no GPU required)")
        print("This tests correctness, not performance.\n")
    else:
        print("\nRunning with real CUDA GPU\n")

    # Test 1: Small scale
    print("\n--- Test 1: Small Scale (10 portfolios, 20 factors) ---")
    benchmark_cuda_vs_numpy(num_portfolios=10, num_risk_factors=20)

    # Test 2: Medium scale
    print("\n--- Test 2: Medium Scale (100 portfolios, 50 factors) ---")
    benchmark_cuda_vs_numpy(num_portfolios=100, num_risk_factors=50)

    # Test 3: Large scale
    print("\n--- Test 3: Large Scale (1000 portfolios, 100 factors) ---")
    benchmark_cuda_vs_numpy(num_portfolios=1000, num_risk_factors=100)

    # Test 4: Gradient computation
    print("\n--- Test 4: Gradient Computation ---")
    benchmark_gradient_cuda(num_portfolios=50, num_risk_factors=30)

    # Test 5: With IR correlations
    print("\n--- Test 5: IR Correlation Test ---")
    np.random.seed(42)
    P, K = 10, 12  # 12 IR tenors
    sensitivities = np.random.randn(P, K) * 1e6
    ir_tenor_idx = np.arange(K, dtype=np.int32)  # Each factor is a different tenor
    risk_weights = np.full(K, 50.0)
    risk_class_idx = np.zeros(K, dtype=np.int32)  # All IR

    im_simple = compute_simm_cuda(sensitivities, risk_weights, risk_class_idx,
                                   use_ir_correlation=False)
    im_corr = compute_simm_cuda(sensitivities, risk_weights, risk_class_idx,
                                 use_ir_correlation=True, ir_tenor_idx=ir_tenor_idx)

    print(f"  IM (no correlation):   {np.mean(im_simple):,.0f}")
    print(f"  IM (with correlation): {np.mean(im_corr):,.0f}")
    print(f"  Difference:            {np.mean(im_corr - im_simple):,.0f}")
    print(f"  (Correlations typically INCREASE margin due to diversification accounting)")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
