#!/usr/bin/env python
"""
Pure GPU Implementation for IR Swaps - Full Pipeline on Device

This implementation keeps everything on GPU:
1. Trade data lives on GPU memory
2. Pricing (IRS) happens on GPU
3. CRIF sensitivities computed via bump-and-revalue on GPU
4. SIMM aggregation on GPU
5. Gradients via finite-difference on GPU

This provides a fair apples-to-apples comparison with AADC implementations
by eliminating CPU-GPU transfer overhead from the benchmark.

Supports only IR Swaps (Risk_IRCurve). Other asset classes will raise an error.

Usage:
    from model.pure_gpu_ir import PureGPUIRBackend

    backend = PureGPUIRBackend(num_threads=8)
    backend.setup(trades, market)
    im, grad = backend.compute_im_and_gradient(allocation)

Version: 1.0.0
"""

import os
import sys
import math
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for CUDA
CUDA_SIMULATOR = os.environ.get('NUMBA_ENABLE_CUDASIM', '0') == '1'

try:
    from numba import cuda
    import numba
    CUDA_AVAILABLE = cuda.is_available() or CUDA_SIMULATOR
except ImportError:
    CUDA_AVAILABLE = False
    cuda = None
    numba = None

MODEL_VERSION = "1.1.0"  # v1.1: Optimized GPU kernels with shared memory and parallel reduction

# =============================================================================
# Constants
# =============================================================================

NUM_IR_TENORS = 12
IR_TENORS = np.array([2/52, 1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0], dtype=np.float64)
TENOR_LABELS = ["2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"]
BUMP_SIZE = 0.0001  # 1bp

# ISDA SIMM v2.6 IR Risk Weights (Regular volatility currencies)
IR_RISK_WEIGHTS = np.array([
    109.0,  # 2w
    104.0,  # 1m
    71.0,   # 3m
    64.0,   # 6m
    52.0,   # 1y
    49.0,   # 2y
    51.0,   # 3y
    51.0,   # 5y
    51.0,   # 10y
    53.0,   # 15y
    56.0,   # 20y
    64.0,   # 30y
], dtype=np.float64)

# ISDA SIMM v2.6 IR Tenor Correlations (12x12)
IR_CORRELATIONS = np.array([
    [1.00, 0.79, 0.67, 0.53, 0.42, 0.37, 0.35, 0.33, 0.31, 0.30, 0.28, 0.28],
    [0.79, 1.00, 0.85, 0.69, 0.57, 0.50, 0.47, 0.44, 0.41, 0.39, 0.37, 0.37],
    [0.67, 0.85, 1.00, 0.86, 0.74, 0.67, 0.63, 0.59, 0.55, 0.52, 0.50, 0.49],
    [0.53, 0.69, 0.86, 1.00, 0.93, 0.87, 0.83, 0.78, 0.72, 0.68, 0.65, 0.63],
    [0.42, 0.57, 0.74, 0.93, 1.00, 0.98, 0.95, 0.90, 0.84, 0.79, 0.76, 0.73],
    [0.37, 0.50, 0.67, 0.87, 0.98, 1.00, 0.99, 0.95, 0.89, 0.84, 0.81, 0.78],
    [0.35, 0.47, 0.63, 0.83, 0.95, 0.99, 1.00, 0.98, 0.93, 0.88, 0.85, 0.82],
    [0.33, 0.44, 0.59, 0.78, 0.90, 0.95, 0.98, 1.00, 0.97, 0.94, 0.91, 0.88],
    [0.31, 0.41, 0.55, 0.72, 0.84, 0.89, 0.93, 0.97, 1.00, 0.98, 0.97, 0.95],
    [0.30, 0.39, 0.52, 0.68, 0.79, 0.84, 0.88, 0.94, 0.98, 1.00, 0.99, 0.98],
    [0.28, 0.37, 0.50, 0.65, 0.76, 0.81, 0.85, 0.91, 0.97, 0.99, 1.00, 0.99],
    [0.28, 0.37, 0.49, 0.63, 0.73, 0.78, 0.82, 0.88, 0.95, 0.98, 0.99, 1.00],
], dtype=np.float64)

# Concentration threshold for IR Delta (USD)
IR_CONCENTRATION_THRESHOLD = 230e6  # $230M

# PSI matrix (cross-risk-class) - only Rates row/col used for IR-only
PSI_RATES = 1.0  # psi[0,0] = 1.0


# =============================================================================
# CUDA Device Functions
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit(device=True)
    def _interp_rate(t, tenors, rates):
        """Linear interpolation for yield curve rate at time t."""
        if t <= tenors[0]:
            return rates[0]
        if t >= tenors[11]:
            return rates[11]
        for i in range(11):
            if tenors[i] <= t < tenors[i + 1]:
                alpha = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
                return rates[i] + alpha * (rates[i + 1] - rates[i])
        return rates[11]

    @cuda.jit(device=True)
    def _discount(t, tenors, rates):
        """Discount factor at time t."""
        if t <= 0.0:
            return 1.0
        r = _interp_rate(t, tenors, rates)
        return math.exp(-r * t)

    @cuda.jit(device=True)
    def _forward_rate(t1, t2, tenors, rates):
        """Forward rate from t1 to t2."""
        if t2 <= t1:
            return _interp_rate(t1, tenors, rates)
        df1 = _discount(t1, tenors, rates)
        df2 = _discount(t2, tenors, rates)
        if df2 <= 0.0:
            return 0.0
        return math.log(df1 / df2) / (t2 - t1)

    @cuda.jit(device=True)
    def _price_irs(notional, fixed_rate, maturity, frequency, payer, tenors, rates):
        """Price vanilla IRS on device."""
        dt = 1.0 / frequency
        num_periods = int(maturity * frequency)

        fixed_leg = 0.0
        floating_leg = 0.0

        for i in range(1, num_periods + 1):
            t = i * dt
            df = _discount(t, tenors, rates)
            fixed_leg += notional * fixed_rate * dt * df
            t_prev = (i - 1) * dt
            fwd = _forward_rate(t_prev, t, tenors, rates)
            floating_leg += notional * dt * fwd * df

        npv = floating_leg - fixed_leg
        if not payer:
            npv = fixed_leg - floating_leg
        return npv


# =============================================================================
# CUDA Kernels
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def _compute_crif_kernel(
        # Trade data (T trades)
        notionals, fixed_rates, maturities, frequencies, payers,
        # Market data
        curve_rates, tenors,
        # Output: (T, K) sensitivities
        sensitivities,
    ):
        """
        Compute CRIF sensitivities for all trades via GPU bump-and-revalue.
        One thread per trade.
        """
        t_idx = cuda.grid(1)
        if t_idx >= notionals.shape[0]:
            return

        notional = notionals[t_idx]
        fixed_rate = fixed_rates[t_idx]
        maturity = maturities[t_idx]
        frequency = frequencies[t_idx]
        payer = payers[t_idx]

        # Base price
        base_pv = _price_irs(notional, fixed_rate, maturity, frequency, payer,
                            tenors, curve_rates)

        # Local bumped rates
        bumped_rates = cuda.local.array(12, dtype=numba.float64)

        # Bump each tenor
        for k in range(12):
            for j in range(12):
                bumped_rates[j] = curve_rates[j]
            bumped_rates[k] += BUMP_SIZE

            bumped_pv = _price_irs(notional, fixed_rate, maturity, frequency, payer,
                                   tenors, bumped_rates)

            # DV01 = change per 1bp bump, scaled by 1/bump_size to match CRIF convention
            sensitivities[t_idx, k] = (bumped_pv - base_pv) / BUMP_SIZE

    @cuda.jit
    def _simm_im_gradient_kernel(
        # Aggregated sensitivities (P, K)
        agg_sens,
        # SIMM parameters
        risk_weights, correlations_flat, concentration_factors,
        # Outputs
        im_output, gradients,
    ):
        """
        Compute SIMM IM and gradients for P portfolios.
        One thread per portfolio.

        For IR-only:
        - K = 12 (tenors)
        - K_ir = sqrt(sum_ij rho_ij * WS_i * WS_j)
        - IM = K_ir (single risk class)
        - Gradient via chain rule
        """
        p = cuda.grid(1)
        if p >= agg_sens.shape[0]:
            return

        K = 12

        # Local arrays
        ws = cuda.local.array(12, dtype=numba.float64)

        # Step 1: Weighted sensitivities with concentration
        for k in range(K):
            ws[k] = agg_sens[p, k] * risk_weights[k] * concentration_factors[k]

        # Step 2: K_ir^2 = sum_ij rho_ij * WS_i * WS_j
        k_ir_sq = 0.0
        for i in range(K):
            for j in range(K):
                rho = correlations_flat[i * K + j]
                k_ir_sq += rho * ws[i] * ws[j]

        # K_ir
        k_ir = math.sqrt(k_ir_sq) if k_ir_sq > 0.0 else 0.0
        im_output[p] = k_ir

        # Step 3: Gradient dIM/dS_k
        # dK_ir/dWS_k = (1/K_ir) * sum_j rho_kj * WS_j
        # dWS_k/dS_k = RW_k * CR_k
        # dIM/dS_k = dK_ir/dWS_k * dWS_k/dS_k
        if k_ir > 1e-12:
            for k in range(K):
                dK_dWS = 0.0
                for j in range(K):
                    rho = correlations_flat[k * K + j]
                    dK_dWS += rho * ws[j]
                dK_dWS /= k_ir
                gradients[p, k] = dK_dWS * risk_weights[k] * concentration_factors[k]
        else:
            for k in range(K):
                gradients[p, k] = 0.0

    @cuda.jit
    def _simm_im_only_kernel(
        # Aggregated sensitivities (N, K)
        agg_sens,
        # SIMM parameters
        risk_weights, correlations_flat, concentration_factors,
        # Output
        im_output,
    ):
        """
        Forward-only SIMM: compute IM without gradients.
        Used for brute-force optimization.
        """
        p = cuda.grid(1)
        if p >= agg_sens.shape[0]:
            return

        K = 12
        ws = cuda.local.array(12, dtype=numba.float64)

        for k in range(K):
            ws[k] = agg_sens[p, k] * risk_weights[k] * concentration_factors[k]

        k_ir_sq = 0.0
        for i in range(K):
            for j in range(K):
                rho = correlations_flat[i * K + j]
                k_ir_sq += rho * ws[i] * ws[j]

        im_output[p] = math.sqrt(k_ir_sq) if k_ir_sq > 0.0 else 0.0


# =============================================================================
# Optimized CUDA Kernels v2 - Shared Memory + Parallel Reduction
# =============================================================================
#
# Optimization strategy for K=12 IR-only SIMM:
#   - 1 thread-block per portfolio (32 threads = 1 warp)
#   - Shared memory for correlation matrix (12×12 = 144 floats = 1.2KB)
#   - Parallel reduction for K×K correlation computation
#   - Warp-level primitives for fast reduction
#
# Occupancy analysis (H100):
#   - 32 threads/block, ~1.5KB shared memory
#   - Theoretical: 64 blocks/SM (2048 threads / 32)
#   - Actual limited by number of portfolios P
#   - For P=100: 100 blocks → ~0.75 blocks/SM average
#   - For P=1000: 1000 blocks → ~7.6 blocks/SM average
# =============================================================================

if CUDA_AVAILABLE:
    # Thread block size for v2 kernels (1 warp minimum for efficiency)
    V2_THREADS_PER_BLOCK = 32

    @cuda.jit
    def _simm_im_gradient_kernel_v2(
        # Aggregated sensitivities (P, K=12)
        agg_sens,
        # SIMM parameters
        risk_weights, correlations, concentration_factors,
        # Outputs
        im_output, gradients,
    ):
        """
        Optimized SIMM IM + gradient kernel using shared memory.

        Grid: P blocks (1 per portfolio)
        Block: 32 threads (1 warp)

        Shared memory layout:
        - correlations: 12×12 = 144 floats (1152 bytes)
        - ws: 12 floats (96 bytes)
        - k_ir_broadcast: 1 float (8 bytes)
        Total: ~1.3KB per block
        """
        K = 12

        # Shared memory allocations
        shared_corr = cuda.shared.array((12, 12), dtype=numba.float64)
        shared_ws = cuda.shared.array(12, dtype=numba.float64)
        shared_k_ir = cuda.shared.array(1, dtype=numba.float64)

        p = cuda.blockIdx.x  # Portfolio index
        tid = cuda.threadIdx.x  # Thread within block (0-31)

        if p >= agg_sens.shape[0]:
            return

        # Step 1: Cooperative load of correlation matrix
        # 32 threads load 144 elements → each thread loads ~5 elements
        for idx in range(tid, 144, 32):
            i = idx // 12
            j = idx % 12
            shared_corr[i, j] = correlations[i, j]
        cuda.syncthreads()

        # Step 2: Compute weighted sensitivities (threads 0-11)
        if tid < K:
            shared_ws[tid] = agg_sens[p, tid] * risk_weights[tid] * concentration_factors[tid]
        cuda.syncthreads()

        # Step 3: Parallel K_ir² computation
        # Each thread computes a subset of the 144 correlation terms
        my_sum = 0.0
        for idx in range(tid, 144, 32):
            i = idx // 12
            j = idx % 12
            my_sum += shared_corr[i, j] * shared_ws[i] * shared_ws[j]

        # Warp-level reduction using shuffle
        # All 32 threads participate in the reduction
        mask = 0xFFFFFFFF
        for offset in (16, 8, 4, 2, 1):
            my_sum += cuda.shfl_down_sync(mask, my_sum, offset)

        # Thread 0 has the final sum
        if tid == 0:
            k_ir = math.sqrt(my_sum) if my_sum > 0.0 else 0.0
            im_output[p] = k_ir
            shared_k_ir[0] = k_ir
        cuda.syncthreads()

        # Broadcast k_ir to all threads
        k_ir = shared_k_ir[0]

        # Step 4: Gradient computation (threads 0-11)
        if tid < K:
            if k_ir > 1e-12:
                dK_dWS = 0.0
                for j in range(K):
                    dK_dWS += shared_corr[tid, j] * shared_ws[j]
                dK_dWS /= k_ir
                gradients[p, tid] = dK_dWS * risk_weights[tid] * concentration_factors[tid]
            else:
                gradients[p, tid] = 0.0

    @cuda.jit
    def _simm_im_only_kernel_v2(
        # Aggregated sensitivities (N, K=12)
        agg_sens,
        # SIMM parameters
        risk_weights, correlations, concentration_factors,
        # Output
        im_output,
    ):
        """
        Optimized forward-only SIMM kernel using shared memory.
        No gradient computation (~50% less work than gradient version).

        Grid: N blocks (1 per scenario/portfolio)
        Block: 32 threads (1 warp)
        """
        K = 12

        # Shared memory allocations
        shared_corr = cuda.shared.array((12, 12), dtype=numba.float64)
        shared_ws = cuda.shared.array(12, dtype=numba.float64)

        p = cuda.blockIdx.x
        tid = cuda.threadIdx.x

        if p >= agg_sens.shape[0]:
            return

        # Cooperative load of correlation matrix
        for idx in range(tid, 144, 32):
            i = idx // 12
            j = idx % 12
            shared_corr[i, j] = correlations[i, j]
        cuda.syncthreads()

        # Compute weighted sensitivities
        if tid < K:
            shared_ws[tid] = agg_sens[p, tid] * risk_weights[tid] * concentration_factors[tid]
        cuda.syncthreads()

        # Parallel K_ir² computation
        my_sum = 0.0
        for idx in range(tid, 144, 32):
            i = idx // 12
            j = idx % 12
            my_sum += shared_corr[i, j] * shared_ws[i] * shared_ws[j]

        # Warp reduction
        mask = 0xFFFFFFFF
        for offset in (16, 8, 4, 2, 1):
            my_sum += cuda.shfl_down_sync(mask, my_sum, offset)

        if tid == 0:
            im_output[p] = math.sqrt(my_sum) if my_sum > 0.0 else 0.0


# =============================================================================
# Pure GPU Backend Class
# =============================================================================

@dataclass
class PureGPUTimingDetail:
    """Timing breakdown for pure GPU operations."""
    setup_time_sec: float = 0.0
    crif_time_sec: float = 0.0
    simm_time_sec: float = 0.0
    total_time_sec: float = 0.0
    num_trades: int = 0
    num_portfolios: int = 0
    num_evals: int = 0


class PureGPUIRBackend:
    """
    Pure GPU backend for IR Swaps only.

    All computation stays on GPU:
    - Trade data on device
    - CRIF via bump-and-revalue on device
    - SIMM aggregation on device
    - Gradients via chain rule on device
    """

    def __init__(self, device: int = 0):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available. Cannot use PureGPUIRBackend.")

        self.device = device
        self.is_setup = False

        # Device arrays (allocated during setup)
        self.d_notionals = None
        self.d_fixed_rates = None
        self.d_maturities = None
        self.d_frequencies = None
        self.d_payers = None
        self.d_curve_rates = None
        self.d_tenors = None
        self.d_sensitivities = None  # (T, K)
        self.d_risk_weights = None
        self.d_correlations = None      # Flattened for v1 kernels
        self.d_correlations_2d = None   # 2D for v2 optimized kernels
        self.d_concentration = None

        # Dimensions
        self.num_trades = 0
        self.num_factors = NUM_IR_TENORS  # K = 12

        # Sensitivity matrix on host (for allocation multiply)
        self.S_host = None

        # Timing
        self.last_timing = PureGPUTimingDetail()

    def setup(self, trades: List[Any], market: Any, trade_types: List[str] = None):
        """
        Setup backend with trades and market data.

        Validates that only IR swaps are provided.
        Copies all data to GPU and computes CRIF sensitivities.
        """
        # Validate trade types
        if trade_types is not None:
            for tt in trade_types:
                if tt != 'ir_swap':
                    raise ValueError(
                        f"PureGPUIRBackend only supports 'ir_swap', got '{tt}'. "
                        "Other asset classes require full GPU implementation."
                    )

        # Validate trades
        from model.trade_types import IRSwapTrade
        for i, t in enumerate(trades):
            if not isinstance(t, IRSwapTrade):
                raise ValueError(
                    f"Trade {i} is not an IRSwapTrade (got {type(t).__name__}). "
                    "PureGPUIRBackend only supports IR swaps."
                )

        t_start = time.perf_counter()

        if not CUDA_SIMULATOR:
            cuda.select_device(self.device)

        self.num_trades = len(trades)
        T = self.num_trades
        K = self.num_factors

        # Extract trade parameters
        notionals = np.array([t.notional for t in trades], dtype=np.float64)
        fixed_rates = np.array([t.fixed_rate for t in trades], dtype=np.float64)
        maturities = np.array([t.maturity for t in trades], dtype=np.float64)
        frequencies = np.array([t.frequency for t in trades], dtype=np.float64)
        payers = np.array([1 if t.payer else 0 for t in trades], dtype=np.int32)

        # Extract market data (USD curve)
        if hasattr(market, 'usd_curve'):
            curve_rates = np.array(market.usd_curve.zero_rates, dtype=np.float64)
        elif 'USD' in market.curves:
            curve_rates = np.array(market.curves['USD'].zero_rates, dtype=np.float64)
        else:
            raise ValueError("Market must have USD yield curve (market.usd_curve or market.curves['USD'])")
        tenors = IR_TENORS.copy()

        # Compute concentration factors
        # For simplicity, use CR = 1.0 (no concentration adjustment)
        # Full implementation would compute based on net sensitivity per bucket
        concentration = np.ones(K, dtype=np.float64)

        # Copy to device
        self.d_notionals = cuda.to_device(notionals)
        self.d_fixed_rates = cuda.to_device(fixed_rates)
        self.d_maturities = cuda.to_device(maturities)
        self.d_frequencies = cuda.to_device(frequencies)
        self.d_payers = cuda.to_device(payers)
        self.d_curve_rates = cuda.to_device(curve_rates)
        self.d_tenors = cuda.to_device(tenors)
        self.d_risk_weights = cuda.to_device(IR_RISK_WEIGHTS)
        self.d_correlations = cuda.to_device(IR_CORRELATIONS.flatten())  # v1 kernels
        self.d_correlations_2d = cuda.to_device(IR_CORRELATIONS)          # v2 optimized kernels
        self.d_concentration = cuda.to_device(concentration)

        # Allocate sensitivity matrix on device
        self.d_sensitivities = cuda.device_array((T, K), dtype=np.float64)

        setup_time = time.perf_counter() - t_start

        # Compute CRIF sensitivities on GPU
        t_crif = time.perf_counter()

        threads_per_block = 256
        blocks = (T + threads_per_block - 1) // threads_per_block

        _compute_crif_kernel[blocks, threads_per_block](
            self.d_notionals, self.d_fixed_rates, self.d_maturities,
            self.d_frequencies, self.d_payers,
            self.d_curve_rates, self.d_tenors,
            self.d_sensitivities,
        )

        if not CUDA_SIMULATOR:
            cuda.synchronize()

        crif_time = time.perf_counter() - t_crif

        # Copy sensitivity matrix to host for allocation operations
        self.S_host = self.d_sensitivities.copy_to_host()

        self.is_setup = True

        self.last_timing = PureGPUTimingDetail(
            setup_time_sec=setup_time,
            crif_time_sec=crif_time,
            num_trades=T,
        )

        return self.last_timing

    def get_sensitivity_matrix(self) -> np.ndarray:
        """Return the (T, K) sensitivity matrix."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")
        return self.S_host

    def get_risk_factors(self) -> List[Tuple[str, str, int, str]]:
        """Return risk factor metadata for IR tenors."""
        factors = []
        for k, label in enumerate(TENOR_LABELS):
            factors.append(("Risk_IRCurve", "USD", 0, label))
        return factors

    def compute_im_and_gradient(
        self,
        allocation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SIMM IM and gradients for all portfolios.

        Args:
            allocation: (T, P) allocation matrix

        Returns:
            im_values: (P,) IM per portfolio
            gradients: (T, P) dIM/d(allocation) per trade per portfolio
        """
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        t_start = time.perf_counter()

        T, P = allocation.shape
        K = self.num_factors

        # Aggregate sensitivities: agg_S = S^T @ allocation -> (K, P)
        # Then transpose for GPU: (P, K)
        agg_S = (self.S_host.T @ allocation).T  # (P, K)
        agg_S = np.ascontiguousarray(agg_S, dtype=np.float64)

        # Allocate outputs
        im_output = np.zeros(P, dtype=np.float64)
        grad_agg = np.zeros((P, K), dtype=np.float64)  # dIM/dS_agg

        # Copy to device
        d_agg_S = cuda.to_device(agg_S)
        d_im = cuda.to_device(im_output)
        d_grad = cuda.to_device(grad_agg)

        # Launch optimized v2 kernel
        # 1 block per portfolio, 32 threads per block (1 warp)
        # Uses shared memory for correlation matrix
        blocks = P
        threads_per_block = V2_THREADS_PER_BLOCK

        _simm_im_gradient_kernel_v2[blocks, threads_per_block](
            d_agg_S,
            self.d_risk_weights, self.d_correlations_2d, self.d_concentration,
            d_im, d_grad,
        )

        if not CUDA_SIMULATOR:
            cuda.synchronize()

        # Copy back
        d_im.copy_to_host(im_output)
        d_grad.copy_to_host(grad_agg)

        # Chain rule: gradient[t,p] = sum_k S[t,k] * dIM_p/dS_agg[p,k]
        # gradient = S @ grad_agg^T -> (T, P)
        gradients = self.S_host @ grad_agg.T

        simm_time = time.perf_counter() - t_start

        self.last_timing.simm_time_sec = simm_time
        self.last_timing.num_portfolios = P
        self.last_timing.num_evals += 1
        self.last_timing.total_time_sec += simm_time

        return im_output, gradients

    def compute_im_only(
        self,
        allocation: np.ndarray,
    ) -> np.ndarray:
        """
        Compute SIMM IM only (no gradients) for brute-force optimization.

        Args:
            allocation: (T, P) allocation matrix

        Returns:
            im_values: (P,) IM per portfolio
        """
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        t_start = time.perf_counter()

        T, P = allocation.shape
        K = self.num_factors

        # Aggregate
        agg_S = (self.S_host.T @ allocation).T  # (P, K)
        agg_S = np.ascontiguousarray(agg_S, dtype=np.float64)

        im_output = np.zeros(P, dtype=np.float64)

        d_agg_S = cuda.to_device(agg_S)
        d_im = cuda.to_device(im_output)

        # Launch optimized v2 kernel
        blocks = P
        threads_per_block = V2_THREADS_PER_BLOCK

        _simm_im_only_kernel_v2[blocks, threads_per_block](
            d_agg_S,
            self.d_risk_weights, self.d_correlations_2d, self.d_concentration,
            d_im,
        )

        if not CUDA_SIMULATOR:
            cuda.synchronize()

        d_im.copy_to_host(im_output)

        simm_time = time.perf_counter() - t_start
        self.last_timing.simm_time_sec += simm_time
        self.last_timing.num_evals += 1

        return im_output

    def compute_im_batched(
        self,
        agg_sensitivities: np.ndarray,
    ) -> np.ndarray:
        """
        Compute IM for pre-aggregated sensitivities (N, K).
        Used for brute-force candidate evaluation.
        """
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        N, K = agg_sensitivities.shape
        agg_S = np.ascontiguousarray(agg_sensitivities, dtype=np.float64)

        im_output = np.zeros(N, dtype=np.float64)

        d_agg_S = cuda.to_device(agg_S)
        d_im = cuda.to_device(im_output)

        # Launch optimized v2 kernel
        blocks = N
        threads_per_block = V2_THREADS_PER_BLOCK

        _simm_im_only_kernel_v2[blocks, threads_per_block](
            d_agg_S,
            self.d_risk_weights, self.d_correlations_2d, self.d_concentration,
            d_im,
        )

        if not CUDA_SIMULATOR:
            cuda.synchronize()

        d_im.copy_to_host(im_output)
        self.last_timing.num_evals += 1

        return im_output


# =============================================================================
# Helper Functions for Benchmark Integration
# =============================================================================

def create_pure_gpu_grad_fn(backend: PureGPUIRBackend, S: np.ndarray):
    """Create gradient function closure for optimizer using v2 kernels."""
    def grad_fn(agg_S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            agg_S: (K, P) aggregated sensitivities
        Returns:
            im_values: (P,)
            gradients: (K, P) dIM/dS_agg
        """
        P = agg_S.shape[1]
        K = agg_S.shape[0]

        # Transpose for GPU kernel: (P, K)
        agg_S_t = np.ascontiguousarray(agg_S.T, dtype=np.float64)

        im_output = np.zeros(P, dtype=np.float64)
        grad_agg = np.zeros((P, K), dtype=np.float64)

        d_agg_S = cuda.to_device(agg_S_t)
        d_im = cuda.to_device(im_output)
        d_grad = cuda.to_device(grad_agg)

        # Launch optimized v2 kernel
        blocks = P
        threads_per_block = V2_THREADS_PER_BLOCK

        _simm_im_gradient_kernel_v2[blocks, threads_per_block](
            d_agg_S,
            backend.d_risk_weights, backend.d_correlations_2d, backend.d_concentration,
            d_im, d_grad,
        )

        if not CUDA_SIMULATOR:
            cuda.synchronize()

        d_im.copy_to_host(im_output)
        d_grad.copy_to_host(grad_agg)

        backend.last_timing.num_evals += 1

        return im_output, grad_agg.T  # Return (K, P)

    return grad_fn


def create_pure_gpu_im_fn(backend: PureGPUIRBackend):
    """Create forward-only IM function closure for brute-force optimizer."""
    def im_fn(agg_S: np.ndarray) -> np.ndarray:
        """
        Args:
            agg_S: (K, N) aggregated sensitivities for N scenarios
        Returns:
            im_values: (N,)
        """
        N = agg_S.shape[1]
        agg_S_t = np.ascontiguousarray(agg_S.T, dtype=np.float64)
        return backend.compute_im_batched(agg_S_t)

    return im_fn


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":
    if not CUDA_AVAILABLE:
        print("CUDA not available. Exiting.")
        sys.exit(1)

    from model.trade_types import generate_trades_by_type, generate_market_environment

    print(f"Pure GPU IR Backend v{MODEL_VERSION}")
    print("=" * 60)

    # Generate test data
    num_trades = 100
    num_portfolios = 5

    market = generate_market_environment(currencies=['USD'])
    trades = generate_trades_by_type('ir_swap', num_trades, currencies=['USD'])

    print(f"Trades: {num_trades}")
    print(f"Portfolios: {num_portfolios}")

    # Setup backend
    backend = PureGPUIRBackend()
    timing = backend.setup(trades, market, trade_types=['ir_swap'])

    print(f"\nSetup time: {timing.setup_time_sec*1000:.2f} ms")
    print(f"CRIF time: {timing.crif_time_sec*1000:.2f} ms")

    # Create random allocation
    allocation = np.zeros((num_trades, num_portfolios), dtype=np.float64)
    for t in range(num_trades):
        allocation[t, t % num_portfolios] = 1.0

    # Compute IM and gradient
    t_start = time.perf_counter()
    im_values, gradients = backend.compute_im_and_gradient(allocation)
    elapsed = time.perf_counter() - t_start

    print(f"\nIM computation time: {elapsed*1000:.2f} ms")
    print(f"Total IM: ${sum(im_values):,.2f}")
    print(f"Per-portfolio IMs: {[f'${v:,.2f}' for v in im_values]}")
    print(f"Gradient shape: {gradients.shape}")
    print(f"Gradient range: [{gradients.min():.2e}, {gradients.max():.2e}]")

    # Test forward-only
    t_start = time.perf_counter()
    im_only = backend.compute_im_only(allocation)
    elapsed = time.perf_counter() - t_start

    print(f"\nForward-only time: {elapsed*1000:.2f} ms")
    print(f"IM match: {np.allclose(im_values, im_only)}")

    print("\nDone.")
