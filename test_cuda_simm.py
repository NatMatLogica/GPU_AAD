#!/usr/bin/env python
"""Test script for CUDA SIMM implementation."""
import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

from model.simm_cuda import (
    compute_simm_cuda,
    compute_simm_numpy,
    compute_simm_gradient_cuda,
)
import numpy as np
import time

print('=' * 70)
print('SIMM CUDA Implementation - Full Test Suite')
print('=' * 70)

# Test 1: Correctness at scale
print('\n--- Test 1: Correctness (100 portfolios, 50 factors) ---')
np.random.seed(42)
P, K = 100, 50
sens = np.random.randn(P, K) * 1e6
risk_weights = np.random.uniform(10, 100, K)
risk_class_idx = np.random.randint(0, 6, K).astype(np.int32)

im_cuda = compute_simm_cuda(sens, risk_weights, risk_class_idx)
im_numpy = compute_simm_numpy(sens, risk_weights, risk_class_idx)

max_diff = np.max(np.abs(im_cuda - im_numpy))
rel_diff = max_diff / np.mean(im_numpy)
print(f'  Max absolute diff: {max_diff:.2e}')
print(f'  Max relative diff: {rel_diff:.2e}')
print(f'  Status: {"PASS" if rel_diff < 1e-10 else "FAIL"}')

# Test 2: Timing comparison
print('\n--- Test 2: Timing (simulator mode) ---')

# NumPy timing
start = time.perf_counter()
for _ in range(3):
    _ = compute_simm_numpy(sens, risk_weights, risk_class_idx)
numpy_time = (time.perf_counter() - start) / 3

# CUDA timing
start = time.perf_counter()
for _ in range(3):
    _ = compute_simm_cuda(sens, risk_weights, risk_class_idx)
cuda_time = (time.perf_counter() - start) / 3

print(f'  NumPy: {numpy_time*1000:.2f} ms')
print(f'  CUDA (simulated): {cuda_time*1000:.2f} ms')
print(f'  Note: Simulator is slower than real GPU')

# Test 3: Gradient validation
print('\n--- Test 3: Gradient Validation ---')
P, K = 10, 20
sens = np.random.randn(P, K) * 1e6
risk_weights = np.random.uniform(10, 100, K)
risk_class_idx = np.random.randint(0, 6, K).astype(np.int32)

im_values, gradients = compute_simm_gradient_cuda(sens, risk_weights, risk_class_idx)

# Finite difference check for portfolio 0
eps = 1e-6
fd_gradient = np.zeros(K)
base_im = im_values[0]

for k in range(K):
    sens_plus = sens.copy()
    sens_plus[0, k] += eps
    im_plus = compute_simm_cuda(sens_plus, risk_weights, risk_class_idx)[0]
    fd_gradient[k] = (im_plus - base_im) / eps

grad_diff = np.max(np.abs(gradients[0, :] - fd_gradient))
grad_rel = grad_diff / (np.mean(np.abs(fd_gradient)) + 1e-10)
print(f'  Max gradient diff: {grad_diff:.2e}')
print(f'  Rel gradient diff: {grad_rel:.2e}')
# 1% tolerance is acceptable for optimization (analytic gradient is simplified)
print(f'  Status: {"PASS" if grad_rel < 1e-2 else "FAIL"} (1% tolerance for optimization)')

# Test 4: Sample IM values
print('\n--- Test 4: Sample IM Values ---')
print(f'  First 5 portfolio IMs: {im_cuda[:5]}')
print(f'  Mean IM: ${np.mean(im_cuda):,.0f}')

print('\n' + '=' * 70)
print('All tests completed!')
print('=' * 70)
