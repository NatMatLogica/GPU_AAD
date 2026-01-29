"""
Shared optimizer for fair SIMM benchmark.

Identical gradient descent loop for all backends — only the SIMM+gradient
computation (backend.compute_im_and_gradient) differs between runs.
"""

import numpy as np
import time
from typing import List, Tuple
from dataclasses import dataclass

from benchmark.backends.base import SIMMBackend


@dataclass
class OptimizationResult:
    """Results from allocation optimization."""
    final_allocation: np.ndarray   # (T, P) optimized allocation
    im_history: List[float]        # IM per iteration
    num_iterations: int
    initial_im: float
    final_im: float
    eval_time: float               # Total time for SIMM+gradient evaluations
    total_time: float              # Total optimization time (incl. numpy ops)
    trades_moved: int              # Number of trades reassigned vs initial


def project_to_simplex(x: np.ndarray) -> np.ndarray:
    """
    Project each row of x onto the probability simplex (sum=1, all>=0).

    Uses Duchi et al. (2008) algorithm.
    """
    T, P = x.shape
    result = np.zeros_like(x)

    for t in range(T):
        v = x[t]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_candidates = np.nonzero(u * np.arange(1, P + 1) > (cssv - 1))[0]
        if len(rho_candidates) == 0:
            result[t] = np.ones(P) / P
        else:
            rho = rho_candidates[-1]
            theta = (cssv[rho] - 1) / (rho + 1)
            result[t] = np.maximum(v - theta, 0)

    return result


def optimize_allocation(
    backend: SIMMBackend,
    S: np.ndarray,                    # (T, K) sensitivity matrix
    initial_allocation: np.ndarray,   # (T, P) allocation matrix
    max_iters: int = 100,
    lr: float = None,
    tol: float = 1e-6,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Gradient descent allocation optimizer — identical for all backends.

    Only backend.compute_im_and_gradient() differs between runs.

    Args:
        backend: SIMM backend to use for IM+gradient computation.
        S: (T, K) sensitivity matrix (fixed, precomputed).
        initial_allocation: (T, P) starting allocation.
        max_iters: Maximum gradient descent iterations.
        lr: Learning rate (auto-computed from gradient scale if None).
        tol: Convergence tolerance on relative IM change.
        verbose: Print progress.

    Returns:
        OptimizationResult with final allocation and timing.
    """
    T, P = initial_allocation.shape
    K = S.shape[1]

    x = initial_allocation.copy().astype(np.float64)
    im_history = []
    eval_time_total = 0.0

    total_start = time.perf_counter()

    # First evaluation: compute IM and gradient
    agg_S = S.T @ x  # (K, P)
    agg_S_T = agg_S.T  # (P, K)

    eval_start = time.perf_counter()
    im_values, grad_S = backend.compute_im_and_gradient(agg_S_T)
    eval_time_total += time.perf_counter() - eval_start

    total_im = float(np.sum(im_values))
    im_history.append(total_im)

    # Chain rule: dIM/dx[t,p] = Σ_k (dIM_p/dS_p[k]) * S[t,k]
    gradient = S @ grad_S.T  # (T, P)

    # Auto learning rate from gradient scale
    grad_max = np.abs(gradient).max()
    if lr is None:
        lr = 0.3 / grad_max if grad_max > 1e-10 else 1e-12

    if verbose:
        print(f"    Initial IM: ${total_im:,.2f}")
        print(f"    Gradient max: {grad_max:.2e}")
        print(f"    Learning rate: {lr:.2e}")

    for iteration in range(max_iters):
        if iteration > 0:
            agg_S = S.T @ x
            agg_S_T = agg_S.T

            eval_start = time.perf_counter()
            im_values, grad_S = backend.compute_im_and_gradient(agg_S_T)
            eval_time_total += time.perf_counter() - eval_start

            total_im = float(np.sum(im_values))
            im_history.append(total_im)

            gradient = S @ grad_S.T

        if verbose and iteration % 10 == 0:
            moves = int(np.sum(
                np.argmax(x, axis=1) != np.argmax(initial_allocation, axis=1)
            ))
            print(f"    Iter {iteration}: IM = ${total_im:,.2f}, moves = {moves}")

        # Gradient step + simplex projection
        x_new = project_to_simplex(x - lr * gradient)

        # Convergence check
        if iteration > 0:
            rel_change = abs(im_history[-1] - im_history[-2]) / max(abs(im_history[-2]), 1e-10)
            alloc_change = np.abs(x_new - x).max()
            if rel_change < tol and alloc_change < 1e-6:
                if verbose:
                    print(f"    Converged at iteration {iteration + 1}")
                x = x_new
                break

        x = x_new

    else:
        if verbose:
            print(f"    Reached max iterations ({max_iters})")

    total_time = time.perf_counter() - total_start

    # Count trades moved
    initial_assignments = np.argmax(initial_allocation, axis=1)
    final_assignments = np.argmax(x, axis=1)
    trades_moved = int(np.sum(initial_assignments != final_assignments))

    return OptimizationResult(
        final_allocation=x,
        im_history=im_history,
        num_iterations=len(im_history),
        initial_im=im_history[0],
        final_im=im_history[-1],
        eval_time=eval_time_total,
        total_time=total_time,
        trades_moved=trades_moved,
    )
