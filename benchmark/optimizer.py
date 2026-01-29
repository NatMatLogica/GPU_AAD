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
    best_x = x.copy()
    im_history = []
    eval_time_total = 0.0

    # Backtracking line search parameters
    LS_BETA = 0.5
    LS_MAX_TRIES = 10

    total_start = time.perf_counter()

    # First evaluation: compute IM and gradient
    agg_S = S.T @ x  # (K, P)
    agg_S_T = agg_S.T  # (P, K)

    eval_start = time.perf_counter()
    im_values, grad_S = backend.compute_im_and_gradient(agg_S_T)
    eval_time_total += time.perf_counter() - eval_start

    total_im = float(np.sum(im_values))
    im_history.append(total_im)
    best_im = total_im

    # Chain rule: dIM/dx[t,p] = Σ_k (dIM_p/dS_p[k]) * S[t,k]
    gradient = S @ grad_S.T  # (T, P)

    # Auto learning rate from gradient scale (conservative)
    grad_max = np.abs(gradient).max()
    if lr is None:
        lr = 1.0 / grad_max if grad_max > 1e-10 else 1e-12

    if verbose:
        print(f"    Initial IM: ${total_im:,.2f}")
        print(f"    Gradient max: {grad_max:.2e}")
        print(f"    Learning rate: {lr:.2e}")

    stalled_count = 0

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

        # Track best solution
        if total_im < best_im:
            best_im = total_im
            best_x = x.copy()
            stalled_count = 0
        else:
            stalled_count += 1

        if verbose and iteration % 10 == 0:
            moves = int(np.sum(
                np.argmax(x, axis=1) != np.argmax(initial_allocation, axis=1)
            ))
            print(f"    Iter {iteration}: IM = ${total_im:,.2f}, best = ${best_im:,.2f}, moves = {moves}")

        # Early exit if stalled
        if stalled_count >= 20:
            if verbose:
                print(f"    Stalled for {stalled_count} iterations, reverting to best")
            x = best_x.copy()
            break

        # Monotone backtracking line search
        step_size = lr
        accepted = False

        for _ in range(LS_MAX_TRIES):
            x_candidate = project_to_simplex(x - step_size * gradient)

            agg_S_c = S.T @ x_candidate
            eval_start = time.perf_counter()
            im_values_c, _ = backend.compute_im_and_gradient(agg_S_c.T)
            eval_time_total += time.perf_counter() - eval_start

            candidate_im = float(np.sum(im_values_c))

            if candidate_im < total_im:
                x = x_candidate
                accepted = True
                break

            step_size *= LS_BETA

        # Convergence check
        if iteration > 0 and len(im_history) >= 2:
            rel_change = abs(im_history[-1] - im_history[-2]) / max(abs(im_history[-2]), 1e-10)
            if rel_change < tol:
                if verbose:
                    print(f"    Converged at iteration {iteration + 1}")
                break

    else:
        if verbose:
            print(f"    Reached max iterations ({max_iters})")

    # Round to integer allocation
    rounded_x = np.zeros_like(best_x)
    for t in range(T):
        rounded_x[t, np.argmax(best_x[t])] = 1.0

    # Greedy local search on integer allocation
    if verbose:
        print(f"    Running greedy local search...")
    greedy_x, greedy_im, greedy_eval = _greedy_local_search(
        backend, S, rounded_x, max_rounds=max_iters, verbose=verbose,
    )
    eval_time_total += greedy_eval

    if greedy_im < best_im:
        best_x = greedy_x
        best_im = greedy_im

    total_time = time.perf_counter() - total_start

    # Count trades moved
    initial_assignments = np.argmax(initial_allocation, axis=1)
    final_assignments = np.argmax(best_x, axis=1)
    trades_moved = int(np.sum(initial_assignments != final_assignments))

    return OptimizationResult(
        final_allocation=best_x,
        im_history=im_history,
        num_iterations=len(im_history),
        initial_im=im_history[0],
        final_im=best_im,
        eval_time=eval_time_total,
        total_time=total_time,
        trades_moved=trades_moved,
    )


def _greedy_local_search(
    backend: SIMMBackend,
    S: np.ndarray,
    integer_allocation: np.ndarray,
    max_rounds: int = 50,
    verbose: bool = True,
) -> tuple:
    """
    Gradient-guided greedy local search for the benchmark optimizer.

    Uses gradient to identify promising single-trade moves, verifies each
    with actual IM evaluation.

    Returns:
        (improved_allocation, final_im, eval_time)
    """
    T, P = integer_allocation.shape
    x = integer_allocation.copy()
    eval_time_total = 0.0
    total_moves = 0

    # Initial evaluation
    agg_S_T = (S.T @ x).T
    eval_start = time.perf_counter()
    im_values, grad_S = backend.compute_im_and_gradient(agg_S_T)
    eval_time_total += time.perf_counter() - eval_start

    current_im = float(np.sum(im_values))
    gradient = S @ grad_S.T  # (T, P)

    for round_idx in range(max_rounds):
        current_assignments = np.argmax(x, axis=1)
        current_grads = gradient[np.arange(T), current_assignments]
        best_targets = np.argmin(gradient, axis=1)

        mask = best_targets != current_assignments
        if not np.any(mask):
            break

        candidate_indices = np.where(mask)[0]
        expected_improvement = current_grads - np.min(gradient, axis=1)
        sorted_candidates = candidate_indices[
            np.argsort(expected_improvement[candidate_indices])[::-1]
        ]

        accepted = False
        max_tries = min(len(sorted_candidates), T // 5 + 5)

        for try_idx in range(max_tries):
            t = sorted_candidates[try_idx]
            from_p = current_assignments[t]
            to_p = best_targets[t]

            x[t, :] = 0.0
            x[t, to_p] = 1.0

            agg_S_T_c = (S.T @ x).T
            eval_start = time.perf_counter()
            im_values_c, _ = backend.compute_im_and_gradient(agg_S_T_c)
            eval_time_total += time.perf_counter() - eval_start

            candidate_im = float(np.sum(im_values_c))

            if candidate_im < current_im:
                improvement = current_im - candidate_im
                current_im = candidate_im
                total_moves += 1
                accepted = True

                if verbose:
                    print(f"    Greedy round {round_idx+1}: move trade {t} "
                          f"(p{from_p}->p{to_p}), IM -${improvement:,.0f}")

                # Recompute gradient
                agg_S_T = (S.T @ x).T
                eval_start = time.perf_counter()
                im_values, grad_S = backend.compute_im_and_gradient(agg_S_T)
                eval_time_total += time.perf_counter() - eval_start
                gradient = S @ grad_S.T
                break
            else:
                x[t, :] = 0.0
                x[t, from_p] = 1.0

        if not accepted:
            break

    if verbose and total_moves > 0:
        print(f"    Greedy search: {total_moves} moves, final IM ${current_im:,.2f}")

    return x, current_im, eval_time_total
