"""
Batched Line Search Optimizer for ISDA SIMM Portfolio Allocation.

Instead of evaluating one step size at a time (serial backtracking),
pre-computes ALL candidate step sizes and evaluates them in a SINGLE
kernel call by treating each (step_size, portfolio) pair as an
independent "portfolio" in the batch.

Current serial: 1 gradient eval + up to 10 serial LS evals = ~11 calls/iter
Batched:        1 gradient eval + 1 batched LS eval         =  2 calls/iter

Version: 1.0.0
"""

MODEL_VERSION = "1.0.0"

import time
import numpy as np
from typing import Callable, Tuple, Optional


def _project_to_simplex(x: np.ndarray) -> np.ndarray:
    """Project each row of x onto the probability simplex (vectorized)."""
    T, P = x.shape
    # Sort descending along columns (each row independently)
    u = np.sort(x, axis=1)[:, ::-1]                    # (T, P)
    cssv = np.cumsum(u, axis=1)                          # (T, P)
    arange = np.arange(1, P + 1)                         # (P,)
    mask = u * arange > (cssv - 1)                       # (T, P) bool
    # rho = last True index per row; if no True, use 0
    rho = P - 1 - np.argmax(mask[:, ::-1], axis=1)      # (T,)
    theta = (cssv[np.arange(T), rho] - 1.0) / (rho + 1) # (T,)
    result = np.maximum(x - theta[:, None], 0.0)
    return result


def _round_to_integer(x: np.ndarray) -> np.ndarray:
    """Round soft allocation to hard assignment (argmax per trade, vectorized)."""
    T, P = x.shape
    result = np.zeros_like(x)
    result[np.arange(T), np.argmax(x, axis=1)] = 1.0
    return result


def _batched_line_search(
    x: np.ndarray,
    direction: np.ndarray,
    S: np.ndarray,
    grad_fn: Callable,
    current_im: float,
    lr: float,
    ls_candidates: int,
    ls_beta: float,
) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Evaluate all candidate step sizes in a single batched kernel call.

    Args:
        x: Current allocation (T, P)
        direction: Descent direction (T, P)
        S: Sensitivity matrix (T, K)
        grad_fn: fn(agg_S: (N, K)) -> (im[N], grad[N, K])
        current_im: Current total IM to beat
        lr: Initial learning rate
        ls_candidates: Number of step sizes to try
        ls_beta: Step size reduction factor between candidates

    Returns:
        (best_x or None, best_im, num_portfolios_evaluated)
    """
    T, P = x.shape
    K = S.shape[1]

    # Pre-compute all candidate step sizes: lr, lr*beta, lr*beta^2, ...
    step_sizes = lr * (ls_beta ** np.arange(ls_candidates))

    # Pre-compute all candidate allocations
    # Broadcast: (ls_candidates, T, P) = (T, P) - (ls_candidates, 1, 1) * (T, P)
    x_shifted = x[None, :, :] - step_sizes[:, None, None] * direction[None, :, :]
    x_candidates = np.zeros_like(x_shifted)
    for i in range(ls_candidates):
        x_candidates[i] = _project_to_simplex(x_shifted[i])

    # Build batched aggregated sensitivities: (ls_candidates * P, K)
    # x_candidates is (ls_candidates, T, P), S is (T, K)
    # For each candidate i: agg[i] = x_candidates[i].T @ S → (P, K)
    # einsum: agg[i, p, k] = sum_t S[t,k] * x[i,t,p]
    agg_batch = np.einsum('itP,tK->iPK', x_candidates, S).reshape(ls_candidates * P, K)

    # Single kernel call for ALL candidates
    im_batch, _ = grad_fn(agg_batch)

    # Reshape and find best
    im_by_candidate = im_batch.reshape(ls_candidates, P)
    total_im_by_candidate = np.sum(im_by_candidate, axis=1)

    # Find best candidate that improves on current IM
    best_idx = np.argmin(total_im_by_candidate)
    best_im = float(total_im_by_candidate[best_idx])

    if best_im < current_im:
        return x_candidates[best_idx], best_im, ls_candidates * P
    else:
        return None, current_im, ls_candidates * P


def optimize_allocation_batched(
    S: np.ndarray,
    initial_allocation: np.ndarray,
    grad_fn: Callable,
    max_iters: int = 100,
    lr: Optional[float] = None,
    tol: float = 1e-6,
    method: str = "gradient_descent",
    ls_candidates: int = 10,
    ls_beta: float = 0.5,
    verbose: bool = False,
    label: str = "",
) -> dict:
    """
    Portfolio allocation optimizer with batched line search.

    Evaluates all line search candidates in a single kernel call,
    reducing kernel dispatch overhead from ~11 calls/iter to 2 calls/iter.

    Args:
        S: Sensitivity matrix (T, K)
        initial_allocation: Initial allocation (T, P), rows sum to 1
        grad_fn: fn(agg_S: (N, K)) -> (im[N], grad[N, K])
        max_iters: Maximum optimization iterations
        lr: Initial learning rate (auto-computed if None)
        tol: Convergence tolerance (relative IM change)
        method: "gradient_descent" or "adam"
        ls_candidates: Number of step sizes to evaluate in parallel
        ls_beta: Step size reduction factor between candidates
        verbose: Print progress
        label: Label for verbose output

    Returns:
        dict with final_allocation, final_im, initial_im, im_history,
        num_iterations, num_evals, num_ls_evals, eval_time, grad_time,
        trades_moved
    """
    T, P = initial_allocation.shape
    K = S.shape[1]

    x = initial_allocation.copy()
    im_history = []
    eval_start = time.perf_counter()
    total_grad_time = 0.0
    num_evals = 0        # Total kernel calls (counted as calls, not portfolios)
    num_ls_evals = 0     # Line search kernel calls only

    # Adam state
    use_adam = (method == "adam")
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # First gradient evaluation
    agg_S_T = np.dot(S.T, x).T  # (P, K)
    grad_start = time.perf_counter()
    im_values, grad_S = grad_fn(agg_S_T)
    total_grad_time += time.perf_counter() - grad_start
    num_evals += 1

    total_im = float(np.sum(im_values))
    im_history.append(total_im)
    gradient = np.dot(S, grad_S.T)  # (T, P)

    # Auto learning rate
    grad_max = np.abs(gradient).max()
    if lr is None:
        if grad_max > 1e-10:
            lr = 1.0 / grad_max if use_adam else 0.3 / grad_max
        else:
            lr = 1e-12

    best_im = total_im
    best_x = x.copy()
    stalled_count = 0

    for iteration in range(max_iters):
        # Refresh gradient (skip on iteration 0 — already computed above)
        if iteration > 0:
            agg_S_T = np.dot(S.T, x).T
            grad_start = time.perf_counter()
            im_values, grad_S = grad_fn(agg_S_T)
            total_grad_time += time.perf_counter() - grad_start
            num_evals += 1

            total_im = float(np.sum(im_values))
            im_history.append(total_im)
            gradient = np.dot(S, grad_S.T)

        if total_im < best_im:
            best_im = total_im
            best_x = x.copy()
            stalled_count = 0
        else:
            stalled_count += 1

        if stalled_count >= 20:
            x = best_x.copy()
            break

        # Compute direction
        if use_adam:
            t_step = iteration + 1
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** t_step)
            v_hat = v / (1 - beta2 ** t_step)
            direction = m_hat / (np.sqrt(v_hat) + eps)
        else:
            direction = gradient

        # Batched line search — single kernel call for all candidates
        grad_start = time.perf_counter()
        accepted_x, accepted_im, n_portfolios = _batched_line_search(
            x, direction, S, grad_fn,
            current_im=total_im,
            lr=lr,
            ls_candidates=ls_candidates,
            ls_beta=ls_beta,
        )
        total_grad_time += time.perf_counter() - grad_start
        num_evals += 1       # 1 batched kernel call
        num_ls_evals += 1

        if accepted_x is not None:
            x = accepted_x

        # Convergence check
        if iteration > 0 and len(im_history) >= 2:
            rel_change = abs(im_history[-1] - im_history[-2]) / max(abs(im_history[-2]), 1e-10)
            if rel_change < tol:
                break

        if verbose and iteration % 10 == 0:
            print(f"    [{label}] iter {iteration}: IM={total_im:,.0f}, "
                  f"lr={lr:.2e}, evals={num_evals}")

    eval_time = time.perf_counter() - eval_start

    # Final rounding and evaluation
    x_final = _round_to_integer(best_x)
    agg_final = np.dot(S.T, x_final).T
    im_final, _ = grad_fn(agg_final)
    num_evals += 1
    final_im = float(np.sum(im_final))

    return {
        'final_allocation': x_final,
        'final_im': final_im,
        'initial_im': im_history[0],
        'im_history': im_history,
        'num_iterations': min(iteration + 1, max_iters),
        'num_evals': num_evals,
        'num_ls_evals': num_ls_evals,
        'eval_time': eval_time,
        'grad_time': total_grad_time,
        'trades_moved': int(np.sum(
            np.argmax(x_final, axis=1) != np.argmax(initial_allocation, axis=1)
        )),
    }
