"""
Optimal Trade Allocation to Minimize Total SIMM.

Uses AADC to:
1. Record allocation kernel: x[T,P] -> total_IM
2. Compute gradients dIM/dx[t,p] via adjoint pass
3. Optimize via gradient descent with simplex projection

The allocation kernel reuses the same ISDA SIMM logic from src/agg_margins.py
but wraps it in an AADC kernel where allocation fractions are the differentiable
inputs.

Mathematical Formulation:
    Input:
      - T trades with CRIF sensitivity vectors s_t (computed via AADC)
      - P portfolios
      - Allocation matrix X where x_tp = fraction of trade t in portfolio p

    Constraints:
      - sum_p x_tp = 1 for all t (each trade fully allocated)
      - x_tp in [0,1] for partial trades, or {0,1} for whole trades

    Portfolio sensitivities:
      - S_p = sum_t x_tp * s_t (aggregated CRIF for portfolio p)

    Portfolio IM (ISDA SIMM v2.6):
      - K_r = sqrt(sum_i (w_i * S_p[i])^2) for each risk class r
      - IM_p = sqrt(sum_r sum_s psi_rs * K_r * K_s) with ISDA correlation matrix

    Objective:
      - Minimize sum_p IM_p (total IM across all portfolios)

Usage:
    from model.simm_allocation_optimizer import reallocate_trades_optimal

    result = reallocate_trades_optimal(
        trades, market, num_portfolios=5,
        initial_allocation=current_allocation,
        allow_partial=False, method='gradient_descent'
    )

Version: 1.0.0
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional

# Version
MODULE_VERSION = "1.0.0"

try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False


# =============================================================================
# Helper Functions
# =============================================================================

def _get_unique_risk_factors(trade_crifs: Dict[str, pd.DataFrame]) -> List[Tuple]:
    """
    Extract unique risk factors across all trades.

    Returns list of (RiskType, Qualifier, Bucket, Label1) tuples that
    uniquely identify each sensitivity in the CRIF.
    """
    factors = set()
    for crif_df in trade_crifs.values():
        for _, row in crif_df.iterrows():
            key = (row['RiskType'], row['Qualifier'], str(row['Bucket']), row['Label1'])
            factors.add(key)
    return sorted(factors)


def _build_sensitivity_matrix(
    trade_crifs: Dict[str, pd.DataFrame],
    trade_ids: List[str],
    risk_factors: List[Tuple],
) -> np.ndarray:
    """
    Build T x K sensitivity matrix where S[t,k] = sensitivity of trade t to factor k.

    Args:
        trade_crifs: Dict mapping trade_id to CRIF DataFrame
        trade_ids: Ordered list of trade IDs
        risk_factors: Ordered list of (RiskType, Qualifier, Bucket, Label1) tuples

    Returns:
        S: numpy array of shape (T, K) where T = num trades, K = num risk factors
    """
    T = len(trade_ids)
    K = len(risk_factors)
    S = np.zeros((T, K))

    factor_to_idx = {f: i for i, f in enumerate(risk_factors)}
    trade_to_idx = {t: i for i, t in enumerate(trade_ids)}

    for trade_id, crif_df in trade_crifs.items():
        if trade_id not in trade_to_idx:
            continue
        t_idx = trade_to_idx[trade_id]
        for _, row in crif_df.iterrows():
            key = (row['RiskType'], row['Qualifier'], str(row['Bucket']), row['Label1'])
            if key in factor_to_idx:
                k_idx = factor_to_idx[key]
                S[t_idx, k_idx] += float(row['Amount'])

    return S


def _map_risk_type_to_class(risk_type: str) -> str:
    """Map CRIF RiskType to SIMM risk class."""
    if risk_type in ("Risk_IRCurve", "Risk_Inflation", "Risk_XCcyBasis",
                     "Risk_IRVol", "Risk_InflationVol"):
        return "Rates"
    elif risk_type in ("Risk_FX", "Risk_FXVol"):
        return "FX"
    elif risk_type in ("Risk_CreditQ", "Risk_CreditVol", "Risk_BaseCorr"):
        return "CreditQ"
    elif risk_type in ("Risk_CreditNonQ", "Risk_CreditVolNonQ"):
        return "CreditNonQ"
    elif risk_type in ("Risk_Equity", "Risk_EquityVol"):
        return "Equity"
    elif risk_type in ("Risk_Commodity", "Risk_CommodityVol"):
        return "Commodity"
    return "Rates"


# IR risk weights by bucket (ISDA SIMM v2.6)
_IR_RISK_WEIGHTS = np.array([
    77, 77, 68, 56, 52, 50, 51, 52, 50, 51, 51, 64
], dtype=float)
_FX_RISK_WEIGHT = 8.4
_EQUITY_RISK_WEIGHT = 25.0
_INFLATION_RISK_WEIGHT = 63.0


def _get_risk_weight(risk_type: str, bucket: str) -> float:
    """Get simplified risk weight for a CRIF entry."""
    if risk_type == "Risk_IRCurve":
        try:
            idx = int(bucket) - 1
            if 0 <= idx < len(_IR_RISK_WEIGHTS):
                return _IR_RISK_WEIGHTS[idx]
        except (ValueError, IndexError):
            pass
        return 50.0
    elif risk_type == "Risk_Inflation":
        return _INFLATION_RISK_WEIGHT
    elif risk_type == "Risk_FX":
        return _FX_RISK_WEIGHT
    elif risk_type == "Risk_FXVol":
        return _FX_RISK_WEIGHT * 0.55
    elif risk_type in ("Risk_Equity", "Risk_EquityVol"):
        return _EQUITY_RISK_WEIGHT
    return 50.0


# Cross-risk-class correlation matrix (psi) - ISDA SIMM v2.6/v2.7
# Order: Rates, CreditQ, CreditNonQ, Equity, Commodity, FX
_PSI_MATRIX = np.array([
    [1.00, 0.04, 0.04, 0.07, 0.37, 0.14],  # Rates
    [0.04, 1.00, 0.54, 0.70, 0.27, 0.37],  # CreditQ
    [0.04, 0.54, 1.00, 0.46, 0.24, 0.15],  # CreditNonQ
    [0.07, 0.70, 0.46, 1.00, 0.35, 0.39],  # Equity
    [0.37, 0.27, 0.24, 0.35, 1.00, 0.35],  # Commodity
    [0.14, 0.37, 0.15, 0.39, 0.35, 1.00],  # FX
])

_RISK_CLASS_ORDER = ['Rates', 'CreditQ', 'CreditNonQ', 'Equity', 'Commodity', 'FX']


# =============================================================================
# AADC Allocation Kernel
# =============================================================================

def record_allocation_im_kernel(
    S: np.ndarray,  # T x K sensitivity matrix
    risk_factors: List[Tuple],  # List of (RiskType, Qualifier, Bucket, Label1)
    num_portfolios: int,
) -> Tuple['aadc.Functions', Dict, Dict, int]:
    """
    Record AADC kernel: allocation fractions -> total IM.

    The kernel computes:
    1. For each portfolio p, aggregate sensitivities: S_p[k] = sum_t x[t,p] * S[t,k]
    2. Apply ISDA SIMM aggregation formula to get IM_p
    3. Sum portfolio IMs: total_im = sum_p IM_p

    Args:
        S: Sensitivity matrix of shape (T, K)
        risk_factors: List of (RiskType, Qualifier, Bucket, Label1) for each column of S
        num_portfolios: Number of portfolios P

    Returns:
        (funcs, x_handles, S_handles, im_output) where:
        - funcs: AADC Functions object
        - x_handles: Dict[(t,p) -> handle] for allocation fraction inputs
        - S_handles: Dict[(t,k) -> handle] for sensitivity inputs (non-diff)
        - im_output: Handle for total_im output
    """
    if not AADC_AVAILABLE:
        raise RuntimeError("AADC is required for allocation optimization")

    T, K = S.shape
    P = num_portfolios

    # Precompute risk class mapping and weights for each factor
    factor_risk_classes = []
    factor_weights = []
    for rt, qual, bucket, label1 in risk_factors:
        rc = _map_risk_type_to_class(rt)
        rw = _get_risk_weight(rt, bucket)
        factor_risk_classes.append(rc)
        factor_weights.append(rw)

    with aadc.record_kernel() as funcs:
        # Mark allocation fractions as differentiable inputs
        x_aadc = []  # T x P
        x_handles = {}
        for t in range(T):
            row = []
            for p in range(P):
                x_tp = aadc.idouble(1.0 / P)  # Initialize uniform
                handle = x_tp.mark_as_input()
                x_handles[(t, p)] = handle
                row.append(x_tp)
            x_aadc.append(row)

        # Mark sensitivities as non-differentiable (constants)
        S_handles = {}
        S_aadc = []
        for t in range(T):
            row = []
            for k in range(K):
                s_tk = aadc.idouble(S[t, k])
                handle = s_tk.mark_as_input_no_diff()
                S_handles[(t, k)] = handle
                row.append(s_tk)
            S_aadc.append(row)

        # Compute portfolio IMs
        portfolio_ims = []
        for p in range(P):
            # Aggregate sensitivities: S_p[k] = sum_t x[t,p] * S[t,k]
            agg_sens = []
            for k in range(K):
                s_pk = aadc.idouble(0.0)
                for t in range(T):
                    s_pk = s_pk + x_aadc[t][p] * S_aadc[t][k]
                agg_sens.append(s_pk)

            # Compute weighted sensitivities per risk class
            risk_class_margins = []

            for rc in _RISK_CLASS_ORDER:
                rc_indices = [k for k in range(K) if factor_risk_classes[k] == rc]
                if not rc_indices:
                    risk_class_margins.append(aadc.idouble(0.0))
                    continue

                ws_sq_sum = aadc.idouble(0.0)
                for k in rc_indices:
                    ws = agg_sens[k] * factor_weights[k]
                    ws_sq_sum = ws_sq_sum + ws * ws

                k_r = np.sqrt(ws_sq_sum)
                risk_class_margins.append(k_r)

            # Cross-risk-class aggregation: IM_p = sqrt(sum_r,s psi_rs * K_r * K_s)
            simm_sq = aadc.idouble(0.0)
            for i in range(6):
                for j in range(6):
                    psi_ij = _PSI_MATRIX[i, j]
                    simm_sq = simm_sq + psi_ij * risk_class_margins[i] * risk_class_margins[j]

            im_p = np.sqrt(simm_sq)
            portfolio_ims.append(im_p)

        # Total IM
        total_im = aadc.idouble(0.0)
        for p in range(P):
            total_im = total_im + portfolio_ims[p]

        im_output = total_im.mark_as_output()

    return funcs, x_handles, S_handles, im_output


# =============================================================================
# Gradient Computation
# =============================================================================

def compute_allocation_gradient(
    funcs: 'aadc.Functions',
    x_handles: Dict,
    S_handles: Dict,
    im_output: int,
    S: np.ndarray,
    current_allocation: np.ndarray,
    num_threads: int,
) -> Tuple[np.ndarray, float]:
    """
    Evaluate kernel and compute gradient dIM/dx[t,p].

    Args:
        funcs: AADC Functions object from record_allocation_im_kernel
        x_handles: Dict[(t,p) -> handle] for allocation inputs
        S_handles: Dict[(t,k) -> handle] for sensitivity inputs
        im_output: Handle for total_im output
        S: Sensitivity matrix of shape (T, K)
        current_allocation: Current allocation matrix of shape (T, P)
        num_threads: Number of AADC worker threads

    Returns:
        (gradient, total_im) where:
        - gradient: Array of shape (T, P) with dIM/dx[t,p]
        - total_im: Scalar total IM value
    """
    T, P = current_allocation.shape
    K = S.shape[1]

    workers = aadc.ThreadPool(num_threads)

    # Set inputs
    inputs = {}
    for t in range(T):
        for p in range(P):
            inputs[x_handles[(t, p)]] = np.array([current_allocation[t, p]])

    for t in range(T):
        for k in range(K):
            inputs[S_handles[(t, k)]] = np.array([S[t, k]])

    # Request gradient w.r.t. allocation fractions
    diff_handles = [x_handles[(t, p)] for t in range(T) for p in range(P)]
    request = {im_output: diff_handles}

    results = aadc.evaluate(funcs, request, inputs, workers)

    total_im = float(results[0][im_output][0])

    gradient = np.zeros((T, P))
    for t in range(T):
        for p in range(P):
            h = x_handles[(t, p)]
            gradient[t, p] = float(results[1][im_output][h][0])

    return gradient, total_im


# =============================================================================
# Simplex Projection
# =============================================================================

def project_to_simplex(x: np.ndarray) -> np.ndarray:
    """
    Project each row of x onto the probability simplex (sum=1, all>=0).

    Uses the algorithm from "Efficient Projections onto the l1-Ball
    for Learning in High Dimensions" by Duchi et al. (2008).

    Args:
        x: Array of shape (T, P)

    Returns:
        Projected array where each row sums to 1 and all values >= 0
    """
    T, P = x.shape
    result = np.zeros_like(x)

    for t in range(T):
        v = x[t]
        # Sort in descending order
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_candidates = np.nonzero(u * np.arange(1, P + 1) > (cssv - 1))[0]
        if len(rho_candidates) == 0:
            # Fallback: uniform distribution
            result[t] = np.ones(P) / P
        else:
            rho = rho_candidates[-1]
            theta = (cssv[rho] - 1) / (rho + 1)
            result[t] = np.maximum(v - theta, 0)

    return result


# =============================================================================
# Optimization Algorithms
# =============================================================================

def optimize_allocation_gradient_descent(
    funcs: 'aadc.Functions',
    x_handles: Dict,
    S_handles: Dict,
    im_output: int,
    S: np.ndarray,
    initial_allocation: np.ndarray,
    num_threads: int,
    max_iters: int = 100,
    lr: float = 1e-12,
    tol: float = 1e-6,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[float], int]:
    """
    Gradient descent with simplex projection.

    For each iteration:
    1. Compute gradient dIM/dx_tp
    2. Take gradient step: x_new = x - lr * grad
    3. Project each row to simplex: sum_p x_tp = 1, x_tp >= 0
    4. Check convergence

    Args:
        funcs: AADC Functions object
        x_handles, S_handles, im_output: From record_allocation_im_kernel
        S: Sensitivity matrix
        initial_allocation: Starting allocation of shape (T, P)
        num_threads: AADC worker threads
        max_iters: Maximum iterations
        lr: Learning rate (small since IM values are large)
        tol: Relative tolerance for convergence
        verbose: Print progress

    Returns:
        (optimal_allocation, im_history, num_iterations)
    """
    x = initial_allocation.copy()
    im_history = []

    for iteration in range(max_iters):
        gradient, total_im = compute_allocation_gradient(
            funcs, x_handles, S_handles, im_output, S, x, num_threads
        )
        im_history.append(total_im)

        if verbose and iteration % 10 == 0:
            print(f"    Iter {iteration}: IM = ${total_im:,.2f}")

        # Gradient step
        x_new = x - lr * gradient

        # Project to simplex (each row sums to 1, all >= 0)
        x_new = project_to_simplex(x_new)

        # Check convergence
        if iteration > 0:
            rel_change = abs(im_history[-1] - im_history[-2]) / max(abs(im_history[-2]), 1e-10)
            if rel_change < tol:
                if verbose:
                    print(f"    Converged at iteration {iteration + 1}")
                return x_new, im_history, iteration + 1

        x = x_new

    if verbose:
        print(f"    Reached max iterations ({max_iters})")
    return x, im_history, max_iters


def optimize_allocation_greedy(
    trade_crifs: Dict[str, pd.DataFrame],
    num_portfolios: int,
    num_threads: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    Greedy assignment: assign each trade to portfolio that minimizes total IM.

    This is a fast heuristic that may not find the global optimum but provides
    a good initial solution or standalone result for smaller problems.

    Algorithm:
    1. Start with all trades in portfolio 0
    2. For each trade, try moving it to each portfolio
    3. Keep the assignment that minimizes total IM
    4. Repeat until no improvement

    Args:
        trade_crifs: Dict mapping trade_id to CRIF DataFrame
        num_portfolios: Number of portfolios
        num_threads: AADC worker threads
        verbose: Print progress

    Returns:
        Allocation matrix of shape (T, P) with one-hot rows
    """
    from model.simm_portfolio_aadc import precompute_all_trade_crifs
    from common.portfolio import run_simm

    trade_ids = list(trade_crifs.keys())
    T = len(trade_ids)
    P = num_portfolios

    # Start with uniform distribution across portfolios
    assignment = np.zeros(T, dtype=int)
    for t in range(T):
        assignment[t] = t % P

    def compute_total_im(assign):
        """Compute total IM for a given assignment."""
        total = 0.0
        for p in range(P):
            p_trades = [trade_ids[t] for t in range(T) if assign[t] == p]
            if not p_trades:
                continue
            # Combine CRIFs
            p_crifs = [trade_crifs[tid] for tid in p_trades]
            p_crif = pd.concat(p_crifs, ignore_index=True)
            _, im, _ = run_simm(p_crif)
            total += im
        return total

    best_im = compute_total_im(assignment)
    if verbose:
        print(f"    Greedy initial IM: ${best_im:,.2f}")

    improved = True
    iteration = 0
    while improved and iteration < 100:
        improved = False
        iteration += 1

        for t in range(T):
            current_p = assignment[t]
            best_move_im = best_im

            for new_p in range(P):
                if new_p == current_p:
                    continue

                # Try moving trade t to portfolio new_p
                test_assign = assignment.copy()
                test_assign[t] = new_p
                test_im = compute_total_im(test_assign)

                if test_im < best_move_im - 1e-6:
                    best_move_im = test_im
                    best_new_p = new_p

            if best_move_im < best_im - 1e-6:
                assignment[t] = best_new_p
                best_im = best_move_im
                improved = True
                if verbose:
                    print(f"    Iter {iteration}: Moved trade {t} to portfolio {best_new_p}, IM = ${best_im:,.2f}")

    # Convert to allocation matrix
    allocation = np.zeros((T, P))
    for t in range(T):
        allocation[t, assignment[t]] = 1.0

    return allocation


def round_to_integer_allocation(continuous_allocation: np.ndarray) -> np.ndarray:
    """
    Round continuous allocation to integer (one-hot per row).

    Each trade is assigned to the portfolio with highest allocation fraction.

    Args:
        continuous_allocation: Array of shape (T, P) with values in [0, 1]

    Returns:
        Array of shape (T, P) with exactly one 1.0 per row
    """
    T, P = continuous_allocation.shape
    result = np.zeros_like(continuous_allocation)

    for t in range(T):
        best_p = np.argmax(continuous_allocation[t])
        result[t, best_p] = 1.0

    return result


# =============================================================================
# Main Entry Point
# =============================================================================

def reallocate_trades_optimal(
    trades: list,
    market,  # MarketEnvironment
    num_portfolios: int,
    initial_allocation: np.ndarray,
    num_threads: int = 8,
    allow_partial: bool = False,
    method: str = 'gradient_descent',
    max_iters: int = 100,
    lr: float = 1e-12,
    tol: float = 1e-6,
    verbose: bool = True,
) -> Dict:
    """
    Full optimization pipeline for trade allocation.

    Steps:
    1. Precompute CRIF for all T trades (AADC: T kernel recordings + evaluations)
    2. Record allocation kernel: x[T x P] -> total_IM (single kernel)
    3. Initialize: from provided initial_allocation
    4. Iterate gradient descent:
       a. Evaluate kernel -> total_IM, gradient dIM/dx[t,p]
       b. x_new = x - lr * gradient
       c. Project each row to simplex (sum_p x[t,p] = 1)
       d. Check convergence
    5. Round to integer (if allow_partial is False)
    6. Recompute final IM to verify

    Args:
        trades: List of trade objects
        market: MarketEnvironment with curves, spots, etc.
        num_portfolios: Number of portfolios P
        initial_allocation: Starting allocation matrix of shape (T, P)
        num_threads: AADC worker threads
        allow_partial: If False, round to integer allocation at the end
        method: 'gradient_descent' or 'greedy'
        max_iters: Maximum optimization iterations
        lr: Learning rate for gradient descent
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        Dict with:
        - final_allocation: np.ndarray (T x P)
        - final_im: float
        - initial_im: float
        - im_history: List[float]
        - num_iterations: int
        - elapsed_time: float
        - trades_moved: int
        - trade_ids: List[str]
        - converged: bool
    """
    if not AADC_AVAILABLE:
        raise RuntimeError("AADC is required for allocation optimization")

    from model.simm_portfolio_aadc import precompute_all_trade_crifs

    start_time = time.perf_counter()

    # Step 1: Precompute all trade CRIFs
    if verbose:
        print("  Precomputing trade CRIFs...")
    trade_crifs = precompute_all_trade_crifs(trades, market, num_threads)

    trade_ids = [t.trade_id for t in trades if t.trade_id in trade_crifs]
    T = len(trade_ids)
    P = num_portfolios

    if T == 0:
        return {
            'final_allocation': initial_allocation,
            'final_im': 0.0,
            'initial_im': 0.0,
            'im_history': [],
            'num_iterations': 0,
            'elapsed_time': time.perf_counter() - start_time,
            'trades_moved': 0,
            'trade_ids': [],
            'converged': True,
        }

    # Filter initial_allocation to match trade_ids
    trade_id_to_idx = {t.trade_id: i for i, t in enumerate(trades)}
    filtered_allocation = np.zeros((T, P))
    for i, tid in enumerate(trade_ids):
        orig_idx = trade_id_to_idx.get(tid)
        if orig_idx is not None and orig_idx < initial_allocation.shape[0]:
            filtered_allocation[i] = initial_allocation[orig_idx]

    # Step 2: Build sensitivity matrix
    risk_factors = _get_unique_risk_factors(trade_crifs)
    S = _build_sensitivity_matrix(trade_crifs, trade_ids, risk_factors)
    K = len(risk_factors)

    if verbose:
        print(f"  Sensitivity matrix: {T} trades x {K} risk factors x {P} portfolios")

    if method == 'greedy':
        # Use greedy algorithm (doesn't need AADC kernel)
        if verbose:
            print("  Running greedy optimization...")
        final_allocation = optimize_allocation_greedy(
            trade_crifs, P, num_threads, verbose
        )
        im_history = []
        num_iters = 0

        # Compute final IM
        from common.portfolio import run_simm
        final_im = 0.0
        initial_im = 0.0
        for p in range(P):
            p_trade_ids = [trade_ids[t] for t in range(T) if filtered_allocation[t, p] > 0.5]
            if p_trade_ids:
                p_crifs = [trade_crifs[tid] for tid in p_trade_ids]
                p_crif = pd.concat(p_crifs, ignore_index=True)
                _, im, _ = run_simm(p_crif)
                initial_im += im

            p_trade_ids_final = [trade_ids[t] for t in range(T) if final_allocation[t, p] > 0.5]
            if p_trade_ids_final:
                p_crifs = [trade_crifs[tid] for tid in p_trade_ids_final]
                p_crif = pd.concat(p_crifs, ignore_index=True)
                _, im, _ = run_simm(p_crif)
                final_im += im

    else:
        # Gradient descent method
        # Step 3: Record allocation kernel
        if verbose:
            print("  Recording allocation kernel...")
        funcs, x_handles, S_handles, im_output = record_allocation_im_kernel(S, risk_factors, P)

        # Step 4: Compute initial IM
        _, initial_im = compute_allocation_gradient(
            funcs, x_handles, S_handles, im_output, S, filtered_allocation, num_threads
        )
        if verbose:
            print(f"  Initial total IM: ${initial_im:,.2f}")

        # Step 5: Optimize
        if verbose:
            print(f"  Running {method} optimization...")
        final_allocation, im_history, num_iters = optimize_allocation_gradient_descent(
            funcs, x_handles, S_handles, im_output, S,
            filtered_allocation, num_threads,
            max_iters=max_iters, lr=lr, tol=tol, verbose=verbose,
        )

        # Step 6: Round if needed
        if not allow_partial:
            final_allocation = round_to_integer_allocation(final_allocation)

        # Step 7: Compute final IM
        _, final_im = compute_allocation_gradient(
            funcs, x_handles, S_handles, im_output, S, final_allocation, num_threads
        )

    # Count moves
    initial_assignments = np.argmax(filtered_allocation, axis=1)
    final_assignments = np.argmax(final_allocation, axis=1)
    trades_moved = int(np.sum(initial_assignments != final_assignments))

    elapsed = time.perf_counter() - start_time

    converged = num_iters < max_iters if method == 'gradient_descent' else True

    return {
        'final_allocation': final_allocation,
        'final_im': final_im,
        'initial_im': initial_im,
        'im_history': im_history,
        'num_iterations': num_iters,
        'elapsed_time': elapsed,
        'trades_moved': trades_moved,
        'trade_ids': trade_ids,
        'converged': converged,
    }


# =============================================================================
# Verification Utilities
# =============================================================================

def verify_allocation_gradient(
    funcs: 'aadc.Functions',
    x_handles: Dict,
    S_handles: Dict,
    im_output: int,
    S: np.ndarray,
    allocation: np.ndarray,
    num_threads: int,
    eps: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Verify AADC gradient via finite differences.

    Args:
        funcs, x_handles, S_handles, im_output: From record_allocation_im_kernel
        S: Sensitivity matrix
        allocation: Current allocation
        num_threads: AADC worker threads
        eps: Finite difference step size

    Returns:
        (max_rel_error, aadc_gradient, fd_gradient)
    """
    T, P = allocation.shape

    # Get AADC gradient
    aadc_grad, base_im = compute_allocation_gradient(
        funcs, x_handles, S_handles, im_output, S, allocation, num_threads
    )

    # Compute finite difference gradient
    fd_grad = np.zeros((T, P))
    for t in range(T):
        for p in range(P):
            x_plus = allocation.copy()
            x_plus[t, p] += eps

            _, im_plus = compute_allocation_gradient(
                funcs, x_handles, S_handles, im_output, S, x_plus, num_threads
            )
            fd_grad[t, p] = (im_plus - base_im) / eps

    # Compute max relative error
    denom = np.maximum(np.abs(aadc_grad), np.abs(fd_grad))
    denom = np.where(denom < 1e-10, 1.0, denom)
    rel_errors = np.abs(aadc_grad - fd_grad) / denom
    max_rel_error = np.max(rel_errors)

    return max_rel_error, aadc_grad, fd_grad


def verify_simplex_projection(allocation: np.ndarray) -> Tuple[bool, float, float]:
    """
    Verify allocation matrix satisfies simplex constraints.

    Args:
        allocation: Array of shape (T, P)

    Returns:
        (is_valid, max_row_sum_error, min_value)
    """
    row_sums = allocation.sum(axis=1)
    max_row_sum_error = np.max(np.abs(row_sums - 1.0))
    min_value = np.min(allocation)
    is_valid = max_row_sum_error < 1e-6 and min_value >= -1e-10

    return is_valid, max_row_sum_error, min_value
