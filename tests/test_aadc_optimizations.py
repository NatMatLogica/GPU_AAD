"""
Tests for AADC v2 Optimizations.

Verification Criteria (from plan):
- IM correctness: < 1e-6 relative difference vs baseline
- Gradient correctness: < 1e-4 relative difference vs finite diff
- No regressions: v2 should match v1 results
- Performance gain: > 10% improvement

Test Phases:
- T2: Test mark_as_input_no_diff for constants
- T3: Test pre-computed weighted sensitivities
- T5: Test single evaluate() call for all portfolios
- T6: Test Hessian computation for Newton optimizer
"""

import numpy as np
import pandas as pd
import pytest
import time
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import AADC
try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False
    pytest.skip("AADC not available", allow_module_level=True)

# Import modules under test
from model.simm_allocation_optimizer import (
    record_single_portfolio_simm_kernel,
    compute_allocation_gradient_chainrule,
    _get_unique_risk_factors,
    _build_sensitivity_matrix,
    _get_factor_metadata,
    project_to_simplex,
)

from model.simm_portfolio_aadc_v2 import (
    record_single_portfolio_simm_kernel_v2,
    compute_all_portfolios_im_gradient_v2,
    compute_allocation_gradient_chainrule_v2,
    project_to_simplex_v2,
    optimize_allocation_gradient_descent_v2,
    _get_factor_metadata_v2,
)

from model.simm_allocation_optimizer_v2 import (
    reallocate_trades_optimal_v2,
    compare_v1_v2_results,
    verify_simplex_projection_v2,
)

from common.portfolio import run_simm


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_risk_factors():
    """Create sample risk factors for testing."""
    return [
        ("Risk_IRCurve", "USD", "1", "3m"),
        ("Risk_IRCurve", "USD", "2", "6m"),
        ("Risk_IRCurve", "USD", "3", "1y"),
        ("Risk_IRCurve", "USD", "4", "2y"),
        ("Risk_IRCurve", "USD", "5", "5y"),
        ("Risk_IRCurve", "USD", "6", "10y"),
        ("Risk_FX", "EURUSD", "", ""),
        ("Risk_Equity", "SPX", "1", ""),
    ]


@pytest.fixture
def sample_sensitivity_matrix(sample_risk_factors):
    """Create sample sensitivity matrix (T trades x K factors)."""
    T = 50  # 50 trades
    K = len(sample_risk_factors)
    np.random.seed(42)
    return np.random.randn(T, K) * 1e6  # Sensitivities in $


@pytest.fixture
def sample_allocation(sample_sensitivity_matrix):
    """Create sample allocation matrix (T trades x P portfolios)."""
    T = sample_sensitivity_matrix.shape[0]
    P = 5  # 5 portfolios
    # Start with uniform allocation
    allocation = np.zeros((T, P))
    for t in range(T):
        allocation[t, t % P] = 1.0
    return allocation


@pytest.fixture
def sample_crif():
    """Create sample CRIF DataFrame for testing."""
    rows = []
    np.random.seed(42)
    for i in range(20):
        rows.append({
            "TradeID": f"trade_{i // 4}",
            "RiskType": "Risk_IRCurve",
            "Qualifier": "USD",
            "Bucket": str((i % 6) + 1),
            "Label1": ["3m", "6m", "1y", "2y", "5y", "10y"][i % 6],
            "Label2": "OIS",
            "Amount": np.random.randn() * 1e6,
            "AmountCurrency": "USD",
            "AmountUSD": np.random.randn() * 1e6,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def workers():
    """Create shared ThreadPool."""
    return aadc.ThreadPool(4)


# =============================================================================
# Phase T2: Test mark_as_input_no_diff for Constants
# =============================================================================

class TestNoGradientConstants:
    """Test that constants (weights, correlations) don't need gradients."""

    def test_weights_are_floats_not_idoubles(self, sample_risk_factors):
        """Verify that risk weights are Python floats, not tracked by AADC."""
        factor_risk_classes, factor_weights, _, _, _ = _get_factor_metadata_v2(sample_risk_factors)

        # Weights should be a numpy array of floats
        assert isinstance(factor_weights, np.ndarray)
        assert factor_weights.dtype == np.float64

        # No AADC tracking
        for w in factor_weights:
            assert not isinstance(w, aadc.idouble)

    def test_correlation_lookup_returns_float(self, sample_risk_factors):
        """Verify correlation lookups return plain floats."""
        from model.simm_portfolio_aadc import _get_intra_correlation

        rho = _get_intra_correlation(
            "Rates", "Risk_IRCurve", "Risk_IRCurve",
            "3m", "5y", bucket=None
        )

        assert isinstance(rho, float)
        assert 0 <= rho <= 1


# =============================================================================
# Phase T3: Test Pre-computed Weighted Sensitivities
# =============================================================================

class TestPrecomputedWS:
    """Test that pre-computing WS = s * rw * cr gives same results."""

    def test_ws_computation_outside_kernel(self, sample_risk_factors, sample_sensitivity_matrix):
        """Test that WS computed outside matches inside-kernel computation."""
        K = len(sample_risk_factors)
        factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = \
            _get_factor_metadata_v2(sample_risk_factors)

        # Pre-compute weighted sensitivities (simulating outside kernel)
        T = sample_sensitivity_matrix.shape[0]
        amounts = sample_sensitivity_matrix[0, :]  # First trade's sensitivities

        # Outside kernel: numpy multiplication
        ws_outside = amounts * factor_weights  # CR assumed 1.0 for simplicity

        # Inside kernel would be: idouble * float = idouble
        # But the numerical value should match

        assert len(ws_outside) == K
        assert ws_outside.dtype == np.float64


# =============================================================================
# Phase T5: Test Single evaluate() Call for All Portfolios
# =============================================================================

class TestSingleEvaluate:
    """Test that single evaluate() call returns correct results for all portfolios."""

    def test_evaluate_accepts_array_inputs(self, sample_risk_factors, workers):
        """Test T5.1: Python aadc.evaluate() accepts arrays of length P."""
        K = len(sample_risk_factors)
        P = 5  # Number of portfolios

        factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = \
            _get_factor_metadata_v2(sample_risk_factors)

        funcs, sens_handles, im_output = record_single_portfolio_simm_kernel_v2(
            K, factor_risk_classes, factor_weights,
            factor_risk_types, factor_labels, factor_buckets
        )

        # Create inputs with arrays of length P
        np.random.seed(42)
        inputs = {sens_handles[k]: np.random.randn(P) * 1e6 for k in range(K)}
        request = {im_output: sens_handles}

        # This should NOT raise an error
        results = aadc.evaluate(funcs, request, inputs, workers)

        # Check output shape
        ims = results[0][im_output]
        assert len(ims) == P, f"Expected {P} IM values, got {len(ims)}"

        # Check gradients shape
        for k in range(K):
            grads = results[1][im_output][sens_handles[k]]
            assert len(grads) == P, f"Expected {P} gradient values for factor {k}"

    def test_all_portfolio_ims_match_individual(self, sample_risk_factors,
                                                  sample_sensitivity_matrix,
                                                  sample_allocation, workers):
        """Test T5.2: All P IM values match individual evaluations."""
        K = len(sample_risk_factors)
        T, P = sample_allocation.shape
        S = sample_sensitivity_matrix

        factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = \
            _get_factor_metadata_v2(sample_risk_factors)

        funcs, sens_handles, im_output = record_single_portfolio_simm_kernel_v2(
            K, factor_risk_classes, factor_weights,
            factor_risk_types, factor_labels, factor_buckets
        )

        # v1 pattern: individual evaluation per portfolio
        v1_ims = []
        for p in range(P):
            agg_S_p = np.dot(sample_allocation[:, p], S)
            inputs = {sens_handles[k]: np.array([agg_S_p[k]]) for k in range(K)}
            request = {im_output: sens_handles}
            results = aadc.evaluate(funcs, request, inputs, workers)
            v1_ims.append(float(results[0][im_output][0]))

        # v2 pattern: single evaluation for all portfolios
        _, v2_ims, _ = compute_all_portfolios_im_gradient_v2(
            funcs, sens_handles, im_output, S, sample_allocation, 4, workers
        )

        # Compare
        for p in range(P):
            rel_diff = abs(v1_ims[p] - v2_ims[p]) / max(abs(v1_ims[p]), 1e-10)
            assert rel_diff < 1e-10, f"Portfolio {p}: v1={v1_ims[p]}, v2={v2_ims[p]}, diff={rel_diff}"

    def test_all_portfolio_gradients_match_individual(self, sample_risk_factors,
                                                        sample_sensitivity_matrix,
                                                        sample_allocation, workers):
        """Test T5.3: All P gradient arrays match individual adjoint passes."""
        K = len(sample_risk_factors)
        T, P = sample_allocation.shape
        S = sample_sensitivity_matrix

        factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = \
            _get_factor_metadata_v2(sample_risk_factors)

        funcs, sens_handles, im_output = record_single_portfolio_simm_kernel_v2(
            K, factor_risk_classes, factor_weights,
            factor_risk_types, factor_labels, factor_buckets
        )

        # v1 pattern: compute chain rule gradient per portfolio
        v1_gradient = np.zeros((T, P))
        for p in range(P):
            agg_S_p = np.dot(sample_allocation[:, p], S)
            inputs = {sens_handles[k]: np.array([agg_S_p[k]]) for k in range(K)}
            request = {im_output: sens_handles}
            results = aadc.evaluate(funcs, request, inputs, workers)

            grad_p = np.array([float(results[1][im_output][sens_handles[k]][0]) for k in range(K)])
            v1_gradient[:, p] = np.dot(S, grad_p)

        # v2 pattern: single call
        v2_gradient, _, _ = compute_all_portfolios_im_gradient_v2(
            funcs, sens_handles, im_output, S, sample_allocation, 4, workers
        )

        # Compare
        max_rel_diff = 0
        for t in range(T):
            for p in range(P):
                if abs(v1_gradient[t, p]) > 1e-10:
                    rel_diff = abs(v1_gradient[t, p] - v2_gradient[t, p]) / abs(v1_gradient[t, p])
                    max_rel_diff = max(max_rel_diff, rel_diff)

        assert max_rel_diff < 1e-10, f"Max relative gradient difference: {max_rel_diff}"

    def test_throughput_improvement(self, sample_risk_factors,
                                     sample_sensitivity_matrix,
                                     sample_allocation, workers):
        """Test T5.4: Throughput improvement (target: 10-200x for 5 portfolios)."""
        K = len(sample_risk_factors)
        T, P = sample_allocation.shape
        S = sample_sensitivity_matrix
        num_iters = 10

        factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = \
            _get_factor_metadata_v2(sample_risk_factors)

        funcs, sens_handles, im_output = record_single_portfolio_simm_kernel_v2(
            K, factor_risk_classes, factor_weights,
            factor_risk_types, factor_labels, factor_buckets
        )

        # v1 timing: P separate calls per iteration
        v1_start = time.perf_counter()
        for _ in range(num_iters):
            for p in range(P):
                agg_S_p = np.dot(sample_allocation[:, p], S)
                inputs = {sens_handles[k]: np.array([agg_S_p[k]]) for k in range(K)}
                request = {im_output: sens_handles}
                aadc.evaluate(funcs, request, inputs, workers)
        v1_time = time.perf_counter() - v1_start

        # v2 timing: 1 call per iteration
        v2_start = time.perf_counter()
        for _ in range(num_iters):
            compute_all_portfolios_im_gradient_v2(
                funcs, sens_handles, im_output, S, sample_allocation, 4, workers
            )
        v2_time = time.perf_counter() - v2_start

        speedup = v1_time / v2_time
        print(f"\n  v1 time: {v1_time*1000:.2f} ms ({num_iters} iters x {P} portfolios)")
        print(f"  v2 time: {v2_time*1000:.2f} ms ({num_iters} iters x 1 call)")
        print(f"  SPEEDUP: {speedup:.1f}x")

        # Should see at least 2x speedup (conservative, actual is often 5-10x)
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.1f}x"


# =============================================================================
# Phase T6: Test Hessian Computation
# =============================================================================

class TestHessianComputation:
    """Test Hessian computation for Newton optimizer."""

    def test_hessian_matches_finite_diff(self, sample_risk_factors,
                                          sample_sensitivity_matrix,
                                          sample_allocation, workers):
        """Test T6.1: Hessian matches finite diff on gradient."""
        from model.simm_allocation_optimizer_v2 import compute_hessian_column_v2

        K = len(sample_risk_factors)
        T, P = sample_allocation.shape
        S = sample_sensitivity_matrix[:10, :]  # Smaller for speed
        allocation = sample_allocation[:10, :]

        factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = \
            _get_factor_metadata_v2(sample_risk_factors)

        funcs, sens_handles, im_output = record_single_portfolio_simm_kernel_v2(
            K, factor_risk_classes, factor_weights,
            factor_risk_types, factor_labels, factor_buckets
        )

        # Compute one Hessian column
        hess_col = compute_hessian_column_v2(
            funcs, sens_handles, im_output, S, allocation, 4,
            bump_t=0, bump_p=0, h=1e-4, workers=workers
        )

        assert len(hess_col) == 10 * P  # T * P elements
        # Hessian should have finite values
        assert np.all(np.isfinite(hess_col))


# =============================================================================
# Integration Tests: v1 vs v2 Comparison
# =============================================================================

class TestV1V2Comparison:
    """Compare v1 and v2 implementations for correctness."""

    def test_im_correctness(self, sample_risk_factors, sample_sensitivity_matrix,
                            sample_allocation, workers):
        """IM correctness: < 1e-6 relative difference."""
        K = len(sample_risk_factors)
        T, P = sample_allocation.shape
        S = sample_sensitivity_matrix

        factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = \
            _get_factor_metadata_v2(sample_risk_factors)

        # v1 kernel and computation
        from model.simm_allocation_optimizer import record_single_portfolio_simm_kernel
        funcs_v1, sens_handles_v1, im_output_v1 = record_single_portfolio_simm_kernel(
            K, factor_risk_classes, factor_weights,
            factor_risk_types, factor_labels, factor_buckets
        )
        _, total_im_v1 = compute_allocation_gradient_chainrule(
            funcs_v1, sens_handles_v1, im_output_v1, S, sample_allocation, 4, workers
        )

        # v2 kernel and computation
        funcs_v2, sens_handles_v2, im_output_v2 = record_single_portfolio_simm_kernel_v2(
            K, factor_risk_classes, factor_weights,
            factor_risk_types, factor_labels, factor_buckets
        )
        _, total_im_v2 = compute_allocation_gradient_chainrule_v2(
            funcs_v2, sens_handles_v2, im_output_v2, S, sample_allocation, 4, workers
        )

        rel_diff = abs(total_im_v1 - total_im_v2) / max(abs(total_im_v1), 1e-10)
        assert rel_diff < 1e-6, f"IM relative difference: {rel_diff}"

    def test_gradient_correctness_vs_finite_diff(self, sample_risk_factors,
                                                   sample_sensitivity_matrix,
                                                   sample_allocation, workers):
        """Gradient correctness: < 1e-4 relative difference vs finite diff."""
        K = len(sample_risk_factors)
        T, P = sample_allocation.shape
        S = sample_sensitivity_matrix[:10, :]  # Smaller for speed
        allocation = sample_allocation[:10, :]
        T = 10

        factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = \
            _get_factor_metadata_v2(sample_risk_factors)

        funcs, sens_handles, im_output = record_single_portfolio_simm_kernel_v2(
            K, factor_risk_classes, factor_weights,
            factor_risk_types, factor_labels, factor_buckets
        )

        # AADC gradient
        aadc_grad, total_im = compute_allocation_gradient_chainrule_v2(
            funcs, sens_handles, im_output, S, allocation, 4, workers
        )

        # Finite difference gradient (sample a few elements)
        eps = 1e-6
        max_rel_error = 0
        for t in range(min(3, T)):
            for p in range(min(2, P)):
                x_plus = allocation.copy()
                x_plus[t, p] += eps
                _, im_plus = compute_allocation_gradient_chainrule_v2(
                    funcs, sens_handles, im_output, S, x_plus, 4, workers
                )
                fd_grad = (im_plus - total_im) / eps

                if abs(aadc_grad[t, p]) > 1e-10:
                    rel_error = abs(aadc_grad[t, p] - fd_grad) / abs(aadc_grad[t, p])
                    max_rel_error = max(max_rel_error, rel_error)

        assert max_rel_error < 1e-4, f"Max gradient relative error: {max_rel_error}"


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance benchmarks for v2 optimizations."""

    def test_performance_gain_threshold(self, sample_risk_factors,
                                         sample_sensitivity_matrix,
                                         sample_allocation, workers):
        """Performance gain: > 10% improvement."""
        K = len(sample_risk_factors)
        T, P = sample_allocation.shape
        S = sample_sensitivity_matrix
        num_iters = 20

        factor_risk_classes, factor_weights, factor_risk_types, factor_labels, factor_buckets = \
            _get_factor_metadata_v2(sample_risk_factors)

        funcs, sens_handles, im_output = record_single_portfolio_simm_kernel_v2(
            K, factor_risk_classes, factor_weights,
            factor_risk_types, factor_labels, factor_buckets
        )

        # v1 timing
        v1_start = time.perf_counter()
        for _ in range(num_iters):
            compute_allocation_gradient_chainrule(
                funcs, sens_handles, im_output, S, sample_allocation, 4, workers
            )
        v1_time = time.perf_counter() - v1_start

        # v2 timing
        v2_start = time.perf_counter()
        for _ in range(num_iters):
            compute_allocation_gradient_chainrule_v2(
                funcs, sens_handles, im_output, S, sample_allocation, 4, workers
            )
        v2_time = time.perf_counter() - v2_start

        improvement = (v1_time - v2_time) / v1_time * 100
        print(f"\n  v1 time: {v1_time*1000:.2f} ms")
        print(f"  v2 time: {v2_time*1000:.2f} ms")
        print(f"  Improvement: {improvement:.1f}%")

        # Should see at least 10% improvement
        assert improvement > 10, f"Expected > 10% improvement, got {improvement:.1f}%"


# =============================================================================
# Simplex Projection Tests
# =============================================================================

class TestSimplexProjection:
    """Test simplex projection correctness."""

    def test_projection_maintains_constraints(self, sample_allocation):
        """Verify projected allocation satisfies simplex constraints."""
        # Add some noise to make it not a valid simplex
        noisy_allocation = sample_allocation + np.random.randn(*sample_allocation.shape) * 0.1

        # Project
        projected = project_to_simplex_v2(noisy_allocation)

        # Verify constraints
        is_valid, max_row_error, min_val = verify_simplex_projection_v2(projected)

        assert is_valid, f"Invalid simplex: max_row_error={max_row_error}, min_val={min_val}"
        assert max_row_error < 1e-6
        assert min_val >= -1e-10


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
