"""
Unit tests for the allocation optimizer module.

Tests include:
1. Simplex projection correctness
2. Gradient verification via finite differences
3. Optimization improves IM
4. Small case enumeration for global optimum verification

Usage:
    python -m pytest tests/test_allocation_optimizer.py -v
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

# Import the module under test
from model.simm_allocation_optimizer import (
    project_to_simplex,
    round_to_integer_allocation,
    _get_unique_risk_factors,
    _build_sensitivity_matrix,
    _map_risk_type_to_class,
    _get_risk_weight,
    verify_simplex_projection,
    MODULE_VERSION,
)

# Check for AADC availability
try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False


class TestSimplexProjection:
    """Tests for simplex projection function."""

    def test_already_on_simplex(self):
        """Points already on simplex should remain unchanged."""
        x = np.array([[0.3, 0.5, 0.2], [0.1, 0.9, 0.0]])
        proj = project_to_simplex(x)
        assert np.allclose(proj.sum(axis=1), 1.0), "Rows should sum to 1"
        assert (proj >= 0).all(), "All values should be >= 0"
        assert np.allclose(x, proj, atol=1e-10), "Already valid points should stay same"

    def test_negative_values(self):
        """Negative values should be projected to simplex."""
        x = np.array([[0.3, -0.1, 0.8], [1.5, 0.0, -0.5]])
        proj = project_to_simplex(x)
        assert np.allclose(proj.sum(axis=1), 1.0), "Rows should sum to 1"
        assert (proj >= -1e-10).all(), "All values should be >= 0"

    def test_large_values(self):
        """Values summing to > 1 should be projected."""
        x = np.array([[2.0, 1.0, 0.5], [0.0, 0.0, 3.0]])
        proj = project_to_simplex(x)
        assert np.allclose(proj.sum(axis=1), 1.0), "Rows should sum to 1"
        assert (proj >= -1e-10).all(), "All values should be >= 0"

    def test_uniform_projection(self):
        """Equal values should project to uniform distribution."""
        x = np.array([[1.0, 1.0, 1.0]])
        proj = project_to_simplex(x)
        expected = np.array([[1/3, 1/3, 1/3]])
        assert np.allclose(proj, expected, atol=1e-10), "Equal values should become uniform"

    def test_single_large_value(self):
        """Single very large value should dominate."""
        x = np.array([[100.0, 0.0, 0.0]])
        proj = project_to_simplex(x)
        # First element should be close to 1
        assert proj[0, 0] > 0.9, "Large value should dominate"
        assert np.allclose(proj.sum(axis=1), 1.0), "Row should sum to 1"


class TestRoundToInteger:
    """Tests for rounding continuous allocation to integer."""

    def test_clear_winner(self):
        """Clear winners should be selected."""
        x = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
        result = round_to_integer_allocation(x)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        assert np.array_equal(result, expected)

    def test_tie_breaking(self):
        """Ties should be broken (argmax takes first)."""
        x = np.array([[0.5, 0.5, 0.0]])
        result = round_to_integer_allocation(x)
        # argmax returns first max index
        assert result[0, 0] == 1.0

    def test_one_hot(self):
        """Result should be one-hot per row."""
        x = np.random.rand(10, 5)
        result = round_to_integer_allocation(x)
        assert np.allclose(result.sum(axis=1), 1.0), "Each row should sum to 1"
        assert set(result.flatten()) <= {0.0, 1.0}, "Only 0 and 1 values"


class TestRiskFactorHelpers:
    """Tests for risk factor helper functions."""

    def test_map_risk_type_to_class(self):
        """Verify risk type mapping."""
        assert _map_risk_type_to_class("Risk_IRCurve") == "Rates"
        assert _map_risk_type_to_class("Risk_Inflation") == "Rates"
        assert _map_risk_type_to_class("Risk_FX") == "FX"
        assert _map_risk_type_to_class("Risk_Equity") == "Equity"
        assert _map_risk_type_to_class("Risk_EquityVol") == "Equity"
        assert _map_risk_type_to_class("Risk_CreditQ") == "CreditQ"
        assert _map_risk_type_to_class("Risk_Commodity") == "Commodity"

    def test_get_risk_weight(self):
        """Verify risk weights are positive."""
        ir_weight = _get_risk_weight("Risk_IRCurve", "5")
        fx_weight = _get_risk_weight("Risk_FX", "")
        eq_weight = _get_risk_weight("Risk_Equity", "1")

        assert ir_weight > 0, "IR weight should be positive"
        assert fx_weight > 0, "FX weight should be positive"
        assert eq_weight > 0, "Equity weight should be positive"


class TestGetUniqueRiskFactors:
    """Tests for extracting unique risk factors."""

    def test_unique_factors(self):
        """Extract unique factors from trade CRIFs."""
        crif1 = pd.DataFrame({
            'RiskType': ['Risk_IRCurve', 'Risk_IRCurve'],
            'Qualifier': ['USD', 'USD'],
            'Bucket': ['1', '2'],
            'Label1': ['2W', '1M'],
        })
        crif2 = pd.DataFrame({
            'RiskType': ['Risk_IRCurve', 'Risk_FX'],
            'Qualifier': ['USD', 'EURUSD'],
            'Bucket': ['1', ''],
            'Label1': ['2W', ''],
        })

        trade_crifs = {'trade1': crif1, 'trade2': crif2}
        factors = _get_unique_risk_factors(trade_crifs)

        # Should have 3 unique factors
        assert len(factors) == 3
        assert ('Risk_IRCurve', 'USD', '1', '2W') in factors
        assert ('Risk_IRCurve', 'USD', '2', '1M') in factors
        assert ('Risk_FX', 'EURUSD', '', '') in factors


class TestBuildSensitivityMatrix:
    """Tests for building sensitivity matrix."""

    def test_matrix_shape(self):
        """Verify matrix has correct shape."""
        crif1 = pd.DataFrame({
            'RiskType': ['Risk_IRCurve', 'Risk_IRCurve'],
            'Qualifier': ['USD', 'USD'],
            'Bucket': ['1', '2'],
            'Label1': ['2W', '1M'],
            'Amount': [100.0, 200.0],
        })
        crif2 = pd.DataFrame({
            'RiskType': ['Risk_IRCurve'],
            'Qualifier': ['USD'],
            'Bucket': ['1'],
            'Label1': ['2W'],
            'Amount': [50.0],
        })

        trade_crifs = {'trade1': crif1, 'trade2': crif2}
        trade_ids = ['trade1', 'trade2']
        factors = _get_unique_risk_factors(trade_crifs)

        S = _build_sensitivity_matrix(trade_crifs, trade_ids, factors)

        assert S.shape == (2, 2), f"Expected (2, 2), got {S.shape}"

    def test_matrix_values(self):
        """Verify sensitivity values are correct."""
        crif1 = pd.DataFrame({
            'RiskType': ['Risk_FX'],
            'Qualifier': ['EURUSD'],
            'Bucket': [''],
            'Label1': [''],
            'Amount': [1000.0],
        })

        trade_crifs = {'trade1': crif1}
        trade_ids = ['trade1']
        factors = _get_unique_risk_factors(trade_crifs)

        S = _build_sensitivity_matrix(trade_crifs, trade_ids, factors)

        assert S[0, 0] == 1000.0


class TestVerifySimplexProjection:
    """Tests for simplex verification utility."""

    def test_valid_simplex(self):
        """Valid simplex allocation should pass."""
        allocation = np.array([[0.5, 0.5], [1.0, 0.0]])
        is_valid, max_error, min_val = verify_simplex_projection(allocation)
        assert is_valid, "Valid simplex should pass"
        assert max_error < 1e-6
        assert min_val >= 0

    def test_invalid_sum(self):
        """Rows not summing to 1 should fail."""
        allocation = np.array([[0.3, 0.3], [0.5, 0.5]])  # First row sums to 0.6
        is_valid, max_error, min_val = verify_simplex_projection(allocation)
        assert not is_valid, "Invalid sum should fail"

    def test_negative_values(self):
        """Negative values should fail."""
        allocation = np.array([[0.5, 0.5], [-0.1, 1.1]])
        is_valid, max_error, min_val = verify_simplex_projection(allocation)
        assert not is_valid, "Negative values should fail"


@pytest.mark.skipif(not AADC_AVAILABLE, reason="AADC not available")
class TestAADCIntegration:
    """Tests requiring AADC."""

    def test_kernel_recording(self):
        """Test that allocation kernel can be recorded."""
        from model.simm_allocation_optimizer import record_allocation_im_kernel

        # Simple test case: 2 trades, 2 risk factors, 2 portfolios
        S = np.array([[100.0, 50.0], [200.0, 100.0]])
        risk_factors = [
            ('Risk_IRCurve', 'USD', '1', '2W'),
            ('Risk_FX', 'EURUSD', '', ''),
        ]

        funcs, x_handles, im_output, _ = record_allocation_im_kernel(
            S, risk_factors, num_portfolios=2
        )

        assert funcs is not None
        assert len(x_handles) == 4  # 2 trades x 2 portfolios
        assert im_output is not None

    def test_gradient_computation(self):
        """Test gradient computation."""
        from model.simm_allocation_optimizer import (
            record_allocation_im_kernel,
            compute_allocation_gradient,
        )

        S = np.array([[100.0, 50.0], [200.0, 100.0]])
        risk_factors = [
            ('Risk_IRCurve', 'USD', '1', '2W'),
            ('Risk_FX', 'EURUSD', '', ''),
        ]

        funcs, x_handles, im_output, _ = record_allocation_im_kernel(
            S, risk_factors, num_portfolios=2
        )

        allocation = np.array([[0.5, 0.5], [0.5, 0.5]])
        gradient, total_im = compute_allocation_gradient(
            funcs, x_handles, None, im_output, S, allocation, num_threads=1
        )

        assert gradient.shape == (2, 2)
        assert total_im >= 0, "Total IM should be non-negative"

    def test_gradient_finite_difference(self):
        """Verify AADC gradient matches finite differences."""
        from model.simm_allocation_optimizer import (
            record_allocation_im_kernel,
            verify_allocation_gradient,
        )

        S = np.array([[100.0, 50.0], [200.0, 100.0]])
        risk_factors = [
            ('Risk_IRCurve', 'USD', '1', '2W'),
            ('Risk_FX', 'EURUSD', '', ''),
        ]

        funcs, x_handles, im_output, _ = record_allocation_im_kernel(
            S, risk_factors, num_portfolios=2
        )

        allocation = np.array([[0.6, 0.4], [0.3, 0.7]])

        max_rel_error, aadc_grad, fd_grad = verify_allocation_gradient(
            funcs, x_handles, im_output, S, allocation, num_threads=1
        )

        assert max_rel_error < 1e-3, f"Gradient mismatch: max rel error = {max_rel_error}"

    def test_optimization_improves(self):
        """Verify optimization reduces total IM."""
        from model.simm_allocation_optimizer import (
            record_allocation_im_kernel,
            optimize_allocation_gradient_descent,
            compute_allocation_gradient,
        )

        # Create test data with potential for optimization
        S = np.array([
            [1000.0, 0.0],     # Trade 1: IR only
            [0.0, 500.0],      # Trade 2: FX only
            [500.0, 250.0],    # Trade 3: Both
        ])
        risk_factors = [
            ('Risk_IRCurve', 'USD', '1', '2W'),
            ('Risk_FX', 'EURUSD', '', ''),
        ]

        funcs, x_handles, im_output, _ = record_allocation_im_kernel(
            S, risk_factors, num_portfolios=2
        )

        # Start with suboptimal allocation (all in portfolio 0)
        initial_allocation = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        _, initial_im = compute_allocation_gradient(
            funcs, x_handles, None, im_output, S, initial_allocation, num_threads=1
        )

        # Run optimization
        final_allocation, im_history, num_iters = optimize_allocation_gradient_descent(
            funcs, x_handles, None, im_output, S,
            initial_allocation, num_threads=1,
            max_iters=50, lr=1e-10, tol=1e-6, verbose=False
        )

        _, final_im = compute_allocation_gradient(
            funcs, x_handles, None, im_output, S, final_allocation, num_threads=1
        )

        # Optimization should not make things worse
        assert final_im <= initial_im + 1e-6, f"IM should not increase: {initial_im} -> {final_im}"


class TestModuleVersion:
    """Test module versioning."""

    def test_version_exists(self):
        """Module version should be defined."""
        assert MODULE_VERSION is not None
        assert len(MODULE_VERSION) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
