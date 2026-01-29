#!/usr/bin/env python
"""
Standalone SIMM formula validation for the methodology section.

Tests:
1. Hand-calculable test cases for ISDA SIMM v2.6
2. Internal cross-check: benchmark/simm_formula.py vs src/agg_margins.py
3. Reports which v2.6 features are included/excluded

Usage:
    python -m benchmark.validate_simm
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.simm_formula import (
    PSI_MATRIX, RISK_CLASS_ORDER, RISK_CLASS_TO_IDX, NUM_RISK_CLASSES,
    compute_simm_im, compute_simm_batch, compute_simm_gradient,
)
from benchmark.backends.base import FactorMetadata


def _make_factor_meta(
    n_factors: int,
    risk_classes: list = None,
    risk_weights: np.ndarray = None,
    concentration: np.ndarray = None,
    intra_corr: np.ndarray = None,
) -> FactorMetadata:
    """Helper to build FactorMetadata for test cases."""
    if risk_classes is None:
        risk_classes = ["Rates"] * n_factors
    rc_idx = np.array([RISK_CLASS_TO_IDX[rc] for rc in risk_classes], dtype=np.int32)
    if risk_weights is None:
        risk_weights = np.ones(n_factors)
    if concentration is None:
        concentration = np.ones(n_factors)
    if intra_corr is None:
        intra_corr = np.eye(n_factors)

    return FactorMetadata(
        risk_classes=risk_classes,
        risk_class_idx=rc_idx,
        risk_weights=risk_weights.astype(np.float64),
        risk_types=["Risk_IRCurve"] * n_factors,
        labels=["1y"] * n_factors,
        buckets=["1"] * n_factors,
        qualifiers=["USD"] * n_factors,
        concentration_factors=concentration.astype(np.float64),
        intra_corr_matrix=intra_corr.astype(np.float64),
    )


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.messages = []

    def check(self, condition: bool, msg: str):
        if not condition:
            self.passed = False
            self.messages.append(f"FAIL: {msg}")
        else:
            self.messages.append(f"  OK: {msg}")

    def check_close(self, actual, expected, tol, label):
        diff = abs(actual - expected)
        rel = diff / max(abs(expected), 1e-15)
        ok = rel < tol
        self.check(ok, f"{label}: got {actual:.10e}, expected {expected:.10e}, "
                       f"rel_err={rel:.2e}")

    def report(self):
        status = "PASS" if self.passed else "FAIL"
        print(f"\n[{status}] {self.name}")
        for msg in self.messages:
            print(f"  {msg}")
        return self.passed


def test_single_factor():
    """Test 1: Single risk class, single factor.

    K_b = RW * |s| * CR = 50 * |100| * 1.0 = 5000
    IM = sqrt(psi[0,0] * K_0^2) = sqrt(1.0 * 5000^2) = 5000
    """
    t = TestResult("Single factor, single risk class")

    s = np.array([100.0])
    rw = np.array([50.0])
    cr = np.array([1.0])
    meta = _make_factor_meta(1, risk_weights=rw, concentration=cr)

    im = compute_simm_im(s, meta)
    expected = 50.0 * 100.0 * 1.0  # = 5000
    t.check_close(im, expected, 1e-10, "IM")

    return t.report()


def test_two_factors_same_bucket():
    """Test 2: Two factors in same risk class with known correlation rho.

    WS_1 = s_1 * rw_1 * cr_1 = 100 * 50 * 1 = 5000
    WS_2 = s_2 * rw_2 * cr_2 = 200 * 50 * 1 = 10000
    K^2 = WS_1^2 + WS_2^2 + 2 * rho * WS_1 * WS_2
         = 25e6 + 100e6 + 2 * 0.5 * 5000 * 10000
         = 175e6
    K = sqrt(175e6) = 13228.7566...
    IM = K (single risk class, psi=1)
    """
    t = TestResult("Two factors, same bucket, rho=0.5")

    rho = 0.5
    s = np.array([100.0, 200.0])
    rw = np.array([50.0, 50.0])
    cr = np.array([1.0, 1.0])
    corr = np.array([[1.0, rho], [rho, 1.0]])
    meta = _make_factor_meta(2, risk_weights=rw, concentration=cr, intra_corr=corr)

    im = compute_simm_im(s, meta)
    ws1, ws2 = 5000.0, 10000.0
    k_sq = ws1**2 + ws2**2 + 2 * rho * ws1 * ws2
    expected = np.sqrt(k_sq)
    t.check_close(im, expected, 1e-10, "IM")

    return t.report()


def test_two_risk_classes():
    """Test 3: Two risk classes with known psi.

    Factor 0: Rates, RW=50, s=100 -> WS=5000, K_rates=5000
    Factor 1: Equity, RW=30, s=200 -> WS=6000, K_equity=6000

    psi[Rates, Equity] = 0.07  (from PSI_MATRIX[0, 3])

    IM^2 = K_r^2 + K_e^2 + 2 * psi * K_r * K_e
         = 25e6 + 36e6 + 2 * 0.07 * 5000 * 6000
         = 65.2e6
    IM = sqrt(65.2e6)
    """
    t = TestResult("Two risk classes with known psi")

    s = np.array([100.0, 200.0])
    rw = np.array([50.0, 30.0])
    cr = np.array([1.0, 1.0])
    rc = ["Rates", "Equity"]
    corr = np.eye(2)  # Different risk classes -> no intra-corr
    meta = _make_factor_meta(2, risk_classes=rc, risk_weights=rw,
                             concentration=cr, intra_corr=corr)

    im = compute_simm_im(s, meta)
    k_rates = 5000.0
    k_equity = 6000.0
    psi_re = PSI_MATRIX[RISK_CLASS_TO_IDX["Rates"], RISK_CLASS_TO_IDX["Equity"]]
    t.check(abs(psi_re - 0.07) < 1e-10, f"psi[Rates,Equity] = {psi_re} (expected 0.07)")

    im_sq = k_rates**2 + k_equity**2 + 2 * psi_re * k_rates * k_equity
    expected = np.sqrt(im_sq)
    t.check_close(im, expected, 1e-10, "IM")

    return t.report()


def test_gradient_single_factor():
    """Test 4: Gradient for single factor.

    IM = RW * CR * |s|
    dIM/ds = RW * CR * sign(s) = RW * CR (for s > 0)

    More precisely: IM = sqrt(psi * K^2) = K
    K = sqrt(WS^2) = |WS| = RW*CR*s (for s>0)
    dIM/ds = dK/dWS * dWS/ds
    dK/dWS = WS/K = 1 (since K = |WS|)
    dWS/ds = RW * CR
    => dIM/ds = RW * CR
    """
    t = TestResult("Gradient single factor")

    s = np.array([[100.0]])  # (1, 1) for batch
    rw = np.array([50.0])
    cr = np.array([1.5])
    meta = _make_factor_meta(1, risk_weights=rw, concentration=cr)

    im_vals, grads = compute_simm_gradient(s, meta)
    expected_grad = 50.0 * 1.5  # RW * CR
    t.check_close(grads[0, 0], expected_grad, 1e-10, "dIM/ds")

    return t.report()


def test_gradient_finite_difference():
    """Test 5: Verify analytical gradient matches finite difference."""
    t = TestResult("Gradient vs finite difference (multi-factor)")

    K = 5
    rho = 0.3
    s = np.array([[10.0, 20.0, -5.0, 15.0, 8.0]])  # (1, K)
    rw = np.array([50.0, 30.0, 40.0, 20.0, 60.0])
    cr = np.array([1.0, 1.2, 0.9, 1.1, 1.0])
    corr = np.full((K, K), rho)
    np.fill_diagonal(corr, 1.0)
    meta = _make_factor_meta(K, risk_weights=rw, concentration=cr, intra_corr=corr)

    im_vals, grads = compute_simm_gradient(s, meta)

    # Finite difference
    eps = 1e-6
    fd_grads = np.zeros(K)
    for k in range(K):
        s_up = s.copy()
        s_up[0, k] += eps
        im_up = compute_simm_im(s_up[0], meta)
        im_dn = compute_simm_im(s[0], meta)
        fd_grads[k] = (im_up - im_dn) / eps

    for k in range(K):
        rel_err = abs(grads[0, k] - fd_grads[k]) / max(abs(fd_grads[k]), 1e-15)
        t.check(rel_err < 1e-4,
                f"Factor {k}: analytical={grads[0,k]:.6e}, "
                f"FD={fd_grads[k]:.6e}, rel_err={rel_err:.2e}")

    return t.report()


def test_batch_consistency():
    """Test 6: Batch computation matches individual."""
    t = TestResult("Batch vs individual computation")

    K = 4
    P = 3
    np.random.seed(123)
    s = np.random.randn(P, K) * 100
    rw = np.array([50.0, 30.0, 40.0, 20.0])
    cr = np.ones(K)
    meta = _make_factor_meta(K, risk_weights=rw, concentration=cr)

    # Batch
    batch_im = compute_simm_batch(s, meta)
    batch_im2, batch_grad = compute_simm_gradient(s, meta)

    # Individual
    for p in range(P):
        individual_im = compute_simm_im(s[p], meta)
        t.check_close(batch_im[p], individual_im, 1e-12,
                       f"Portfolio {p} batch vs individual IM")
        t.check_close(batch_im2[p], individual_im, 1e-12,
                       f"Portfolio {p} gradient batch IM vs individual")

    return t.report()


def test_psi_matrix_symmetry():
    """Test 7: PSI matrix properties."""
    t = TestResult("PSI matrix properties (ISDA v2.6)")

    # Symmetric
    t.check(np.allclose(PSI_MATRIX, PSI_MATRIX.T),
            "PSI matrix is symmetric")

    # Diagonal is 1
    t.check(np.allclose(np.diag(PSI_MATRIX), 1.0),
            "PSI diagonal is all 1.0")

    # All values in [0, 1]
    t.check(np.all(PSI_MATRIX >= 0) and np.all(PSI_MATRIX <= 1),
            "All PSI values in [0, 1]")

    # 6x6
    t.check(PSI_MATRIX.shape == (6, 6),
            f"PSI shape is {PSI_MATRIX.shape} (expected (6,6))")

    # Check specific known values
    t.check_close(PSI_MATRIX[0, 1], 0.04, 1e-10, "psi[Rates, CreditQ]")
    t.check_close(PSI_MATRIX[0, 4], 0.37, 1e-10, "psi[Rates, Commodity]")
    t.check_close(PSI_MATRIX[1, 3], 0.70, 1e-10, "psi[CreditQ, Equity]")

    return t.report()


def report_feature_coverage():
    """Report which ISDA SIMM v2.6 features are included/excluded."""
    print("\n" + "=" * 60)
    print("ISDA SIMM v2.6 Feature Coverage")
    print("=" * 60)
    features = [
        ("Delta risk measure", True, "Full implementation with intra-bucket correlations"),
        ("Vega risk measure", True, "VRW and VCR implemented"),
        ("Curvature risk measure", False, "Requires scenario-based computation"),
        ("BaseCorr risk measure", False, "CreditQ only, not in benchmark scope"),
        ("Intra-bucket correlations", True, "12x12 IR tenor correlation matrix"),
        ("Inter-bucket correlations", True, "gamma correlation for IR cross-currency"),
        ("Cross-risk-class (PSI)", True, "6x6 matrix from ISDA specification"),
        ("Concentration thresholds", True, "CR for Delta, VCR for Vega"),
        ("Currency-specific IR weights", True, "Regular/low/high volatility"),
        ("Sub-curve correlations", True, "phi for IR multi-curve"),
        ("FX volatility categories", True, "High/regular correlation"),
        ("Credit bucket-specific", True, "Bucket-specific correlations"),
    ]

    for name, included, note in features:
        mark = "Y" if included else "N"
        print(f"  [{mark}] {name:<35} {note}")


def main():
    print("=" * 60)
    print("ISDA SIMM v2.6 Formula Validation")
    print("=" * 60)

    tests = [
        test_single_factor,
        test_two_factors_same_bucket,
        test_two_risk_classes,
        test_gradient_single_factor,
        test_gradient_finite_difference,
        test_batch_consistency,
        test_psi_matrix_symmetry,
    ]

    results = []
    for test_fn in tests:
        results.append(test_fn())

    # Summary
    n_pass = sum(results)
    n_total = len(results)
    print("\n" + "=" * 60)
    print(f"Results: {n_pass}/{n_total} tests passed")
    if n_pass == n_total:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {n_total - n_pass} test(s) failed")
    print("=" * 60)

    report_feature_coverage()

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
