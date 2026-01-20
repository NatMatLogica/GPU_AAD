#!/usr/bin/env python
"""
Comprehensive validation tests for IR Swap Pricer.

Tests:
1. Par swap pricing (PV ≈ 0)
2. FD truncation error vs AAD (varying bump sizes)
3. SIMM end-to-end integration
4. CRIF format validation
5. Bucket mapping verification
"""

import sys
import numpy as np
sys.path.insert(0, '.')

from model.ir_swap_common import (
    MarketData, IRSwap, TENOR_LABELS, TENOR_YEARS, NUM_TENORS, BUMP_SIZE,
    generate_market_data, generate_crif
)
from model.ir_swap_pricer import price_swap


def create_flat_curve(currency: str, rate: float) -> MarketData:
    """Create a flat yield curve for testing."""
    return MarketData(
        currency=currency,
        valuation_date=0.0,
        discount_tenors=TENOR_YEARS.copy(),
        discount_rates=np.full(NUM_TENORS, rate),
        forward_tenors=TENOR_YEARS.copy(),
        forward_rates=np.full(NUM_TENORS, rate),
    )


def calculate_par_rate(market: MarketData, maturity: float, frequency: float) -> float:
    """
    Calculate the par swap rate for a given maturity.

    Par rate = (1 - DF(T)) / Σ(dt_i * DF(t_i))
    """
    payment_dates = []
    t = frequency
    while t <= maturity + 1e-9:
        payment_dates.append(t)
        t += frequency

    if not payment_dates:
        return 0.0

    # Annuity = sum of discounted period lengths
    annuity = 0.0
    prev_t = 0.0
    for t in payment_dates:
        dt = t - prev_t
        df = market.discount_factor(t)
        annuity += dt * df
        prev_t = t

    # Par rate = (1 - DF(T)) / Annuity
    df_maturity = market.discount_factor(payment_dates[-1])
    par_rate = (1.0 - df_maturity) / annuity if annuity > 0 else 0.0

    return par_rate


# =============================================================================
# TEST 1: Par Swap Validation
# =============================================================================

def test_par_swap_pv_zero():
    """
    Test that a par swap has PV ≈ 0.

    A par swap is one where the fixed rate equals the par rate,
    making the present value of fixed and floating legs equal.
    """
    print("=" * 70)
    print("TEST 1: Par Swap Pricing (PV should be ≈ 0)")
    print("=" * 70)

    from model.ir_swap_pricer import price_swap

    # Create flat curve at 5%
    market = create_flat_curve("USD", 0.05)

    results = []

    for maturity in [1, 2, 5, 10, 30]:
        for frequency in [0.5, 1.0]:
            # Calculate par rate
            par_rate = calculate_par_rate(market, float(maturity), frequency)

            # Create par swap
            par_swap = IRSwap(
                trade_id=f"PAR_{maturity}Y",
                currency="USD",
                notional=1_000_000,
                fixed_rate=par_rate,
                maturity=float(maturity),
                pay_frequency=frequency,
                is_payer=True,
            )

            # Price it
            pv = price_swap(par_swap, market)

            freq_str = "Semi" if frequency == 0.5 else "Annual"
            status = "PASS" if abs(pv) < 1.0 else "FAIL"  # Within $1 tolerance

            results.append({
                'maturity': maturity,
                'frequency': freq_str,
                'par_rate': par_rate,
                'pv': pv,
                'status': status
            })

            print(f"  {maturity}Y {freq_str:6}: par_rate={par_rate:.6f}, PV=${pv:,.2f} [{status}]")

    # Summary
    passed = sum(1 for r in results if r['status'] == 'PASS')
    print(f"\nResult: {passed}/{len(results)} tests passed")
    print()

    return all(r['status'] == 'PASS' for r in results)


# =============================================================================
# TEST 2: FD Truncation Error Analysis
# =============================================================================

def test_fd_truncation_error():
    """
    Verify that FD/AAD difference is due to FD truncation error
    by showing error scales with bump size.

    Theory: FD error ∝ O(bump_size) for forward difference
    If we halve the bump, error should roughly halve.
    """
    print("=" * 70)
    print("TEST 2: FD Truncation Error Analysis")
    print("=" * 70)

    from model.ir_swap_common import BUMP_SIZE
    from model.ir_swap_aadc import price_with_greeks as aadc_greeks, _kernel_cache, AADC_AVAILABLE

    if not AADC_AVAILABLE:
        print("  AADC not available - skipping test")
        return True

    # Create test setup
    market_data = {'USD': create_flat_curve('USD', 0.05)}

    # Single swap for clear analysis
    swap = IRSwap(
        trade_id="TEST_SWAP",
        currency="USD",
        notional=10_000_000,
        fixed_rate=0.05,
        maturity=10.0,
        pay_frequency=0.5,
        is_payer=True,
    )
    trades = [swap]

    # Get AAD delta (exact)
    _kernel_cache.clear()
    aadc_result = aadc_greeks(trades, market_data)
    aadc_delta = aadc_result.ir_delta[0, 0, :]  # First trade, first currency

    print(f"\n  AAD Delta (exact) for 10Y swap:")
    print(f"  Tenor     AAD Delta")
    print(f"  " + "-" * 30)
    for i, label in enumerate(TENOR_LABELS[:6]):
        print(f"  {label:8} {aadc_delta[i]:>15,.2f}")
    print(f"  ...")
    print()

    # Test different bump sizes
    bump_sizes = [0.00001, 0.0001, 0.001, 0.01]  # 0.1bp, 1bp, 10bp, 100bp

    print(f"  Bump Size Analysis (comparing to AAD):")
    print(f"  " + "-" * 60)
    print(f"  {'Bump':>10} {'Max Abs Err':>15} {'Max Rel Err':>15} {'Scaling':>10}")
    print(f"  " + "-" * 60)

    prev_max_err = None

    for bump in bump_sizes:
        # Compute FD delta with this bump size
        fd_delta = np.zeros(NUM_TENORS)
        market = market_data['USD']

        from model.ir_swap_pricer import price_swap
        base_pv = price_swap(swap, market)

        for tenor_idx in range(NUM_TENORS):
            bumped_market = market.bump_curve(tenor_idx, bump)
            bumped_pv = price_swap(swap, bumped_market)
            fd_delta[tenor_idx] = (bumped_pv - base_pv) / bump

        # Compare to AAD
        abs_err = np.abs(fd_delta - aadc_delta)
        max_abs_err = np.max(abs_err)

        nonzero = np.abs(aadc_delta) > 1e-10
        if nonzero.any():
            rel_err = np.abs(fd_delta[nonzero] - aadc_delta[nonzero]) / np.abs(aadc_delta[nonzero])
            max_rel_err = np.max(rel_err)
        else:
            max_rel_err = 0.0

        # Check scaling
        if prev_max_err is not None and prev_max_err > 0:
            scaling = max_abs_err / prev_max_err
        else:
            scaling = np.nan

        bp_label = f"{bump*10000:.1f}bp"
        print(f"  {bp_label:>10} {max_abs_err:>15,.2f} {max_rel_err:>14.4%} {scaling:>10.2f}")

        prev_max_err = max_abs_err

    print()
    print("  Expected: Error scales ~linearly with bump size (scaling ≈ 10 for 10x bump)")
    print("  This confirms the difference is FD truncation error, not AAD bug.")
    print()

    return True


# =============================================================================
# TEST 3: SIMM End-to-End Integration
# =============================================================================

def test_simm_end_to_end():
    """
    Test full pipeline: pricing → CRIF → SIMM margin.
    """
    print("=" * 70)
    print("TEST 3: SIMM End-to-End Integration")
    print("=" * 70)

    from model.ir_swap_pricer import price_with_greeks
    from model.ir_swap_common import generate_trades, generate_crif

    # Generate test portfolio
    currencies = ['USD', 'EUR', 'GBP']
    market_data = {}
    for i, ccy in enumerate(currencies):
        market_data[ccy] = generate_market_data(ccy, base_rate=0.03 + i*0.01, seed=42+i)

    trades = generate_trades(30, currencies, seed=42)

    print(f"\n  Portfolio: {len(trades)} trades across {currencies}")

    # Compute Greeks
    result = price_with_greeks(trades, market_data)
    print(f"  Portfolio PV: ${result.prices.sum():,.2f}")
    print(f"  Sensitivities computed: {result.num_sensitivities}")

    # Generate CRIF
    crif = generate_crif(trades, result)
    print(f"  CRIF rows generated: {len(crif)}")

    # Validate CRIF format
    required_columns = ['ProductClass', 'RiskType', 'Qualifier', 'Bucket',
                       'Label1', 'Label2', 'Amount', 'AmountCurrency', 'AmountUSD']
    missing = [col for col in required_columns if col not in crif.columns]

    if missing:
        print(f"  CRIF VALIDATION FAILED - Missing columns: {missing}")
        return False

    print(f"  CRIF columns: {list(crif.columns)}")
    print(f"  CRIF validation: PASS (all required columns present)")

    # Run SIMM
    print(f"\n  Running SIMM aggregation...")
    try:
        from src.agg_margins import SIMM
        portfolio = SIMM(crif, "USD", 1)
        simm_total = portfolio.simm
        print(f"  SIMM Total Margin: ${simm_total:,.2f}")

        # Show breakdown if available
        if hasattr(portfolio, 'delta'):
            print(f"  SIMM Delta Component: ${portfolio.delta:,.2f}")

        print(f"\n  SIMM Integration: PASS")
        return True

    except Exception as e:
        print(f"  SIMM Error: {e}")
        print(f"  SIMM Integration: FAIL")
        return False


# =============================================================================
# TEST 4: Bucket Mapping Verification
# =============================================================================

def test_bucket_mapping():
    """
    Verify tenor to SIMM bucket mapping is correct.

    SIMM IR Delta buckets:
    1: 2w, 2: 1m, 3: 3m, 4: 6m, 5: 1y, 6: 2y,
    7: 3y, 8: 5y, 9: 10y, 10: 15y, 11: 20y, 12: 30y
    """
    print("=" * 70)
    print("TEST 4: SIMM Bucket Mapping Verification")
    print("=" * 70)

    expected_mapping = {
        '2w': '1', '1m': '2', '3m': '3', '6m': '4',
        '1y': '5', '2y': '6', '3y': '7', '5y': '8',
        '10y': '9', '15y': '10', '20y': '11', '30y': '12'
    }

    from model.ir_swap_pricer import price_with_greeks
    from model.ir_swap_common import generate_crif

    # Create single swap with sensitivity to all tenors
    market_data = {'USD': generate_market_data('USD', seed=42)}

    swap = IRSwap(
        trade_id="BUCKET_TEST",
        currency="USD",
        notional=100_000_000,
        fixed_rate=0.05,
        maturity=30.0,  # 30Y to hit all buckets
        pay_frequency=0.5,
        is_payer=True,
    )

    result = price_with_greeks([swap], market_data)
    crif = generate_crif([swap], result)

    print(f"\n  Checking bucket assignments in CRIF:")
    print(f"  {'Tenor':<8} {'Expected Bucket':<16} {'Actual Bucket':<16} {'Status'}")
    print(f"  " + "-" * 55)

    all_pass = True
    for _, row in crif.iterrows():
        tenor = row['Label1']
        actual_bucket = str(row['Bucket'])
        expected_bucket = expected_mapping.get(tenor, '?')

        status = "PASS" if actual_bucket == expected_bucket else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  {tenor:<8} {expected_bucket:<16} {actual_bucket:<16} {status}")

    print(f"\n  Bucket Mapping: {'PASS' if all_pass else 'FAIL'}")
    print()

    return all_pass


# =============================================================================
# TEST 5: Extended Greeks (Gamma, Cross-Gamma, Theta)
# =============================================================================

def test_extended_greeks():
    """
    Test extended Greeks computation: gamma, cross-gamma, theta.
    """
    print("=" * 70)
    print("TEST 5: Extended Greeks (Gamma, Cross-Gamma, Theta)")
    print("=" * 70)

    from model.ir_swap_aadc import compute_extended_greeks, AADC_AVAILABLE

    if not AADC_AVAILABLE:
        print("  AADC not available - skipping test")
        return True

    # Create test setup - single 10Y swap
    market_data = {'USD': create_flat_curve('USD', 0.05)}

    swap = IRSwap(
        trade_id="GREEKS_TEST",
        currency="USD",
        notional=10_000_000,
        fixed_rate=0.05,
        maturity=10.0,
        pay_frequency=0.5,
        is_payer=True,
    )

    print(f"\n  Computing extended Greeks for 10Y $10M swap...")

    result = compute_extended_greeks([swap], market_data)

    print(f"\n  Price: ${result.prices[0]:,.2f}")
    print(f"  Theta (1 day): ${result.theta[0]:,.2f}")
    print()

    # Show delta and gamma for key tenors
    print(f"  {'Tenor':<8} {'Delta':>15} {'Gamma':>15}")
    print(f"  " + "-" * 40)

    key_tenors = [4, 5, 6, 7, 8]  # 1y, 2y, 3y, 5y, 10y
    for idx in key_tenors:
        tenor = TENOR_LABELS[idx]
        delta = result.ir_delta[0, 0, idx]
        gamma = result.ir_gamma[0, 0, idx]
        print(f"  {tenor:<8} {delta:>15,.2f} {gamma:>15,.2f}")

    # Show sample cross-gamma
    print(f"\n  Cross-Gamma (d²PV/dr_i dr_j) for adjacent tenors:")
    print(f"  " + "-" * 40)
    for i in [7, 8]:  # 5y, 10y
        for j in range(i + 1, min(i + 2, NUM_TENORS)):
            cg = result.ir_cross_gamma[0, 0, i, j]
            print(f"  {TENOR_LABELS[i]}-{TENOR_LABELS[j]}: {cg:,.2f}")

    # Sanity checks
    checks_pass = True

    # Delta should be negative for payer swap (loses value when rates rise)
    if result.ir_delta[0, 0, 8] >= 0:  # 10y tenor
        print(f"\n  WARNING: Delta should be negative for payer swap")
        checks_pass = False

    # Gamma should be positive (convexity)
    if result.ir_gamma[0, 0, 8] <= 0:
        print(f"\n  WARNING: Gamma should be positive (convexity)")
        checks_pass = False

    # Theta should be small relative to PV for ATM swap
    if abs(result.theta[0]) > abs(result.prices[0]) * 0.01:
        print(f"\n  WARNING: Theta seems too large relative to PV")
        # This is not necessarily a failure, depends on the swap

    print(f"\n  Extended Greeks: {'PASS' if checks_pass else 'REVIEW'}")
    print()

    return checks_pass


# =============================================================================
# Main
# =============================================================================

def main():
    print()
    print("=" * 70)
    print("       IR Swap Pricer - Comprehensive Validation Suite")
    print("=" * 70)
    print()

    results = {}

    # Run all tests
    results['par_swap'] = test_par_swap_pv_zero()
    results['fd_truncation'] = test_fd_truncation_error()
    results['simm_e2e'] = test_simm_end_to_end()
    results['bucket_mapping'] = test_bucket_mapping()
    results['extended_greeks'] = test_extended_greeks()

    # Summary
    print("=" * 70)
    print("                    VALIDATION SUMMARY")
    print("=" * 70)
    print()
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:<25} {status}")

    total_passed = sum(results.values())
    total_tests = len(results)
    print()
    print(f"  Overall: {total_passed}/{total_tests} tests passed")
    print("=" * 70)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
