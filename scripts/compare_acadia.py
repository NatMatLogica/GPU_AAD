#!/usr/bin/env python3
"""
Compare our ISDA SIMM implementation against AcadiaSoft simm-lib test cases.

This script:
1. Parses AcadiaSoft's test sensitivities from SimmCrifMixin.java
2. Converts to our CRIF DataFrame format
3. Runs our SIMM calculation
4. Compares results against their expected values
5. Measures and compares performance

AcadiaSoft uses SIMM v2.5, we use v2.6 - expect some parameter differences.
They have both ONE_DAY and TEN_DAY tests - we compare against TEN_DAY (standard SIMM).

Reference test cases from AcadiaSoft:
- testC67: All S_IR (46 IR sensitivities) -> $11,126,437,227 (10-day)
- testC80: All S_FX (12 FX sensitivities) -> $45,609,126,471 (10-day)
"""

import pandas as pd
import numpy as np
import time
import re
import sys
from pathlib import Path

# Add project root to path
try:
    PROJECT_ROOT = Path(__file__).parent
except NameError:
    PROJECT_ROOT = Path('/home/natashamanito/ISDA-SIMM')
sys.path.insert(0, str(PROJECT_ROOT))

from src.agg_margins import SIMM
from src.wnc import set_simm_version, get_simm_version

# =============================================================================
# ACADIA TEST SENSITIVITIES (parsed from SimmCrifMixin.java)
# Format: (productClass, riskType, qualifier, bucket, label1, label2, amountUSD)
# =============================================================================

# S_IR: 46 Interest Rate sensitivities
S_IR = [
    ("RatesFX", "Risk_IRCurve", "USD", "1", "2w", "OIS", 4000000),
    ("RatesFX", "Risk_IRCurve", "USD", "1", "3m", "Municipal", -3000000),
    ("RatesFX", "Risk_IRCurve", "USD", "1", "1y", "Municipal", 2000000),
    ("RatesFX", "Risk_IRCurve", "USD", "1", "1y", "Prime", 3000000),
    ("RatesFX", "Risk_IRCurve", "USD", "1", "1y", "Prime", -1000000),
    ("RatesFX", "Risk_IRCurve", "EUR", "1", "3y", "Libor3m", -2000000),
    ("RatesFX", "Risk_IRCurve", "EUR", "1", "3y", "Libor6m", 5000000),
    ("RatesFX", "Risk_IRCurve", "EUR", "1", "5y", "Libor12m", 10000000),
    ("RatesFX", "Risk_IRCurve", "EUR", "1", "5y", "Libor12m", 25000000),
    ("RatesFX", "Risk_IRCurve", "EUR", "1", "10y", "Libor12m", 35000000),
    ("RatesFX", "Risk_IRCurve", "AUD", "1", "1m", "Libor3m", 2000000),
    ("RatesFX", "Risk_IRCurve", "AUD", "1", "6m", "Libor3m", 3000000),
    ("RatesFX", "Risk_IRCurve", "AUD", "1", "2y", "Libor3m", -2000000),
    ("RatesFX", "Risk_IRCurve", "CHF", "1", "15y", "Libor6m", -4000000),
    ("RatesFX", "Risk_IRCurve", "CHF", "1", "20y", "Libor6m", 10000000),
    ("RatesFX", "Risk_IRCurve", "CHF", "1", "30y", "Libor6m", 18000000),
    ("RatesFX", "Risk_Inflation", "CHF", "", "", "", -10000000),
    ("RatesFX", "Risk_XCcyBasis", "CHF", "", "", "", 30000000),
    ("RatesFX", "Risk_IRCurve", "JPY", "2", "2w", "Libor1m", -1000000),
    ("RatesFX", "Risk_IRCurve", "JPY", "2", "1m", "Libor1m", -1500000),
    ("RatesFX", "Risk_IRCurve", "JPY", "2", "3m", "Libor3m", 1500000),
    ("RatesFX", "Risk_IRCurve", "JPY", "2", "6m", "Libor3m", 2000000),
    ("RatesFX", "Risk_IRCurve", "JPY", "2", "1y", "Libor6m", 3000000),
    ("RatesFX", "Risk_IRCurve", "JPY", "2", "2y", "Libor6m", 4000000),
    ("RatesFX", "Risk_IRCurve", "JPY", "2", "3y", "Libor6m", 5000000),
    ("RatesFX", "Risk_IRCurve", "JPY", "2", "5y", "Libor12m", 20000000),
    ("RatesFX", "Risk_IRCurve", "JPY", "2", "10y", "Libor12m", 30000000),
    ("RatesFX", "Risk_IRCurve", "JPY", "2", "15y", "Libor12m", -1000000),
    ("RatesFX", "Risk_IRCurve", "JPY", "2", "20y", "Libor12m", -2000000),
    ("RatesFX", "Risk_IRCurve", "JPY", "2", "30y", "Libor12m", 3000000),
    ("RatesFX", "Risk_Inflation", "JPY", "", "", "", 5000000),
    ("RatesFX", "Risk_XCcyBasis", "JPY", "", "", "", 500000),
    ("RatesFX", "Risk_IRCurve", "CNY", "3", "2w", "OIS", 1000000),
    ("RatesFX", "Risk_IRCurve", "CNY", "3", "1m", "OIS", 1500000),
    ("RatesFX", "Risk_IRCurve", "CNY", "3", "3m", "Libor1m", -500000),
    ("RatesFX", "Risk_IRCurve", "CNY", "3", "6m", "Libor3m", -1000000),
    ("RatesFX", "Risk_IRCurve", "MXN", "3", "1y", "Libor6m", 9000000),
    ("RatesFX", "Risk_IRCurve", "MXN", "3", "2y", "Libor12m", 10000000),
    ("RatesFX", "Risk_IRCurve", "MXN", "3", "3y", "OIS", -500000),
    ("RatesFX", "Risk_IRCurve", "MXN", "3", "5y", "OIS", -1000000),
    # Note: S_IR_41-46 have ProductClass="Credit" in Acadia's test
    ("Credit", "Risk_IRCurve", "BRL", "3", "10y", "Libor6m", 14000000),
    ("Credit", "Risk_IRCurve", "BRL", "3", "15y", "Libor6m", 30000000),
    ("Credit", "Risk_IRCurve", "BRL", "3", "20y", "Libor12m", -800000),
    ("Credit", "Risk_IRCurve", "BRL", "3", "30y", "Libor12m", -800000),
    ("Credit", "Risk_Inflation", "BRL", "", "", "", 2000000),
    ("Credit", "Risk_XCcyBasis", "BRL", "", "", "", -1000000),
]

# S_FX: 12 FX sensitivities
S_FX = [
    ("RatesFX", "Risk_FX", "EUR", "", "", "", 50000000),
    ("RatesFX", "Risk_FX", "EUR", "", "", "", -50000000),
    ("RatesFX", "Risk_FX", "EUR", "", "", "", -5000000000),
    ("RatesFX", "Risk_FX", "USD", "", "", "", 610000000),
    ("RatesFX", "Risk_FX", "GBP", "", "", "", 910000000),
    ("RatesFX", "Risk_FX", "EUR", "", "", "", -900000000),
    ("RatesFX", "Risk_FX", "CNY", "", "", "", -200000000),
    ("RatesFX", "Risk_FX", "KRW", "", "", "", 210000000),
    ("RatesFX", "Risk_FX", "TRY", "", "", "", 80000000),
    ("RatesFX", "Risk_FX", "BRL", "", "", "", -300000000),
    ("Credit", "Risk_FX", "BRL", "", "", "", 41000000),
    ("Credit", "Risk_FX", "QAR", "", "", "", -40000000),
]

# Single sensitivity tests for validation
SINGLE_TESTS = [
    # (sensitivities, expected_1day, expected_10day, description)
    ([("RatesFX", "Risk_IRCurve", "USD", "1", "2w", "OIS", 4000000)],
     76000000, None, "S_IR_1: USD 2w OIS +4M"),
    ([("RatesFX", "Risk_IRCurve", "EUR", "1", "3y", "Libor6m", 5000000)],
     80000000, None, "S_IR_7: EUR 3y Libor6m +5M"),
]

# =============================================================================
# EXPECTED RESULTS FROM ACADIA TESTS
# =============================================================================

EXPECTED_RESULTS = {
    # Test name: (sensitivities, expected_1day, expected_10day)
    "All_IR": (S_IR, 3134574486, 11126437227),
    "All_FX": (S_FX, 10242651353, 45609126471),
}


def sensitivities_to_crif(sensitivities: list) -> pd.DataFrame:
    """
    Convert list of sensitivity tuples to CRIF DataFrame.

    Args:
        sensitivities: List of (productClass, riskType, qualifier, bucket, label1, label2, amountUSD)

    Returns:
        CRIF DataFrame with standard columns
    """
    rows = []
    for i, sens in enumerate(sensitivities):
        productClass, riskType, qualifier, bucket, label1, label2, amountUSD = sens
        rows.append({
            'TradeID': f'T{i+1}',
            'ProductClass': productClass,
            'RiskType': riskType,
            'Qualifier': qualifier,
            'Bucket': bucket if bucket else '',
            'Label1': label1 if label1 else '',
            'Label2': label2 if label2 else '',
            'Amount': amountUSD,
            'AmountCurrency': 'USD',
            'AmountUSD': amountUSD,
        })

    return pd.DataFrame(rows)


def run_our_simm(crif: pd.DataFrame) -> tuple:
    """
    Run our SIMM implementation on CRIF data.

    Returns:
        (simm_total, elapsed_time_ms, breakdown_dict)
    """
    start = time.perf_counter()

    # Use our src/agg_margins.py SIMM class
    simm_calc = SIMM(crif, calculation_currency='USD', exchange_rate=1.0)

    elapsed = (time.perf_counter() - start) * 1000

    return simm_calc.simm, elapsed, simm_calc.simm_break_down


def compare_single_sensitivity(sens_tuple, expected_1day, description):
    """Test a single sensitivity and compare weighted sensitivity calculation."""
    crif = sensitivities_to_crif([sens_tuple])

    # Run our calculation
    our_result, elapsed, _ = run_our_simm(crif)

    # Our result is 10-day, so scale to compare with 1-day
    # ISDA specifies sqrt(10) scaling for holding period
    our_1day_approx = our_result / np.sqrt(10)

    # Calculate ratio
    ratio = our_result / expected_1day if expected_1day else 0
    ratio_1day = our_1day_approx / expected_1day if expected_1day else 0

    print(f"\n{description}")
    print(f"  Acadia 1-day:    ${expected_1day:,.0f}")
    print(f"  Our result:      ${our_result:,.0f} (10-day)")
    print(f"  Our ÷ sqrt(10):  ${our_1day_approx:,.0f}")
    print(f"  Ratio (10d/1d):  {ratio:.2f}x")
    print(f"  Time:            {elapsed:.1f}ms")


def compare_portfolio(name, sensitivities, expected_1day, expected_10day):
    """Compare portfolio-level SIMM calculation."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  Sensitivities: {len(sensitivities)}")
    print(f"{'='*60}")

    crif = sensitivities_to_crif(sensitivities)

    # Run our calculation
    our_result, elapsed, breakdown = run_our_simm(crif)

    # Compare with expected values
    diff_pct_10d = ((our_result - expected_10day) / expected_10day * 100) if expected_10day else 0
    diff_pct_1d = ((our_result - expected_1day) / expected_1day * 100) if expected_1day else 0

    print(f"\nResults:")
    print(f"  Acadia 1-day:  ${expected_1day:>20,.0f}")
    print(f"  Acadia 10-day: ${expected_10day:>20,.0f}")
    print(f"  Our result:    ${our_result:>20,.0f}")
    print(f"  Diff vs 10d:   {diff_pct_10d:>+20.2f}%")
    print(f"  Ratio (ours/10d): {our_result/expected_10day:.4f}" if expected_10day else "")
    print(f"\nPerformance:")
    print(f"  Time: {elapsed:.2f}ms")

    # Analyze version differences
    # Acadia v2.5 vs our v2.6 - different risk weights
    print(f"\n  Note: Differences expected due to SIMM v2.5 vs v2.6 parameters")

    return our_result, expected_10day, elapsed


def run_performance_benchmark(sensitivities, num_iterations=100):
    """Run performance benchmark with multiple iterations."""
    crif = sensitivities_to_crif(sensitivities)

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        simm_calc = SIMM(crif, calculation_currency='USD', exchange_rate=1.0)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'sensitivities': len(sensitivities),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare SIMM implementations')
    parser.add_argument('--version', '-v', type=str, default='2.5',
                        choices=['2.3', '2.4', '2.5', '2.6', '2.7'],
                        help='SIMM version to use (default: 2.5 to match Acadia)')
    args = parser.parse_args()

    # Set SIMM version
    set_simm_version(args.version)

    print("="*70)
    print("ISDA SIMM Comparison: Our Implementation vs AcadiaSoft simm-lib")
    print("="*70)
    print(f"\nAcadiaSoft: SIMM v2.5 (Java)")
    print(f"Our impl:   SIMM v{args.version} (Python)")
    if args.version == '2.5':
        print("\nUsing v2.5 parameters for apples-to-apples comparison!")
    else:
        print(f"\nNote: Parameter differences between v2.5 and v{args.version} will cause numeric differences.")
        print("      Use --version 2.5 for exact comparison.")

    # Run portfolio comparisons
    results = []
    for name, (sensitivities, exp_1d, exp_10d) in EXPECTED_RESULTS.items():
        our_result, expected, elapsed = compare_portfolio(name, sensitivities, exp_1d, exp_10d)
        results.append({
            'name': name,
            'our_result': our_result,
            'expected': expected,
            'elapsed_ms': elapsed,
            'num_sens': len(sensitivities),
        })

    # Performance benchmark
    print("\n" + "="*70)
    print("Performance Benchmark (100 iterations)")
    print("="*70)

    for name, (sensitivities, _, _) in EXPECTED_RESULTS.items():
        stats = run_performance_benchmark(sensitivities, num_iterations=100)
        print(f"\n{name} ({stats['sensitivities']} sensitivities):")
        print(f"  Mean:  {stats['mean_ms']:.2f}ms")
        print(f"  Std:   {stats['std_ms']:.2f}ms")
        print(f"  Range: [{stats['min_ms']:.2f}, {stats['max_ms']:.2f}]ms")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    print("\n| Test    | Sensitivities | Acadia 10-day    | Our Result       | Diff %  | Time   |")
    print("|---------|---------------|------------------|------------------|---------|--------|")
    for r in results:
        diff_pct = (r['our_result'] - r['expected']) / r['expected'] * 100 if r['expected'] else 0
        print(f"| {r['name']:<7} | {r['num_sens']:>13} | ${r['expected']:>14,.0f} | ${r['our_result']:>14,.0f} | {diff_pct:>+6.2f}% | {r['elapsed_ms']:>5.1f}ms |")

    print("\n" + "="*70)
    print("Analysis of Differences")
    print("="*70)
    if args.version == '2.5':
        print("""
Using SIMM v2.5 parameters for exact comparison with AcadiaSoft.

If results still differ, possible causes:
- Different aggregation order or rounding
- Different handling of edge cases
- Bucket/tenor mapping differences

Check individual risk class breakdowns to isolate discrepancies.
""")
    else:
        print(f"""
The differences are expected due to SIMM version changes:

v2.5 (Acadia) vs v{args.version} (Ours):
- Risk weights recalibrated each version
- Correlation parameters updated
- Concentration thresholds adjusted

For exact comparison, run: python compare_acadia.py --version 2.5
""")


if __name__ == "__main__":
    main()
