"""
What-If Analytics Module - Fast Margin Attribution and Scenario Analysis.

This module demonstrates the power of AADC for what-if analysis:

THE PROBLEM:
    "Which trades are consuming the most margin?"

    Naive approach: For each trade, compute IM without it, then subtract.
    - N trades × O(SIMM) = O(N²) for large portfolios
    - 1000 trades × 1 sec each = 17 minutes!

AADC SOLUTION:
    Use the gradient ∂IM/∂sensitivity to compute ALL trade contributions at once.
    - Compute gradient once: O(K) where K = risk factors
    - Each trade's contribution: gradient · trade_sensitivities = O(K)
    - Total: O(N×K) ≈ O(N)
    - 1000 trades = ~1 second!

PRODUCTION USE CASES:
    1. Daily margin attribution report
    2. "What if I unwind my top 10 contributors?"
    3. Hedge recommendation: "Which trade to add to reduce margin?"
    4. Stress attribution: "Which trades hurt most under stress?"

Version: 1.0.0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time

import sys
sys.path.insert(0, '.')


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TradeMarginContribution:
    """
    Margin contribution of a single trade.

    There are TWO ways to think about trade contribution:

    1. INCREMENTAL: IM(portfolio) - IM(portfolio without trade)
       - "How much would margin drop if I removed this trade?"
       - Can be negative (trade provides netting benefit)

    2. MARGINAL (first-order): gradient · trade_sensitivities
       - Linear approximation using AADC gradient
       - Fast to compute
       - Accurate when trades are small relative to portfolio

    For well-diversified portfolios, these are approximately equal.
    """
    trade_id: str

    # Sensitivities
    gross_sensitivity: float        # Sum of |sensitivities|
    net_sensitivity: float          # Sum of sensitivities (signed)

    # Margin contribution
    marginal_contribution: float    # gradient · sensitivities (AADC)
    incremental_contribution: float # Exact leave-one-out (optional, slow)

    # Percentage of total
    contribution_pct: float

    # Is this trade adding or reducing margin?
    is_margin_additive: bool        # True if removing trade would reduce IM


@dataclass
class MarginAttributionReport:
    """
    Full margin attribution report for a portfolio.

    Shows which trades are consuming margin and which provide netting benefit.
    """
    # Portfolio summary
    total_im: float
    num_trades: int
    num_risk_factors: int

    # Attribution by trade
    trade_contributions: List[TradeMarginContribution]

    # Top contributors
    top_margin_consumers: List[str]     # Trades adding most margin
    top_margin_reducers: List[str]      # Trades providing most netting

    # Computation stats
    computation_method: str             # "aadc_gradient" or "full_recalc"
    computation_time_ms: float

    # For comparison
    naive_time_estimate_ms: float       # Estimated time for O(N²) approach


@dataclass
class WhatIfScenarioResult:
    """
    Result of a what-if scenario analysis.
    """
    scenario_name: str
    description: str

    # Before/after
    current_im: float
    scenario_im: float
    im_change: float
    im_change_pct: float

    # Details
    trades_affected: List[str]
    computation_time_ms: float


# =============================================================================
# AADC-Accelerated Margin Attribution
# =============================================================================

def compute_margin_attribution_aadc(
    portfolio_crif: pd.DataFrame,
    trade_sensitivities: Dict[str, pd.DataFrame],
    num_threads: int = 4
) -> MarginAttributionReport:
    """
    Compute margin attribution for all trades using AADC gradient.

    This is the FAST method - O(N×K) instead of O(N²).

    The key insight:
        marginal_contribution[trade] ≈ Σ_k gradient[k] × sensitivity[trade, k]

    Where gradient[k] = ∂IM/∂S[k] is computed ONCE via AADC.

    Args:
        portfolio_crif: Aggregated CRIF for entire portfolio
        trade_sensitivities: Dict mapping trade_id -> CRIF for that trade
        num_threads: Threads for AADC

    Returns:
        MarginAttributionReport with all trade contributions

    Why this works:
        SIMM is approximately linear in sensitivities for small perturbations.
        The gradient captures this linear relationship.
        For a trade with sensitivities S_trade, its marginal contribution is:
            contribution ≈ Σ_k (∂IM/∂S_k) × S_trade[k]
    """
    start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # Step 1: Compute AADC gradient for portfolio
    # -------------------------------------------------------------------------
    try:
        from model.simm_portfolio_aadc import compute_im_gradient_aadc

        n = len(portfolio_crif)
        gradient, total_im, _, _ = compute_im_gradient_aadc(
            portfolio_crif, num_threads
        )
        method = "aadc_gradient"

    except Exception as e:
        # Fallback: compute total IM without gradient
        print(f"Warning: AADC unavailable ({e}), using numerical approximation")
        from model.pretrade_analytics import calculate_simm_margin

        total_im = calculate_simm_margin(portfolio_crif)
        n = len(portfolio_crif)

        # Approximate gradient using risk weights
        gradient = np.ones(n) * 0.005  # Rough approximation
        method = "approximate"

    # -------------------------------------------------------------------------
    # Step 2: Build mapping from CRIF rows to risk factors
    # -------------------------------------------------------------------------
    # Create index for fast lookup
    factor_to_idx = {}
    for i, row in portfolio_crif.iterrows():
        key = (row['RiskType'], row['Qualifier'], row['Bucket'], row['Label1'])
        factor_to_idx[key] = i

    # -------------------------------------------------------------------------
    # Step 3: Compute contribution for each trade
    # -------------------------------------------------------------------------
    contributions = []

    for trade_id, trade_crif in trade_sensitivities.items():
        # Sum gradient × sensitivity for this trade's risk factors
        marginal_contrib = 0.0
        gross_sens = 0.0
        net_sens = 0.0

        for _, row in trade_crif.iterrows():
            key = (row['RiskType'], row['Qualifier'], row['Bucket'], row['Label1'])
            sens = row['AmountUSD']

            gross_sens += abs(sens)
            net_sens += sens

            if key in factor_to_idx:
                idx = factor_to_idx[key]
                marginal_contrib += gradient[idx] * sens

        # Contribution as percentage of total
        contrib_pct = (marginal_contrib / total_im * 100) if total_im > 0 else 0

        contributions.append(TradeMarginContribution(
            trade_id=trade_id,
            gross_sensitivity=gross_sens,
            net_sensitivity=net_sens,
            marginal_contribution=marginal_contrib,
            incremental_contribution=marginal_contrib,  # Approximation
            contribution_pct=contrib_pct,
            is_margin_additive=marginal_contrib > 0
        ))

    # -------------------------------------------------------------------------
    # Step 4: Sort and identify top contributors
    # -------------------------------------------------------------------------
    # Sort by absolute contribution (largest first)
    contributions.sort(key=lambda x: abs(x.marginal_contribution), reverse=True)

    # Top margin consumers (positive contribution)
    consumers = [c.trade_id for c in contributions if c.is_margin_additive][:10]

    # Top margin reducers (negative contribution - provide netting)
    reducers = [c.trade_id for c in contributions if not c.is_margin_additive][:10]

    computation_time = (time.perf_counter() - start_time) * 1000

    # Estimate naive approach time: N trades × ~100ms per SIMM calc
    naive_estimate = len(trade_sensitivities) * 100  # ms

    return MarginAttributionReport(
        total_im=total_im,
        num_trades=len(trade_sensitivities),
        num_risk_factors=n,
        trade_contributions=contributions,
        top_margin_consumers=consumers,
        top_margin_reducers=reducers,
        computation_method=method,
        computation_time_ms=computation_time,
        naive_time_estimate_ms=naive_estimate
    )


def compute_margin_attribution_naive(
    portfolio_crif: pd.DataFrame,
    trade_sensitivities: Dict[str, pd.DataFrame],
) -> MarginAttributionReport:
    """
    Compute margin attribution using naive leave-one-out method.

    This is the SLOW method - O(N²) - for comparison.

    For each trade:
        1. Remove trade's sensitivities from portfolio CRIF
        2. Recalculate full SIMM
        3. Contribution = IM(full) - IM(without trade)

    This is exact but slow: N trades × O(SIMM) each.
    """
    from model.pretrade_analytics import calculate_simm_margin

    start_time = time.perf_counter()

    # Full portfolio IM
    total_im = calculate_simm_margin(portfolio_crif)

    contributions = []

    for trade_id, trade_crif in trade_sensitivities.items():
        # Create portfolio CRIF without this trade's sensitivities
        # Match by (RiskType, Qualifier, Bucket, Label1) and subtract amounts

        modified_crif = portfolio_crif.copy()

        for _, trade_row in trade_crif.iterrows():
            # Find matching row in portfolio CRIF
            mask = (
                (modified_crif['RiskType'] == trade_row['RiskType']) &
                (modified_crif['Qualifier'] == trade_row['Qualifier']) &
                (modified_crif['Bucket'] == trade_row['Bucket']) &
                (modified_crif['Label1'] == trade_row['Label1'])
            )

            if mask.any():
                # Subtract this trade's sensitivity
                modified_crif.loc[mask, 'AmountUSD'] -= trade_row['AmountUSD']
                modified_crif.loc[mask, 'Amount'] -= trade_row['Amount']

        # Remove zero rows
        modified_crif = modified_crif[modified_crif['AmountUSD'].abs() > 1e-10]

        # Calculate IM without this trade
        im_without = calculate_simm_margin(modified_crif) if len(modified_crif) > 0 else 0

        # Incremental contribution
        incr_contrib = total_im - im_without

        # Calculate gross/net sensitivity
        gross_sens = trade_crif['AmountUSD'].abs().sum()
        net_sens = trade_crif['AmountUSD'].sum()

        contrib_pct = (incr_contrib / total_im * 100) if total_im > 0 else 0

        contributions.append(TradeMarginContribution(
            trade_id=trade_id,
            gross_sensitivity=gross_sens,
            net_sensitivity=net_sens,
            marginal_contribution=incr_contrib,
            incremental_contribution=incr_contrib,
            contribution_pct=contrib_pct,
            is_margin_additive=incr_contrib > 0
        ))

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x.incremental_contribution), reverse=True)

    consumers = [c.trade_id for c in contributions if c.is_margin_additive][:10]
    reducers = [c.trade_id for c in contributions if not c.is_margin_additive][:10]

    computation_time = (time.perf_counter() - start_time) * 1000

    return MarginAttributionReport(
        total_im=total_im,
        num_trades=len(trade_sensitivities),
        num_risk_factors=len(portfolio_crif),
        trade_contributions=contributions,
        top_margin_consumers=consumers,
        top_margin_reducers=reducers,
        computation_method="full_recalc",
        computation_time_ms=computation_time,
        naive_time_estimate_ms=computation_time  # This IS the naive time
    )


# =============================================================================
# What-If Scenarios
# =============================================================================

def whatif_unwind_top_contributors(
    portfolio_crif: pd.DataFrame,
    trade_sensitivities: Dict[str, pd.DataFrame],
    attribution: MarginAttributionReport,
    num_to_unwind: int = 5
) -> WhatIfScenarioResult:
    """
    What-if: Unwind the top N margin-consuming trades.

    This is a common question: "What if I unwind my riskiest positions?"

    Args:
        portfolio_crif: Current portfolio CRIF
        trade_sensitivities: Per-trade CRIFs
        attribution: Pre-computed margin attribution
        num_to_unwind: Number of top contributors to unwind

    Returns:
        WhatIfScenarioResult showing margin impact
    """
    from model.pretrade_analytics import calculate_simm_margin

    start_time = time.perf_counter()

    current_im = attribution.total_im

    # Get top contributors to unwind
    trades_to_unwind = attribution.top_margin_consumers[:num_to_unwind]

    # Create modified CRIF without these trades
    modified_crif = portfolio_crif.copy()

    for trade_id in trades_to_unwind:
        if trade_id not in trade_sensitivities:
            continue

        trade_crif = trade_sensitivities[trade_id]

        for _, trade_row in trade_crif.iterrows():
            mask = (
                (modified_crif['RiskType'] == trade_row['RiskType']) &
                (modified_crif['Qualifier'] == trade_row['Qualifier']) &
                (modified_crif['Bucket'] == trade_row['Bucket']) &
                (modified_crif['Label1'] == trade_row['Label1'])
            )

            if mask.any():
                modified_crif.loc[mask, 'AmountUSD'] -= trade_row['AmountUSD']
                modified_crif.loc[mask, 'Amount'] -= trade_row['Amount']

    # Remove near-zero rows
    modified_crif = modified_crif[modified_crif['AmountUSD'].abs() > 1e-10]

    # Calculate new IM
    scenario_im = calculate_simm_margin(modified_crif) if len(modified_crif) > 0 else 0

    im_change = scenario_im - current_im
    im_change_pct = (im_change / current_im * 100) if current_im > 0 else 0

    computation_time = (time.perf_counter() - start_time) * 1000

    return WhatIfScenarioResult(
        scenario_name=f"Unwind Top {num_to_unwind} Contributors",
        description=f"Remove {len(trades_to_unwind)} highest margin-consuming trades",
        current_im=current_im,
        scenario_im=scenario_im,
        im_change=im_change,
        im_change_pct=im_change_pct,
        trades_affected=trades_to_unwind,
        computation_time_ms=computation_time
    )


def whatif_add_hedge(
    portfolio_crif: pd.DataFrame,
    hedge_crif: pd.DataFrame,
    hedge_description: str = "Proposed Hedge"
) -> WhatIfScenarioResult:
    """
    What-if: Add a hedging trade to the portfolio.

    Common use case: "If I add this offsetting swap, how much margin do I save?"

    Args:
        portfolio_crif: Current portfolio CRIF
        hedge_crif: CRIF for the proposed hedge
        hedge_description: Description of the hedge

    Returns:
        WhatIfScenarioResult showing margin impact
    """
    from model.pretrade_analytics import calculate_simm_margin

    start_time = time.perf_counter()

    current_im = calculate_simm_margin(portfolio_crif)

    # Combine portfolio with hedge
    combined_crif = pd.concat([portfolio_crif, hedge_crif], ignore_index=True)

    # Aggregate sensitivities (net them)
    agg_cols = ['RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'AmountCurrency']
    if 'ProductClass' in combined_crif.columns:
        agg_cols = ['ProductClass'] + agg_cols
    combined_crif = combined_crif.groupby(
        agg_cols,
        as_index=False
    ).agg({
        'TradeID': 'first',
        'Amount': 'sum',
        'AmountUSD': 'sum'
    })

    scenario_im = calculate_simm_margin(combined_crif)

    im_change = scenario_im - current_im
    im_change_pct = (im_change / current_im * 100) if current_im > 0 else 0

    computation_time = (time.perf_counter() - start_time) * 1000

    # Get trade IDs from hedge
    hedge_trades = hedge_crif['TradeID'].unique().tolist()

    return WhatIfScenarioResult(
        scenario_name=f"Add Hedge: {hedge_description}",
        description=f"Add {len(hedge_trades)} hedging trade(s) to reduce margin",
        current_im=current_im,
        scenario_im=scenario_im,
        im_change=im_change,
        im_change_pct=im_change_pct,
        trades_affected=hedge_trades,
        computation_time_ms=computation_time
    )


def whatif_stress_scenario(
    portfolio_crif: pd.DataFrame,
    shock_factors: Dict[str, float],
    scenario_name: str = "Custom Stress"
) -> WhatIfScenarioResult:
    """
    What-if: Apply stress shocks to sensitivities.

    Common use case: "What happens to my margin if rates spike 100bp?"

    Args:
        portfolio_crif: Current portfolio CRIF
        shock_factors: Dict of RiskType -> multiplicative shock
                       e.g., {"Risk_IRCurve": 1.5} means 50% increase
        scenario_name: Name for this scenario

    Returns:
        WhatIfScenarioResult with stressed margin
    """
    from model.pretrade_analytics import calculate_simm_margin

    start_time = time.perf_counter()

    current_im = calculate_simm_margin(portfolio_crif)

    # Apply shocks
    stressed_crif = portfolio_crif.copy()

    for risk_type, shock in shock_factors.items():
        mask = stressed_crif['RiskType'] == risk_type
        if mask.any():
            stressed_crif.loc[mask, 'Amount'] *= shock
            stressed_crif.loc[mask, 'AmountUSD'] *= shock

    scenario_im = calculate_simm_margin(stressed_crif)

    im_change = scenario_im - current_im
    im_change_pct = (im_change / current_im * 100) if current_im > 0 else 0

    computation_time = (time.perf_counter() - start_time) * 1000

    return WhatIfScenarioResult(
        scenario_name=scenario_name,
        description=f"Apply shocks: {shock_factors}",
        current_im=current_im,
        scenario_im=scenario_im,
        im_change=im_change,
        im_change_pct=im_change_pct,
        trades_affected=[],  # Affects all trades
        computation_time_ms=computation_time
    )


# =============================================================================
# Reporting
# =============================================================================

def print_attribution_report(report: MarginAttributionReport, top_n: int = 10):
    """Print formatted margin attribution report."""
    print("\n" + "=" * 80)
    print("                      MARGIN ATTRIBUTION REPORT")
    print("=" * 80)

    print(f"\nPortfolio Summary:")
    print(f"  Total IM:          ${report.total_im:>15,.0f}")
    print(f"  Number of trades:  {report.num_trades:>15,}")
    print(f"  Risk factors:      {report.num_risk_factors:>15,}")

    print(f"\nComputation:")
    print(f"  Method:            {report.computation_method}")
    print(f"  Time:              {report.computation_time_ms:>10.1f} ms")
    print(f"  Naive estimate:    {report.naive_time_estimate_ms:>10,.0f} ms")
    if report.naive_time_estimate_ms > 0:
        speedup = report.naive_time_estimate_ms / max(report.computation_time_ms, 1)
        print(f"  Speedup:           {speedup:>10.0f}x")

    # Top margin consumers
    print("\n" + "-" * 80)
    print(f"TOP {top_n} MARGIN CONSUMERS (trades adding to margin)")
    print("-" * 80)
    print(f"{'Trade ID':<15} {'Contribution':>18} {'% of Total':>12} {'Net Sens':>18}")
    print("-" * 80)

    consumers = [c for c in report.trade_contributions if c.is_margin_additive][:top_n]
    for c in consumers:
        print(f"{c.trade_id:<15} ${c.marginal_contribution:>17,.0f} {c.contribution_pct:>11.1f}% ${c.net_sensitivity:>17,.0f}")

    # Top margin reducers (netting benefit)
    reducers = [c for c in report.trade_contributions if not c.is_margin_additive][:top_n]
    if reducers:
        print("\n" + "-" * 80)
        print(f"TOP {min(top_n, len(reducers))} MARGIN REDUCERS (trades providing netting benefit)")
        print("-" * 80)
        print(f"{'Trade ID':<15} {'Contribution':>18} {'% of Total':>12} {'Net Sens':>18}")
        print("-" * 80)

        for c in reducers:
            print(f"{c.trade_id:<15} ${c.marginal_contribution:>17,.0f} {c.contribution_pct:>11.1f}% ${c.net_sensitivity:>17,.0f}")

    # Summary
    total_additive = sum(c.marginal_contribution for c in report.trade_contributions if c.is_margin_additive)
    total_reducing = sum(c.marginal_contribution for c in report.trade_contributions if not c.is_margin_additive)

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"  Trades adding margin:     {len(consumers):>5} trades, ${total_additive:>15,.0f}")
    print(f"  Trades reducing margin:   {len(reducers):>5} trades, ${total_reducing:>15,.0f}")
    print(f"  Net (should ≈ Total IM):          ${total_additive + total_reducing:>15,.0f}")
    print("=" * 80)


def print_whatif_result(result: WhatIfScenarioResult):
    """Print formatted what-if result."""
    print("\n" + "-" * 60)
    print(f"WHAT-IF: {result.scenario_name}")
    print("-" * 60)
    print(f"  {result.description}")
    print()
    print(f"  Current IM:    ${result.current_im:>15,.0f}")
    print(f"  Scenario IM:   ${result.scenario_im:>15,.0f}")
    print(f"  Change:        ${result.im_change:>15,.0f} ({result.im_change_pct:+.1f}%)")
    if result.trades_affected:
        print(f"  Trades:        {', '.join(result.trades_affected[:5])}")
        if len(result.trades_affected) > 5:
            print(f"                 ... and {len(result.trades_affected) - 5} more")
    print(f"  Compute time:  {result.computation_time_ms:.1f} ms")
    print("-" * 60)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    """
    Demo: Margin Attribution and What-If Analysis

    Shows:
    1. Fast AADC-based margin attribution (vs slow naive method)
    2. What-if scenarios: unwind, hedge, stress
    """
    print("=" * 80)
    print("       WHAT-IF ANALYTICS DEMO")
    print("       Margin Attribution & Scenario Analysis")
    print("=" * 80)

    # =========================================================================
    # Setup: Generate test portfolio
    # =========================================================================

    try:
        from model.trade_types import generate_market_environment, generate_trades_by_type
        from model.simm_portfolio_aadc import compute_crif_aadc

        print("\nGenerating test portfolio...")

        currencies = ['USD', 'EUR', 'GBP']
        market = generate_market_environment(currencies, seed=42)

        # Generate 50 trades (enough to show speedup, not too slow for demo)
        trades = generate_trades_by_type('ir_swap', 50, currencies, seed=42)

        # Compute CRIF for each trade individually
        trade_sensitivities = {}
        all_crif_rows = []

        for trade in trades:
            trade_crif, _, _ = compute_crif_aadc([trade], market, num_threads=4)
            trade_sensitivities[trade.trade_id] = trade_crif
            all_crif_rows.append(trade_crif)

        # Aggregate into portfolio CRIF
        portfolio_crif = pd.concat(all_crif_rows, ignore_index=True)
        portfolio_crif = portfolio_crif.groupby(
            ['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'AmountCurrency'],
            as_index=False
        ).agg({
            'TradeID': lambda x: ','.join(x.unique()),
            'Amount': 'sum',
            'AmountUSD': 'sum'
        })

        print(f"  Trades: {len(trades)}")
        print(f"  Risk factors: {len(portfolio_crif)}")

        # =====================================================================
        # Demo 1: Margin Attribution (AADC vs Naive)
        # =====================================================================

        print("\n" + "=" * 80)
        print("DEMO 1: MARGIN ATTRIBUTION")
        print("Which trades are consuming the most margin?")
        print("=" * 80)

        # AADC method (fast)
        print("\n>>> Running AADC gradient method...")
        attribution_aadc = compute_margin_attribution_aadc(
            portfolio_crif,
            trade_sensitivities,
            num_threads=4
        )
        print_attribution_report(attribution_aadc)

        # Naive method (slow) - only run for small portfolios
        if len(trades) <= 20:
            print("\n>>> Running naive leave-one-out method (for comparison)...")
            attribution_naive = compute_margin_attribution_naive(
                portfolio_crif,
                trade_sensitivities
            )
            print(f"\nNaive method took: {attribution_naive.computation_time_ms:.0f} ms")
            print(f"AADC method took:  {attribution_aadc.computation_time_ms:.0f} ms")
            speedup = attribution_naive.computation_time_ms / max(attribution_aadc.computation_time_ms, 1)
            print(f"Speedup: {speedup:.1f}x")
        else:
            print(f"\n(Skipping naive method for {len(trades)} trades - would take ~{len(trades)*100/1000:.0f} seconds)")

        # =====================================================================
        # Demo 2: What-If Scenarios
        # =====================================================================

        print("\n" + "=" * 80)
        print("DEMO 2: WHAT-IF SCENARIOS")
        print("=" * 80)

        # Scenario 1: Unwind top 5 contributors
        print("\n>>> Scenario: Unwind top 5 margin consumers")
        unwind_result = whatif_unwind_top_contributors(
            portfolio_crif,
            trade_sensitivities,
            attribution_aadc,
            num_to_unwind=5
        )
        print_whatif_result(unwind_result)

        # Scenario 2: Add a hedge (opposite direction swap)
        print("\n>>> Scenario: Add offsetting hedge")
        if attribution_aadc.top_margin_consumers:
            # Create a hedge that offsets the largest contributor
            top_trade_id = attribution_aadc.top_margin_consumers[0]
            top_trade_crif = trade_sensitivities[top_trade_id].copy()

            # Flip the sign (offsetting position)
            hedge_crif = top_trade_crif.copy()
            hedge_crif['Amount'] *= -1
            hedge_crif['AmountUSD'] *= -1
            hedge_crif['TradeID'] = 'HEDGE_' + top_trade_id

            hedge_result = whatif_add_hedge(
                portfolio_crif,
                hedge_crif,
                f"Offset {top_trade_id}"
            )
            print_whatif_result(hedge_result)
        else:
            print("No margin-additive trades found; skipping hedge scenario")

        # Scenario 3: Stress test (rates +50%)
        print("\n>>> Scenario: Stress test - IR sensitivities +50%")
        stress_result = whatif_stress_scenario(
            portfolio_crif,
            {"Risk_IRCurve": 1.5},
            "IR Stress +50%"
        )
        print_whatif_result(stress_result)

    except ImportError as e:
        print(f"\nNote: Full demo requires AADC. Running simplified demo...")
        print(f"(Error: {e})")

        # Create simple test data
        portfolio_crif = pd.DataFrame({
            'TradeID': ['T1,T2', 'T1,T2', 'T3', 'T3'],
            'RiskType': ['Risk_IRCurve'] * 4,
            'Qualifier': ['USD'] * 4,
            'Bucket': ['9', '10', '9', '10'],
            'Label1': ['10y', '15y', '10y', '15y'],
            'Label2': [''] * 4,
            'Amount': [20e6, 15e6, -10e6, -8e6],
            'AmountCurrency': ['USD'] * 4,
            'AmountUSD': [20e6, 15e6, -10e6, -8e6],
        })

        trade_sensitivities = {
            'T1': pd.DataFrame({
                'TradeID': ['T1', 'T1'],
                'RiskType': ['Risk_IRCurve', 'Risk_IRCurve'],
                'Qualifier': ['USD', 'USD'],
                'Bucket': ['9', '10'],
                'Label1': ['10y', '15y'],
                'Label2': ['', ''],
                'Amount': [12e6, 9e6],
                'AmountCurrency': ['USD', 'USD'],
                'AmountUSD': [12e6, 9e6],
            }),
            'T2': pd.DataFrame({
                'TradeID': ['T2', 'T2'],
                'RiskType': ['Risk_IRCurve', 'Risk_IRCurve'],
                'Qualifier': ['USD', 'USD'],
                'Bucket': ['9', '10'],
                'Label1': ['10y', '15y'],
                'Label2': ['', ''],
                'Amount': [8e6, 6e6],
                'AmountCurrency': ['USD', 'USD'],
                'AmountUSD': [8e6, 6e6],
            }),
            'T3': pd.DataFrame({
                'TradeID': ['T3', 'T3'],
                'RiskType': ['Risk_IRCurve', 'Risk_IRCurve'],
                'Qualifier': ['USD', 'USD'],
                'Bucket': ['9', '10'],
                'Label1': ['10y', '15y'],
                'Label2': ['', ''],
                'Amount': [-10e6, -8e6],
                'AmountCurrency': ['USD', 'USD'],
                'AmountUSD': [-10e6, -8e6],
            }),
        }

        print("\nRunning with simplified test data...")

        # Run naive attribution (AADC not available)
        attribution = compute_margin_attribution_naive(portfolio_crif, trade_sensitivities)
        print_attribution_report(attribution)

        # Stress test
        stress_result = whatif_stress_scenario(
            portfolio_crif,
            {"Risk_IRCurve": 1.5},
            "IR Stress +50%"
        )
        print_whatif_result(stress_result)

    print("\n" + "=" * 80)
    print("                         DEMO COMPLETE")
    print("=" * 80)
