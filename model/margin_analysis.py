"""
Margin Analysis - Stress Margin and Incremental Margin Analysis.

This module provides tools for:
1. Stress margin: Apply shocks to SIMM inputs and recalculate margin
2. Incremental margin: Calculate marginal contribution of trades to portfolio margin

Version: 1.0.0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from copy import deepcopy

import sys
sys.path.insert(0, '.')

from model.ir_swap_common import (
    IRSwap, MarketData, GreeksResult,
    TENOR_LABELS, TENOR_YEARS, NUM_TENORS,
    generate_crif
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ShockScenario:
    """
    Definition of a stress scenario for margin calculation.

    Shocks are applied multiplicatively: shocked_value = base_value * (1 + shock)
    For additive shocks (e.g., parallel shift), use shock_type='additive'.
    """
    name: str
    description: str

    # IR curve shocks by tenor (dict of tenor_label -> shock factor)
    # e.g., {"2y": 0.50, "5y": 0.50, "10y": 0.50} for 50% increase
    ir_shocks: Dict[str, float] = field(default_factory=dict)

    # Global IR shock (applied to all tenors if not overridden)
    ir_parallel_shock: float = 0.0

    # Steepening shock: short end vs long end
    # Positive = steepening (long rates up more than short)
    ir_steepening_shock: float = 0.0

    # Vol shock for vega sensitivities
    vol_shock: float = 0.0

    # FX shock (global)
    fx_shock: float = 0.0

    # Credit spread shock
    credit_shock: float = 0.0

    # Shock type: 'multiplicative' or 'additive'
    shock_type: str = 'multiplicative'


@dataclass
class StressMarginResult:
    """Result from stress margin calculation."""
    scenario_name: str
    base_margin: float
    stressed_margin: float
    margin_change: float
    margin_change_pct: float

    # Breakdown by risk class (optional)
    base_breakdown: Optional[Dict[str, float]] = None
    stressed_breakdown: Optional[Dict[str, float]] = None


@dataclass
class IncrementalMarginResult:
    """Result from incremental margin calculation."""
    trade_id: str
    portfolio_margin: float
    margin_without_trade: float
    incremental_margin: float
    incremental_margin_pct: float

    # Whether adding this trade increases or decreases margin
    is_margin_additive: bool = True


@dataclass
class WhatIfResult:
    """Result from what-if analysis (adding new trades)."""
    scenario_name: str
    current_margin: float
    new_margin: float
    margin_change: float
    margin_change_pct: float
    trades_added: List[str]
    trades_removed: List[str] = field(default_factory=list)


# =============================================================================
# Predefined Stress Scenarios
# =============================================================================

STANDARD_SCENARIOS = {
    "parallel_up_100bp": ShockScenario(
        name="parallel_up_100bp",
        description="Parallel rate shift +100bp",
        ir_parallel_shock=1.0,  # 100% increase to sensitivities
        shock_type='multiplicative'
    ),
    "parallel_down_100bp": ShockScenario(
        name="parallel_down_100bp",
        description="Parallel rate shift -100bp",
        ir_parallel_shock=-0.5,  # 50% decrease (sensitivities halved)
        shock_type='multiplicative'
    ),
    "steepener_50bp": ShockScenario(
        name="steepener_50bp",
        description="Curve steepening: 2y -25bp, 10y +25bp, 30y +50bp",
        ir_shocks={
            "2w": -0.25, "1m": -0.25, "3m": -0.20, "6m": -0.15,
            "1y": -0.10, "2y": -0.05, "3y": 0.0, "5y": 0.10,
            "10y": 0.25, "15y": 0.35, "20y": 0.45, "30y": 0.50
        },
        shock_type='multiplicative'
    ),
    "flattener_50bp": ShockScenario(
        name="flattener_50bp",
        description="Curve flattening: 2y +25bp, 10y -25bp, 30y -50bp",
        ir_shocks={
            "2w": 0.25, "1m": 0.25, "3m": 0.20, "6m": 0.15,
            "1y": 0.10, "2y": 0.05, "3y": 0.0, "5y": -0.10,
            "10y": -0.25, "15y": -0.35, "20y": -0.45, "30y": -0.50
        },
        shock_type='multiplicative'
    ),
    "vol_up_25pct": ShockScenario(
        name="vol_up_25pct",
        description="Volatility shock +25%",
        vol_shock=0.25,
        shock_type='multiplicative'
    ),
    "credit_widen_50pct": ShockScenario(
        name="credit_widen_50pct",
        description="Credit spreads widen 50%",
        credit_shock=0.50,
        shock_type='multiplicative'
    ),
    "crisis_scenario": ShockScenario(
        name="crisis_scenario",
        description="Crisis: rates +200bp, vol +50%, credit +100%",
        ir_parallel_shock=2.0,
        vol_shock=0.50,
        credit_shock=1.0,
        shock_type='multiplicative'
    ),
}


# =============================================================================
# Stress Margin Functions
# =============================================================================

def apply_shocks_to_crif(
    crif: pd.DataFrame,
    scenario: ShockScenario
) -> pd.DataFrame:
    """
    Apply shock scenario to CRIF sensitivities.

    Args:
        crif: CRIF DataFrame with sensitivities
        scenario: ShockScenario to apply

    Returns:
        Shocked CRIF DataFrame
    """
    shocked_crif = crif.copy()

    # Apply IR curve shocks
    ir_mask = shocked_crif['RiskType'].isin(['Risk_IRCurve', 'Risk_Inflation', 'Risk_XCcyBasis'])

    if ir_mask.any():
        for idx in shocked_crif[ir_mask].index:
            tenor = shocked_crif.loc[idx, 'Label1']

            # Get shock for this tenor
            if tenor in scenario.ir_shocks:
                shock = scenario.ir_shocks[tenor]
            else:
                shock = scenario.ir_parallel_shock

            # Apply shock
            if scenario.shock_type == 'multiplicative':
                shocked_crif.loc[idx, 'Amount'] *= (1 + shock)
                shocked_crif.loc[idx, 'AmountUSD'] *= (1 + shock)
            else:  # additive
                shocked_crif.loc[idx, 'Amount'] += shock
                shocked_crif.loc[idx, 'AmountUSD'] += shock

    # Apply vol shocks
    vol_mask = shocked_crif['RiskType'].isin(['Risk_IRVol', 'Risk_InflationVol', 'Risk_FXVol',
                                               'Risk_EquityVol', 'Risk_CommodityVol',
                                               'Risk_CreditVol', 'Risk_CreditVolNonQ'])
    if vol_mask.any() and scenario.vol_shock != 0:
        if scenario.shock_type == 'multiplicative':
            shocked_crif.loc[vol_mask, 'Amount'] *= (1 + scenario.vol_shock)
            shocked_crif.loc[vol_mask, 'AmountUSD'] *= (1 + scenario.vol_shock)
        else:
            shocked_crif.loc[vol_mask, 'Amount'] += scenario.vol_shock
            shocked_crif.loc[vol_mask, 'AmountUSD'] += scenario.vol_shock

    # Apply FX shocks
    fx_mask = shocked_crif['RiskType'] == 'Risk_FX'
    if fx_mask.any() and scenario.fx_shock != 0:
        if scenario.shock_type == 'multiplicative':
            shocked_crif.loc[fx_mask, 'Amount'] *= (1 + scenario.fx_shock)
            shocked_crif.loc[fx_mask, 'AmountUSD'] *= (1 + scenario.fx_shock)
        else:
            shocked_crif.loc[fx_mask, 'Amount'] += scenario.fx_shock
            shocked_crif.loc[fx_mask, 'AmountUSD'] += scenario.fx_shock

    # Apply credit shocks
    credit_mask = shocked_crif['RiskType'].isin(['Risk_CreditQ', 'Risk_CreditNonQ'])
    if credit_mask.any() and scenario.credit_shock != 0:
        if scenario.shock_type == 'multiplicative':
            shocked_crif.loc[credit_mask, 'Amount'] *= (1 + scenario.credit_shock)
            shocked_crif.loc[credit_mask, 'AmountUSD'] *= (1 + scenario.credit_shock)
        else:
            shocked_crif.loc[credit_mask, 'Amount'] += scenario.credit_shock
            shocked_crif.loc[credit_mask, 'AmountUSD'] += scenario.credit_shock

    return shocked_crif


def calculate_simm_margin(crif: pd.DataFrame, calculation_currency: str = 'USD') -> float:
    """
    Calculate SIMM margin from CRIF.

    This is a wrapper around the existing SIMM calculation.
    """
    try:
        from src.agg_margins import SIMM
        simm = SIMM(crif, calculation_currency, exchange_rate=1.0)
        return simm.simm
    except Exception as e:
        # Fallback: simple sum of weighted sensitivities (for testing)
        # This is NOT a proper SIMM calculation
        total = 0.0
        ir_mask = crif['RiskType'].isin(['Risk_IRCurve', 'Risk_Inflation'])
        if ir_mask.any():
            # Approximate IR margin (sum of absolute sensitivities * risk weight)
            # Risk weights vary by tenor, using average ~0.5%
            total += abs(crif.loc[ir_mask, 'AmountUSD'].sum()) * 0.005
        return total


def compute_stress_margin(
    crif: pd.DataFrame,
    scenario: Union[ShockScenario, str],
    calculation_currency: str = 'USD'
) -> StressMarginResult:
    """
    Compute stressed SIMM margin for a given scenario.

    Args:
        crif: CRIF DataFrame with base sensitivities
        scenario: ShockScenario or name of predefined scenario
        calculation_currency: Currency for margin calculation

    Returns:
        StressMarginResult with base and stressed margin
    """
    # Resolve scenario name to object
    if isinstance(scenario, str):
        if scenario not in STANDARD_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(STANDARD_SCENARIOS.keys())}")
        scenario = STANDARD_SCENARIOS[scenario]

    # Calculate base margin
    base_margin = calculate_simm_margin(crif, calculation_currency)

    # Apply shocks
    shocked_crif = apply_shocks_to_crif(crif, scenario)

    # Calculate stressed margin
    stressed_margin = calculate_simm_margin(shocked_crif, calculation_currency)

    # Calculate change
    margin_change = stressed_margin - base_margin
    margin_change_pct = (margin_change / base_margin * 100) if base_margin != 0 else 0.0

    return StressMarginResult(
        scenario_name=scenario.name,
        base_margin=base_margin,
        stressed_margin=stressed_margin,
        margin_change=margin_change,
        margin_change_pct=margin_change_pct
    )


def compute_stress_margin_suite(
    crif: pd.DataFrame,
    scenarios: List[Union[ShockScenario, str]] = None,
    calculation_currency: str = 'USD'
) -> List[StressMarginResult]:
    """
    Compute stressed margin for multiple scenarios.

    Args:
        crif: CRIF DataFrame with base sensitivities
        scenarios: List of scenarios (defaults to all standard scenarios)
        calculation_currency: Currency for margin calculation

    Returns:
        List of StressMarginResult
    """
    if scenarios is None:
        scenarios = list(STANDARD_SCENARIOS.keys())

    results = []
    for scenario in scenarios:
        result = compute_stress_margin(crif, scenario, calculation_currency)
        results.append(result)

    return results


# =============================================================================
# Incremental Margin Functions
# =============================================================================

def compute_incremental_margin(
    trades: List[IRSwap],
    greeks_result: GreeksResult,
    trade_idx: int,
    calculation_currency: str = 'USD'
) -> IncrementalMarginResult:
    """
    Compute incremental margin contribution of a single trade.

    Incremental margin = Portfolio margin - Margin without this trade

    Args:
        trades: List of all trades
        greeks_result: GreeksResult for all trades
        trade_idx: Index of trade to analyze
        calculation_currency: Currency for margin calculation

    Returns:
        IncrementalMarginResult
    """
    # Generate full portfolio CRIF
    full_crif = generate_crif(trades, greeks_result)

    # Calculate full portfolio margin
    portfolio_margin = calculate_simm_margin(full_crif, calculation_currency)

    # Create CRIF without the specified trade
    trade_id = trades[trade_idx].trade_id

    # Filter out the trade by creating a modified greeks result
    # We zero out the deltas for this trade
    modified_deltas = greeks_result.ir_delta.copy()
    modified_deltas[trade_idx, :, :] = 0.0

    modified_greeks = GreeksResult(
        prices=greeks_result.prices.copy(),
        ir_delta=modified_deltas,
        currencies=greeks_result.currencies,
        tenor_labels=greeks_result.tenor_labels,
        eval_time=0.0,
        first_run_time=0.0,
        steady_state_time=0.0,
        recording_time=0.0,
        num_evals=0,
        num_sensitivities=0,
        num_bumps=0
    )

    # Generate CRIF without the trade
    crif_without = generate_crif(trades, modified_greeks)

    # Calculate margin without trade
    margin_without = calculate_simm_margin(crif_without, calculation_currency)

    # Calculate incremental margin
    incremental_margin = portfolio_margin - margin_without
    incremental_margin_pct = (incremental_margin / portfolio_margin * 100) if portfolio_margin != 0 else 0.0

    return IncrementalMarginResult(
        trade_id=trade_id,
        portfolio_margin=portfolio_margin,
        margin_without_trade=margin_without,
        incremental_margin=incremental_margin,
        incremental_margin_pct=incremental_margin_pct,
        is_margin_additive=incremental_margin > 0
    )


def compute_all_incremental_margins(
    trades: List[IRSwap],
    greeks_result: GreeksResult,
    calculation_currency: str = 'USD',
    top_n: int = None
) -> List[IncrementalMarginResult]:
    """
    Compute incremental margin for all trades in portfolio.

    Args:
        trades: List of all trades
        greeks_result: GreeksResult for all trades
        calculation_currency: Currency for margin calculation
        top_n: Return only top N contributors (by absolute incremental margin)

    Returns:
        List of IncrementalMarginResult, sorted by absolute incremental margin
    """
    results = []

    # Generate full CRIF once
    full_crif = generate_crif(trades, greeks_result)
    portfolio_margin = calculate_simm_margin(full_crif, calculation_currency)

    for i, trade in enumerate(trades):
        # Create modified deltas (zero out this trade)
        modified_deltas = greeks_result.ir_delta.copy()
        modified_deltas[i, :, :] = 0.0

        modified_greeks = GreeksResult(
            prices=greeks_result.prices.copy(),
            ir_delta=modified_deltas,
            currencies=greeks_result.currencies,
            tenor_labels=greeks_result.tenor_labels,
            eval_time=0.0,
            first_run_time=0.0,
            steady_state_time=0.0,
            recording_time=0.0,
            num_evals=0,
            num_sensitivities=0,
            num_bumps=0
        )

        crif_without = generate_crif(trades, modified_greeks)
        margin_without = calculate_simm_margin(crif_without, calculation_currency)

        incremental_margin = portfolio_margin - margin_without
        incremental_margin_pct = (incremental_margin / portfolio_margin * 100) if portfolio_margin != 0 else 0.0

        results.append(IncrementalMarginResult(
            trade_id=trade.trade_id,
            portfolio_margin=portfolio_margin,
            margin_without_trade=margin_without,
            incremental_margin=incremental_margin,
            incremental_margin_pct=incremental_margin_pct,
            is_margin_additive=incremental_margin > 0
        ))

    # Sort by absolute incremental margin (largest first)
    results.sort(key=lambda x: abs(x.incremental_margin), reverse=True)

    if top_n is not None:
        results = results[:top_n]

    return results


# =============================================================================
# What-If Analysis Functions
# =============================================================================

def compute_whatif_add_trades(
    existing_trades: List[IRSwap],
    existing_greeks: GreeksResult,
    new_trades: List[IRSwap],
    new_greeks: GreeksResult,
    calculation_currency: str = 'USD'
) -> WhatIfResult:
    """
    Compute margin impact of adding new trades to portfolio.

    Args:
        existing_trades: Current portfolio trades
        existing_greeks: GreeksResult for current portfolio
        new_trades: Trades to add
        new_greeks: GreeksResult for new trades
        calculation_currency: Currency for margin calculation

    Returns:
        WhatIfResult with margin impact
    """
    # Current portfolio margin
    current_crif = generate_crif(existing_trades, existing_greeks)
    current_margin = calculate_simm_margin(current_crif, calculation_currency)

    # Combined portfolio
    # Combine deltas (need to align currencies)
    all_currencies = list(set(existing_greeks.currencies) | set(new_greeks.currencies))

    # For simplicity, concatenate CRIFs directly
    new_crif = generate_crif(new_trades, new_greeks)
    combined_crif = pd.concat([current_crif, new_crif], ignore_index=True)

    new_margin = calculate_simm_margin(combined_crif, calculation_currency)

    margin_change = new_margin - current_margin
    margin_change_pct = (margin_change / current_margin * 100) if current_margin != 0 else 0.0

    return WhatIfResult(
        scenario_name=f"Add {len(new_trades)} trades",
        current_margin=current_margin,
        new_margin=new_margin,
        margin_change=margin_change,
        margin_change_pct=margin_change_pct,
        trades_added=[t.trade_id for t in new_trades]
    )


def compute_whatif_remove_trades(
    trades: List[IRSwap],
    greeks_result: GreeksResult,
    trade_ids_to_remove: List[str],
    calculation_currency: str = 'USD'
) -> WhatIfResult:
    """
    Compute margin impact of removing trades from portfolio.

    Args:
        trades: Current portfolio trades
        greeks_result: GreeksResult for current portfolio
        trade_ids_to_remove: Trade IDs to remove
        calculation_currency: Currency for margin calculation

    Returns:
        WhatIfResult with margin impact
    """
    # Current portfolio margin
    current_crif = generate_crif(trades, greeks_result)
    current_margin = calculate_simm_margin(current_crif, calculation_currency)

    # Find indices of trades to remove
    trade_id_to_idx = {t.trade_id: i for i, t in enumerate(trades)}
    remove_indices = set()
    for tid in trade_ids_to_remove:
        if tid in trade_id_to_idx:
            remove_indices.add(trade_id_to_idx[tid])

    # Create modified deltas (zero out removed trades)
    modified_deltas = greeks_result.ir_delta.copy()
    for idx in remove_indices:
        modified_deltas[idx, :, :] = 0.0

    modified_greeks = GreeksResult(
        prices=greeks_result.prices.copy(),
        ir_delta=modified_deltas,
        currencies=greeks_result.currencies,
        tenor_labels=greeks_result.tenor_labels,
        eval_time=0.0,
        first_run_time=0.0,
        steady_state_time=0.0,
        recording_time=0.0,
        num_evals=0,
        num_sensitivities=0,
        num_bumps=0
    )

    new_crif = generate_crif(trades, modified_greeks)
    new_margin = calculate_simm_margin(new_crif, calculation_currency)

    margin_change = new_margin - current_margin
    margin_change_pct = (margin_change / current_margin * 100) if current_margin != 0 else 0.0

    return WhatIfResult(
        scenario_name=f"Remove {len(trade_ids_to_remove)} trades",
        current_margin=current_margin,
        new_margin=new_margin,
        margin_change=margin_change,
        margin_change_pct=margin_change_pct,
        trades_added=[],
        trades_removed=trade_ids_to_remove
    )


# =============================================================================
# Reporting Functions
# =============================================================================

def print_stress_report(results: List[StressMarginResult]):
    """Print formatted stress margin report."""
    print("\n" + "=" * 70)
    print("                    STRESS MARGIN REPORT")
    print("=" * 70)

    if not results:
        print("No results to display.")
        return

    base_margin = results[0].base_margin
    print(f"\nBase Margin: ${base_margin:,.0f}")
    print("\n" + "-" * 70)
    print(f"{'Scenario':<25} {'Stressed':>15} {'Change':>15} {'Change %':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r.scenario_name:<25} ${r.stressed_margin:>14,.0f} ${r.margin_change:>14,.0f} {r.margin_change_pct:>9.1f}%")

    print("=" * 70)


def print_incremental_report(results: List[IncrementalMarginResult], top_n: int = 10):
    """Print formatted incremental margin report."""
    print("\n" + "=" * 70)
    print("                  INCREMENTAL MARGIN REPORT")
    print("=" * 70)

    if not results:
        print("No results to display.")
        return

    portfolio_margin = results[0].portfolio_margin
    print(f"\nPortfolio Margin: ${portfolio_margin:,.0f}")
    print(f"\nTop {min(top_n, len(results))} margin contributors:")
    print("\n" + "-" * 70)
    print(f"{'Trade ID':<15} {'Incr. Margin':>15} {'Margin %':>10} {'Direction':>12}")
    print("-" * 70)

    for r in results[:top_n]:
        direction = "Additive" if r.is_margin_additive else "Offsetting"
        print(f"{r.trade_id:<15} ${r.incremental_margin:>14,.0f} {r.incremental_margin_pct:>9.1f}% {direction:>12}")

    # Summary stats
    additive = sum(1 for r in results if r.is_margin_additive)
    offsetting = len(results) - additive

    print("-" * 70)
    print(f"\nSummary: {additive} additive trades, {offsetting} offsetting trades")
    print("=" * 70)


# =============================================================================
# Demo / Testing
# =============================================================================

if __name__ == "__main__":
    from model.ir_swap_common import generate_trades, generate_market_data
    from model.ir_swap_aadc import price_with_greeks, AADC_AVAILABLE
    from model.ir_swap_pricer import price_with_greeks as baseline_greeks

    print("=" * 70)
    print("       Margin Analysis Demo - Stress & Incremental Margin")
    print("=" * 70)

    # Generate test portfolio
    currencies = ['USD', 'EUR', 'GBP']
    trades = generate_trades(50, currencies, seed=42)
    market_data = {ccy: generate_market_data(ccy, seed=42+i) for i, ccy in enumerate(currencies)}

    # Price with greeks
    print("\nPricing portfolio...")
    if AADC_AVAILABLE:
        greeks = price_with_greeks(trades, market_data)
    else:
        greeks = baseline_greeks(trades, market_data)

    # Generate CRIF
    crif = generate_crif(trades, greeks)
    print(f"Generated CRIF with {len(crif)} sensitivities")

    # === Stress Margin Demo ===
    print("\n" + "=" * 70)
    print("STRESS MARGIN ANALYSIS")
    print("=" * 70)

    stress_results = compute_stress_margin_suite(crif)
    print_stress_report(stress_results)

    # === Incremental Margin Demo ===
    print("\n" + "=" * 70)
    print("INCREMENTAL MARGIN ANALYSIS")
    print("=" * 70)

    incr_results = compute_all_incremental_margins(trades, greeks, top_n=10)
    print_incremental_report(incr_results)

    # === What-If Demo ===
    print("\n" + "=" * 70)
    print("WHAT-IF ANALYSIS: Remove Top 3 Contributors")
    print("=" * 70)

    top_3_ids = [r.trade_id for r in incr_results[:3]]
    whatif_result = compute_whatif_remove_trades(trades, greeks, top_3_ids)

    print(f"\nScenario: {whatif_result.scenario_name}")
    print(f"Current Margin:  ${whatif_result.current_margin:,.0f}")
    print(f"New Margin:      ${whatif_result.new_margin:,.0f}")
    print(f"Margin Change:   ${whatif_result.margin_change:,.0f} ({whatif_result.margin_change_pct:.1f}%)")
    print(f"Trades Removed:  {whatif_result.trades_removed}")
