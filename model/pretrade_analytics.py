"""
Pre-Trade Analytics Module - Marginal IM and Bilateral vs Cleared Analysis.

This module provides real-time decision support for trade routing:

1. MARGINAL IM CALCULATOR
   - Given existing portfolios at multiple counterparties
   - Compute marginal IM impact of a proposed new trade at each counterparty
   - Uses AADC gradients for O(1) computation instead of O(N) full recalculation

2. BILATERAL VS CLEARED COMPARISON
   - Compare ISDA SIMM (bilateral) vs CCP margin (cleared)
   - Help decide: "Should I clear this trade or keep it bilateral?"

Production Use Case:
    Trader wants to execute a $500M 10Y EUR swap.
    Question: Route to Goldman (bilateral), JPM (bilateral), or LCH (cleared)?

    This module answers that in milliseconds using pre-computed gradients.

Version: 1.0.0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from enum import Enum
import time

import sys
sys.path.insert(0, '.')


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class MarginalIMResult:
    """
    Result of marginal IM calculation for a single counterparty.

    Marginal IM = IM impact of adding a new trade to an existing portfolio.
    This is NOT the same as standalone IM (trade in isolation).

    Due to netting effects:
    - Marginal IM can be NEGATIVE (trade offsets existing risk)
    - Marginal IM is typically < standalone IM (diversification benefit)
    - Marginal IM depends on what's already in the portfolio
    """
    counterparty: str

    # Current state (before adding new trade)
    current_im: float
    current_trade_count: int

    # Impact of new trade
    marginal_im: float              # IM change from adding this trade
    new_im: float                   # Total IM after adding trade

    # Comparison metrics
    standalone_im: float            # IM if trade was alone (no netting)
    netting_benefit: float          # standalone_im - marginal_im (positive = good)
    netting_benefit_pct: float      # netting_benefit / standalone_im * 100

    # Computation method
    used_gradient: bool = True      # True = fast gradient method, False = full recalc


@dataclass
class CounterpartyRoutingRecommendation:
    """
    Recommendation for where to route a new trade.

    Compares marginal IM across all available counterparties
    and recommends the one that minimizes total margin.
    """
    # The new trade being analyzed
    trade_description: str

    # Results per counterparty
    counterparty_results: List[MarginalIMResult]

    # Recommendation
    recommended_counterparty: str
    recommended_marginal_im: float

    # Comparison to worst option
    worst_counterparty: str
    worst_marginal_im: float
    margin_savings: float           # worst - best

    # Timing
    computation_time_ms: float


@dataclass
class BilateralVsClearedResult:
    """
    Comparison of bilateral (SIMM) vs cleared (CCP) margin.

    Helps answer: "Should I clear this trade or keep it bilateral?"

    Typical findings:
    - CCP margin is 30-50% lower than SIMM for vanilla swaps
    - But clearing has other costs (membership fees, default fund)
    - Bilateral allows more netting with existing portfolio
    """
    # Trade being analyzed
    trade_description: str

    # Bilateral (ISDA SIMM)
    bilateral_im: float
    bilateral_counterparty: str
    bilateral_marginal_im: float    # If adding to existing bilateral portfolio

    # Cleared (CCP)
    cleared_im: float
    clearing_venue: str             # LCH, CME, Eurex, etc.
    cleared_marginal_im: float      # If adding to existing cleared portfolio

    # Comparison
    im_difference: float            # bilateral - cleared (positive = clearing cheaper)
    im_difference_pct: float

    # Recommendation
    recommendation: str             # "CLEAR", "BILATERAL", or "INDIFFERENT"
    rationale: str


class ClearingVenue(Enum):
    """Supported clearing venues with their characteristics."""
    LCH = "LCH"
    CME = "CME"
    EUREX = "EUREX"
    JSCC = "JSCC"


# =============================================================================
# CCP Margin Models (Simplified SPAN-like calculations)
# =============================================================================

# CCP-specific parameters (simplified for PoC)
# In production, these would come from actual CCP margin methodology docs
CCP_PARAMS = {
    ClearingVenue.LCH: {
        "name": "LCH SwapClear",
        "ir_risk_weight": 0.0018,       # ~18bp per $1M DV01
        "holding_period_days": 5,        # 5-day VaR
        "confidence_level": 0.99,
        "addon_factor": 1.2,             # Add-on for model risk
        "min_margin_usd": 10000,         # Minimum margin
        "supported_currencies": ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD"],
    },
    ClearingVenue.CME: {
        "name": "CME Clearing",
        "ir_risk_weight": 0.0020,        # Slightly higher than LCH
        "holding_period_days": 5,
        "confidence_level": 0.99,
        "addon_factor": 1.25,
        "min_margin_usd": 10000,
        "supported_currencies": ["USD", "EUR", "GBP", "JPY"],
    },
    ClearingVenue.EUREX: {
        "name": "Eurex Clearing",
        "ir_risk_weight": 0.0019,
        "holding_period_days": 5,
        "confidence_level": 0.99,
        "addon_factor": 1.2,
        "min_margin_usd": 10000,
        "supported_currencies": ["EUR", "USD", "GBP", "CHF"],
    },
}

# SIMM vs CCP: Why CCP margin is typically lower
# ================================================
# 1. SIMM uses 10-day holding period, CCP uses 5-day
#    → CCP margin ~30% lower just from this
#
# 2. SIMM has conservative correlations (assume less netting)
#    → CCP models actual portfolio-level netting
#
# 3. CCP has daily variation margin, reducing gap risk
#    → SIMM assumes less frequent margining
#
# 4. CCP has mutualized default fund as backstop
#    → Can take more risk in IM calculation


def calculate_ccp_margin(
    crif: pd.DataFrame,
    venue: ClearingVenue = ClearingVenue.LCH,
    existing_cleared_crif: pd.DataFrame = None
) -> Tuple[float, Dict]:
    """
    Calculate CCP margin for a trade or portfolio.

    This is a SIMPLIFIED model based on public CCP methodology documents.
    Real CCP margin calculations are more complex and proprietary.

    The key insight: CCP margin is simpler than SIMM because:
    - Single netting set (all trades at that CCP)
    - Daily variation margin reduces gap risk
    - 5-day holding period vs SIMM's 10-day
    - More aggressive netting assumptions

    Args:
        crif: CRIF DataFrame with sensitivities for the trade/portfolio
        venue: Which CCP (LCH, CME, EUREX)
        existing_cleared_crif: Optional - existing cleared portfolio at this CCP
                               If provided, calculates marginal margin

    Returns:
        (margin_amount, breakdown_dict)

    Methodology (simplified SPAN-like):
    1. Aggregate net sensitivities by risk factor
    2. Apply CCP-specific risk weights
    3. Compute sqrt(sum of squared weighted sensitivities)
    4. Apply holding period and add-on factors
    """
    params = CCP_PARAMS[venue]
    breakdown = {
        "venue": params["name"],
        "components": {}
    }

    # -------------------------------------------------------------------------
    # Step 1: Combine with existing cleared portfolio (if any)
    # -------------------------------------------------------------------------
    if existing_cleared_crif is not None and len(existing_cleared_crif) > 0:
        combined_crif = pd.concat([existing_cleared_crif, crif], ignore_index=True)
    else:
        combined_crif = crif.copy()

    # -------------------------------------------------------------------------
    # Step 2: Aggregate sensitivities by risk factor
    # -------------------------------------------------------------------------
    # Group by (RiskType, Qualifier, Bucket, Label1) and sum amounts
    # This gives us net sensitivity per risk factor

    if len(combined_crif) == 0:
        return params["min_margin_usd"], breakdown

    agg_sens = combined_crif.groupby(
        ['RiskType', 'Qualifier', 'Bucket', 'Label1'],
        as_index=False
    )['AmountUSD'].sum()

    # -------------------------------------------------------------------------
    # Step 3: Calculate margin by risk type
    # -------------------------------------------------------------------------
    total_margin_squared = 0.0

    # --- Interest Rate Risk ---
    # CCP IR margin: simpler than SIMM, just net DV01 × risk weight
    ir_mask = agg_sens['RiskType'].isin(['Risk_IRCurve', 'Risk_Inflation', 'Risk_XCcyBasis'])
    if ir_mask.any():
        ir_sens = agg_sens.loc[ir_mask, 'AmountUSD'].values

        # Net sensitivity (signed sum - allows offsetting)
        ir_net = np.sum(ir_sens)

        # Gross sensitivity (for diversification penalty)
        ir_gross = np.sum(np.abs(ir_sens))

        # CCP uses blend: mostly net, small gross component
        # This rewards netting but penalizes concentration
        ir_effective = 0.8 * abs(ir_net) + 0.2 * ir_gross

        ir_margin = ir_effective * params["ir_risk_weight"]
        total_margin_squared += ir_margin ** 2

        breakdown["components"]["IR"] = {
            "net_sensitivity": ir_net,
            "gross_sensitivity": ir_gross,
            "effective_sensitivity": ir_effective,
            "margin": ir_margin
        }

    # --- FX Risk ---
    fx_mask = agg_sens['RiskType'] == 'Risk_FX'
    if fx_mask.any():
        fx_sens = agg_sens.loc[fx_mask, 'AmountUSD'].values
        fx_net = np.sum(fx_sens)

        # FX risk weight: ~2% for major pairs, higher for EM
        fx_risk_weight = 0.02
        fx_margin = abs(fx_net) * fx_risk_weight
        total_margin_squared += fx_margin ** 2

        breakdown["components"]["FX"] = {
            "net_sensitivity": fx_net,
            "margin": fx_margin
        }

    # --- Equity Risk ---
    eq_mask = agg_sens['RiskType'] == 'Risk_Equity'
    if eq_mask.any():
        eq_sens = agg_sens.loc[eq_mask, 'AmountUSD'].values
        eq_net = np.sum(eq_sens)

        # Equity risk weight: ~15-25% depending on liquidity
        eq_risk_weight = 0.20
        eq_margin = abs(eq_net) * eq_risk_weight
        total_margin_squared += eq_margin ** 2

        breakdown["components"]["Equity"] = {
            "net_sensitivity": eq_net,
            "margin": eq_margin
        }

    # --- Vega Risk ---
    # CCP vega margin is simpler than SIMM
    vega_mask = agg_sens['RiskType'].str.contains('Vol', na=False)
    if vega_mask.any():
        vega_sens = agg_sens.loc[vega_mask, 'AmountUSD'].values
        vega_net = np.sum(np.abs(vega_sens))  # Vega doesn't net as well

        vega_risk_weight = 0.10  # 10% of vega01
        vega_margin = vega_net * vega_risk_weight
        total_margin_squared += vega_margin ** 2

        breakdown["components"]["Vega"] = {
            "gross_sensitivity": vega_net,
            "margin": vega_margin
        }

    # -------------------------------------------------------------------------
    # Step 4: Combine with sqrt (assumes partial correlation)
    # -------------------------------------------------------------------------
    # Using sqrt aggregation assumes ~0 correlation between risk types
    # This is aggressive but typical for CCPs
    base_margin = np.sqrt(total_margin_squared)

    # -------------------------------------------------------------------------
    # Step 5: Apply add-ons and floors
    # -------------------------------------------------------------------------
    # Add-on for model uncertainty
    margin_with_addon = base_margin * params["addon_factor"]

    # Apply minimum
    final_margin = max(margin_with_addon, params["min_margin_usd"])

    breakdown["base_margin"] = base_margin
    breakdown["addon_factor"] = params["addon_factor"]
    breakdown["final_margin"] = final_margin

    return final_margin, breakdown


# =============================================================================
# SIMM Margin Calculation (wrapper around existing implementation)
# =============================================================================

def calculate_simm_margin(crif: pd.DataFrame, calculation_currency: str = 'USD') -> float:
    """
    Calculate ISDA SIMM margin from CRIF.

    This wraps the existing SIMM implementation in src/agg_margins.py.

    SIMM is more conservative than CCP margin because:
    - 10-day holding period (vs 5-day for CCP)
    - Conservative correlations (less netting benefit assumed)
    - Designed for bilateral relationships (no mutualized default fund)
    """
    try:
        from src.agg_margins import SIMM
        simm = SIMM(crif, calculation_currency, exchange_rate=1.0)
        return simm.simm
    except Exception as e:
        # Fallback: simplified calculation for testing
        # This is NOT a proper SIMM - just for PoC when full SIMM unavailable
        print(f"Warning: Using simplified SIMM proxy (error: {e})")

        total = 0.0

        # IR component
        ir_mask = crif['RiskType'].isin(['Risk_IRCurve', 'Risk_Inflation'])
        if ir_mask.any():
            ir_sens = crif.loc[ir_mask, 'AmountUSD'].values
            # SIMM is more conservative: less netting, higher weights
            ir_net = abs(np.sum(ir_sens))
            ir_gross = np.sum(np.abs(ir_sens))
            # SIMM netting factor ~0.5 (less aggressive than CCP's 0.8)
            ir_effective = 0.5 * ir_net + 0.5 * ir_gross
            total += ir_effective * 0.005  # ~50bp risk weight

        # FX component
        fx_mask = crif['RiskType'] == 'Risk_FX'
        if fx_mask.any():
            fx_sens = abs(crif.loc[fx_mask, 'AmountUSD'].sum())
            total += fx_sens * 0.04  # 4% for FX

        return total


# =============================================================================
# Marginal IM Calculator (the main production use case)
# =============================================================================

def compute_marginal_im_gradient(
    portfolio_crif: pd.DataFrame,
    num_threads: int = 4
) -> Tuple[Dict[str, float], np.ndarray, float]:
    """
    Compute the gradient dIM/dSensitivity for a portfolio.

    This gradient allows O(1) marginal IM calculation for any new trade,
    instead of O(N) full SIMM recalculation.

    The math:
        marginal_IM(new_trade) ≈ Σ_k gradient[k] × new_trade_sensitivity[k]

    This is a first-order approximation that works well when:
    - New trade is small relative to portfolio
    - Portfolio is diversified

    Args:
        portfolio_crif: CRIF DataFrame for existing portfolio
        num_threads: Threads for AADC evaluation

    Returns:
        (gradient_by_factor, gradient_array, current_im)
        - gradient_by_factor: Dict mapping (RiskType, Qualifier, Bucket, Label1) -> gradient
        - gradient_array: Raw gradient array aligned with CRIF rows
        - current_im: Current portfolio IM
    """
    try:
        # Use AADC for exact gradient computation
        from model.simm_portfolio_aadc import record_simm_kernel, compute_im_gradient_aadc
        import aadc

        # Record SIMM kernel
        funcs, sens_handles, im_output, record_time = record_simm_kernel(portfolio_crif)

        # Compute gradient via adjoint
        workers = aadc.ThreadPool(num_threads)
        n = len(portfolio_crif)
        amounts = portfolio_crif["AmountUSD"].values
        inputs = {sens_handles[i]: np.array([float(amounts[i])]) for i in range(n)}

        request = {im_output: sens_handles}
        results = aadc.evaluate(funcs, request, inputs, workers)

        current_im = float(results[0][im_output][0])

        # Extract gradients
        gradient_array = np.zeros(n)
        for i in range(n):
            if sens_handles[i] in results[0]:
                gradient_array[i] = float(results[0][sens_handles[i]][0])

        # Build gradient-by-factor dictionary for easy lookup
        gradient_by_factor = {}
        for i, row in portfolio_crif.iterrows():
            key = (row['RiskType'], row['Qualifier'], row['Bucket'], row['Label1'])
            gradient_by_factor[key] = gradient_array[i]

        return gradient_by_factor, gradient_array, current_im

    except Exception as e:
        # Fallback: numerical gradient (slower but works without AADC)
        print(f"Warning: Using numerical gradient (AADC unavailable: {e})")

        current_im = calculate_simm_margin(portfolio_crif)

        # Compute numerical gradient by bumping each sensitivity
        epsilon = 1e-6
        gradient_array = np.zeros(len(portfolio_crif))

        for i in range(len(portfolio_crif)):
            bumped_crif = portfolio_crif.copy()
            bumped_crif.loc[i, 'AmountUSD'] += epsilon
            bumped_im = calculate_simm_margin(bumped_crif)
            gradient_array[i] = (bumped_im - current_im) / epsilon

        gradient_by_factor = {}
        for i, row in portfolio_crif.iterrows():
            key = (row['RiskType'], row['Qualifier'], row['Bucket'], row['Label1'])
            gradient_by_factor[key] = gradient_array[i]

        return gradient_by_factor, gradient_array, current_im


def compute_marginal_im_fast(
    new_trade_crif: pd.DataFrame,
    portfolio_gradient: Dict[Tuple, float],
    portfolio_current_im: float
) -> float:
    """
    Fast marginal IM calculation using pre-computed gradient.

    This is O(K) where K = number of risk factors in new trade,
    instead of O(N) for full SIMM recalculation.

    For a typical trade with ~20 risk factors, this is ~1000x faster
    than full recalculation for a portfolio with 20,000 risk factors.

    Args:
        new_trade_crif: CRIF for the new trade
        portfolio_gradient: Pre-computed dIM/dSens for portfolio
        portfolio_current_im: Current portfolio IM

    Returns:
        Estimated marginal IM for the new trade

    Note: This is a LINEAR approximation. For very large trades
    relative to portfolio, use full recalculation for accuracy.
    """
    marginal_im = 0.0

    for _, row in new_trade_crif.iterrows():
        key = (row['RiskType'], row['Qualifier'], row['Bucket'], row['Label1'])
        sensitivity = row['AmountUSD']

        # Get gradient for this risk factor
        # If factor not in portfolio, gradient is approximately the risk weight
        if key in portfolio_gradient:
            gradient = portfolio_gradient[key]
        else:
            # New risk factor - use standalone approximation
            # This is the derivative of IM with respect to a new factor
            gradient = estimate_standalone_gradient(row['RiskType'])

        marginal_im += gradient * sensitivity

    return marginal_im


def estimate_standalone_gradient(risk_type: str) -> float:
    """
    Estimate gradient for a risk factor not in the portfolio.

    When adding a completely new risk factor, the marginal IM
    is approximately the standalone IM, which equals:

        risk_weight × sensitivity

    So the gradient (dIM/dSens) ≈ risk_weight
    """
    # SIMM risk weights by risk type (approximate)
    risk_weights = {
        'Risk_IRCurve': 0.005,      # ~50bp for IR
        'Risk_Inflation': 0.005,
        'Risk_XCcyBasis': 0.003,
        'Risk_FX': 0.04,            # ~4% for FX
        'Risk_Equity': 0.20,        # ~20% for equity
        'Risk_Commodity': 0.15,
        'Risk_CreditQ': 0.02,       # ~2% for IG credit
        'Risk_CreditNonQ': 0.05,    # ~5% for HY credit
    }

    # Default for vega and other types
    if 'Vol' in risk_type:
        return 0.10  # 10% for vega

    return risk_weights.get(risk_type, 0.01)


def analyze_trade_routing(
    new_trade_crif: pd.DataFrame,
    counterparty_portfolios: Dict[str, pd.DataFrame],
    counterparty_gradients: Dict[str, Dict] = None,
    num_threads: int = 4,
    use_gradient: bool = True
) -> CounterpartyRoutingRecommendation:
    """
    Analyze which counterparty to route a new trade to.

    This is THE key production function for pre-trade decision support.

    Example usage:
        # Trader wants to execute a 10Y EUR swap
        new_trade = generate_trade_crif(...)

        # Existing portfolios at each counterparty
        portfolios = {
            "Goldman": goldman_crif,
            "JPM": jpm_crif,
            "Citi": citi_crif,
        }

        # Get routing recommendation
        result = analyze_trade_routing(new_trade, portfolios)
        print(f"Route to: {result.recommended_counterparty}")
        print(f"Margin savings vs worst: ${result.margin_savings:,.0f}")

    Args:
        new_trade_crif: CRIF for the proposed new trade
        counterparty_portfolios: Dict of counterparty name -> CRIF DataFrame
        counterparty_gradients: Optional pre-computed gradients (for speed)
        num_threads: Threads for AADC
        use_gradient: If True, use fast gradient method; if False, full recalc

    Returns:
        CounterpartyRoutingRecommendation with analysis and recommendation
    """
    start_time = time.perf_counter()

    results = []

    # Calculate standalone IM (trade in isolation) for comparison
    standalone_im = calculate_simm_margin(new_trade_crif)

    for cp_name, cp_crif in counterparty_portfolios.items():
        # Get or compute gradient
        if use_gradient:
            if counterparty_gradients and cp_name in counterparty_gradients:
                gradient = counterparty_gradients[cp_name]['gradient']
                current_im = counterparty_gradients[cp_name]['current_im']
            else:
                gradient, _, current_im = compute_marginal_im_gradient(cp_crif, num_threads)

            # Fast marginal IM calculation
            marginal_im = compute_marginal_im_fast(new_trade_crif, gradient, current_im)
            new_im = current_im + marginal_im
            used_gradient = True
        else:
            # Full recalculation (slower but exact)
            current_im = calculate_simm_margin(cp_crif)
            combined_crif = pd.concat([cp_crif, new_trade_crif], ignore_index=True)
            new_im = calculate_simm_margin(combined_crif)
            marginal_im = new_im - current_im
            used_gradient = False

        # Calculate netting benefit
        netting_benefit = standalone_im - marginal_im
        netting_benefit_pct = (netting_benefit / standalone_im * 100) if standalone_im > 0 else 0

        results.append(MarginalIMResult(
            counterparty=cp_name,
            current_im=current_im,
            current_trade_count=len(cp_crif),
            marginal_im=marginal_im,
            new_im=new_im,
            standalone_im=standalone_im,
            netting_benefit=netting_benefit,
            netting_benefit_pct=netting_benefit_pct,
            used_gradient=used_gradient
        ))

    # Sort by marginal IM (lowest first)
    results.sort(key=lambda x: x.marginal_im)

    # Build recommendation
    best = results[0]
    worst = results[-1]

    computation_time = (time.perf_counter() - start_time) * 1000

    # Build trade description from CRIF
    trade_types = new_trade_crif['RiskType'].unique().tolist()
    trade_desc = f"Trade with {len(new_trade_crif)} sensitivities ({', '.join(trade_types[:3])})"

    return CounterpartyRoutingRecommendation(
        trade_description=trade_desc,
        counterparty_results=results,
        recommended_counterparty=best.counterparty,
        recommended_marginal_im=best.marginal_im,
        worst_counterparty=worst.counterparty,
        worst_marginal_im=worst.marginal_im,
        margin_savings=worst.marginal_im - best.marginal_im,
        computation_time_ms=computation_time
    )


# =============================================================================
# Bilateral vs Cleared Comparison
# =============================================================================

def compare_bilateral_vs_cleared(
    new_trade_crif: pd.DataFrame,
    bilateral_portfolio: pd.DataFrame = None,
    bilateral_counterparty: str = "Bilateral",
    cleared_portfolio: pd.DataFrame = None,
    clearing_venue: ClearingVenue = ClearingVenue.LCH,
    num_threads: int = 4
) -> BilateralVsClearedResult:
    """
    Compare bilateral (SIMM) vs cleared (CCP) margin for a trade.

    This helps answer the fundamental question:
    "Should I clear this trade or keep it bilateral?"

    Factors that favor CLEARING:
    - Lower margin (CCP typically 30-50% less)
    - Reduced counterparty credit risk
    - Standardized collateral
    - Access to CCP netting pool

    Factors that favor BILATERAL:
    - Better netting with existing bilateral portfolio
    - No CCP membership fees / default fund
    - More flexibility in collateral
    - No novation to CCP required

    Args:
        new_trade_crif: CRIF for the trade being analyzed
        bilateral_portfolio: Existing bilateral portfolio (optional)
        bilateral_counterparty: Name for bilateral option
        cleared_portfolio: Existing cleared portfolio at this CCP (optional)
        clearing_venue: Which CCP to compare
        num_threads: Threads for AADC

    Returns:
        BilateralVsClearedResult with comparison and recommendation
    """
    # -------------------------------------------------------------------------
    # Calculate BILATERAL margin (SIMM)
    # -------------------------------------------------------------------------
    if bilateral_portfolio is not None and len(bilateral_portfolio) > 0:
        # Marginal margin on existing portfolio
        bilateral_current = calculate_simm_margin(bilateral_portfolio)
        combined_bilateral = pd.concat([bilateral_portfolio, new_trade_crif], ignore_index=True)
        bilateral_new = calculate_simm_margin(combined_bilateral)
        bilateral_marginal = bilateral_new - bilateral_current
    else:
        # Standalone margin
        bilateral_current = 0.0
        bilateral_new = calculate_simm_margin(new_trade_crif)
        bilateral_marginal = bilateral_new

    bilateral_im = bilateral_new if bilateral_portfolio is None else bilateral_marginal

    # -------------------------------------------------------------------------
    # Calculate CLEARED margin (CCP)
    # -------------------------------------------------------------------------
    cleared_im, cleared_breakdown = calculate_ccp_margin(
        new_trade_crif,
        venue=clearing_venue,
        existing_cleared_crif=cleared_portfolio
    )

    if cleared_portfolio is not None and len(cleared_portfolio) > 0:
        # Calculate marginal
        cleared_current, _ = calculate_ccp_margin(cleared_portfolio, venue=clearing_venue)
        cleared_marginal = cleared_im - cleared_current
    else:
        cleared_current = 0.0
        cleared_marginal = cleared_im

    # -------------------------------------------------------------------------
    # Compare and recommend
    # -------------------------------------------------------------------------
    im_difference = bilateral_marginal - cleared_marginal
    im_difference_pct = (im_difference / bilateral_marginal * 100) if bilateral_marginal != 0 else 0

    # Decision thresholds
    # Clearing saves money if difference > 10% (to account for other clearing costs)
    CLEARING_THRESHOLD = 0.10
    INDIFFERENCE_BAND = 0.05

    if im_difference / max(bilateral_marginal, 1) > CLEARING_THRESHOLD:
        recommendation = "CLEAR"
        rationale = (
            f"Clearing saves ${im_difference:,.0f} ({im_difference_pct:.1f}%) in margin. "
            f"CCP margin is ${cleared_marginal:,.0f} vs bilateral ${bilateral_marginal:,.0f}. "
            f"Recommend clearing at {clearing_venue.value} unless other factors dominate."
        )
    elif im_difference / max(bilateral_marginal, 1) < -CLEARING_THRESHOLD:
        recommendation = "BILATERAL"
        rationale = (
            f"Bilateral saves ${-im_difference:,.0f} ({-im_difference_pct:.1f}%) due to netting. "
            f"Existing bilateral portfolio provides netting benefit. "
            f"Recommend bilateral with {bilateral_counterparty}."
        )
    else:
        recommendation = "INDIFFERENT"
        rationale = (
            f"Margin difference is small (${abs(im_difference):,.0f}, {abs(im_difference_pct):.1f}%). "
            f"Decision should be based on other factors: credit risk, operational preference, "
            f"existing relationships."
        )

    # Build trade description
    trade_types = new_trade_crif['RiskType'].unique().tolist()
    trade_desc = f"Trade with {len(new_trade_crif)} sensitivities"

    return BilateralVsClearedResult(
        trade_description=trade_desc,
        bilateral_im=bilateral_new,
        bilateral_counterparty=bilateral_counterparty,
        bilateral_marginal_im=bilateral_marginal,
        cleared_im=cleared_im,
        clearing_venue=clearing_venue.value,
        cleared_marginal_im=cleared_marginal,
        im_difference=im_difference,
        im_difference_pct=im_difference_pct,
        recommendation=recommendation,
        rationale=rationale
    )


# =============================================================================
# Reporting Functions
# =============================================================================

def print_routing_recommendation(result: CounterpartyRoutingRecommendation):
    """Print formatted routing recommendation."""
    print("\n" + "=" * 75)
    print("                    PRE-TRADE ROUTING ANALYSIS")
    print("=" * 75)

    print(f"\nTrade: {result.trade_description}")
    print(f"Computation time: {result.computation_time_ms:.1f} ms")

    print("\n" + "-" * 75)
    print(f"{'Counterparty':<20} {'Current IM':>15} {'Marginal IM':>15} {'New IM':>15} {'Netting %':>10}")
    print("-" * 75)

    for r in result.counterparty_results:
        print(f"{r.counterparty:<20} ${r.current_im:>14,.0f} ${r.marginal_im:>14,.0f} ${r.new_im:>14,.0f} {r.netting_benefit_pct:>9.1f}%")

    print("-" * 75)
    print(f"\nStandalone IM (no netting): ${result.counterparty_results[0].standalone_im:,.0f}")

    print("\n" + "=" * 75)
    print("                         RECOMMENDATION")
    print("=" * 75)
    print(f"\n  >>> Route to: {result.recommended_counterparty}")
    print(f"  >>> Marginal IM: ${result.recommended_marginal_im:,.0f}")
    print(f"  >>> Savings vs {result.worst_counterparty}: ${result.margin_savings:,.0f}")
    print("=" * 75)


def print_bilateral_vs_cleared(result: BilateralVsClearedResult):
    """Print formatted bilateral vs cleared comparison."""
    print("\n" + "=" * 75)
    print("                 BILATERAL vs CLEARED COMPARISON")
    print("=" * 75)

    print(f"\nTrade: {result.trade_description}")

    print("\n" + "-" * 75)
    print(f"{'Option':<25} {'Marginal IM':>20} {'Total IM':>20}")
    print("-" * 75)
    print(f"{'Bilateral (' + result.bilateral_counterparty + ')':<25} ${result.bilateral_marginal_im:>19,.0f} ${result.bilateral_im:>19,.0f}")
    print(f"{'Cleared (' + result.clearing_venue + ')':<25} ${result.cleared_marginal_im:>19,.0f} ${result.cleared_im:>19,.0f}")
    print("-" * 75)
    print(f"{'Difference (Bilateral - Cleared)':<25} ${result.im_difference:>19,.0f} ({result.im_difference_pct:>+.1f}%)")

    print("\n" + "=" * 75)
    print(f"                    RECOMMENDATION: {result.recommendation}")
    print("=" * 75)
    print(f"\n{result.rationale}")
    print("=" * 75)


# =============================================================================
# Demo / Testing
# =============================================================================

if __name__ == "__main__":
    """
    Demo showing pre-trade analytics in action.

    Scenario: Asset manager with portfolios at 3 bilateral counterparties
    wants to execute a new 10Y USD swap. Where should they route it?
    """
    print("=" * 75)
    print("       PRE-TRADE ANALYTICS DEMO")
    print("       Marginal IM & Bilateral vs Cleared Analysis")
    print("=" * 75)

    # =========================================================================
    # Setup: Generate sample portfolios and new trade
    # =========================================================================

    try:
        from model.trade_types import (
            generate_market_environment,
            generate_trades_by_type,
            IRSwapTrade
        )
        from model.simm_portfolio_aadc import compute_crif_aadc

        print("\nGenerating sample portfolios...")

        currencies = ['USD', 'EUR', 'GBP']
        market = generate_market_environment(currencies, seed=42)

        # Create portfolios at 3 counterparties with different characteristics
        #
        # Goldman: Large, diversified portfolio (lots of netting potential)
        # JPM: Medium portfolio, mostly pay-fixed (directional)
        # Citi: Small portfolio, receive-fixed

        counterparty_portfolios = {}

        # Goldman: 100 diversified IR swaps
        goldman_trades = generate_trades_by_type('ir_swap', 100, currencies, seed=42)
        goldman_crif, _, _ = compute_crif_aadc(goldman_trades, market, num_threads=4)
        counterparty_portfolios['Goldman'] = goldman_crif
        print(f"  Goldman: {len(goldman_trades)} trades, {len(goldman_crif)} sensitivities")

        # JPM: 50 pay-fixed swaps (directional exposure)
        jpm_trades = generate_trades_by_type('ir_swap', 50, currencies, seed=43)
        # Make them all pay-fixed
        for t in jpm_trades:
            t.payer = True
        jpm_crif, _, _ = compute_crif_aadc(jpm_trades, market, num_threads=4)
        counterparty_portfolios['JPM'] = jpm_crif
        print(f"  JPM: {len(jpm_trades)} trades (pay-fixed), {len(jpm_crif)} sensitivities")

        # Citi: 30 receive-fixed swaps
        citi_trades = generate_trades_by_type('ir_swap', 30, currencies, seed=44)
        for t in citi_trades:
            t.payer = False
        citi_crif, _, _ = compute_crif_aadc(citi_trades, market, num_threads=4)
        counterparty_portfolios['Citi'] = citi_crif
        print(f"  Citi: {len(citi_trades)} trades (receive-fixed), {len(citi_crif)} sensitivities")

        # New trade: 10Y USD receive-fixed swap
        # This should net well with JPM (pay-fixed) but add to Citi (also receive)
        new_trade = [IRSwapTrade(
            trade_id="NEW_10Y_USD",
            notional=100_000_000,  # $100M notional
            currency="USD",
            maturity=10.0,
            fixed_rate=0.035,
            frequency=2,
            payer=False  # Receive fixed
        )]
        new_trade_crif, _, _ = compute_crif_aadc(new_trade, market, num_threads=4)
        print(f"\n  New trade: 10Y USD receive-fixed swap, $100M notional")
        print(f"             {len(new_trade_crif)} sensitivities")

        # =====================================================================
        # Demo 1: Counterparty Routing
        # =====================================================================

        print("\n" + "=" * 75)
        print("DEMO 1: COUNTERPARTY ROUTING")
        print("Which counterparty should we route the new trade to?")
        print("=" * 75)

        routing_result = analyze_trade_routing(
            new_trade_crif,
            counterparty_portfolios,
            num_threads=4,
            use_gradient=False  # Use full recalc for accuracy in demo
        )

        print_routing_recommendation(routing_result)

        # =====================================================================
        # Demo 2: Bilateral vs Cleared
        # =====================================================================

        print("\n" + "=" * 75)
        print("DEMO 2: BILATERAL vs CLEARED")
        print("Should we clear this trade at LCH or keep it bilateral?")
        print("=" * 75)

        # Compare: bilateral at JPM (best netting) vs cleared at LCH
        bilateral_cleared_result = compare_bilateral_vs_cleared(
            new_trade_crif,
            bilateral_portfolio=counterparty_portfolios['JPM'],
            bilateral_counterparty="JPM",
            cleared_portfolio=None,  # Assume no existing cleared portfolio
            clearing_venue=ClearingVenue.LCH
        )

        print_bilateral_vs_cleared(bilateral_cleared_result)

        # =====================================================================
        # Demo 3: Show netting benefit visualization
        # =====================================================================

        print("\n" + "=" * 75)
        print("DEMO 3: NETTING BENEFIT ANALYSIS")
        print("=" * 75)

        print("\nThe new trade is RECEIVE-FIXED. How does this net with each portfolio?")
        print()
        print("  Goldman: Mixed portfolio (pay + receive) -> Moderate netting")
        print("  JPM:     All PAY-FIXED -> EXCELLENT netting (opposite direction)")
        print("  Citi:    All RECEIVE-FIXED -> POOR netting (same direction)")
        print()
        print("This explains why JPM gives the lowest marginal IM!")

    except ImportError as e:
        print(f"\nNote: Full demo requires AADC. Running simplified demo...")
        print(f"(Error: {e})")

        # Create simple test CRIFs manually
        print("\nCreating simplified test data...")

        # Simple portfolio CRIF
        portfolio_crif = pd.DataFrame({
            'TradeID': ['T1', 'T1', 'T2', 'T2'],
            'RiskType': ['Risk_IRCurve', 'Risk_IRCurve', 'Risk_IRCurve', 'Risk_IRCurve'],
            'Qualifier': ['USD', 'USD', 'USD', 'USD'],
            'Bucket': ['10', '11', '10', '11'],
            'Label1': ['10y', '15y', '10y', '15y'],
            'Label2': ['', '', '', ''],
            'Amount': [1e6, 0.5e6, -0.8e6, -0.4e6],
            'AmountCurrency': ['USD', 'USD', 'USD', 'USD'],
            'AmountUSD': [1e6, 0.5e6, -0.8e6, -0.4e6],
        })

        # New trade CRIF
        new_trade_crif = pd.DataFrame({
            'TradeID': ['NEW', 'NEW'],
            'RiskType': ['Risk_IRCurve', 'Risk_IRCurve'],
            'Qualifier': ['USD', 'USD'],
            'Bucket': ['10', '11'],
            'Label1': ['10y', '15y'],
            'Label2': ['', ''],
            'Amount': [-0.5e6, -0.25e6],  # Receive fixed
            'AmountCurrency': ['USD', 'USD'],
            'AmountUSD': [-0.5e6, -0.25e6],
        })

        print("\nBilateral vs Cleared comparison:")
        result = compare_bilateral_vs_cleared(
            new_trade_crif,
            bilateral_portfolio=portfolio_crif,
            bilateral_counterparty="Test Bank",
            clearing_venue=ClearingVenue.LCH
        )
        print_bilateral_vs_cleared(result)

    print("\n" + "=" * 75)
    print("                         DEMO COMPLETE")
    print("=" * 75)
