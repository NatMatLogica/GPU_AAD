# Margin Optimization Framework - Implementation Plan

## Executive Summary

This document outlines the architecture and implementation plan for a **Margin Optimization Framework** that extends the existing ISDA-SIMM calculator to support:

1. Multi-entity trade allocation simulation
2. Stress margin (shocked SIMM inputs)
3. Custodian/Clearer optimization
4. Portfolio-level margin analytics
5. Transaction structuring recommendations

## 1. Problem Statement

Financial institutions face significant challenges in managing Initial Margin (IM):

- **Regulatory pressure**: UMR (Uncleared Margin Rules) requires bilateral margin posting
- **Capital efficiency**: Margin consumes capital and liquidity
- **Operational complexity**: Multiple counterparties, custodians, and clearing houses
- **Optimization opportunity**: Trade allocation and structuring can significantly reduce margin

### Key Questions to Answer:
1. Where should I book a new trade to minimize incremental margin?
2. What's the margin impact of novating trades between entities?
3. How does margin change under stress scenarios?
4. What's the optimal allocation of my portfolio across custodians/clearers?

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MARGIN OPTIMIZATION FRAMEWORK                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   PRICING    │───▶│  SENSITIVITY │───▶│    MARGIN CALCULATOR     │  │
│  │    ENGINE    │    │   ENGINE     │    │  (SIMM / CCP / Stressed) │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│         │                   │                        │                   │
│         ▼                   ▼                        ▼                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ALLOCATION OPTIMIZER                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │  Entity     │  │  Netting    │  │  Constraint             │  │   │
│  │  │  Model      │  │  Set Model  │  │  Engine                 │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                         │
│                                ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    SIMULATION ENGINE                             │   │
│  │  • Monte Carlo margin projection                                 │   │
│  │  • Stress scenarios                                              │   │
│  │  • What-if analysis                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                         │
│                                ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ANALYTICS & REPORTING                         │   │
│  │  • Marginal contribution    • Netting benefit                    │   │
│  │  • Allocation recommendations • MVA calculation                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Design

### 3.1 Entity & Netting Set Model

```python
@dataclass
class LegalEntity:
    """Represents a legal entity that can hold trades."""
    entity_id: str
    name: str
    jurisdiction: str
    entity_type: str  # "bank", "dealer", "fund", "ccp"


@dataclass
class Counterparty:
    """Counterparty with CSA parameters."""
    counterparty_id: str
    name: str
    entity: LegalEntity
    csa_params: 'CSAParameters'


@dataclass
class CSAParameters:
    """Credit Support Annex parameters."""
    threshold: float = 0.0          # Amount below which no margin posted
    mta: float = 500_000            # Minimum Transfer Amount
    rounding: float = 100_000       # Rounding amount
    initial_margin_type: str = "SIMM"  # "SIMM", "Schedule", "CCP"
    variation_margin: bool = True
    eligible_collateral: List[str] = field(default_factory=lambda: ["USD", "EUR", "UST"])
    haircuts: Dict[str, float] = field(default_factory=dict)


@dataclass
class NettingSet:
    """A netting set groups trades for margin calculation."""
    netting_set_id: str
    our_entity: LegalEntity
    counterparty: Counterparty
    trades: List[Trade]
    csa: CSAParameters

    def calculate_simm(self) -> float:
        """Calculate SIMM for this netting set."""
        pass

    def calculate_margin(self, margin_type: str = "SIMM") -> float:
        """Calculate margin based on type (SIMM, Schedule, CCP)."""
        pass
```

### 3.2 Stress Margin Module

Extend SIMM to support stressed scenarios:

```python
@dataclass
class StressScenario:
    """Defines a stress scenario for margin calculation."""
    scenario_id: str
    name: str
    description: str

    # Risk weight multipliers by risk class
    rw_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "IR": 1.0, "FX": 1.0, "Credit": 1.0, "Equity": 1.0, "Commodity": 1.0
    })

    # Correlation shocks
    correlation_shift: float = 0.0  # -1 to +1, added to correlations

    # Specific bucket shocks
    bucket_shocks: Dict[str, Dict[str, float]] = field(default_factory=dict)


# Predefined scenarios
STRESS_SCENARIOS = {
    "base": StressScenario("base", "Base Case", "No stress"),

    "high_vol": StressScenario(
        "high_vol", "High Volatility",
        "2x risk weights across all classes",
        rw_multipliers={"IR": 2.0, "FX": 2.0, "Credit": 2.0, "Equity": 2.0, "Commodity": 2.0}
    ),

    "correlation_stress": StressScenario(
        "corr_stress", "Correlation Stress",
        "Correlations increase by 25%",
        correlation_shift=0.25
    ),

    "ir_shock": StressScenario(
        "ir_shock", "IR Rate Shock",
        "IR risk weights 3x, others 1.5x",
        rw_multipliers={"IR": 3.0, "FX": 1.5, "Credit": 1.5, "Equity": 1.5, "Commodity": 1.5}
    ),

    "credit_crisis": StressScenario(
        "credit_crisis", "Credit Crisis",
        "Credit 4x, correlation +50%",
        rw_multipliers={"Credit": 4.0},
        correlation_shift=0.5
    ),
}


class StressedSIMM:
    """SIMM calculator with stress scenario support."""

    def __init__(self, base_simm: SIMM, scenario: StressScenario):
        self.base_simm = base_simm
        self.scenario = scenario

    def calculate(self, crif: pd.DataFrame) -> Dict[str, float]:
        """Calculate SIMM under stress scenario."""
        # Apply risk weight multipliers
        stressed_crif = self._apply_rw_shocks(crif)

        # Apply correlation shifts
        stressed_corr = self._apply_correlation_shocks()

        # Calculate with stressed parameters
        result = self.base_simm.calculate_with_params(
            stressed_crif, stressed_corr
        )

        return {
            "base_simm": self.base_simm.calculate(crif),
            "stressed_simm": result,
            "stress_add_on": result - self.base_simm.calculate(crif),
            "scenario": self.scenario.name
        }
```

### 3.3 Allocation Optimizer

```python
class AllocationOptimizer:
    """Optimizes trade allocation across entities/counterparties."""

    def __init__(
        self,
        entities: List[LegalEntity],
        counterparties: List[Counterparty],
        netting_sets: List[NettingSet]
    ):
        self.entities = entities
        self.counterparties = counterparties
        self.netting_sets = netting_sets

    def optimize_allocation(
        self,
        trades: List[Trade],
        objective: str = "min_margin",  # "min_margin", "min_mva", "balanced"
        constraints: Optional[Dict] = None
    ) -> AllocationResult:
        """
        Find optimal allocation of trades across netting sets.

        Args:
            trades: Trades to allocate
            objective: Optimization objective
            constraints: Limits on allocation (e.g., max per entity)

        Returns:
            AllocationResult with recommendations
        """
        pass

    def marginal_margin(
        self,
        netting_set: NettingSet,
        new_trade: Trade
    ) -> float:
        """Calculate marginal margin impact of adding a trade."""
        base_margin = netting_set.calculate_simm()

        # Temporarily add trade
        netting_set.trades.append(new_trade)
        new_margin = netting_set.calculate_simm()
        netting_set.trades.pop()

        return new_margin - base_margin

    def novation_analysis(
        self,
        trade: Trade,
        from_ns: NettingSet,
        to_ns: NettingSet
    ) -> NovationResult:
        """Analyze margin impact of novating a trade."""
        # Current state
        current_total = from_ns.calculate_simm() + to_ns.calculate_simm()

        # After novation
        from_ns.trades.remove(trade)
        to_ns.trades.append(trade)
        new_total = from_ns.calculate_simm() + to_ns.calculate_simm()

        # Restore
        to_ns.trades.remove(trade)
        from_ns.trades.append(trade)

        return NovationResult(
            margin_before=current_total,
            margin_after=new_total,
            margin_saving=current_total - new_total,
            recommendation="novate" if new_total < current_total else "keep"
        )

    def what_if_analysis(
        self,
        scenario: AllocationScenario
    ) -> WhatIfResult:
        """Run what-if analysis for a given allocation scenario."""
        pass


@dataclass
class AllocationResult:
    """Result of allocation optimization."""
    total_margin: float
    margin_by_netting_set: Dict[str, float]
    trade_allocations: Dict[str, str]  # trade_id -> netting_set_id
    netting_benefit: float
    recommendations: List[str]
```

### 3.4 Simulation Engine

```python
class MarginSimulator:
    """Simulates margin under various scenarios."""

    def __init__(
        self,
        pricing_engine: PricingEngine,
        sensitivity_engine: SensitivityEngine,
        simm_calculator: SIMM
    ):
        self.pricing = pricing_engine
        self.sensitivities = sensitivity_engine
        self.simm = simm_calculator

    def simulate_margin_paths(
        self,
        portfolio: Portfolio,
        market_scenarios: List[MarketScenario],
        time_horizon: int = 252,  # Trading days
        num_paths: int = 1000
    ) -> MarginPathResult:
        """
        Monte Carlo simulation of future margin requirements.

        Used for:
        - MVA (Margin Valuation Adjustment) calculation
        - Liquidity planning
        - Stress testing
        """
        margin_paths = np.zeros((num_paths, time_horizon))

        for path in range(num_paths):
            for t in range(time_horizon):
                # Evolve market
                market_t = self._evolve_market(market_scenarios, t, path)

                # Reprice portfolio
                sensitivities = self.sensitivities.compute(portfolio, market_t)

                # Calculate SIMM
                margin_paths[path, t] = self.simm.calculate(sensitivities)

        return MarginPathResult(
            paths=margin_paths,
            mean_path=margin_paths.mean(axis=0),
            percentile_95=np.percentile(margin_paths, 95, axis=0),
            percentile_99=np.percentile(margin_paths, 99, axis=0),
            mva=self._calculate_mva(margin_paths)
        )

    def stress_test(
        self,
        portfolio: Portfolio,
        scenarios: List[StressScenario]
    ) -> StressTestResult:
        """Run stress test across multiple scenarios."""
        results = {}

        for scenario in scenarios:
            stressed_simm = StressedSIMM(self.simm, scenario)
            results[scenario.scenario_id] = stressed_simm.calculate(
                portfolio.get_crif()
            )

        return StressTestResult(
            base_margin=results.get("base", {}).get("base_simm", 0),
            stressed_margins=results,
            max_stress=max(r["stressed_simm"] for r in results.values()),
            scenario_ranking=self._rank_scenarios(results)
        )

    def _calculate_mva(
        self,
        margin_paths: np.ndarray,
        funding_spread: float = 0.01,  # 100bps
        discount_rate: float = 0.03
    ) -> float:
        """Calculate Margin Valuation Adjustment."""
        # MVA = integral of Expected_Margin * Funding_Spread * Discount_Factor
        dt = 1 / 252
        expected_margin = margin_paths.mean(axis=0)
        discount_factors = np.exp(-discount_rate * np.arange(len(expected_margin)) * dt)

        mva = np.sum(expected_margin * funding_spread * discount_factors * dt)
        return mva
```

### 3.5 Portfolio Analytics

```python
class MarginAnalytics:
    """Advanced analytics for margin optimization."""

    def __init__(self, simm_calculator: SIMM):
        self.simm = simm_calculator

    def marginal_contribution(
        self,
        portfolio: Portfolio
    ) -> pd.DataFrame:
        """
        Calculate marginal SIMM contribution per trade.

        For each trade, compute: SIMM(portfolio) - SIMM(portfolio - trade)
        """
        total_simm = self.simm.calculate(portfolio.get_crif())

        contributions = []
        for trade in portfolio.trades:
            # Remove trade temporarily
            portfolio.trades.remove(trade)
            simm_without = self.simm.calculate(portfolio.get_crif())
            portfolio.trades.append(trade)

            contributions.append({
                "trade_id": trade.trade_id,
                "marginal_simm": total_simm - simm_without,
                "standalone_simm": self._standalone_simm(trade),
                "netting_benefit": self._standalone_simm(trade) - (total_simm - simm_without)
            })

        return pd.DataFrame(contributions)

    def netting_benefit_analysis(
        self,
        netting_sets: List[NettingSet]
    ) -> NettingBenefitResult:
        """Analyze netting benefit across netting sets."""
        standalone_total = sum(
            self._standalone_simm(trade)
            for ns in netting_sets
            for trade in ns.trades
        )

        netted_total = sum(
            ns.calculate_simm()
            for ns in netting_sets
        )

        return NettingBenefitResult(
            standalone_simm=standalone_total,
            netted_simm=netted_total,
            netting_benefit=standalone_total - netted_total,
            netting_benefit_pct=(standalone_total - netted_total) / standalone_total * 100,
            by_netting_set={
                ns.netting_set_id: {
                    "standalone": sum(self._standalone_simm(t) for t in ns.trades),
                    "netted": ns.calculate_simm()
                }
                for ns in netting_sets
            }
        )

    def clearing_vs_bilateral(
        self,
        trades: List[Trade],
        bilateral_csa: CSAParameters,
        ccp_margin_model: CCPMarginModel
    ) -> ClearingAnalysisResult:
        """Compare margin for bilateral vs cleared execution."""
        pass

    def optimal_compression(
        self,
        portfolio: Portfolio
    ) -> CompressionResult:
        """Identify trades that could be compressed to reduce margin."""
        pass
```

---

## 4. Implementation Phases

### Phase 1: Foundation (2-3 weeks)
- [ ] Entity and Netting Set data model
- [ ] CSA parameter configuration
- [ ] Multi-netting-set SIMM calculation
- [ ] Basic marginal margin calculation

### Phase 2: Stress Margin (1-2 weeks)
- [ ] Stress scenario definition
- [ ] Risk weight multiplier application
- [ ] Correlation shock implementation
- [ ] Stress reporting

### Phase 3: Allocation Optimizer (2-3 weeks)
- [ ] Trade allocation engine
- [ ] Novation analysis
- [ ] What-if scenarios
- [ ] Constraint handling

### Phase 4: Simulation Engine (2-3 weeks)
- [ ] Monte Carlo margin paths
- [ ] MVA calculation
- [ ] Stress testing framework
- [ ] Scenario management

### Phase 5: Analytics & Reporting (1-2 weeks)
- [ ] Marginal contribution analysis
- [ ] Netting benefit reporting
- [ ] Clearing vs bilateral comparison
- [ ] Dashboard/visualization

---

## 5. AADC Integration Points

The following computations are candidates for AADC acceleration:

| Computation | Current Method | AADC Benefit |
|-------------|----------------|--------------|
| Trade Sensitivities | Bump-and-revalue | 10-50x speedup |
| Marginal Margin | Full recalc per trade | Adjoint gives all at once |
| Stress Margin | Multiple scenarios | Batch kernel evaluation |
| MVA Simulation | MC over time horizon | Vectorized path simulation |
| Allocation Optimization | Gradient-free | Gradient-based optimization |

### AADC Kernel for Marginal Contribution

```python
def record_marginal_simm_kernel(num_trades: int):
    """
    Record kernel that computes marginal SIMM for all trades simultaneously.

    Instead of N separate calculations (remove each trade, recalculate),
    use AAD to get sensitivities of SIMM to each trade's contribution.
    """
    with aadc.record_kernel():
        # Mark trade contributions as inputs
        trade_contributions = []
        for i in range(num_trades):
            contrib = aadc.idouble(1.0)  # 1.0 = trade included
            contrib.mark_as_input()
            trade_contributions.append(contrib)

        # Weighted CRIF sensitivities
        weighted_crif = compute_weighted_crif(trade_contributions)

        # SIMM calculation
        simm = calculate_simm(weighted_crif)
        simm_output = simm.mark_as_output()

    # Evaluate with all trades included
    # Derivatives give marginal contribution!
    return kernel, simm_output
```

---

## 6. Data Requirements

### Input Data
1. **Trade Data**: Full trade population with booking entity
2. **Counterparty Data**: Legal hierarchy, CSA parameters
3. **Market Data**: Curves, volatilities for pricing
4. **SIMM Parameters**: Risk weights, correlations (already have)
5. **Stress Scenarios**: Regulatory and internal scenarios

### Output Data
1. **SIMM by Netting Set**: Current margin requirements
2. **Stress SIMM**: Margin under stress scenarios
3. **Allocation Recommendations**: Optimal trade placement
4. **MVA Estimates**: Cost of margin over time
5. **Audit Trail**: Full calculation breakdown

---

## 7. Success Metrics

| Metric | Target |
|--------|--------|
| Margin reduction from optimization | 10-30% |
| Stress scenario coverage | 100% regulatory scenarios |
| Calculation time (1000 trades) | < 5 seconds |
| What-if scenario time | < 1 second |
| MVA simulation (252 days, 1000 paths) | < 30 seconds |

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Model risk in stress scenarios | Backtesting, regulatory alignment |
| Optimization may be locally optimal | Multiple starting points, constraints |
| Performance at scale | AADC acceleration, caching |
| Data quality | Validation layer, reconciliation |

---

## 9. Recommended Starting Point

For initial demonstration, recommend implementing:

1. **Entity/Netting Set Model** - Foundation for everything else
2. **Stress Margin** - High client visibility, relatively simple
3. **Marginal Contribution** - Immediate value for allocation decisions

This gives a working prototype that demonstrates:
- Multi-entity margin calculation
- Stress scenarios
- Trade-level margin attribution

From there, the optimization and simulation components can be added incrementally.
