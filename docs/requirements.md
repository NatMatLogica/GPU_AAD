# Client Requirements - Margin Optimization Platform

## Overview

Requirements gathered from client discussions regarding margin optimization capabilities for a $40B AUM portfolio.

---

## 1. Margin Agreement Simulation

### 1.1 Relocation Analysis
- Simulate impact of relocating trades between margin agreements
- Compare margin under different CSA/collateral terms
- Model novation scenarios (moving trades between counterparties)

### 1.2 Bilateral vs Cleared
- Support both bilateral (CSA) and cleared (CCP) margin calculations
- Model the trade-off between bilateral and cleared execution
- Account for different margin methodologies:
  - Bilateral: ISDA SIMM
  - Cleared: CCP-specific (LCH, CME, Eurex)

---

## 2. Product Coverage

### 2.1 Current Scope
- Interest Rate Swaps (vanilla fixed-for-floating)

### 2.2 Required Extensions
- **Derivatives (Devs) products**:
  - Swaptions (for Vol Arb strategies)
  - Caps/Floors
  - Cross-currency swaps
  - Basis swaps

- **Vol Arb specific**:
  - Client currently uses **QuantLib** for Vol Arb margin agreements
  - Need to integrate or replicate QuantLib pricing for consistency

---

## 3. Portfolio-Level Pricing & Risk

### 3.1 Aggregation
- Price and risk at portfolio level, not just trade-by-trade
- Support portfolio-level Greeks (delta, gamma, vega)
- Netting set awareness for margin calculations

### 3.2 What-If Analysis
- Add/remove trades and see margin impact
- Incremental margin contribution per trade
- Marginal VaR contribution

---

## 4. Stress Margin

### 4.1 Margin Input Shocks
- Shock underlying SIMM inputs (sensitivities) and recalculate margin
- Scenario-based stress testing:
  - Parallel rate shifts
  - Curve steepening/flattening
  - Vol shocks
  - Credit spread widening

### 4.2 SIMM Extensions
- Stressed SIMM (using stressed risk weights)
- Concentration add-ons under stress
- Procyclicality analysis (margin changes through market stress)

---

## 5. Custodian & Counterparty Allocation Optimization

### 5.1 Multi-Custodian Simulation
- Simulate allocation of trades across multiple custodians
- Each custodian may have different:
  - Margin methodologies
  - Netting sets
  - Collateral eligibility
  - Haircuts

### 5.2 Counterparty Optimization
- Perturbations of notional affect margin differently per counterparty
- Optimize trade allocation to minimize total margin
- Consider:
  - Netting benefits
  - Diversification effects
  - Concentration penalties

### 5.3 Transaction Structuring
- **Core problem**: How to structure transactions to limit additional margin required
- Decision variables:
  - Which counterparty to face
  - Which custodian to use
  - Cleared vs bilateral
  - Trade sizing and splitting

---

## 6. Simulation Dimensions

The following dimensions need to be varied in margin optimization:

| Dimension | Description |
|-----------|-------------|
| Counterparty | Which entity to face for each trade |
| Custodian | Where collateral is held |
| Clearing venue | CCP vs bilateral, which CCP |
| Netting set | How trades are grouped for netting |
| Collateral type | Cash, securities, currency |
| Trade notional | Size of each position |
| Trade tenor | Maturity of each position |
| Execution timing | When to execute (margin call timing) |

---

## 7. Integration Requirements

### 7.1 QuantLib Integration
- Client uses QuantLib for Vol Arb margin agreements
- Options:
  1. Call QuantLib from Python for swaption pricing
  2. Replicate QuantLib models in AADC for consistency
  3. Hybrid: QuantLib for exotic, AADC for vanilla

### 7.2 Data Feeds
- Market data for curves, vols
- Counterparty/custodian configuration
- CSA terms database

---

## 8. Priority Ranking

| Priority | Requirement | Complexity | Value |
|----------|-------------|------------|-------|
| P0 | Stress margin (SIMM input shocks) | Medium | High |
| P0 | Portfolio-level aggregation | Low | High |
| P1 | Custodian allocation simulation | High | High |
| P1 | Bilateral vs cleared comparison | Medium | High |
| P2 | Swaption pricing (Vol Arb) | High | Medium |
| P2 | QuantLib integration | Medium | Medium |
| P3 | Full optimization solver | Very High | High |

---

## 9. Current State vs Requirements

| Capability | Current | Required | Gap |
|------------|---------|----------|-----|
| IR Swap pricing | ✓ | ✓ | - |
| SIMM calculation | ✓ | ✓ | - |
| AAD sensitivities | ✓ | ✓ | - |
| Gamma/Curvature | ✓ | ✓ | - |
| Swaptions | ✗ | ✓ | **Gap** |
| Stress margin | ✗ | ✓ | **Gap** |
| Multi-custodian | ✗ | ✓ | **Gap** |
| Allocation optimization | ✗ | ✓ | **Gap** |
| QuantLib integration | ✗ | ✓ | **Gap** |

---

## 10. Proposed Roadmap

### Phase 1: Foundation (Current)
- ✓ IR Swap pricer with AADC
- ✓ SIMM integration
- ✓ Validation framework

### Phase 2: Stress & Scenarios
- Stress margin (shock SIMM inputs)
- Scenario generation framework
- Portfolio-level what-if

### Phase 3: Multi-Entity
- Multi-custodian modeling
- Counterparty margin comparison
- Netting set optimization

### Phase 4: Products & Integration
- Swaption pricing (Vol Arb)
- QuantLib integration
- Cross-currency swaps

### Phase 5: Optimization
- Allocation optimization solver
- Transaction structuring recommendations
- Margin efficiency scoring
