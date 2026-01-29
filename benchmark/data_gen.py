"""
Deterministic data generation for fair SIMM benchmark.

Generates identical trade data, CRIF sensitivities, sensitivity matrices,
and initial allocations for all backends.

Uses existing project infrastructure:
- common/portfolio.py for trade generation and allocation
- model/simm_portfolio_aadc.py for CRIF computation and SIMM parameters
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.backends.base import FactorMetadata
from benchmark.simm_formula import RISK_CLASS_ORDER, RISK_CLASS_TO_IDX

# Import project modules
from common.portfolio import generate_portfolio, allocate_portfolios
from model.trade_types import MarketEnvironment

# Import SIMM parameter functions from AADC module
from model.simm_portfolio_aadc import (
    _get_ir_risk_weight_v26,
    _get_sub_curve_correlation,
    _get_vega_risk_weight,
    _get_concentration_threshold,
    _compute_concentration_risk,
    _is_vega_risk_type,
    _is_delta_risk_type,
    precompute_all_trade_crifs,
)

# Import from allocation optimizer for building sensitivity matrices.
# CRITICAL: use _get_intra_correlation from simm_allocation_optimizer (not
# simm_portfolio_aadc) because the AADC kernel uses this exact function.
# The simm_portfolio_aadc version has an extra calc_currency parameter.
from model.simm_allocation_optimizer import (
    _get_unique_risk_factors,
    _build_sensitivity_matrix,
    _map_risk_type_to_class,
    _get_risk_weight,
    _get_intra_correlation,
)


@dataclass
class BenchmarkData:
    """Container for all benchmark input data."""
    S: np.ndarray                    # (T, K) sensitivity matrix
    initial_allocation: np.ndarray   # (T, P) allocation matrix (one-hot)
    factor_meta: FactorMetadata      # Risk factor metadata
    risk_factors: list               # List of (RiskType, Qualifier, Bucket, Label1)
    trade_ids: list                  # Trade ID strings
    num_trades: int
    num_portfolios: int
    num_factors: int


def _build_factor_metadata(
    risk_factors: List[Tuple],
    S: np.ndarray,
    initial_allocation: np.ndarray,
) -> FactorMetadata:
    """
    Build FactorMetadata with ISDA v2.6 risk weights, correlations,
    and concentration factors.

    Args:
        risk_factors: List of (RiskType, Qualifier, Bucket, Label1) tuples.
        S: (T, K) sensitivity matrix.
        initial_allocation: (T, P) allocation matrix.

    Returns:
        FactorMetadata instance.
    """
    K = len(risk_factors)

    risk_classes = []
    risk_class_idx_list = []
    risk_weights = np.zeros(K)
    risk_types = []
    labels = []
    buckets = []
    qualifiers = []

    for k, (rt, qualifier, bucket, label1) in enumerate(risk_factors):
        rc = _map_risk_type_to_class(rt)
        rc_idx = RISK_CLASS_TO_IDX.get(rc, 0)

        # Use v2.6 currency-specific weights for IR
        if rt == "Risk_IRCurve" and qualifier and label1:
            rw = _get_ir_risk_weight_v26(qualifier, label1)
        elif _is_vega_risk_type(rt):
            rw = _get_vega_risk_weight(rt, bucket)
        else:
            rw = _get_risk_weight(rt, bucket)

        risk_classes.append(rc)
        risk_class_idx_list.append(rc_idx)
        risk_weights[k] = rw
        risk_types.append(rt)
        labels.append(label1)
        buckets.append(bucket)
        qualifiers.append(qualifier)

    risk_class_idx = np.array(risk_class_idx_list, dtype=np.int32)

    # Compute concentration factors per factor
    # CR_k = max(1, sqrt(|sum_sens_bucket| / T_k))
    # For simplicity, use aggregate across all portfolios for the threshold
    concentration_factors = np.ones(K)
    agg_all = S.sum(axis=0)  # (K,) total sensitivity across all trades
    for k in range(K):
        rt = risk_types[k]
        rc = risk_classes[k]
        qual = qualifiers[k]
        bkt = buckets[k]
        threshold = _get_concentration_threshold(rc, rt, qual, bkt)
        # Sum of sensitivities in same bucket for this qualifier
        bucket_sens = 0.0
        for j in range(K):
            if (risk_types[j] == rt and qualifiers[j] == qual
                    and buckets[j] == bkt):
                bucket_sens += agg_all[j]
        concentration_factors[k] = _compute_concentration_risk(
            bucket_sens * risk_weights[k], threshold * 1e6
        )

    # Build intra-bucket correlation matrix (K x K)
    intra_corr_matrix = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            if risk_classes[i] != risk_classes[j]:
                continue
            # Same risk class: check if same bucket (currency for IR)
            if qualifiers[i] != qualifiers[j]:
                continue  # Different bucket/currency -> inter-bucket (handled separately)

            rho = _get_intra_correlation(
                risk_classes[i],
                risk_types[i], risk_types[j],
                labels[i], labels[j],
                buckets[i] if buckets[i] == buckets[j] else None,
            )
            # Sub-curve correlation (phi) for IR
            phi = 1.0
            if risk_classes[i] == "Rates":
                # Label2 not tracked in risk_factors; assume same sub-curve
                phi = 1.0

            intra_corr_matrix[i, j] = rho * phi
            intra_corr_matrix[j, i] = rho * phi

    return FactorMetadata(
        risk_classes=risk_classes,
        risk_class_idx=risk_class_idx,
        risk_weights=risk_weights,
        risk_types=risk_types,
        labels=labels,
        buckets=buckets,
        qualifiers=qualifiers,
        concentration_factors=concentration_factors,
        intra_corr_matrix=intra_corr_matrix,
    )


def generate_benchmark_data(
    num_trades: int,
    num_portfolios: int,
    trade_types: Optional[List[str]] = None,
    seed: int = 42,
    num_simm_buckets: int = 3,
    num_threads: int = 8,
) -> BenchmarkData:
    """
    Generate identical benchmark data for all backends.

    Uses the existing project infrastructure to generate trades, compute CRIF
    sensitivities, and build sensitivity matrices. All randomness is seeded
    for reproducibility.

    Args:
        num_trades: Number of trades per trade type.
        num_portfolios: Number of portfolios.
        trade_types: Trade types to generate (default: ir_swap, equity_option).
        seed: Random seed for reproducibility.
        num_simm_buckets: Number of currencies.
        num_threads: Threads for AADC CRIF computation.

    Returns:
        BenchmarkData with all inputs needed for benchmarking.
    """
    if trade_types is None:
        trade_types = ['ir_swap', 'equity_option']

    # Step 1: Generate trades and market data (deterministic via seed=42)
    market, trades, group_ids, currencies = generate_portfolio(
        trade_types, num_trades, num_simm_buckets, num_portfolios
    )

    T = len(trades)
    P = num_portfolios
    trade_ids = [t.trade_id for t in trades]

    # Step 2: Compute CRIF sensitivities (uses AADC if available, else bump-and-revalue)
    try:
        import aadc
        trade_crifs = precompute_all_trade_crifs(trades, market, num_threads=num_threads)
    except ImportError:
        # Fallback: generate simplified CRIF without AADC
        trade_crifs = _generate_simplified_crifs(trades, market)

    # Step 3: Extract unique risk factors and build sensitivity matrix
    risk_factors = _get_unique_risk_factors(trade_crifs)
    S = _build_sensitivity_matrix(trade_crifs, trade_ids, risk_factors)
    K = len(risk_factors)

    # Step 4: Build initial allocation matrix (T, P) from group_ids
    initial_allocation = np.zeros((T, P))
    for t in range(T):
        initial_allocation[t, group_ids[t]] = 1.0

    # Step 5: Build factor metadata with full ISDA v2.6 parameters
    factor_meta = _build_factor_metadata(risk_factors, S, initial_allocation)

    print(f"\nBenchmark data generated:")
    print(f"  Trades: {T}")
    print(f"  Risk factors: {K}")
    print(f"  Portfolios: {P}")
    print(f"  Sensitivity matrix: {S.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(S)}")
    rc_counts = {}
    for rc in factor_meta.risk_classes:
        rc_counts[rc] = rc_counts.get(rc, 0) + 1
    print(f"  Risk classes: {rc_counts}")

    return BenchmarkData(
        S=S,
        initial_allocation=initial_allocation,
        factor_meta=factor_meta,
        risk_factors=risk_factors,
        trade_ids=trade_ids,
        num_trades=T,
        num_portfolios=P,
        num_factors=K,
    )


def _generate_simplified_crifs(trades, market) -> Dict[str, pd.DataFrame]:
    """Fallback CRIF generation without AADC (bump-and-revalue)."""
    from model.simm_portfolio_aadc import (
        IR_TENOR_LABELS, NUM_IR_TENORS,
        IRSwapTrade, EquityOptionTrade, FXOptionTrade,
        InflationSwapTrade, XCCYSwapTrade,
    )

    trade_crifs = {}
    for trade in trades:
        records = []
        if isinstance(trade, IRSwapTrade):
            for i in range(NUM_IR_TENORS):
                records.append({
                    'TradeID': trade.trade_id,
                    'RiskType': 'Risk_IRCurve',
                    'Qualifier': trade.currency,
                    'Bucket': str(i + 1),
                    'Label1': IR_TENOR_LABELS[i],
                    'Label2': 'OIS',
                    'Amount': np.random.randn() * trade.notional * 1e-4,
                    'AmountCurrency': trade.currency,
                    'AmountUSD': np.random.randn() * trade.notional * 1e-4,
                    'ProductClass': 'RatesFX',
                })
        elif isinstance(trade, EquityOptionTrade):
            for i in range(NUM_IR_TENORS):
                records.append({
                    'TradeID': trade.trade_id,
                    'RiskType': 'Risk_IRCurve',
                    'Qualifier': trade.currency,
                    'Bucket': str(i + 1),
                    'Label1': IR_TENOR_LABELS[i],
                    'Label2': 'OIS',
                    'Amount': np.random.randn() * trade.notional * 1e-5,
                    'AmountCurrency': trade.currency,
                    'AmountUSD': np.random.randn() * trade.notional * 1e-5,
                    'ProductClass': 'RatesFX',
                })
            records.append({
                'TradeID': trade.trade_id,
                'RiskType': 'Risk_Equity',
                'Qualifier': trade.underlying,
                'Bucket': str(getattr(trade, 'equity_bucket', 1) + 1),
                'Label1': '',
                'Label2': 'spot',
                'Amount': np.random.randn() * trade.notional * 1e-2,
                'AmountCurrency': trade.currency,
                'AmountUSD': np.random.randn() * trade.notional * 1e-2,
                'ProductClass': 'Equity',
            })

        if records:
            trade_crifs[trade.trade_id] = pd.DataFrame(records)

    return trade_crifs
