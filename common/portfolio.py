"""
Shared portfolio construction, logging, and result formatting
for both baseline and AADC SIMM implementations.
"""

import numpy as np
import pandas as pd
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.trade_types import (
    MarketEnvironment, generate_market_environment, generate_trades_by_type,
)
from src.agg_margins import SIMM

# =============================================================================
# Constants
# =============================================================================

SUPPORTED_TRADE_TYPES = ["ir_swap", "equity_option", "fx_option", "inflation_swap", "xccy_swap"]
CURRENCY_LIST = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "SEK", "NOK", "DKK"]

LOG_DIR = PROJECT_ROOT / "data"
LOG_FILE = LOG_DIR / "execution_log_portfolio.csv"
CRIF_LOG_FILE = LOG_DIR / "crif_output.csv"
TRADE_CONTRIB_FILE = LOG_DIR / "trade_contributions.csv"

SIMM_RISK_CLASSES = ["Rates", "FX", "CreditQ", "CreditNonQ", "Equity", "Commodity"]

LOG_COLUMNS = [
    "timestamp", "model_name", "model_version",
    "trade_types",
    "num_trades", "num_simm_buckets", "num_portfolios", "num_threads",
    # Timing
    "crif_time_sec", "simm_time_sec", "im_sens_time_sec",
    # Group and results
    "group_id", "num_group_trades",
    "im_result",
    # Gradient info
    "num_im_sensitivities",
    # Reallocation (optional, empty when not used)
    "reallocate_n",
    "reallocate_time_sec",
    "im_after_realloc",
    "im_realloc_estimate",
    "realloc_estimate_matches",
    # Optimization (optional, empty when not used)
    "optimize_method",
    "optimize_time_sec",
    "optimize_initial_im",
    "optimize_final_im",
    "optimize_trades_moved",
    "optimize_converged",
    "status",
]


# =============================================================================
# Portfolio Construction
# =============================================================================

def parse_common_args(description: str) -> argparse.Namespace:
    """Parse common CLI arguments shared by baseline and AADC."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--trade-types", type=str, default="ir_swap",
                        help="Comma-separated trade types (ir_swap,equity_option,fx_option,inflation_swap,xccy_swap)")
    parser.add_argument("--trades", type=int, default=10,
                        help="Number of trades per type")
    parser.add_argument("--simm-buckets", type=int, default=2,
                        help="Number of currencies (SIMM IR buckets)")
    parser.add_argument("--portfolios", type=int, default=5,
                        help="Number of portfolio groups")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of threads")
    parser.add_argument("--reallocate", type=int, default=None,
                        help="Number of trades to reallocate using gradient info (AADC only)")
    parser.add_argument("--no-refresh-gradients", action="store_true",
                        help="Disable iterative gradient refresh during reallocation (v2.6.0 default: enabled)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run full allocation optimization (AADC only)")
    parser.add_argument("--method", type=str, default="auto",
                        choices=["auto", "gradient_descent", "greedy"],
                        help="Optimization method: auto (picks based on size), gradient_descent, greedy")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Allow partial trade allocation across portfolios")
    parser.add_argument("--max-iters", type=int, default=100,
                        help="Maximum optimization iterations (default: 100)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for gradient descent (default: auto-computed from gradient scale)")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Convergence tolerance (default: 1e-6)")
    return parser.parse_args()


def allocate_portfolios(num_trades: int, num_portfolios: int) -> np.ndarray:
    """Deterministic random allocation of trades to portfolio groups."""
    np.random.seed(42)
    return np.random.randint(0, num_portfolios, num_trades)


def generate_portfolio(trade_types: List[str], num_trades: int,
                       num_simm_buckets: int, num_portfolios: int):
    """
    Generate market data, trades, and portfolio allocation.

    Returns:
        (market, trades, group_ids, currencies)
    """
    for tt in trade_types:
        if tt not in SUPPORTED_TRADE_TYPES:
            print(f"Error: unsupported trade type '{tt}'. Supported: {SUPPORTED_TRADE_TYPES}")
            sys.exit(1)

    currencies = CURRENCY_LIST[:num_simm_buckets]

    print("Generating market data...")
    market = generate_market_environment(currencies, seed=42)

    trades = []
    for tt in trade_types:
        print(f"Generating {num_trades} {tt} trades...")
        tt_trades = generate_trades_by_type(tt, num_trades, currencies, seed=42)
        trades.extend(tt_trades)

    if not trades:
        print("Error: no trades generated (no supported trade types)")
        sys.exit(1)

    num_trades_actual = len(trades)
    print(f"  Total trades: {num_trades_actual}")

    group_ids = allocate_portfolios(num_trades_actual, num_portfolios)
    print(f"Portfolio allocation:")
    for g in range(num_portfolios):
        count = int(np.sum(group_ids == g))
        print(f"  Group {g}: {count} trades")
    print()

    return market, trades, group_ids, currencies


# =============================================================================
# SIMM Computation
# =============================================================================

def run_simm(group_crif: pd.DataFrame):
    """
    Run SIMM aggregation on a group's CRIF.

    Returns:
        (portfolio, base_im, simm_time)
    """
    simm_input = group_crif.drop(columns=["TradeID"])
    start = time.perf_counter()
    portfolio = SIMM(simm_input, "USD", 1)
    simm_time = time.perf_counter() - start
    base_im = portfolio.simm
    return portfolio, base_im, simm_time


# =============================================================================
# Per-Trade SIMM Contribution (Euler Decomposition)
# =============================================================================

def compute_trade_contributions(group_crif: pd.DataFrame, gradient: np.ndarray) -> Dict[str, float]:
    """
    Compute per-trade SIMM contribution via Euler decomposition.

    For each trade, contribution = sum of (dIM/ds_i * s_i) for all sensitivities
    belonging to that trade. By Euler's theorem for homogeneous functions,
    the sum of all trade contributions equals the total IM.

    Args:
        group_crif: CRIF DataFrame with TradeID and Amount columns
        gradient: Array of dIM/ds_i for each CRIF row

    Returns:
        Dict mapping TradeID -> SIMM contribution (USD)
    """
    contributions = {}
    for i in range(len(group_crif)):
        trade_id = group_crif.iloc[i]["TradeID"]
        sensitivity = float(group_crif.iloc[i]["Amount"])
        grad_i = gradient[i]
        contrib = grad_i * sensitivity
        contributions[trade_id] = contributions.get(trade_id, 0.0) + contrib
    return contributions


def print_trade_contributions(contributions: Dict[str, float], group_id, base_im: float):
    """Print per-trade SIMM contributions for a group."""
    if not contributions:
        return
    print(f"  Group {group_id} Trade Contributions (Euler decomposition):")
    print(f"    {'TradeID':<20} {'Contribution (USD)':>20} {'% of IM':>10}")
    print(f"    {'-'*52}")
    sorted_trades = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    for trade_id, contrib in sorted_trades:
        pct = (contrib / base_im * 100) if base_im > 0 else 0.0
        print(f"    {trade_id:<20} {contrib:>20,.2f} {pct:>9.2f}%")
    euler_sum = sum(contributions.values())
    print(f"    {'-'*52}")
    print(f"    {'Sum':<20} {euler_sum:>20,.2f} (IM = {base_im:>,.2f})")
    print()


def save_trade_contributions_log(all_contributions: List[Dict]):
    """
    Save per-trade contribution data to CSV.

    Args:
        all_contributions: List of dicts with keys: group_id, trade_id, contribution, im_total
    """
    if not all_contributions:
        return
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_contributions)
        df.to_csv(TRADE_CONTRIB_FILE, index=False)
        print(f"Trade contributions logged: {len(df)} rows -> {TRADE_CONTRIB_FILE}")
    except OSError as e:
        print(f"Warning: could not write trade contributions log: {e}")


# =============================================================================
# Logging
# =============================================================================

def write_log(log_rows: List[dict]):
    """Append rows to the execution log CSV, handling column evolution."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(log_rows, columns=LOG_COLUMNS)
        if LOG_FILE.exists():
            # Read existing header to check for column alignment
            existing_df = pd.read_csv(LOG_FILE, nrows=0)
            if set(existing_df.columns) != set(LOG_COLUMNS):
                # Columns changed: reload, merge, rewrite
                existing_df = pd.read_csv(LOG_FILE)
                combined = pd.concat([existing_df, df], ignore_index=True)
                combined = combined.reindex(columns=LOG_COLUMNS)
                combined.to_csv(LOG_FILE, mode="w", header=True, index=False)
            else:
                df.to_csv(LOG_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(LOG_FILE, mode="w", header=True, index=False)
    except OSError as e:
        print(f"Warning: could not write execution log: {e}")


def save_crif_log(all_crifs: List[pd.DataFrame]):
    """Save combined CRIF DataFrames to CSV."""
    if all_crifs:
        try:
            combined_crif = pd.concat(all_crifs, ignore_index=True)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            combined_crif.to_csv(CRIF_LOG_FILE, index=False)
            print(f"CRIF logged: {len(combined_crif)} rows -> {CRIF_LOG_FILE}")
        except OSError as e:
            print(f"Warning: could not write CRIF log: {e}")


# =============================================================================
# Results Printing
# =============================================================================

def print_results_table(group_results: List[dict], num_trades_actual: int):
    """Print per-group results table and return aggregate totals."""
    print(f"{'Group':<7} {'Trades':>6} {'IM (USD)':>18} {'CRIF (s)':>10} "
          f"{'SIMM (s)':>10} {'# Sens':>7} {'dIM/ds (s)':>10}")
    print("-" * 74)
    total_im = 0.0
    total_crif_time = 0.0
    total_simm_time = 0.0
    total_grad_time = 0.0
    total_num_sens = 0
    for res in group_results:
        print(f"  {res['group_id']:<5} {res['num_group_trades']:>6} "
              f"{res['im_result']:>18,.2f} {res['crif_time_sec']:>10.3f} "
              f"{res['simm_time_sec']:>10.6f} {res['num_im_sensitivities']:>7} "
              f"{res['im_sens_time_sec']:>10.3f}")
        total_im += res["im_result"]
        total_crif_time += res["crif_time_sec"]
        total_simm_time += res["simm_time_sec"]
        total_grad_time += res["im_sens_time_sec"]
        total_num_sens += res["num_im_sensitivities"]
    print("-" * 74)
    print(f"  {'ALL':<5} {num_trades_actual:>6} "
          f"{total_im:>18,.2f} {total_crif_time:>10.3f} "
          f"{total_simm_time:>10.6f} {total_num_sens:>7} "
          f"{total_grad_time:>10.3f}")
    print()

    return {
        "total_im": total_im,
        "total_crif_time": total_crif_time,
        "total_simm_time": total_simm_time,
        "total_grad_time": total_grad_time,
        "total_num_sens": total_num_sens,
    }


# =============================================================================
# Log Row Construction
# =============================================================================

def build_log_rows(
    group_results: List[dict],
    totals: dict,
    model_name: str,
    model_version: str,
    trade_types_str: str,
    num_trades_actual: int,
    num_simm_buckets: int,
    num_portfolios: int,
    num_threads: int,
) -> List[dict]:
    """
    Build execution log rows (one per group + aggregate).
    """
    timestamp = datetime.now().isoformat()

    log_rows = []
    for res in group_results:
        row = {
            "timestamp": timestamp,
            "model_name": model_name,
            "model_version": model_version,
            "trade_types": trade_types_str,
            "num_trades": num_trades_actual,
            "num_simm_buckets": num_simm_buckets,
            "num_portfolios": num_portfolios,
            "num_threads": num_threads,
            "group_id": res["group_id"],
            "num_group_trades": res["num_group_trades"],
            "im_result": res["im_result"],
            "crif_time_sec": res["crif_time_sec"],
            "simm_time_sec": res["simm_time_sec"],
            "im_sens_time_sec": res["im_sens_time_sec"],
            "num_im_sensitivities": res["num_im_sensitivities"],
            "reallocate_n": res.get("reallocate_n", ""),
            "reallocate_time_sec": res.get("reallocate_time_sec", ""),
            "im_after_realloc": res.get("im_after_realloc", ""),
            "im_realloc_estimate": res.get("im_realloc_estimate", ""),
            "realloc_estimate_matches": res.get("realloc_estimate_matches", ""),
            "optimize_method": res.get("optimize_method", ""),
            "optimize_time_sec": res.get("optimize_time_sec", ""),
            "optimize_initial_im": res.get("optimize_initial_im", ""),
            "optimize_final_im": res.get("optimize_final_im", ""),
            "optimize_trades_moved": res.get("optimize_trades_moved", ""),
            "optimize_converged": res.get("optimize_converged", ""),
            "status": "success",
        }
        log_rows.append(row)

    # Aggregate row
    agg_row = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_version": model_version,
        "trade_types": trade_types_str,
        "num_trades": num_trades_actual,
        "num_simm_buckets": num_simm_buckets,
        "num_portfolios": num_portfolios,
        "num_threads": num_threads,
        "group_id": "ALL",
        "num_group_trades": num_trades_actual,
        "im_result": totals["total_im"],
        "crif_time_sec": totals["total_crif_time"],
        "simm_time_sec": totals["total_simm_time"],
        "im_sens_time_sec": totals["total_grad_time"],
        "num_im_sensitivities": totals["total_num_sens"],
        "reallocate_n": totals.get("reallocate_n", ""),
        "reallocate_time_sec": totals.get("reallocate_time_sec", ""),
        "im_after_realloc": totals.get("im_after_realloc", ""),
        "im_realloc_estimate": totals.get("im_realloc_estimate", ""),
        "realloc_estimate_matches": totals.get("realloc_estimate_matches", ""),
        "optimize_method": totals.get("optimize_method", ""),
        "optimize_time_sec": totals.get("optimize_time_sec", ""),
        "optimize_initial_im": totals.get("optimize_initial_im", ""),
        "optimize_final_im": totals.get("optimize_final_im", ""),
        "optimize_trades_moved": totals.get("optimize_trades_moved", ""),
        "optimize_converged": totals.get("optimize_converged", ""),
        "status": "success",
    }
    log_rows.append(agg_row)

    return log_rows


def make_empty_group_result(group_id: int, num_group_trades: int = 0,
                            crif_time: float = 0.0) -> dict:
    """Create a group result dict with zero values (for empty groups)."""
    return {
        "group_id": group_id,
        "num_group_trades": num_group_trades,
        "im_result": 0.0,
        "crif_time_sec": crif_time,
        "simm_time_sec": 0.0,
        "im_sens_time_sec": 0.0,
        "num_im_sensitivities": 0,
    }
