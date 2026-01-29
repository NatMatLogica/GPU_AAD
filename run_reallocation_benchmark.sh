#!/bin/bash
#
# SIMM Reallocation Benchmark: Test margin reduction via gradient-guided reallocation
#
# Runs multiple configurations with increasing reallocation counts to show:
# - Total IM before/after reallocation
# - IM reduction (absolute and %)
# - Trades moved
# - Gradient estimate accuracy
#

set -e

cd "$(dirname "$0")"

TRADES=2000       # Per type
PORTFOLIOS=10
SIMM_BUCKETS=5
THREADS=8

echo "========================================================================"
echo "  SIMM Reallocation Benchmark - Margin Optimization"
echo "========================================================================"
echo ""

# Mixed portfolio for good reallocation opportunities
TRADE_TYPES="ir_swap,equity_option,fx_option"

echo ">>> Configuration: $TRADES trades/type x 3 types = $(($TRADES * 3)) total"
echo ">>> Portfolios: $PORTFOLIOS, SIMM buckets: $SIMM_BUCKETS"
echo ""

# Baseline (no reallocation)
echo "--- Baseline (no reallocation) ---"
python -m model.simm_portfolio_aadc \
    --trades $TRADES \
    --simm-buckets $SIMM_BUCKETS \
    --portfolios $PORTFOLIOS \
    --threads $THREADS \
    --trade-types $TRADE_TYPES
echo ""

# Reallocate 10 trades
echo "--- Reallocate 10 trades ---"
python -m model.simm_portfolio_aadc \
    --trades $TRADES \
    --simm-buckets $SIMM_BUCKETS \
    --portfolios $PORTFOLIOS \
    --threads $THREADS \
    --trade-types $TRADE_TYPES \
    --reallocate 10
echo ""

# Reallocate 50 trades
echo "--- Reallocate 50 trades ---"
python -m model.simm_portfolio_aadc \
    --trades $TRADES \
    --simm-buckets $SIMM_BUCKETS \
    --portfolios $PORTFOLIOS \
    --threads $THREADS \
    --trade-types $TRADE_TYPES \
    --reallocate 50
echo ""

# Reallocate 100 trades
echo "--- Reallocate 100 trades ---"
python -m model.simm_portfolio_aadc \
    --trades $TRADES \
    --simm-buckets $SIMM_BUCKETS \
    --portfolios $PORTFOLIOS \
    --threads $THREADS \
    --trade-types $TRADE_TYPES \
    --reallocate 100
echo ""

# Reallocate 200 trades
echo "--- Reallocate 200 trades ---"
python -m model.simm_portfolio_aadc \
    --trades $TRADES \
    --simm-buckets $SIMM_BUCKETS \
    --portfolios $PORTFOLIOS \
    --threads $THREADS \
    --trade-types $TRADE_TYPES \
    --reallocate 200
echo ""

echo "========================================================================"
echo "  Reallocation benchmark complete."
echo "  Results in data/execution_log_portfolio.csv"
echo ""
echo "  Key columns to analyze:"
echo "    - im_before_realloc, im_after_realloc"
echo "    - realloc_trades_moved"
echo "    - realloc_im_reduction, realloc_im_reduction_pct"
echo "    - realloc_estimate_matches"
echo "========================================================================"
