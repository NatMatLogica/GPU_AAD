#!/bin/bash
#
# Quick test: small scale for verification (100 trades, 5 portfolios)
#

set -e

cd "$(dirname "$0")"

echo ">>> Quick test: 100 trades (20/type x 5 types), 5 portfolios"
python -m model.simm_portfolio_aadc \
    --trades 20 \
    --simm-buckets 3 \
    --portfolios 5 \
    --threads 4 \
    --trade-types ir_swap,equity_option,fx_option,inflation_swap,xccy_swap \
    --reallocate 5

echo ""
echo ">>> Check log output:"
tail -1 data/execution_log_portfolio.csv | tr ',' '\n' | head -30
