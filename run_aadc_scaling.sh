#!/bin/bash
# AADC Scaling Benchmarks
# Usage: ./run_aadc_scaling.sh [section]
#   section: trades=trade scaling, K=risk factor scaling, P=portfolio scaling, all=everything

set -e
cd "$(dirname "$0")"
source venv/bin/activate

SECTION="${1:-all}"

run_trade_scaling() {
  echo "=== AADC Trade Scaling (T: 100 → 50,000) ==="
  python benchmark_typical_day.py --trades 100 --portfolios 5 --threads 16
  python benchmark_typical_day.py --trades 500 --portfolios 5 --threads 16
  python benchmark_typical_day.py --trades 1000 --portfolios 5 --threads 16
  python benchmark_typical_day.py --trades 2000 --portfolios 5 --threads 16
  python benchmark_typical_day.py --trades 5000 --portfolios 5 --threads 16
  python benchmark_typical_day.py --trades 10000 --portfolios 5 --threads 16
  python benchmark_typical_day.py --trades 20000 --portfolios 5 --threads 16
  python benchmark_typical_day.py --trades 50000 --portfolios 5 --threads 16
}

run_k_scaling() {
  echo "=== AADC Risk Factor Scaling (K: 20 → 170) ==="
  # K~20
  python benchmark_typical_day.py --trades 1000 --portfolios 5 --threads 16 --simm-buckets 2
  # K~30
  python benchmark_typical_day.py --trades 1000 --portfolios 5 --threads 16 --simm-buckets 3
  # K~70
  python benchmark_typical_day.py --trades 1000 --portfolios 5 --threads 16 \
    --trade-types ir_swap,fx_option
  # K~110
  python benchmark_typical_day.py --trades 1000 --portfolios 5 --threads 16 \
    --trade-types ir_swap,fx_option,equity_option
  # K~170
  python benchmark_typical_day.py --trades 1000 --portfolios 5 --threads 16 \
    --trade-types ir_swap,fx_option,equity_option,inflation_swap,xccy_swap
}

run_portfolio_scaling() {
  echo "=== AADC Portfolio Scaling (P: 3 → 100) ==="
  python benchmark_typical_day.py --trades 5000 --portfolios 3 --threads 16
  python benchmark_typical_day.py --trades 5000 --portfolios 10 --threads 16
  python benchmark_typical_day.py --trades 5000 --portfolios 25 --threads 16
  python benchmark_typical_day.py --trades 5000 --portfolios 50 --threads 16
  python benchmark_typical_day.py --trades 5000 --portfolios 100 --threads 16
}

case "$SECTION" in
  trades|T) run_trade_scaling ;;
  K|k) run_k_scaling ;;
  P|p) run_portfolio_scaling ;;
  all)
    run_trade_scaling
    run_k_scaling
    run_portfolio_scaling
    ;;
  *)
    echo "Usage: $0 [trades|K|P|all]"
    exit 1
    ;;
esac

echo "=== AADC scaling benchmarks complete ==="
echo "Results in: data/typical_day.csv"
echo ""
echo "Key columns: num_trades, num_portfolios, num_risk_factors, backend, evals_per_sec"
