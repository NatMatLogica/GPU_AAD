#!/bin/bash
# SIMM Benchmark Suite - Full Run
# Usage: ./run_benchmarks.sh [section]
#   section: 1=IR-only, 2=Portfolio-scaling, 3=Multi-asset, all=everything (default)

set -e
cd "$(dirname "$0")"
source venv/bin/activate

SECTION="${1:-all}"

run_section1() {
  echo "=== Section 1: IR-Only Trade Scaling ==="
  python benchmark_trading_workflow.py --trades 50 --portfolios 3 --threads 16
  python benchmark_trading_workflow.py --trades 100 --portfolios 3 --threads 16
  python benchmark_trading_workflow.py --trades 500 --portfolios 5 --threads 16
  python benchmark_trading_workflow.py --trades 1000 --portfolios 5 --threads 16
  python benchmark_trading_workflow.py --trades 2000 --portfolios 10 --threads 16
  python benchmark_trading_workflow.py --trades 5000 --portfolios 15 --threads 16
  python benchmark_trading_workflow.py --trades 10000 --portfolios 20 --threads 16
}

run_section2() {
  echo "=== Section 2: Portfolio Scaling (T=2000) ==="
  python benchmark_trading_workflow.py --trades 2000 --portfolios 3 --threads 16
  python benchmark_trading_workflow.py --trades 2000 --portfolios 20 --threads 16
  python benchmark_trading_workflow.py --trades 2000 --portfolios 50 --threads 16
}

run_section3() {
  echo "=== Section 3: Multi-Asset ==="
  python benchmark_trading_workflow.py --trades 100 --portfolios 3 --threads 16 \
    --trade-types ir_swap,equity_option,inflation_swap,fx_option,xccy_swap
  python benchmark_trading_workflow.py --trades 1000 --portfolios 5 --threads 16 \
    --trade-types ir_swap,fx_option
  python benchmark_trading_workflow.py --trades 4000 --portfolios 10 --threads 16 \
    --trade-types ir_swap,equity_option
  python benchmark_trading_workflow.py --trades 12000 --portfolios 10 --threads 16 \
    --trade-types ir_swap,fx_option,equity_option
  python benchmark_trading_workflow.py --trades 24000 --portfolios 15 --threads 16 \
    --trade-types ir_swap,fx_option,equity_option
}

case "$SECTION" in
  1) run_section1 ;;
  2) run_section2 ;;
  3) run_section3 ;;
  all)
    run_section1
    run_section2
    run_section3
    ;;
  *)
    echo "Usage: $0 [1|2|3|all]"
    exit 1
    ;;
esac

echo "=== Benchmarks complete. Results in data/execution_log_portfolio.csv ==="
