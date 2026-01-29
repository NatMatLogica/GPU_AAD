#!/bin/bash
#
# SIMM Portfolio Benchmark: 10K trades, 10 portfolios, multiple trade combinations
#
# Usage: ./run_benchmark_10k.sh [--reallocate N] [--optimize]
#

set -e

cd "$(dirname "$0")"

TRADES=2000       # 2000 per type x 5 types = 10K total trades
PORTFOLIOS=10
SIMM_BUCKETS=5    # 5 currencies
THREADS=8

# Parse optional arguments
REALLOCATE_ARG=""
OPTIMIZE_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --reallocate)
            REALLOCATE_ARG="--reallocate $2"
            shift 2
            ;;
        --optimize)
            OPTIMIZE_ARG="--optimize"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "  SIMM Portfolio Benchmark - 10K Trades"
echo "========================================================================"
echo "  Trades per type: $TRADES"
echo "  Portfolios:      $PORTFOLIOS"
echo "  SIMM buckets:    $SIMM_BUCKETS"
echo "  Threads:         $THREADS"
echo "  Extra args:      $REALLOCATE_ARG $OPTIMIZE_ARG"
echo "========================================================================"
echo ""

# Test 1: All trade types (10K total = 2K each x 5 types)
echo ">>> Test 1: All trade types (ir_swap,equity_option,fx_option,inflation_swap,xccy_swap)"
python -m model.simm_portfolio_aadc \
    --trades $TRADES \
    --simm-buckets $SIMM_BUCKETS \
    --portfolios $PORTFOLIOS \
    --threads $THREADS \
    --trade-types ir_swap,equity_option,fx_option,inflation_swap,xccy_swap \
    $REALLOCATE_ARG $OPTIMIZE_ARG
echo ""

# Test 2: IR-only portfolio (10K IR swaps)
echo ">>> Test 2: IR swaps only (10K)"
python -m model.simm_portfolio_aadc \
    --trades 10000 \
    --simm-buckets $SIMM_BUCKETS \
    --portfolios $PORTFOLIOS \
    --threads $THREADS \
    --trade-types ir_swap \
    $REALLOCATE_ARG $OPTIMIZE_ARG
echo ""

# Test 3: Mixed rates (IR + FX + inflation)
echo ">>> Test 3: Mixed rates (ir_swap,fx_option,inflation_swap) - ~10K trades"
python -m model.simm_portfolio_aadc \
    --trades 3334 \
    --simm-buckets $SIMM_BUCKETS \
    --portfolios $PORTFOLIOS \
    --threads $THREADS \
    --trade-types ir_swap,fx_option,inflation_swap \
    $REALLOCATE_ARG $OPTIMIZE_ARG
echo ""

# Test 4: Equity-heavy (equity + IR)
echo ">>> Test 4: Equity-heavy (ir_swap,equity_option) - 10K trades"
python -m model.simm_portfolio_aadc \
    --trades 5000 \
    --simm-buckets $SIMM_BUCKETS \
    --portfolios $PORTFOLIOS \
    --threads $THREADS \
    --trade-types ir_swap,equity_option \
    $REALLOCATE_ARG $OPTIMIZE_ARG
echo ""

# Test 5: Cross-currency (xccy + fx)
echo ">>> Test 5: Cross-currency (xccy_swap,fx_option) - 10K trades"
python -m model.simm_portfolio_aadc \
    --trades 5000 \
    --simm-buckets $SIMM_BUCKETS \
    --portfolios $PORTFOLIOS \
    --threads $THREADS \
    --trade-types xccy_swap,fx_option \
    $REALLOCATE_ARG $OPTIMIZE_ARG
echo ""

echo "========================================================================"
echo "  Benchmark complete. Results logged to data/execution_log_portfolio.csv"
echo "========================================================================"
