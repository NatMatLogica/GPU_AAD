#!/bin/bash
# ISDA-SIMM Benchmark Script - Section 6.4: End-to-End Pipeline Comparison
# Version: 3.3.0
#
# This script reproduces the benchmarks documented in SIMM-implementation-aadc.md Section 6.4
#
# Usage:
#   cd ~/ISDA-SIMM
#   chmod +x run_benchmarks_6_4.sh
#   ./run_benchmarks_6_4.sh

set -e

echo "============================================================"
echo "ISDA-SIMM Benchmark Script - Section 6.4"
echo "End-to-End Pipeline: Baseline vs AADC v3.3.0"
echo "============================================================"
echo ""

cd ~/ISDA-SIMM
source venv/bin/activate

echo "Starting benchmarks at $(date)"
echo ""

# ------------------------------------------------------------
# Test 1: Baseline - 100 IR trades, 5 portfolios
# ------------------------------------------------------------
echo "[1/4] Running Baseline: 100 IR trades, 5 portfolios, 1 thread..."
echo "      (This includes gradient via bump-and-revalue - may take ~5 minutes)"
python -m model.simm_portfolio_baseline \
    --trades 100 \
    --portfolios 5 \
    --threads 1 \
    --trade-types ir_swap
echo "      Done."
echo ""

# ------------------------------------------------------------
# Test 2: AADC v3.3.0 - 100 IR trades, 5 portfolios
# ------------------------------------------------------------
echo "[2/4] Running AADC v3.3.0: 100 IR trades, 5 portfolios, 8 threads..."
python -m model.simm_portfolio_aadc \
    --trades 100 \
    --portfolios 5 \
    --threads 8 \
    --trade-types ir_swap
echo "      Done."
echo ""

# ------------------------------------------------------------
# Test 3: Baseline - 200 multi-asset trades (100 IR + 100 EQ)
# ------------------------------------------------------------
echo "[3/4] Running Baseline: 200 multi-asset trades (IR+EQ), 5 portfolios, 1 thread..."
echo "      (This includes gradient via bump-and-revalue - may take ~10 minutes)"
python -m model.simm_portfolio_baseline \
    --trades 100 \
    --portfolios 5 \
    --threads 1 \
    --trade-types ir_swap,equity_option
echo "      Done."
echo ""

# ------------------------------------------------------------
# Test 4: AADC v3.3.0 - 200 multi-asset trades (100 IR + 100 EQ)
# ------------------------------------------------------------
echo "[4/4] Running AADC v3.3.0: 200 multi-asset trades (IR+EQ), 5 portfolios, 8 threads..."
python -m model.simm_portfolio_aadc \
    --trades 100 \
    --portfolios 5 \
    --threads 8 \
    --trade-types ir_swap,equity_option
echo "      Done."
echo ""

echo "============================================================"
echo "Benchmarks completed at $(date)"
echo "============================================================"
echo ""
echo "Results logged to: data/execution_log_portfolio.csv"
echo ""
echo "To view recent results:"
echo "  tail -10 data/execution_log_portfolio.csv"
echo ""
echo "To extract summary (last 4 ALL rows):"
echo "  grep ',ALL,' data/execution_log_portfolio.csv | tail -4"
