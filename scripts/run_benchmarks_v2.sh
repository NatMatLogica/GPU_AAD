#!/usr/bin/env bash
#
# run_benchmarks_v2.sh — Run all SIMM-implementation-aadc.md benchmarks
# affected by the v2 batched-evaluate migration.
#
# All AADC paths now use batched evaluate() (single call for all portfolios).
# This script validates correctness and captures performance data.
#
# Usage:
#   chmod +x run_benchmarks_v2.sh
#   ./run_benchmarks_v2.sh [--threads N] [--quick]
#
# Options:
#   --threads N   Override thread count (default: 8)
#   --quick       Run reduced-size benchmarks for quick validation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
THREADS=8
QUICK=false
LOGFILE="data/benchmark_v2_$(date +%Y%m%d_%H%M%S).log"
PASS=0
FAIL=0
SKIP=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --quick)   QUICK=true; shift ;;
        *)         echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Trade counts: full vs quick
if $QUICK; then
    T_SMALL=50;  T_MED=100;  T_LARGE=200; ITERS=20
else
    T_SMALL=100; T_MED=500;  T_LARGE=1000; ITERS=100
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGFILE"; }

run_step() {
    local label="$1"; shift
    log "--- START: $label"
    log "CMD: $*"
    local start_sec=$SECONDS
    if "$@" >> "$LOGFILE" 2>&1; then
        local elapsed=$(( SECONDS - start_sec ))
        log "--- PASS: $label (${elapsed}s)"
        PASS=$((PASS + 1))
    else
        local rc=$?
        local elapsed=$(( SECONDS - start_sec ))
        log "--- FAIL: $label (rc=$rc, ${elapsed}s)"
        FAIL=$((FAIL + 1))
    fi
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
log "=========================================="
log " SIMM v2 Batched-Evaluate Benchmark Suite"
log "=========================================="
log "Date:    $(date -Iseconds)"
log "Dir:     $SCRIPT_DIR"
log "Threads: $THREADS"
log "Quick:   $QUICK"
log "Log:     $LOGFILE"
log ""

if [[ -f venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
    log "Activated venv: $(python --version)"
else
    log "WARNING: No venv found, using system Python: $(python3 --version)"
fi

# Smoke-test AADC import
if ! python -c "import aadc; print(f'AADC {aadc.__version__}')" >> "$LOGFILE" 2>&1; then
    log "ERROR: AADC not importable — AADC benchmarks will fail"
fi

log ""
log "============ Section 1: AADC Portfolio (affected by v2 change) ============"
log ""

# ---------------------------------------------------------------------------
# 1. Basic SIMM with AADC — IR swaps only
#    Affected: per-group gradient now uses batched evaluate()
# ---------------------------------------------------------------------------
run_step "AADC basic IR swaps (${T_SMALL} trades, 5 portfolios)" \
    python -m model.simm_portfolio_aadc \
        --trades "$T_SMALL" --portfolios 5 --threads "$THREADS" \
        --trade-types ir_swap

# ---------------------------------------------------------------------------
# 2. Multi-asset portfolio (no optimization)
#    Affected: per-group gradient batching across 3 trade types
# ---------------------------------------------------------------------------
run_step "AADC multi-asset (${T_SMALL} trades, 5 portfolios, 3 types)" \
    python -m model.simm_portfolio_aadc \
        --trades "$T_SMALL" --portfolios 5 --threads "$THREADS" \
        --trade-types ir_swap,equity_option,fx_option

# ---------------------------------------------------------------------------
# 3. Multi-asset with reallocation
#    Affected: batched gradient, then reallocation uses gradient info
# ---------------------------------------------------------------------------
run_step "AADC multi-asset + reallocation (${T_MED} trades)" \
    python -m model.simm_portfolio_aadc \
        --trades "$T_MED" --portfolios 5 --threads "$THREADS" \
        --trade-types ir_swap,equity_option,fx_option \
        --reallocate 20

# ---------------------------------------------------------------------------
# 4. Full gradient descent optimization (v2 optimizer)
#    Affected: now calls reallocate_trades_optimal_v2 with batched evaluate
# ---------------------------------------------------------------------------
run_step "AADC optimization gradient_descent (${T_LARGE} trades, ${ITERS} iters)" \
    python -m model.simm_portfolio_aadc \
        --trades "$T_LARGE" --portfolios 5 --threads "$THREADS" \
        --trade-types ir_swap,equity_option \
        --optimize --method gradient_descent --max-iters "$ITERS"

# ---------------------------------------------------------------------------
# 5. Newton/BFGS optimization (v2 optimizer only)
#    New capability: v2 optimizer supports method=newton with BFGS
# ---------------------------------------------------------------------------
run_step "AADC optimization newton/BFGS (${T_MED} trades)" \
    python -m model.simm_portfolio_aadc \
        --trades "$T_MED" --portfolios 5 --threads "$THREADS" \
        --trade-types ir_swap \
        --optimize --method newton --max-iters 50

log ""
log "============ Section 2: Analytics modules (use AADC gradient) ============"
log ""

# ---------------------------------------------------------------------------
# 6. What-if analytics
#    Uses compute_im_gradient_aadc (single-portfolio, not directly batched
#    but benefits from kernel caching improvements)
# ---------------------------------------------------------------------------
run_step "What-if analytics" \
    python -m model.whatif_analytics

# ---------------------------------------------------------------------------
# 7. Pre-trade analytics
#    Uses record_simm_kernel + aadc.evaluate (single-portfolio)
# ---------------------------------------------------------------------------
run_step "Pre-trade analytics" \
    python -m model.pretrade_analytics

log ""
log "============ Section 3: Baseline (unaffected — reference only) ============"
log ""

# ---------------------------------------------------------------------------
# 8. Baseline (no AADC, bump-and-revalue) — for comparison
# ---------------------------------------------------------------------------
run_step "Baseline IR swaps (${T_SMALL} trades)" \
    python -m model.simm_portfolio_baseline \
        --trades "$T_SMALL" --portfolios 5 --threads 1 \
        --trade-types ir_swap

log ""
log "============ Section 4: Thread scaling (v2 batched evaluate) ============"
log ""

# ---------------------------------------------------------------------------
# 9. Thread scaling for optimization (v2 batched evaluate)
#    This is where batched evaluate should show better scaling than v1
# ---------------------------------------------------------------------------
for t in 1 4 8; do
    run_step "Thread scaling: optimize ${T_MED} trades, ${t} threads" \
        python -m model.simm_portfolio_aadc \
            --trades "$T_MED" --portfolios 5 --threads "$t" \
            --trade-types ir_swap \
            --optimize --method gradient_descent --max-iters 50
done

log ""
log "============ Section 5: v1 vs v2 comparison ============"
log ""

# ---------------------------------------------------------------------------
# 10. v1 vs v2 benchmark (if the benchmark script exists)
# ---------------------------------------------------------------------------
if [[ -f benchmark_aadc_v1_vs_v2.py ]]; then
    run_step "v1 vs v2 benchmark (${T_MED} trades, 5 portfolios)" \
        python benchmark_aadc_v1_vs_v2.py \
            --trades "$T_MED" --portfolios 5 --threads "$THREADS"
else
    log "--- SKIP: benchmark_aadc_v1_vs_v2.py not found"
    SKIP=$((SKIP + 1))
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log ""
log "=========================================="
log " Benchmark Summary"
log "=========================================="
log "  Passed: $PASS"
log "  Failed: $FAIL"
log "  Skipped: $SKIP"
log "  Log:    $LOGFILE"
log "  Exec log: data/execution_log_portfolio.csv"
log "=========================================="

if [[ $FAIL -gt 0 ]]; then
    log "WARNING: $FAIL benchmark(s) failed — check $LOGFILE for details"
    exit 1
fi

log "All benchmarks passed."
exit 0
