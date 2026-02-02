#!/usr/bin/env python
"""
Trading Day Workflow Benchmark: AADC Python vs GPU (CUDA) vs C++ AADC

Simulates a full trading day across 5 timed stages, comparing three backends:
AADC Python, GPU CUDA, and C++ AADC. All backends use the FULL ISDA SIMM v2.6
formula with intra-bucket correlations, concentration factors, and cross-risk-class
aggregation.

Key AADC advantage: record kernel once at start of day, reuse for every
margin computation (attribution, pre-trade, what-if, optimization).

Stages:
  1. 7:00 AM - Start of Day: Portfolio Setup & Kernel Recording
  2. 8:00 AM - Morning Risk Report: Margin Attribution
  3. 9AM-4PM - Intraday Trading: Pre-Trade Checks (50 new trades)
  4. 2:00 PM - What-If Scenarios (stress, unwind, hedge, IM ladder)
  5. 5:00 PM - EOD: Portfolio Optimization

Usage:
    python benchmark_trading_workflow.py --trades 1000 --portfolios 5 --threads 8
    python benchmark_trading_workflow.py --trades 5000 --portfolios 10 \\
        --new-trades 50 --optimize-iters 100 --refresh-interval 10

Version: 2.7.0
"""

MODEL_VERSION = "2.7.0"

import sys
import os
import re
import io
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


class TeeWriter:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, file_path):
        self._stdout = sys.stdout
        self._file = open(file_path, 'w')

    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._stdout

sys.path.insert(0, str(Path(__file__).parent))

# Check AADC
try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False

# Check CUDA
CUDA_SIMULATOR = os.environ.get('NUMBA_ENABLE_CUDASIM', '0') == '1'
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available() or CUDA_SIMULATOR
except ImportError:
    CUDA_AVAILABLE = False

# Check C++ AADC binary
CPP_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "build", "simm_optimizer")
CPP_AVAILABLE = os.path.isfile(CPP_BINARY) and os.access(CPP_BINARY, os.X_OK)

# Project imports
from common.portfolio import generate_portfolio, write_log
from model.trade_types import generate_trades_by_type, compute_crif_for_trade, MarketEnvironment
from model.simm_portfolio_aadc import (
    precompute_all_trade_crifs,
    _get_ir_risk_weight_v26,
    _map_risk_type_to_class,
    _is_delta_risk_type,
    _is_vega_risk_type,
    _get_vega_risk_weight,
    _get_risk_weight,
    _get_intra_correlation,
    _get_concentration_threshold,
    _compute_concentration_risk,
    PSI_MATRIX,
)
from Weights_and_Corr.v2_6 import (
    ir_gamma_diff_ccy, cr_gamma_diff_ccy,
    creditQ_corr_non_res, equity_corr_non_res, commodity_corr_non_res,
)
from model.simm_allocation_optimizer import (
    _get_unique_risk_factors,
    _build_sensitivity_matrix,
)
from model.simm_portfolio_aadc_v2 import (
    record_single_portfolio_simm_kernel_v2_full,
    _get_factor_metadata_v2_full,
    compute_all_portfolios_im_gradient_v2,
)
from model.simm_portfolio_cuda import (
    compute_simm_and_gradient_cuda,
    compute_simm_im_only_cuda,
    preallocate_gpu_arrays,
)
from benchmark_aadc_vs_gpu import _build_benchmark_log_row

MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_trading_workflow.md")

CURRENCY_LIST = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "SEK", "NOK", "DKK"]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class StepResult:
    step_name: str
    step_time: str
    aadc_time_sec: Optional[float] = None
    gpu_time_sec: Optional[float] = None
    cpp_time_sec: Optional[float] = None
    bf_time_sec: Optional[float] = None   # Brute-force (forward-only GPU, no gradients)
    aadc_evals: int = 0
    gpu_evals: int = 0
    cpp_evals: int = 0
    bf_evals: int = 0
    aadc_kernel_recordings: int = 0   # Number of kernel recordings (AADC only)
    aadc_kernel_reuses: int = 0       # Evaluations that reused existing kernel
    details: dict = field(default_factory=dict)


@dataclass
class KernelEconomics:
    recording_time_sec: float = 0.0
    total_recordings: int = 0         # How many times kernel was recorded
    total_aadc_evals: int = 0         # Total aadc.evaluate() calls
    total_gpu_evals: int = 0          # Total GPU kernel launches
    total_bf_evals: int = 0           # Total brute-force (forward-only) GPU evals
    total_aadc_kernel_reuses: int = 0 # Evals that reused a recorded kernel
    cumulative_aadc_time: float = 0.0
    cumulative_gpu_time: float = 0.0
    cumulative_bf_time: float = 0.0
    cpp_recording_time_sec: float = 0.0
    total_cpp_evals: int = 0
    cumulative_cpp_time: float = 0.0

    def update(self, step: StepResult):
        self.total_aadc_evals += step.aadc_evals
        self.total_gpu_evals += step.gpu_evals
        self.total_cpp_evals += step.cpp_evals
        self.total_bf_evals += step.bf_evals
        self.total_recordings += step.aadc_kernel_recordings
        self.total_aadc_kernel_reuses += step.aadc_kernel_reuses
        if step.aadc_time_sec is not None:
            self.cumulative_aadc_time += step.aadc_time_sec
        if step.gpu_time_sec is not None:
            self.cumulative_gpu_time += step.gpu_time_sec
        if step.cpp_time_sec is not None:
            self.cumulative_cpp_time += step.cpp_time_sec
        if step.bf_time_sec is not None:
            self.cumulative_bf_time += step.bf_time_sec

    @property
    def amortized_recording_ms(self) -> float:
        if self.total_aadc_evals == 0:
            return self.recording_time_sec * 1000
        return (self.recording_time_sec / self.total_aadc_evals) * 1000


# =============================================================================
# C++ AADC Backend (subprocess)
# =============================================================================

def _run_cpp_mode(mode, num_trades, num_portfolios, num_threads, seed,
                  max_iters=100, extra_args=None, input_dir=None):
    """Run C++ AADC binary for a given mode and parse stdout for timing metrics."""
    if not CPP_AVAILABLE:
        return None

    cmd = [
        CPP_BINARY,
        "--trades", str(num_trades),
        "--portfolios", str(num_portfolios),
        "--threads", str(num_threads),
        "--seed", str(seed),
        "--mode", mode,
        "--max-iters", str(max_iters),
        "--no-greedy",
    ]
    if input_dir:
        cmd.extend(["--input-dir", input_dir])
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"    C++ {mode} failed (exit code {result.returncode})")
            if result.stderr:
                print(f"      stderr: {result.stderr[:200]}")
            return None

        stdout = result.stdout
        parsed = {"mode": mode}

        # Common fields (all modes)
        m = re.search(r"Recording time:\s+([\d.]+)\s*ms", stdout)
        if m:
            parsed["recording_time_ms"] = float(m.group(1))

        # CRIF time — match "Time: X ms" after CRIF section
        m = re.search(r"CRIF.*?Time:\s+([\d.]+)\s*ms", stdout, re.DOTALL)
        if not m:
            m = re.search(r"Time:\s+([\d.]+)\s*ms", stdout)
        if m:
            parsed["crif_time_ms"] = float(m.group(1))

        m = re.search(r"Dimensions:\s*T=(\d+)\s*x\s*K=(\d+)", stdout)
        if m:
            parsed["cpp_T"] = int(m.group(1))
            parsed["cpp_K"] = int(m.group(2))

        m = re.search(r"Total wall time:\s+([\d.]+)\s*ms", stdout)
        if m:
            parsed["total_wall_time_ms"] = float(m.group(1))

        # Mode-specific fields
        if mode == "attribution":
            m = re.search(r"Total IM:\s+\$([\d,.]+)", stdout)
            if m:
                parsed["total_im"] = float(m.group(1).replace(",", ""))
            m = re.search(r"Eval time:\s+([\d.]+)\s*ms", stdout)
            if m:
                parsed["eval_time_ms"] = float(m.group(1))
            m = re.search(r"Euler check \(ratio\):\s+([\d.]+)", stdout)
            if m:
                parsed["euler_ratio"] = float(m.group(1))

        elif mode == "whatif":
            scenarios = {}
            m = re.search(
                r"\[1\] Unwind.*?Base IM:\s+\$([\d,.]+).*?Scenario IM:\s+\$([\d,.]+)"
                r".*?IM change:.*?\(([\d.+-]+)%\)",
                stdout, re.DOTALL)
            if m:
                scenarios["unwind"] = {
                    "base_im": float(m.group(1).replace(",", "")),
                    "scenario_im": float(m.group(2).replace(",", "")),
                    "change_pct": float(m.group(3)),
                }
            m = re.search(
                r"\[2\] Stress IR.*?Base IM:\s+\$([\d,.]+).*?Scenario IM:\s+\$([\d,.]+)"
                r".*?IM change:.*?\(([\d.+-]+)%\)",
                stdout, re.DOTALL)
            if m:
                scenarios["stress_ir"] = {
                    "base_im": float(m.group(1).replace(",", "")),
                    "scenario_im": float(m.group(2).replace(",", "")),
                    "change_pct": float(m.group(3)),
                }
            m = re.search(
                r"\[3\] Stress Equity.*?Base IM:\s+\$([\d,.]+).*?Scenario IM:\s+\$([\d,.]+)"
                r".*?IM change:.*?\(([\d.+-]+)%\)",
                stdout, re.DOTALL)
            if m:
                scenarios["stress_eq"] = {
                    "base_im": float(m.group(1).replace(",", "")),
                    "scenario_im": float(m.group(2).replace(",", "")),
                    "change_pct": float(m.group(3)),
                }
            m = re.search(r"Marginal IM:\s+\$([\d,.]+)", stdout)
            if m:
                scenarios["marginal_im"] = float(m.group(1).replace(",", ""))
            parsed["scenarios"] = scenarios
            m = re.search(r"Eval time:\s+([\d.]+)\s*ms", stdout)
            if m:
                parsed["eval_time_ms"] = float(m.group(1))

        elif mode == "pretrade":
            m = re.search(r"Eval time:\s+([\d.]+)\s*ms", stdout)
            if m:
                parsed["routing_eval_ms"] = float(m.group(1))
            m = re.search(r"Best portfolio:\s+(\d+)", stdout)
            if m:
                parsed["best_portfolio"] = int(m.group(1))

        elif mode == "optimize":
            m = re.search(r"Initial IM:\s+\$([\d,.]+)", stdout)
            if m:
                parsed["initial_im"] = float(m.group(1).replace(",", ""))
            m = re.search(r"Final IM:\s+\$([\d,.]+)", stdout)
            if m:
                parsed["final_im"] = float(m.group(1).replace(",", ""))
            m = re.search(r"Trades moved:\s+(\d+)", stdout)
            if m:
                parsed["trades_moved"] = int(m.group(1))
            m = re.search(r"Iterations:\s+(\d+)", stdout)
            if m:
                parsed["iterations"] = int(m.group(1))
            m = re.search(r"Optimization eval:\s+([\d.]+)\s*ms", stdout)
            if m:
                parsed["optimization_eval_ms"] = float(m.group(1))
            m = re.search(r"Kernel recording:\s+([\d.]+)\s*ms", stdout)
            if m:
                parsed["kernel_recording_ms"] = float(m.group(1))

        return parsed

    except subprocess.TimeoutExpired:
        print(f"    C++ {mode} timed out (600s)")
        return None
    except Exception as e:
        print(f"    C++ {mode} error: {e}")
        return None


def _run_all_cpp_modes(num_trades, num_portfolios, num_threads, optimize_iters,
                       seed=42, input_dir=None):
    """Run C++ AADC binary in --mode all (single process, shared import/kernel)."""
    if not CPP_AVAILABLE:
        return {}

    print("\n  Running C++ AADC backend (single process, --mode all)...")

    cmd = [
        CPP_BINARY,
        "--trades", str(num_trades),
        "--portfolios", str(num_portfolios),
        "--threads", str(num_threads),
        "--seed", str(seed),
        "--mode", "all",
        "--max-iters", str(optimize_iters),
    ]
    if input_dir:
        cmd.extend(["--input-dir", input_dir])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"    C++ all mode failed (exit code {result.returncode})")
            if result.stderr:
                print(f"      stderr: {result.stderr[:300]}")
            return {}

        stdout = result.stdout
        cpp_results = {}

        # Parse recording time (shared across all modes)
        m = re.search(r"Recording time:\s+([\d.]+)\s*ms", stdout)
        recording_ms = float(m.group(1)) if m else 0

        m = re.search(r"Total wall time:\s+([\d.]+)\s*ms", stdout)
        total_wall_ms = float(m.group(1)) if m else 0

        # Parse [ATTRIBUTION] section
        attr_section = re.search(r"\[ATTRIBUTION\](.*?)\[WHATIF\]", stdout, re.DOTALL)
        if attr_section:
            s = attr_section.group(1)
            parsed = {"recording_time_ms": recording_ms}
            m = re.search(r"Total IM:\s+\$([\d,.]+)", s)
            if m: parsed["total_im"] = float(m.group(1).replace(",", ""))
            m = re.search(r"Euler check \(ratio\):\s+([\d.]+)", s)
            if m: parsed["euler_ratio"] = float(m.group(1))
            m = re.search(r"Eval time:\s+([\d.]+)\s*ms", s)
            if m: parsed["eval_time_ms"] = float(m.group(1))
            cpp_results["attribution"] = parsed
            print(f"    attribution: {parsed.get('eval_time_ms', 0):.2f} ms eval")

        # Parse [WHATIF] section
        wi_section = re.search(r"\[WHATIF\](.*?)\[PRETRADE\]", stdout, re.DOTALL)
        if wi_section:
            s = wi_section.group(1)
            parsed = {"recording_time_ms": recording_ms, "scenarios": {}}
            m = re.search(r"Eval time:\s+([\d.]+)\s*ms", s)
            if m: parsed["eval_time_ms"] = float(m.group(1))
            m = re.search(r"Marginal IM:\s+\$([\d,.]+)", s)
            if m: parsed["marginal_im"] = float(m.group(1).replace(",", ""))
            # Unwind scenario
            m = re.search(r"Unwind top \d+: Base=\$([\d,.]+)\s+Scenario=\$([\d,.]+)\s+\(([\d.+-]+)%\)", s)
            if m:
                parsed["scenarios"]["unwind"] = {
                    "base_im": float(m.group(1).replace(",", "")),
                    "scenario_im": float(m.group(2).replace(",", "")),
                    "change_pct": float(m.group(3)),
                }
            # Stress IR
            m = re.search(r"Stress IR [\d.]+x: Scenario=\$([\d,.]+)\s+\(([\d.+-]+)%\)", s)
            if m:
                parsed["scenarios"]["stress_ir"] = {
                    "scenario_im": float(m.group(1).replace(",", "")),
                    "change_pct": float(m.group(2)),
                }
            # Stress Equity
            m = re.search(r"Stress Equity [\d.]+x: Scenario=\$([\d,.]+)\s+\(([\d.+-]+)%\)", s)
            if m:
                parsed["scenarios"]["stress_eq"] = {
                    "scenario_im": float(m.group(1).replace(",", "")),
                    "change_pct": float(m.group(2)),
                }
            cpp_results["whatif"] = parsed
            print(f"    whatif: {parsed.get('eval_time_ms', 0):.2f} ms eval")

        # Parse [PRETRADE] section
        pt_section = re.search(r"\[PRETRADE\](.*?)\[OPTIMIZE\]", stdout, re.DOTALL)
        if pt_section:
            s = pt_section.group(1)
            parsed = {"recording_time_ms": recording_ms}
            m = re.search(r"Best portfolio:\s+(\d+)", s)
            if m: parsed["best_portfolio"] = int(m.group(1))
            m = re.search(r"Eval time:\s+([\d.]+)\s*ms", s)
            if m: parsed["routing_eval_ms"] = float(m.group(1))
            cpp_results["pretrade"] = parsed
            print(f"    pretrade: {parsed.get('routing_eval_ms', 0):.2f} ms eval")

        # Parse [OPTIMIZE] section
        opt_section = re.search(r"\[OPTIMIZE\](.*?)(?:Recording time:|$)", stdout, re.DOTALL)
        if opt_section:
            s = opt_section.group(1)
            parsed = {"recording_time_ms": recording_ms}
            m = re.search(r"Initial IM:\s+\$([\d,.]+)", s)
            if m: parsed["initial_im"] = float(m.group(1).replace(",", ""))
            # Use findall to get the LAST occurrence (summary line, not sub-optimizer)
            finals = re.findall(r"Final IM:\s+\$([\d,.]+)", s)
            if finals: parsed["final_im"] = float(finals[-1].replace(",", ""))
            moved_all = re.findall(r"Trades moved:\s+(\d+)", s)
            if moved_all: parsed["trades_moved"] = int(moved_all[-1])
            m = re.search(r"Iterations:\s+(\d+)", s)
            if m: parsed["iterations"] = int(m.group(1))
            m = re.search(r"Optimization eval:\s+([\d.]+)\s*ms", s)
            if m: parsed["optimization_eval_ms"] = float(m.group(1))
            m = re.search(r"Optimization wall:\s+([\d.]+)\s*ms", s)
            if m: parsed["optimization_wall_ms"] = float(m.group(1))
            cpp_results["optimize"] = parsed
            wall = parsed.get('optimization_wall_ms', parsed.get('optimization_eval_ms', 0))
            kern = parsed.get('optimization_eval_ms', 0)
            print(f"    optimize: {wall:.2f} ms wall ({kern:.2f} ms kernel eval)")

        print(f"    Total wall time: {total_wall_ms:.1f} ms "
              f"(import+recording+all evals in ONE process)")

        return cpp_results

    except subprocess.TimeoutExpired:
        print(f"    C++ all mode timed out (600s)")
        return {}
    except Exception as e:
        print(f"    C++ all mode error: {e}")
        return {}


def _apply_cpp_results(steps, economics, cpp_results):
    """Map C++ mode results to workflow steps and update economics."""
    if not cpp_results:
        return

    s1, s2, s3, s4 = steps[:4]
    eod_steps = steps[4:]

    # Step 1 (Portfolio Setup): C++ first eval time (recording stored separately)
    # Use attribution's eval_time as the first IM evaluation
    rec_ms = 0
    if "attribution" in cpp_results:
        rec_ms = cpp_results["attribution"].get("recording_time_ms", 0)
    elif "optimize" in cpp_results:
        rec_ms = cpp_results["optimize"].get("recording_time_ms",
                  cpp_results["optimize"].get("kernel_recording_ms", 0))
    economics.cpp_recording_time_sec = rec_ms / 1000.0

    if "attribution" in cpp_results:
        r = cpp_results["attribution"]
        eval_ms = r.get("eval_time_ms", 0)
        s1.cpp_time_sec = eval_ms / 1000.0
        s1.cpp_evals = 1

    # Step 2 (Attribution): C++ attribution eval time
    if "attribution" in cpp_results:
        r = cpp_results["attribution"]
        s2.cpp_time_sec = r.get("eval_time_ms", 0) / 1000.0
        s2.cpp_evals = 1

    # Step 3 (Pre-Trade): C++ counterparty routing eval time
    if "pretrade" in cpp_results:
        r = cpp_results["pretrade"]
        s3.cpp_time_sec = r.get("routing_eval_ms", 0) / 1000.0
        s3.cpp_evals = 1

    # Step 4 (What-If): C++ whatif eval time
    if "whatif" in cpp_results:
        r = cpp_results["whatif"]
        eval_ms = r.get("eval_time_ms", 0)
        if eval_ms == 0:
            # Fallback: estimate from total wall time minus setup overhead
            total_ms = r.get("total_wall_time_ms", 0)
            setup_ms = r.get("recording_time_ms", 0) + r.get("crif_time_ms", 0)
            eval_ms = max(0, total_ms - setup_ms)
        s4.cpp_time_sec = eval_ms / 1000.0
        s4.cpp_evals = 4  # unwind + stress_ir + stress_eq + marginal

    # Step 5 (Optimization): C++ uses GD — apply to matching EOD step
    if "optimize" in cpp_results:
        r = cpp_results["optimize"]
        opt_ms = r.get("optimization_wall_ms", r.get("optimization_eval_ms", 0))
        iters = r.get("iterations", 0)
        # Find the GD EOD step (C++ uses gradient_descent)
        target = None
        for es in eod_steps:
            if es.details.get("method") == "gradient_descent":
                target = es
                break
        if target is None and eod_steps:
            target = eod_steps[0]  # fallback to first EOD step
        if target is not None:
            target.cpp_time_sec = opt_ms / 1000.0
            target.cpp_evals = iters + 2  # init eval + iterations + final eval
            target.details["cpp"] = cpp_results["optimize"]

    # Store raw C++ parsed results in step details for comparison output
    if "attribution" in cpp_results:
        s1.details["cpp"] = cpp_results["attribution"]
        s2.details["cpp"] = cpp_results["attribution"]  # Attribution step uses same data
    if "pretrade" in cpp_results:
        s3.details["cpp"] = cpp_results["pretrade"]
    if "whatif" in cpp_results:
        s4.details["cpp"] = cpp_results["whatif"]

    # Update economics totals
    for s in steps:
        economics.total_cpp_evals += s.cpp_evals
        if s.cpp_time_sec is not None:
            economics.cumulative_cpp_time += s.cpp_time_sec


# =============================================================================
# Shared Setup — Full ISDA SIMM v2.6
# =============================================================================

def setup_portfolio_and_kernel(
    num_trades, num_portfolios, trade_types, num_simm_buckets, num_threads
) -> dict:
    """
    Generate portfolio, CRIFs, sensitivity matrix, and prepare both backends
    using the full ISDA SIMM v2.6 formula (correlations + concentration).
    """
    market, trades, group_ids, currencies = generate_portfolio(
        trade_types, num_trades, num_simm_buckets, num_portfolios
    )
    T = len(trades)
    P = num_portfolios

    # Compute CRIFs (shared — uses AADC for pricing if available)
    workers = aadc.ThreadPool(num_threads) if AADC_AVAILABLE else None
    crif_start = time.perf_counter()
    trade_crifs = precompute_all_trade_crifs(trades, market, num_threads, workers)
    crif_time = time.perf_counter() - crif_start

    # Build sensitivity matrix (only trades with CRIF entries)
    risk_factors = _get_unique_risk_factors(trade_crifs)
    trade_ids = sorted(trade_crifs.keys())
    S = _build_sensitivity_matrix(trade_crifs, trade_ids, risk_factors)
    T = len(trade_ids)  # May be < len(trades) if some trades produce no CRIF
    K = len(risk_factors)

    # Risk weights, risk class indices, and risk measure indices
    risk_class_map = {"Rates": 0, "CreditQ": 1, "CreditNonQ": 2,
                      "Equity": 3, "Commodity": 4, "FX": 5}
    risk_weights = np.zeros(K, dtype=np.float64)
    risk_class_idx = np.zeros(K, dtype=np.int32)
    risk_measure_idx = np.zeros(K, dtype=np.int32)

    # Per-factor metadata for bucket building
    factor_rc_name = []
    factor_bucket_key = []
    factor_bucket_num = []

    for k, (rt, qualifier, bucket, label1) in enumerate(risk_factors):
        rc = _map_risk_type_to_class(rt)
        risk_class_idx[k] = risk_class_map.get(rc, 0)

        is_vega = _is_vega_risk_type(rt)
        risk_measure_idx[k] = 1 if is_vega else 0

        if rt == "Risk_IRCurve" and qualifier and label1:
            risk_weights[k] = _get_ir_risk_weight_v26(qualifier, label1)
        elif is_vega:
            risk_weights[k] = _get_vega_risk_weight(rt, bucket)
        else:
            risk_weights[k] = _get_risk_weight(rt, bucket)

        # Bucket key (currency for Rates/FX, bucket number for others)
        if rc in ("Rates", "FX"):
            bkey = qualifier if qualifier else ""
        else:
            bkey = str(bucket) if bucket else "0"

        factor_rc_name.append(rc)
        factor_bucket_key.append(bkey)
        try:
            factor_bucket_num.append(int(bucket) if bucket else 0)
        except (ValueError, TypeError):
            factor_bucket_num.append(0)

    # --- Assign unique bucket IDs per (rc, rm, bucket_key) ---
    bucket_map = {}
    for k in range(K):
        key = (risk_class_idx[k], risk_measure_idx[k], factor_bucket_key[k])
        if key not in bucket_map:
            bucket_map[key] = len(bucket_map)

    B = len(bucket_map)
    bucket_id = np.zeros(K, dtype=np.int32)
    bucket_rc = np.zeros(B, dtype=np.int32)
    bucket_rm = np.zeros(B, dtype=np.int32)
    bucket_num = np.zeros(B, dtype=np.int32)

    for k in range(K):
        key = (risk_class_idx[k], risk_measure_idx[k], factor_bucket_key[k])
        bid = bucket_map[key]
        bucket_id[k] = bid
        bucket_rc[bid] = risk_class_idx[k]
        bucket_rm[bid] = risk_measure_idx[k]
        bucket_num[bid] = factor_bucket_num[k]

    # Compute concentration factors from CRIF (same method as AADC factor_metadata)
    combined_crif = pd.concat(list(trade_crifs.values()), ignore_index=True)
    factor_metadata = _get_factor_metadata_v2_full(risk_factors, combined_crif)
    concentration_factors = np.array([fm.cr for fm in factor_metadata], dtype=np.float64)

    # --- Intra-bucket correlation matrix (K×K, 0 for cross-bucket) ---
    intra_corr_flat = np.zeros(K * K, dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if bucket_id[i] != bucket_id[j]:
                continue
            if i == j:
                intra_corr_flat[i * K + j] = 1.0
            else:
                rho = _get_intra_correlation(
                    factor_rc_name[i],
                    risk_factors[i][0], risk_factors[j][0],
                    risk_factors[i][3], risk_factors[j][3],
                    factor_bucket_key[i],
                )
                intra_corr_flat[i * K + j] = rho

    # --- Inter-bucket gamma matrix (B×B) with g_bc correction ---
    _eq_inter = np.array([list(row) for row in equity_corr_non_res])
    _cm_inter = np.array([list(row) for row in commodity_corr_non_res])
    _cq_inter = np.array([list(row) for row in creditQ_corr_non_res])

    bucket_cr_rep = np.ones(B, dtype=np.float64)
    for k in range(K):
        bucket_cr_rep[bucket_id[k]] = concentration_factors[k]

    bucket_gamma_flat = np.zeros(B * B, dtype=np.float64)
    for bi in range(B):
        for bj in range(B):
            if bi == bj:
                continue
            if bucket_rc[bi] != bucket_rc[bj] or bucket_rm[bi] != bucket_rm[bj]:
                continue
            rc = bucket_rc[bi]
            b1, b2 = bucket_num[bi], bucket_num[bj]
            gamma = 0.0
            if rc == 0:  # Rates
                gamma = ir_gamma_diff_ccy
            elif rc == 1:  # CreditQ
                if 1 <= b1 <= 12 and 1 <= b2 <= 12:
                    gamma = _cq_inter[b1 - 1, b2 - 1]
                else:
                    gamma = 0.5
            elif rc == 2:  # CreditNonQ
                gamma = cr_gamma_diff_ccy
            elif rc == 3:  # Equity
                if 1 <= b1 <= 12 and 1 <= b2 <= 12:
                    gamma = _eq_inter[b1 - 1, b2 - 1]
            elif rc == 4:  # Commodity
                if 1 <= b1 <= 17 and 1 <= b2 <= 17:
                    gamma = _cm_inter[b1 - 1, b2 - 1]
            if gamma != 0.0:
                cr_b, cr_c = bucket_cr_rep[bi], bucket_cr_rep[bj]
                g_bc = min(cr_b, cr_c) / max(cr_b, cr_c) if max(cr_b, cr_c) > 0 else 1.0
                bucket_gamma_flat[bi * B + bj] = gamma * g_bc

    # Also build the K×K intra_corr_matrix (used by AADC path for compatibility)
    intra_corr_matrix = intra_corr_flat.reshape(K, K).copy()
    # Add inter-bucket IR gamma into the K×K matrix for AADC path
    for i in range(K):
        rc_i = factor_rc_name[i]
        if rc_i != "Rates":
            continue
        qual_i = risk_factors[i][1]
        for j in range(i + 1, K):
            if factor_rc_name[j] != "Rates":
                continue
            qual_j = risk_factors[j][1]
            if qual_i == qual_j:
                continue
            cr_i, cr_j = concentration_factors[i], concentration_factors[j]
            g_bc = min(cr_i, cr_j) / max(cr_i, cr_j) if max(cr_i, cr_j) > 0 else 1.0
            intra_corr_matrix[i, j] = ir_gamma_diff_ccy * g_bc
            intra_corr_matrix[j, i] = ir_gamma_diff_ccy * g_bc

    # Initial allocation
    initial_allocation = np.zeros((T, P), dtype=np.float64)
    trade_id_to_idx = {tid: i for i, tid in enumerate(trade_ids)}
    for i, trade in enumerate(trades):
        if trade.trade_id in trade_id_to_idx:
            t_idx = trade_id_to_idx[trade.trade_id]
            initial_allocation[t_idx, group_ids[i]] = 1.0

    # =========================================================================
    # AADC: Record full ISDA v2.6 kernel (one-time cost)
    # =========================================================================
    funcs = sens_handles = im_output = None
    rec_time = 0.0
    if AADC_AVAILABLE:
        rec_start = time.perf_counter()
        funcs, sens_handles, im_output = record_single_portfolio_simm_kernel_v2_full(
            K, factor_metadata
        )
        rec_time = time.perf_counter() - rec_start

    # =========================================================================
    # GPU: Pre-allocate constants on device (avoids repeated H2D transfers)
    # =========================================================================
    gpu_constants = None
    if CUDA_AVAILABLE:
        gpu_constants = preallocate_gpu_arrays(
            risk_weights, concentration_factors, bucket_id, risk_measure_idx,
            bucket_rc, bucket_rm, intra_corr_flat, bucket_gamma_flat,
            B,
        )

    active_corrs = int(np.sum(intra_corr_flat != 0) - K) // 2  # Exclude diagonal

    return {
        "market": market, "trades": trades, "group_ids": group_ids,
        "currencies": currencies, "trade_crifs": trade_crifs,
        "S": S, "risk_factors": risk_factors, "trade_ids": trade_ids,
        "risk_weights": risk_weights, "risk_class_idx": risk_class_idx,
        "risk_measure_idx": risk_measure_idx,
        "intra_corr_matrix": intra_corr_matrix,
        "intra_corr_flat": intra_corr_flat,
        "bucket_id": bucket_id, "bucket_rc": bucket_rc, "bucket_rm": bucket_rm,
        "bucket_gamma_flat": bucket_gamma_flat, "num_buckets": B,
        "concentration_factors": concentration_factors,
        "initial_allocation": initial_allocation,
        "factor_metadata": factor_metadata,
        # AADC kernel
        "funcs": funcs, "sens_handles": sens_handles, "im_output": im_output,
        "workers": workers, "rec_time": rec_time,
        # GPU constants
        "gpu_constants": gpu_constants,
        # Dimensions
        "T": T, "P": P, "K": K, "B": B, "num_threads": num_threads,
        "crif_time": crif_time, "active_corrs": active_corrs,
    }


# =============================================================================
# Data Export for C++ Backend (apples-to-apples comparison)
# =============================================================================

TENOR_LABEL_TO_IDX = {
    "2w": 0, "1m": 1, "3m": 2, "6m": 3, "1y": 4, "2y": 5,
    "3y": 6, "5y": 7, "10y": 8, "15y": 9, "20y": 10, "30y": 11,
}


def _export_shared_data(ctx, output_dir):
    """
    Export sensitivity matrix, factor metadata, allocation, and trade IDs
    to CSV files for consumption by the C++ AADC backend.
    """
    os.makedirs(output_dir, exist_ok=True)

    S = ctx["S"]
    T, K = S.shape
    risk_factors = ctx["risk_factors"]
    factor_metadata = ctx["factor_metadata"]
    trade_ids = ctx["trade_ids"]
    allocation = ctx["initial_allocation"]
    P = allocation.shape[1]

    # 1. sensitivity_matrix.csv — write header then bulk numpy save
    sm_path = os.path.join(output_dir, "sensitivity_matrix.csv")
    with open(sm_path, "w") as f:
        f.write(f"{T},{K}\n")
    with open(sm_path, "ab") as f:
        np.savetxt(f, S, delimiter=",", fmt="%.15g")

    # 2. risk_factors.csv
    with open(os.path.join(output_dir, "risk_factors.csv"), "w") as f:
        f.write("risk_type,qualifier,bucket,label1\n")
        for rt, qualifier, bucket, label1 in risk_factors:
            bstr = str(bucket) if bucket else "0"
            f.write(f"{rt},{qualifier},{bstr},{label1}\n")

    # 3. factor_metadata.csv
    with open(os.path.join(output_dir, "factor_metadata.csv"), "w") as f:
        f.write("risk_class,risk_measure,risk_type,qualifier,bucket,label1,"
                "tenor_idx,weight,cr,bucket_key\n")
        for fm in factor_metadata:
            rc_name = fm.risk_class
            rm_name = "Vega" if fm.is_vega else "Delta"
            bstr = str(fm.bucket) if fm.bucket else "0"
            tenor_idx = -1
            if fm.risk_type in ("Risk_IRCurve", "Risk_IRVol") and fm.label1:
                tenor_idx = TENOR_LABEL_TO_IDX.get(fm.label1, -1)
            f.write(f"{rc_name},{rm_name},{fm.risk_type},{fm.qualifier},"
                    f"{bstr},{fm.label1},{tenor_idx},"
                    f"{fm.weight:.15g},{fm.cr:.15g},{fm.bucket_key}\n")

    # 4. allocation.csv — write header then bulk numpy save
    alloc_path = os.path.join(output_dir, "allocation.csv")
    with open(alloc_path, "w") as f:
        f.write(f"{T},{P}\n")
    with open(alloc_path, "ab") as f:
        np.savetxt(f, allocation, delimiter=",", fmt="%.15g")

    # 5. trade_ids.csv
    with open(os.path.join(output_dir, "trade_ids.csv"), "w") as f:
        for tid in trade_ids:
            f.write(f"{tid}\n")

    # Verify file integrity
    with open(sm_path, "r") as f:
        header_line = f.readline().strip()
        row_count = sum(1 for _ in f)
    if row_count != T:
        raise RuntimeError(
            f"sensitivity_matrix.csv integrity check failed: header says T={T} "
            f"but file has {row_count} data rows"
        )

    return output_dir


# =============================================================================
# Gradient Functions — Full ISDA SIMM v2.6
# =============================================================================

def _eval_aadc(ctx, agg_S_T):
    """
    Evaluate full ISDA AADC kernel for P portfolios in single evaluate() call.
    Returns (im_values[P], grad_S[P,K]).
    Reuses the recorded kernel from setup — no re-recording.
    """
    P, K = agg_S_T.shape
    workers = ctx["workers"]
    if workers is None:
        workers = aadc.ThreadPool(ctx["num_threads"])

    inputs = {ctx["sens_handles"][k]: agg_S_T[:, k] for k in range(K)}
    request = {ctx["im_output"]: ctx["sens_handles"]}

    results = aadc.evaluate(ctx["funcs"], request, inputs, workers)

    im_values = np.array(results[0][ctx["im_output"]])
    gradients = np.zeros((P, K))
    for k in range(K):
        gradients[:, k] = results[1][ctx["im_output"]][ctx["sens_handles"][k]]

    return im_values, gradients


def _eval_gpu(ctx, agg_S_T):
    """
    Evaluate full ISDA GPU kernel for P portfolios.
    Returns (im_values[P], grad_S[P,K]).
    Uses pre-allocated GPU constants (bucket structure, correlations, etc.).
    """
    return compute_simm_and_gradient_cuda(
        agg_S_T, ctx["risk_weights"], ctx["concentration_factors"],
        ctx["bucket_id"], ctx["risk_measure_idx"],
        ctx["bucket_rc"], ctx["bucket_rm"],
        ctx["intra_corr_flat"], ctx["bucket_gamma_flat"],
        ctx["num_buckets"],
        gpu_arrays=ctx["gpu_constants"],
    )


def _make_aadc_grad_fn(ctx):
    """Closure for AADC gradient evaluation (kernel reuse)."""
    def fn(agg_S_T):
        return _eval_aadc(ctx, agg_S_T)
    return fn


def _make_gpu_grad_fn(ctx):
    """Closure for GPU gradient evaluation."""
    def fn(agg_S_T):
        return _eval_gpu(ctx, agg_S_T)
    return fn


# =============================================================================
# Optimization (shared logic, different gradient backends)
# =============================================================================

def _project_to_simplex(x):
    """Project each row of x onto the probability simplex (vectorized)."""
    T, P = x.shape
    u = np.sort(x, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1)
    arange = np.arange(1, P + 1)
    mask = u * arange > (cssv - 1)
    rho = P - 1 - np.argmax(mask[:, ::-1], axis=1)
    theta = (cssv[np.arange(T), rho] - 1.0) / (rho + 1)
    return np.maximum(x - theta[:, None], 0.0)


def _round_to_integer(x):
    T, P = x.shape
    result = np.zeros_like(x)
    result[np.arange(T), np.argmax(x, axis=1)] = 1.0
    return result


def optimize_allocation(
    S, initial_allocation, grad_fn,
    max_iters=100, lr=None, tol=1e-6, verbose=False, label="",
    method="gradient_descent",
):
    """Allocation optimizer with method selection (gradient_descent or adam)."""
    T, P = initial_allocation.shape

    x = initial_allocation.copy()
    im_history = []
    eval_start = time.perf_counter()
    total_grad_time = 0.0
    num_evals = 0

    # Adam moment estimates
    use_adam = (method == "adam")
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # First evaluation
    agg_S_T = np.dot(S.T, x).T  # (P, K)
    grad_start = time.perf_counter()
    im_values, grad_S = grad_fn(agg_S_T)
    total_grad_time += time.perf_counter() - grad_start
    num_evals += 1

    total_im = float(np.sum(im_values))
    im_history.append(total_im)
    gradient = np.dot(S, grad_S.T)  # (T, P)

    grad_max = np.abs(gradient).max()
    if lr is None:
        if grad_max > 1e-10:
            lr = 1.0 / grad_max if use_adam else 0.3 / grad_max
        else:
            lr = 1e-12

    best_im = total_im
    best_x = x.copy()
    stalled_count = 0

    LS_BETA = 0.5
    LS_MAX_TRIES = 10

    for iteration in range(max_iters):
        if iteration > 0:
            agg_S_T = np.dot(S.T, x).T
            grad_start = time.perf_counter()
            im_values, grad_S = grad_fn(agg_S_T)
            total_grad_time += time.perf_counter() - grad_start
            num_evals += 1

            total_im = float(np.sum(im_values))
            im_history.append(total_im)
            gradient = np.dot(S, grad_S.T)

        if total_im < best_im:
            best_im = total_im
            best_x = x.copy()
            stalled_count = 0
        else:
            stalled_count += 1

        if stalled_count >= 20:
            x = best_x.copy()
            break

        # Compute step direction
        if use_adam:
            t_step = iteration + 1
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** t_step)
            v_hat = v / (1 - beta2 ** t_step)
            direction = m_hat / (np.sqrt(v_hat) + eps)
        else:
            direction = gradient

        # Backtracking line search
        step_size = lr
        for _ in range(LS_MAX_TRIES):
            x_candidate = _project_to_simplex(x - step_size * direction)
            agg_S_c = np.dot(S.T, x_candidate).T
            grad_start = time.perf_counter()
            im_values_c, _ = grad_fn(agg_S_c)
            total_grad_time += time.perf_counter() - grad_start
            num_evals += 1
            candidate_im = float(np.sum(im_values_c))

            if candidate_im < total_im:
                x = x_candidate
                break
            step_size *= LS_BETA

        if iteration > 0 and len(im_history) >= 2:
            rel_change = abs(im_history[-1] - im_history[-2]) / max(abs(im_history[-2]), 1e-10)
            if rel_change < tol:
                break

    eval_time = time.perf_counter() - eval_start
    x_final = _round_to_integer(best_x)

    # Final IM
    agg_final = np.dot(S.T, x_final).T
    im_final, _ = grad_fn(agg_final)
    num_evals += 1
    final_im = float(np.sum(im_final))

    return {
        'final_allocation': x_final,
        'final_im': final_im,
        'initial_im': im_history[0],
        'im_history': im_history,
        'num_iterations': min(iteration + 1, max_iters),
        'num_evals': num_evals,
        'eval_time': eval_time,
        'grad_time': total_grad_time,
        'trades_moved': int(np.sum(
            np.argmax(x_final, axis=1) != np.argmax(initial_allocation, axis=1)
        )),
    }


# =============================================================================
# Greedy Local Search (mirrors C++ greedyLocalSearch in allocation_optimizer.h)
# =============================================================================

def greedy_local_search(
    S, integer_allocation, grad_fn,
    max_rounds=50, verbose=False, label="",
):
    """
    Gradient-guided greedy local search on integer (one-hot) allocations.
    Mirrors C++ greedyLocalSearch() in allocation_optimizer.h:746.

    Uses AADC/GPU gradients to predict promising discrete trade moves,
    validates top-k candidates with actual evaluation, and iterates
    with gradient refresh until no improvement is found.
    """
    T, P = integer_allocation.shape
    K = S.shape[1]
    x = integer_allocation.copy()
    best_x = x.copy()
    im_history = []
    total_eval_time = 0.0
    num_evals = 0

    top_k = min(T, max(20, T // 10))

    # Initial evaluation
    agg_S_T = np.dot(S.T, x).T  # (P, K)
    t0 = time.perf_counter()
    im_values, grad_S = grad_fn(agg_S_T)
    total_eval_time += time.perf_counter() - t0
    num_evals += 1

    total_im = float(np.sum(im_values))
    best_im = total_im
    initial_im = total_im
    im_history.append(total_im)

    # Trade-level gradient: (T, P)
    gradient = np.dot(S, grad_S.T)
    curr_assignments = np.argmax(x, axis=1)  # (T,)

    if verbose:
        print(f"     [{label} Greedy] Initial IM: ${total_im:,.0f} (top_k={top_k})")

    for round_idx in range(max_rounds):
        # Predicted deltas: delta[t,p] = gradient[t,p] - gradient[t, curr_p]
        g_curr = gradient[np.arange(T), curr_assignments]  # (T,)
        deltas = gradient - g_curr[:, None]  # (T, P)
        deltas[np.arange(T), curr_assignments] = 0.0  # no self-move

        # Candidates with negative predicted delta (improvements)
        trade_indices, new_p_indices = np.where(deltas < 0.0)

        if len(trade_indices) == 0:
            if verbose:
                print(f"     [{label} Greedy] No improving candidates, "
                      f"stopping at round {round_idx}")
            break

        predicted_deltas = deltas[trade_indices, new_p_indices]

        # Top-k most negative deltas
        n_try = min(top_k, len(predicted_deltas))
        top_indices = np.argpartition(predicted_deltas, n_try)[:n_try]
        top_indices = top_indices[np.argsort(predicted_deltas[top_indices])]

        improved = False
        accepted = 0
        moved_this_round = set()

        for idx in top_indices:
            t = int(trade_indices[idx])
            new_p = int(new_p_indices[idx])
            curr_p = int(curr_assignments[t])

            if t in moved_this_round:
                continue

            # Apply move
            x[t, curr_p] = 0.0
            x[t, new_p] = 1.0

            # Validate with actual evaluation
            agg_S_T = np.dot(S.T, x).T
            t0 = time.perf_counter()
            im_trial, _ = grad_fn(agg_S_T)
            total_eval_time += time.perf_counter() - t0
            num_evals += 1
            trial_im = float(np.sum(im_trial))

            if trial_im < total_im:
                # Accept
                total_im = trial_im
                curr_assignments[t] = new_p
                improved = True
                accepted += 1
                moved_this_round.add(t)
                if total_im < best_im:
                    best_im = total_im
                    best_x = x.copy()
            else:
                # Revert
                x[t, new_p] = 0.0
                x[t, curr_p] = 1.0

        im_history.append(total_im)

        if verbose:
            print(f"     [{label} Greedy] Round {round_idx}: "
                  f"IM = ${total_im:,.0f}, tried {n_try}, accepted {accepted}")

        if not improved:
            if verbose:
                print(f"     [{label} Greedy] No improvement validated, "
                      f"stopping at round {round_idx}")
            break

        # Refresh gradients for next round
        agg_S_T = np.dot(S.T, x).T
        t0 = time.perf_counter()
        im_values, grad_S = grad_fn(agg_S_T)
        total_eval_time += time.perf_counter() - t0
        num_evals += 1
        total_im = float(np.sum(im_values))
        gradient = np.dot(S, grad_S.T)

    trades_moved = int(np.sum(
        np.argmax(best_x, axis=1) != np.argmax(integer_allocation, axis=1)
    ))

    if verbose:
        reduction = 100.0 * (1.0 - best_im / initial_im) if initial_im > 0 else 0.0
        print(f"     [{label} Greedy] Final IM: ${best_im:,.0f} "
              f"(reduction: {reduction:.1f}%)")

    return {
        'final_allocation': best_x,
        'final_im': best_im,
        'initial_im': initial_im,
        'im_history': im_history,
        'num_evals': num_evals,
        'eval_time': total_eval_time,
        'trades_moved': trades_moved,
    }


# =============================================================================
# GPU Brute-Force Search (forward-only kernel, no gradients)
# =============================================================================

def _make_gpu_im_only_fn(ctx):
    """Closure for forward-only GPU IM evaluation (no gradients)."""
    def fn(sensitivities):
        return compute_simm_im_only_cuda(
            sensitivities, ctx["risk_weights"], ctx["concentration_factors"],
            ctx["bucket_id"], ctx["risk_measure_idx"],
            ctx["bucket_rc"], ctx["bucket_rm"],
            ctx["intra_corr_flat"], ctx["bucket_gamma_flat"],
            ctx["num_buckets"], gpu_arrays=ctx["gpu_constants"],
        )
    return fn


def brute_force_gpu_search(
    S, integer_allocation, ctx,
    max_rounds=50, verbose=False, label="",
) -> dict:
    """
    Brute-force optimizer: evaluates ALL T×(P-1) single-trade moves per round
    using a forward-only GPU kernel (no gradients needed).

    Each candidate move (trade t: src → dst) only changes 2 portfolios.
    We stack the 2C changed rows into a single GPU launch, compute delta IM,
    and pick the best improving move.
    """
    T, P = integer_allocation.shape
    K = S.shape[1]
    x = integer_allocation.copy()
    im_history = []
    total_eval_time = 0.0
    num_evals = 0

    im_fn = _make_gpu_im_only_fn(ctx)

    # Baseline IM
    agg_S = np.dot(S.T, x).T  # (P, K)
    t0 = time.perf_counter()
    base_ims = im_fn(agg_S)  # (P,)
    total_eval_time += time.perf_counter() - t0
    num_evals += 1

    total_im = float(np.sum(base_ims))
    initial_im = total_im
    best_im = total_im
    best_x = x.copy()
    im_history.append(total_im)

    if verbose:
        print(f"     [{label} BruteForce] Initial IM: ${total_im:,.0f}")

    for round_idx in range(max_rounds):
        curr = np.argmax(x, axis=1)  # (T,)

        # Generate ALL T×(P-1) candidate moves (vectorized)
        all_trades = np.repeat(np.arange(T), P)     # (T*P,)
        all_targets = np.tile(np.arange(P), T)       # (T*P,)
        mask = all_targets != np.repeat(curr, P)
        trade_idx = all_trades[mask]                  # (C,)
        dst_p = all_targets[mask]                     # (C,)
        src_p = curr[trade_idx]                       # (C,)
        C = len(trade_idx)

        if C == 0:
            break

        # Incremental aggregation: only 2 portfolios change per candidate
        new_src_agg = agg_S[src_p] - S[trade_idx]    # (C, K)
        new_dst_agg = agg_S[dst_p] + S[trade_idx]    # (C, K)

        # Single GPU launch for all 2C rows
        batch = np.vstack([new_src_agg, new_dst_agg])  # (2C, K)
        t0 = time.perf_counter()
        batch_ims = im_fn(batch)                        # (2C,)
        total_eval_time += time.perf_counter() - t0
        num_evals += 1

        # Compute delta per candidate
        new_src_ims = batch_ims[:C]
        new_dst_ims = batch_ims[C:]
        delta = (new_src_ims - base_ims[src_p]) + (new_dst_ims - base_ims[dst_p])

        # Pick best improving move
        best_cand = np.argmin(delta)
        best_delta = float(delta[best_cand])

        if best_delta >= 0.0:
            if verbose:
                print(f"     [{label} BruteForce] Round {round_idx}: "
                      f"no improving move, stopping")
            break

        # Apply the best move
        t_move = int(trade_idx[best_cand])
        from_p = int(src_p[best_cand])
        to_p = int(dst_p[best_cand])

        x[t_move, from_p] = 0.0
        x[t_move, to_p] = 1.0

        # Update aggregated sensitivities incrementally
        agg_S[from_p] -= S[t_move]
        agg_S[to_p] += S[t_move]

        # Update base IMs for changed portfolios
        base_ims[from_p] = float(new_src_ims[best_cand])
        base_ims[to_p] = float(new_dst_ims[best_cand])

        total_im += best_delta
        im_history.append(total_im)

        if total_im < best_im:
            best_im = total_im
            best_x = x.copy()

        if verbose:
            print(f"     [{label} BruteForce] Round {round_idx}: "
                  f"move trade {t_move} (p{from_p}->p{to_p}), "
                  f"delta=${best_delta:,.0f}, IM=${total_im:,.0f}")

    trades_moved = int(np.sum(
        np.argmax(best_x, axis=1) != np.argmax(integer_allocation, axis=1)
    ))

    if verbose:
        reduction = 100.0 * (1.0 - best_im / initial_im) if initial_im > 0 else 0.0
        print(f"     [{label} BruteForce] Final IM: ${best_im:,.0f} "
              f"(reduction: {reduction:.1f}%, moves: {trades_moved})")

    return {
        'final_allocation': best_x,
        'final_im': best_im,
        'initial_im': initial_im,
        'im_history': im_history,
        'num_evals': num_evals,
        'eval_time': total_eval_time,
        'trades_moved': trades_moved,
        'num_iterations': len(im_history) - 1,
    }


# =============================================================================
# Step 1: Portfolio Setup
# =============================================================================

def step1_portfolio_setup(ctx, verbose=True):
    """7:00 AM - Portfolio setup, kernel recording, first IM evaluation."""
    S = ctx["S"]
    allocation = ctx["initial_allocation"]
    agg_S_T = np.dot(S.T, allocation).T  # (P, K)

    result = StepResult("Portfolio Setup", "7:00 AM")

    # AADC: first evaluation (reuses recorded kernel)
    aadc_ims = aadc_grads = None
    if AADC_AVAILABLE:
        start = time.perf_counter()
        aadc_ims, aadc_grads = _eval_aadc(ctx, agg_S_T)
        result.aadc_time_sec = time.perf_counter() - start
        result.aadc_evals = 1
        result.aadc_kernel_recordings = 1  # Recorded during setup
        result.aadc_kernel_reuses = 1      # This eval reuses the recorded kernel

    # GPU: first evaluation
    gpu_ims = gpu_grads = None
    if CUDA_AVAILABLE:
        start = time.perf_counter()
        gpu_ims, gpu_grads = _eval_gpu(ctx, agg_S_T)
        result.gpu_time_sec = time.perf_counter() - start
        result.gpu_evals = 1

    # Brute-force (forward-only GPU, no gradients)
    bf_ims = None
    if CUDA_AVAILABLE:
        im_fn = _make_gpu_im_only_fn(ctx)
        start = time.perf_counter()
        bf_ims = im_fn(agg_S_T)
        result.bf_time_sec = time.perf_counter() - start
        result.bf_evals = 1

    ims = aadc_ims if aadc_ims is not None else gpu_ims
    if ims is None:
        ims = bf_ims
    result.details = {
        "total_im": float(np.sum(ims)) if ims is not None else 0.0,
        "per_portfolio_im": list(ims) if ims is not None else [],
        "aadc_ims": aadc_ims, "aadc_grads": aadc_grads,
        "gpu_ims": gpu_ims, "gpu_grads": gpu_grads,
        "bf_ims": bf_ims,
        "agg_S_T": agg_S_T,
    }

    if verbose:
        _print_step_header(result)
        print(f"     CRIF computation (shared):   {ctx['crif_time']:.3f} s")
        if AADC_AVAILABLE:
            print(f"     AADC Py kernel recording:    {ctx['rec_time']*1000:.2f} ms")
        print(f"     Correlations: {ctx['active_corrs']} non-trivial pairs")
        cr = ctx["concentration_factors"]
        print(f"     Concentration: min={cr.min():.2f}, max={cr.max():.2f}")
        _print_step_times(result)
        print(f"     Total IM: ${result.details['total_im']:,.2f}")
        # Verify AADC vs GPU match
        if aadc_ims is not None and gpu_ims is not None:
            diff = np.max(np.abs(aadc_ims - gpu_ims))
            rel = diff / max(np.max(np.abs(aadc_ims)), 1e-10)
            print(f"     AADC vs GPU match: max rel diff = {rel:.2e}")

    return result


# =============================================================================
# Step 2: Margin Attribution
# =============================================================================

def step2_margin_attribution(ctx, prev_result, verbose=True):
    """8:00 AM - Gradient-based trade contribution analysis (Euler decomposition)."""
    S = ctx["S"]
    allocation = ctx["initial_allocation"]
    T, K = S.shape
    assignments = np.argmax(allocation, axis=1)

    result = StepResult("Margin Attribution", "8:00 AM")

    def _compute_attribution(grad_S):
        """Euler decomposition: contribution[t] = S[t,:] . grad_S[p(t),:]"""
        return np.sum(S * grad_S[assignments, :], axis=1)

    # AADC: reuses gradient from Step 1 (no new kernel eval)
    aadc_contribs = None
    if AADC_AVAILABLE:
        start = time.perf_counter()
        aadc_contribs = _compute_attribution(prev_result.details["aadc_grads"])
        result.aadc_time_sec = time.perf_counter() - start
        result.aadc_evals = 1               # 1 attribution result (from cached gradient)
        result.aadc_kernel_reuses = 0       # Just numpy on cached gradient

    # GPU: reuses gradient from Step 1
    gpu_contribs = None
    if CUDA_AVAILABLE:
        start = time.perf_counter()
        gpu_contribs = _compute_attribution(prev_result.details["gpu_grads"])
        result.gpu_time_sec = time.perf_counter() - start
        result.gpu_evals = 1               # 1 attribution result (from cached gradient)

    # Brute-force: bump-and-revalue (remove each trade, recompute IM)
    bf_contribs = None
    if CUDA_AVAILABLE:
        agg_S_T = prev_result.details["agg_S_T"]   # (P, K)
        base_ims = prev_result.details.get("bf_ims")
        if base_ims is None:
            base_ims = prev_result.details.get("gpu_ims")
        if base_ims is not None:
            im_fn = _make_gpu_im_only_fn(ctx)
            start = time.perf_counter()
            # For each trade t in portfolio p, compute IM of p without t
            removed_agg = agg_S_T[assignments] - S  # (T, K): each row = portfolio agg minus trade
            removed_ims = im_fn(removed_agg)         # (T,) single GPU launch
            bf_contribs = base_ims[assignments] - removed_ims  # contribution per trade
            result.bf_time_sec = time.perf_counter() - start
            result.bf_evals = 1  # single batched GPU launch

    contribs = aadc_contribs if aadc_contribs is not None else gpu_contribs
    if contribs is None:
        contribs = bf_contribs
    total_im = prev_result.details["total_im"]

    if contribs is not None:
        sorted_idx = np.argsort(contribs)
        top_consumers = sorted_idx[-5:][::-1]
        top_reducers = sorted_idx[:5]
        euler_sum = float(np.sum(contribs))
        euler_error_pct = abs(euler_sum - total_im) / max(abs(total_im), 1e-10) * 100
    else:
        top_consumers = top_reducers = []
        euler_sum = euler_error_pct = 0.0

    result.details = {
        "contributions": contribs,
        "top_consumers": top_consumers,
        "top_reducers": top_reducers,
        "euler_sum": euler_sum,
        "euler_error_pct": euler_error_pct,
        "total_im": total_im,
    }

    if verbose:
        _print_step_header(result)
        _print_step_times(result)
        if contribs is not None:
            trade_ids = ctx["trade_ids"]
            tc = top_consumers[0]
            tr = top_reducers[0]
            print(f"     Top consumer: {trade_ids[tc]} "
                  f"(${contribs[tc]:,.0f}, {contribs[tc]/total_im*100:.1f}%)")
            print(f"     Top reducer:  {trade_ids[tr]} "
                  f"(${contribs[tr]:,.0f}, {contribs[tr]/total_im*100:.1f}%)")
            print(f"     Euler check:  {euler_error_pct:.4f}% error")
            print(f"     Kernel reuse: gradient from Step 1 (0 new evals)")

    return result


# =============================================================================
# Step 3: Intraday Pre-Trade Checks
# =============================================================================

def step3_intraday_trading(ctx, prev_step1, num_new_trades=50,
                           refresh_interval=10, verbose=True):
    """9AM-4PM - Route new trades to optimal counterparty."""
    S = ctx["S"]
    allocation = ctx["initial_allocation"].copy()
    risk_factors = ctx["risk_factors"]
    market = ctx["market"]
    K = ctx["K"]
    P = ctx["P"]
    T = ctx["T"]

    result = StepResult("Intraday Pre-Trade", "9AM-4PM")

    # Generate new trades with different seed
    new_trades = generate_trades_by_type(
        "ir_swap", num_new_trades, ctx["currencies"][:3], seed=9999
    )

    # Compute CRIF for new trades and build sensitivity vectors
    new_trade_sens = []
    rf_index = {rf: k for k, rf in enumerate(risk_factors)}
    for trade in new_trades:
        crif_rows = compute_crif_for_trade(trade, market)
        s_new = np.zeros(K, dtype=np.float64)
        for row in crif_rows:
            key = (row["RiskType"], row.get("Qualifier", ""),
                   row.get("Bucket", ""), row.get("Label1", ""))
            if key in rf_index:
                s_new[rf_index[key]] += row["AmountUSD"]
        new_trade_sens.append(s_new)

    def _route_trades(grad_fn, label):
        """Route new trades using gradient-based marginal IM."""
        local_S = S.copy()
        local_alloc = allocation.copy()

        # Extend for new trades
        local_S = np.vstack([local_S, np.zeros((num_new_trades, K))])
        local_alloc = np.vstack([local_alloc, np.zeros((num_new_trades, P))])

        routing = []
        num_evals = 0
        grad_S = None
        total_time = 0.0

        for i, s_new in enumerate(new_trade_sens):
            if i % refresh_interval == 0:
                agg_S = np.dot(local_S.T, local_alloc)
                start = time.perf_counter()
                im_vals, grad_S = grad_fn(agg_S.T)
                total_time += time.perf_counter() - start
                num_evals += 1

            # Marginal IM for each portfolio
            marginal_ims = grad_S @ s_new  # (P,)
            best_p = int(np.argmin(marginal_ims))

            # "Execute" trade
            t_new = T + i
            local_S[t_new, :] = s_new
            local_alloc[t_new, best_p] = 1.0
            routing.append((i, best_p, float(marginal_ims[best_p])))

        return routing, num_evals, total_time

    # AADC (reuses recorded kernel for every gradient refresh)
    aadc_routing = None
    if AADC_AVAILABLE:
        grad_fn = _make_aadc_grad_fn(ctx)
        aadc_routing, aadc_evals, aadc_time = _route_trades(grad_fn, "AADC")
        result.aadc_time_sec = aadc_time
        result.aadc_evals = aadc_evals
        result.aadc_kernel_reuses = aadc_evals  # Every eval reuses the kernel

    # GPU
    gpu_routing = None
    if CUDA_AVAILABLE:
        grad_fn = _make_gpu_grad_fn(ctx)
        gpu_routing, gpu_evals, gpu_time = _route_trades(grad_fn, "GPU")
        result.gpu_time_sec = gpu_time
        result.gpu_evals = gpu_evals

    # Brute-force: try all P portfolios per trade (forward-only GPU)
    bf_routing = None
    if CUDA_AVAILABLE:
        im_fn = _make_gpu_im_only_fn(ctx)
        local_S = S.copy()
        local_alloc = allocation.copy()
        local_S = np.vstack([local_S, np.zeros((num_new_trades, K))])
        local_alloc = np.vstack([local_alloc, np.zeros((num_new_trades, P))])

        bf_routing = []
        bf_evals = 0
        bf_time = 0.0

        for i, s_new in enumerate(new_trade_sens):
            # Compute current agg_S
            agg_S = np.dot(local_S.T, local_alloc).T  # (P, K)

            # Build P candidate rows: agg_S[p] + s_new for each p
            candidates = agg_S + s_new[np.newaxis, :]  # (P, K) — broadcast

            start = time.perf_counter()
            cand_ims = im_fn(candidates)  # (P,) single GPU launch
            bf_time += time.perf_counter() - start
            bf_evals += 1

            # Current portfolio IMs (from agg_S)
            start = time.perf_counter()
            base_ims = im_fn(agg_S)  # (P,)
            bf_time += time.perf_counter() - start
            bf_evals += 1

            marginal_ims = cand_ims - base_ims  # (P,)
            best_p = int(np.argmin(marginal_ims))

            t_new = T + i
            local_S[t_new, :] = s_new
            local_alloc[t_new, best_p] = 1.0
            bf_routing.append((i, best_p, float(marginal_ims[best_p])))

        result.bf_time_sec = bf_time
        result.bf_evals = bf_evals

    routing = aadc_routing if aadc_routing is not None else gpu_routing
    if routing is None:
        routing = bf_routing
    if routing:
        portfolio_counts = np.zeros(P, dtype=int)
        for _, best_p, _ in routing:
            portfolio_counts[best_p] += 1
        result.details = {
            "num_new_trades": num_new_trades,
            "refresh_interval": refresh_interval,
            "routing": routing,
            "portfolio_counts": portfolio_counts,
        }
    else:
        result.details = {"num_new_trades": num_new_trades}

    if verbose:
        _print_step_header(result)
        _print_step_times(result)
        evals = result.aadc_evals or result.gpu_evals
        print(f"     {num_new_trades} trades routed ({evals} gradient refreshes)")
        if routing:
            counts_str = ", ".join(f"P{i}:{c}" for i, c in enumerate(portfolio_counts) if c > 0)
            print(f"     Distribution: {counts_str}")
            print(f"     Kernel reuse: {evals} evals, 0 re-recordings")

    return result


# =============================================================================
# Step 4: What-If Scenarios
# =============================================================================

def step4_whatif_scenarios(ctx, step1_result, step2_result, verbose=True):
    """2:00 PM - Stress tests and scenario analysis."""
    agg_S_T = step1_result.details["agg_S_T"]
    risk_class_idx = ctx["risk_class_idx"]
    K = ctx["K"]
    S = ctx["S"]
    allocation = ctx["initial_allocation"]
    total_im = step1_result.details["total_im"]

    result = StepResult("What-If Scenarios", "2:00 PM")

    def _run_scenarios(grad_fn, label):
        scenarios = {}
        total_time = 0.0
        num_evals = 0

        # (a) Stress: Rates +50bp
        shocked = agg_S_T.copy()
        for k in range(K):
            if risk_class_idx[k] == 0:
                shocked[:, k] *= 1.5
        start = time.perf_counter()
        stress_ims, _ = grad_fn(shocked)
        total_time += time.perf_counter() - start
        num_evals += 1
        stress_im = float(np.sum(stress_ims))
        scenarios["stress_50bp"] = {
            "im": stress_im,
            "change_pct": (stress_im - total_im) / total_im * 100,
        }

        # (b) Unwind top 5 contributors
        contribs = step2_result.details.get("contributions")
        if contribs is not None:
            top5 = step2_result.details["top_consumers"][:5]
            mod_alloc = allocation.copy()
            for t_idx in top5:
                mod_alloc[t_idx, :] = 0.0
            agg_mod = np.dot(S.T, mod_alloc).T
            start = time.perf_counter()
            unwind_ims, _ = grad_fn(agg_mod)
            total_time += time.perf_counter() - start
            num_evals += 1
            unwind_im = float(np.sum(unwind_ims))
            scenarios["unwind_top5"] = {
                "im": unwind_im,
                "change_pct": (unwind_im - total_im) / total_im * 100,
            }

        # (c) Add hedge (reverse top contributor)
        if contribs is not None:
            top1 = step2_result.details["top_consumers"][0]
            top1_portfolio = np.argmax(allocation[top1])
            hedge_sens = -S[top1, :]
            hedged = agg_S_T.copy()
            hedged[top1_portfolio, :] += hedge_sens
            start = time.perf_counter()
            hedge_ims, _ = grad_fn(hedged)
            total_time += time.perf_counter() - start
            num_evals += 1
            hedge_im = float(np.sum(hedge_ims))
            scenarios["add_hedge"] = {
                "im": hedge_im,
                "change_pct": (hedge_im - total_im) / total_im * 100,
            }

        # (d) IM ladder
        shock_levels = [0.5, 0.75, 1.0, 1.25, 1.5]
        ladder = []
        for shock in shock_levels:
            shocked = agg_S_T.copy()
            for k in range(K):
                if risk_class_idx[k] == 0:
                    shocked[:, k] *= shock
            start = time.perf_counter()
            ladder_ims, _ = grad_fn(shocked)
            total_time += time.perf_counter() - start
            num_evals += 1
            ladder.append(float(np.sum(ladder_ims)))
        scenarios["im_ladder"] = {
            "shock_levels": shock_levels,
            "im_values": ladder,
        }

        return scenarios, num_evals, total_time

    # AADC (all evals reuse recorded kernel)
    aadc_scenarios = None
    if AADC_AVAILABLE:
        grad_fn = _make_aadc_grad_fn(ctx)
        aadc_scenarios, aadc_evals, aadc_time = _run_scenarios(grad_fn, "AADC")
        result.aadc_time_sec = aadc_time
        result.aadc_evals = aadc_evals
        result.aadc_kernel_reuses = aadc_evals

    # GPU
    gpu_scenarios = None
    if CUDA_AVAILABLE:
        grad_fn = _make_gpu_grad_fn(ctx)
        gpu_scenarios, gpu_evals, gpu_time = _run_scenarios(grad_fn, "GPU")
        result.gpu_time_sec = gpu_time
        result.gpu_evals = gpu_evals

    # Brute-force: forward-only GPU (same scenarios, no gradient computation)
    bf_scenarios = None
    if CUDA_AVAILABLE:
        im_fn = _make_gpu_im_only_fn(ctx)
        bf_grad_fn = lambda agg: (im_fn(agg), None)  # wrap as grad_fn shape
        bf_scenarios, bf_evals, bf_time = _run_scenarios(bf_grad_fn, "BF")
        result.bf_time_sec = bf_time
        result.bf_evals = bf_evals

    scenarios = aadc_scenarios if aadc_scenarios is not None else gpu_scenarios
    if scenarios is None:
        scenarios = bf_scenarios
    result.details = {"scenarios": scenarios, "base_im": total_im}

    if verbose:
        _print_step_header(result)
        _print_step_times(result)
        evals = result.aadc_evals or result.gpu_evals
        print(f"     {evals} scenario evaluations (kernel reuse, 0 re-recordings)")
        if scenarios:
            if "stress_50bp" in scenarios:
                s = scenarios["stress_50bp"]
                print(f"     Stress +50bp:  ${s['im']:,.0f} ({s['change_pct']:+.1f}%)")
            if "unwind_top5" in scenarios:
                s = scenarios["unwind_top5"]
                print(f"     Unwind top 5:  ${s['im']:,.0f} ({s['change_pct']:+.1f}%)")
            if "add_hedge" in scenarios:
                s = scenarios["add_hedge"]
                print(f"     Add hedge:     ${s['im']:,.0f} ({s['change_pct']:+.1f}%)")
            if "im_ladder" in scenarios:
                ladder = scenarios["im_ladder"]
                ladder_str = ", ".join(f"${v:,.0f}" for v in ladder["im_values"])
                print(f"     IM ladder:     [{ladder_str}]")

    return result


# =============================================================================
# Step 5: EOD Optimization
# =============================================================================

def step5_eod_optimization(ctx, max_iters=100, verbose=True, exclude=None):
    """5:00 PM - Portfolio optimization. Runs all methods not in exclude list."""
    S = ctx["S"]
    allocation = ctx["initial_allocation"]
    exclude = exclude or []
    results = []

    ALL_METHODS = [
        ("gradient_descent", "EOD: GD"),
        ("adam", "EOD: Adam"),
        ("gpu_brute_force", "EOD: Brute-Force"),
    ]

    for method, step_name in ALL_METHODS:
        if method in exclude:
            continue

        result = StepResult(step_name, "5:00 PM")

        if method == "gpu_brute_force":
            # GPU brute-force: forward-only kernel, no gradients
            gpu_opt = None
            aadc_opt = None
            if CUDA_AVAILABLE:
                start = time.perf_counter()
                gpu_opt = brute_force_gpu_search(
                    S, allocation, ctx,
                    max_rounds=max_rounds_for_brute_force(max_iters),
                    verbose=verbose, label="GPU",
                )
                result.bf_time_sec = time.perf_counter() - start
                result.bf_evals = gpu_opt["num_evals"]
            else:
                if verbose:
                    print("     GPU brute-force requires CUDA. Skipping.")
        else:
            # AADC: Phase 1 (continuous) + Phase 2 (greedy)
            aadc_opt = None
            if AADC_AVAILABLE:
                grad_fn = _make_aadc_grad_fn(ctx)
                start = time.perf_counter()
                aadc_opt = optimize_allocation(
                    S, allocation, grad_fn,
                    max_iters=max_iters, verbose=False, label="AADC",
                    method=method,
                )
                # Phase 2: greedy local search on rounded integer result
                greedy = greedy_local_search(
                    S, aadc_opt['final_allocation'], grad_fn,
                    max_rounds=50, verbose=verbose, label="AADC",
                )
                if greedy['final_im'] < aadc_opt['final_im']:
                    aadc_opt['final_allocation'] = greedy['final_allocation']
                    aadc_opt['final_im'] = greedy['final_im']
                    aadc_opt['trades_moved'] = greedy['trades_moved']
                aadc_opt['num_evals'] += greedy['num_evals']
                result.aadc_time_sec = time.perf_counter() - start
                result.aadc_evals = aadc_opt["num_evals"]
                result.aadc_kernel_reuses = aadc_opt["num_evals"]  # All reuse

            # GPU: Phase 1 (continuous) + Phase 2 (greedy)
            gpu_opt = None
            if CUDA_AVAILABLE:
                grad_fn = _make_gpu_grad_fn(ctx)
                start = time.perf_counter()
                gpu_opt = optimize_allocation(
                    S, allocation, grad_fn,
                    max_iters=max_iters, verbose=False, label="GPU",
                    method=method,
                )
                # Phase 2: greedy local search on rounded integer result
                greedy = greedy_local_search(
                    S, gpu_opt['final_allocation'], grad_fn,
                    max_rounds=50, verbose=verbose, label="GPU",
                )
                if greedy['final_im'] < gpu_opt['final_im']:
                    gpu_opt['final_allocation'] = greedy['final_allocation']
                    gpu_opt['final_im'] = greedy['final_im']
                    gpu_opt['trades_moved'] = greedy['trades_moved']
                gpu_opt['num_evals'] += greedy['num_evals']
                result.gpu_time_sec = time.perf_counter() - start
                result.gpu_evals = gpu_opt["num_evals"]

        opt = aadc_opt if aadc_opt is not None else gpu_opt
        if opt:
            reduction_pct = (opt["initial_im"] - opt["final_im"]) / opt["initial_im"] * 100
            result.details = {
                "initial_im": opt["initial_im"],
                "final_im": opt["final_im"],
                "reduction_pct": reduction_pct,
                "trades_moved": opt["trades_moved"],
                "iterations": opt["num_iterations"],
                "aadc_opt": aadc_opt,
                "gpu_opt": gpu_opt,
                "method": method,
            }
        else:
            result.details = {"method": method}

        if verbose:
            _print_step_header(result)
            _print_step_times(result)
            if opt:
                print(f"     Initial IM:  ${opt['initial_im']:,.0f}")
                print(f"     Final IM:    ${opt['final_im']:,.0f} "
                      f"(reduction: {reduction_pct:.1f}%)")
                print(f"     Trades moved: {opt['trades_moved']}, "
                      f"Iterations: {opt['num_iterations']}"
                      + ("" if method == "gpu_brute_force" else " + greedy"))
                if AADC_AVAILABLE and method != "gpu_brute_force":
                    print(f"     AADC Py: {result.aadc_evals} evals, "
                          f"all kernel reuse (0 re-recordings)")

        results.append(result)

    return results


def max_rounds_for_brute_force(max_iters):
    """Map --optimize-iters to brute-force max_rounds (1 move per round)."""
    return max_iters


# =============================================================================
# Output Helpers
# =============================================================================

def _print_step_header(result: StepResult):
    print(f"\n  {result.step_time} - {result.step_name}")
    print(f"  {'-' * 60}")


def _fmt_time(t):
    if t is None:
        return "N/A"
    if t < 0.001:
        return f"{t*1e6:.0f} us"
    if t < 1.0:
        return f"{t*1000:.2f} ms"
    return f"{t:.3f} s"


def _print_step_times(result: StepResult):
    aadc_str = _fmt_time(result.aadc_time_sec)
    gpu_str = _fmt_time(result.gpu_time_sec)
    cpp_str = _fmt_time(result.cpp_time_sec)
    bf_str = _fmt_time(result.bf_time_sec)
    speedup = ""
    if result.aadc_time_sec and result.gpu_time_sec and result.gpu_time_sec > 0:
        ratio = result.aadc_time_sec / result.gpu_time_sec
        if ratio > 1:
            speedup = f"  (GPU {ratio:.1f}x faster)"
        else:
            speedup = f"  (AADC {1/ratio:.1f}x faster)"
    line = f"     AADC Py: {aadc_str:<12}  GPU: {gpu_str:<12}"
    if result.bf_time_sec is not None:
        line += f"  BF: {bf_str:<12}"
    if result.cpp_time_sec is not None:
        line += f"  C++ AADC: {cpp_str:<12}"
    line += speedup
    print(line)


def _match_label(py_val, cpp_val, tol_pct=1.0):
    """Return MATCH/DIFF label comparing two values. tol_pct is % tolerance."""
    if py_val is None or cpp_val is None:
        return ""
    if py_val == 0 and cpp_val == 0:
        return "MATCH"
    ref = max(abs(py_val), abs(cpp_val), 1e-10)
    diff_pct = abs(py_val - cpp_val) / ref * 100
    if diff_pct < tol_pct:
        return "MATCH"
    return f"DIFF {diff_pct:.2f}%"


def _fmt_im(val):
    """Format IM value compactly."""
    return f"${val:,.0f}"


def print_cpp_vs_python_comparison(steps, ctx):
    """Reprint per-step results with all three backends (AADC Py, GPU, C++ AADC)."""
    has_any = any("cpp" in s.details for s in steps)
    if not has_any:
        return

    s1, s2, s3, s4 = steps[:4]
    eod_steps = steps[4:]

    def _print_im_row(label, py_val, cpp_val):
        """Print a row comparing IM values."""
        py_s = _fmt_im(py_val) if py_val is not None else "N/A"
        cpp_s = _fmt_im(cpp_val) if cpp_val is not None else "N/A"
        ml = _match_label(py_val, cpp_val) if py_val is not None and cpp_val is not None else ""
        print(f"    {label:<16} Py {py_s:>22}  |  C++ {cpp_s:>22}  [{ml}]")

    print("\n" + "=" * 78)
    print("  STEP-BY-STEP RESULTS: AADC Py vs GPU vs C++ AADC")
    print("=" * 78)

    # --- Step 1: Portfolio Setup ---
    cpp_attr = s1.details.get("cpp", {})
    print(f"\n  7:00 AM - Portfolio Setup")
    print(f"  {'-' * 72}")
    print(f"     CRIF computation (shared):   {ctx['crif_time']:.3f} s")
    if AADC_AVAILABLE:
        print(f"     AADC Py kernel recording:    {ctx['rec_time']*1000:.2f} ms")
    _print_step_times(s1)
    print(f"     Total IM: ${s1.details['total_im']:,.2f}")
    aadc_ims = s1.details.get("aadc_ims")
    gpu_ims = s1.details.get("gpu_ims")
    if aadc_ims is not None and gpu_ims is not None:
        diff = np.max(np.abs(aadc_ims - gpu_ims))
        rel = diff / max(np.max(np.abs(aadc_ims)), 1e-10)
        print(f"     AADC vs GPU match: max rel diff = {rel:.2e}")

    # --- Step 2: Margin Attribution ---
    cpp_attr = s2.details.get("cpp", {})
    contribs = s2.details.get("contributions")
    total_im = s2.details.get("total_im", 0)
    print(f"\n  8:00 AM - Margin Attribution")
    print(f"  {'-' * 72}")
    _print_step_times(s2)
    if contribs is not None:
        trade_ids = ctx["trade_ids"]
        tc = s2.details["top_consumers"][0]
        tr = s2.details["top_reducers"][0]
        euler_error_pct = s2.details.get("euler_error_pct", 0)
        print(f"     Top consumer: {trade_ids[tc]} "
              f"(${contribs[tc]:,.0f}, {contribs[tc]/total_im*100:.1f}%)")
        print(f"     Top reducer:  {trade_ids[tr]} "
              f"(${contribs[tr]:,.0f}, {contribs[tr]/total_im*100:.1f}%)")
        print(f"     Euler check:  {euler_error_pct:.4f}% error")
    if cpp_attr:
        cpp_im = cpp_attr.get("total_im")
        cpp_euler = cpp_attr.get("euler_ratio")
        if cpp_im is not None:
            _print_im_row("Total IM:", total_im, cpp_im)
            if total_im and abs(total_im - cpp_im) / max(abs(total_im), 1e-10) > 0.01:
                print(f"    {'':>16} (C++ attribution: single-portfolio; see EOD for full match)")
        if cpp_euler is not None:
            py_euler_sum = s2.details.get("euler_sum", 0)
            py_euler_ratio = py_euler_sum / total_im if total_im and total_im > 0 else None
            if py_euler_ratio is not None:
                ml = _match_label(py_euler_ratio, cpp_euler)
                print(f"    {'Euler ratio:':<16} Py {py_euler_ratio:>22.6f}  |  C++ {cpp_euler:>22.6f}  [{ml}]")

    # --- Step 3: Intraday Pre-Trade ---
    print(f"\n  9AM-4PM - Intraday Pre-Trade")
    print(f"  {'-' * 72}")
    _print_step_times(s3)
    d3 = s3.details
    evals = s3.aadc_evals or s3.gpu_evals
    print(f"     {d3.get('num_new_trades', 0)} trades routed ({evals} gradient refreshes)")
    if "portfolio_counts" in d3:
        counts_str = ", ".join(f"P{i}:{c}" for i, c in enumerate(d3["portfolio_counts"]) if c > 0)
        print(f"     Distribution: {counts_str}")

    # --- Step 4: What-If Scenarios ---
    cpp_wi = s4.details.get("cpp", {})
    py_scenarios = s4.details.get("scenarios", {})
    print(f"\n  2:00 PM - What-If Scenarios")
    print(f"  {'-' * 72}")
    _print_step_times(s4)
    evals = s4.aadc_evals or s4.gpu_evals
    print(f"     {evals} scenario evaluations (kernel reuse, 0 re-recordings)")
    if py_scenarios:
        if "stress_50bp" in py_scenarios:
            s = py_scenarios["stress_50bp"]
            py_stress_im = s['im']
            line = f"     Stress +50bp:  ${py_stress_im:,.0f} ({s['change_pct']:+.1f}%)"
            cpp_stress = cpp_wi.get("scenarios", {}).get("stress_ir", {})
            if cpp_stress:
                cpp_v = cpp_stress.get("scenario_im")
                ml = _match_label(py_stress_im, cpp_v)
                line += f"    C++: ${cpp_v:,.0f} [{ml}]"
            print(line)
        if "unwind_top5" in py_scenarios:
            s = py_scenarios["unwind_top5"]
            py_unwind_im = s['im']
            line = f"     Unwind top 5:  ${py_unwind_im:,.0f} ({s['change_pct']:+.1f}%)"
            cpp_unwind = cpp_wi.get("scenarios", {}).get("unwind", {})
            if cpp_unwind:
                cpp_v = cpp_unwind.get("scenario_im")
                ml = _match_label(py_unwind_im, cpp_v)
                line += f"    C++: ${cpp_v:,.0f} [{ml}]"
            print(line)
        if "add_hedge" in py_scenarios:
            s = py_scenarios["add_hedge"]
            print(f"     Add hedge:     ${s['im']:,.0f} ({s['change_pct']:+.1f}%)")
        if "im_ladder" in py_scenarios:
            ladder = py_scenarios["im_ladder"]
            ladder_str = ", ".join(f"${v:,.0f}" for v in ladder["im_values"])
            print(f"     IM ladder:     [{ladder_str}]")
    if cpp_wi:
        cpp_stress_eq = cpp_wi.get("scenarios", {}).get("stress_eq", {})
        if cpp_stress_eq:
            print(f"     Stress Equity: C++ ${cpp_stress_eq.get('scenario_im', 0):,.0f} "
                  f"({cpp_stress_eq.get('change_pct', 0):+.1f}%)")
        cpp_marginal = cpp_wi.get("marginal_im")
        if cpp_marginal is not None:
            print(f"     Marginal IM:   C++ ${cpp_marginal:,.0f}")

    # --- Step 5: EOD Optimization (one section per method) ---
    for es in eod_steps:
        cpp_opt = es.details.get("cpp", {})
        print(f"\n  5:00 PM - {es.step_name}")
        print(f"  {'-' * 72}")
        _print_step_times(es)
        d5 = es.details
        if "initial_im" in d5:
            py_init = d5["initial_im"]
            py_final = d5["final_im"]
            reduction_pct = d5.get("reduction_pct", 0)
            print(f"     Initial IM:  ${py_init:,.0f}")
            print(f"     Final IM:    ${py_final:,.0f} "
                  f"(reduction: {reduction_pct:.1f}%)")
            print(f"     Trades moved: {d5.get('trades_moved', 0)}, "
                  f"Iterations: {d5.get('iterations', 0)}")
            if cpp_opt:
                cpp_init = cpp_opt.get("initial_im")
                cpp_final = cpp_opt.get("final_im")
                if cpp_init is not None:
                    _print_im_row("Initial IM:", py_init, cpp_init)
                if cpp_final is not None:
                    _print_im_row("Final IM:", py_final, cpp_final)
                cpp_moved = cpp_opt.get("trades_moved")
                if cpp_moved is not None:
                    match = "MATCH" if d5.get("trades_moved") == cpp_moved else \
                        f"DIFF (Py={d5.get('trades_moved')}, C++={cpp_moved})"
                    print(f"    {'Trades moved:':<16} Py {d5.get('trades_moved', 0):>22}  "
                          f"|  C++ {cpp_moved:>22}  [{match}]")

    print("\n" + "=" * 78)


def print_workflow_summary(steps, economics, config):
    """Print the daily kernel economics summary."""
    print("\n" + "=" * 70)
    print("  DAILY KERNEL ECONOMICS (Full ISDA SIMM v2.6)")
    print("=" * 70)
    print(f"  Kernel recordings:                {economics.total_recordings}")
    print(f"  AADC Py recording (1-time):       {economics.recording_time_sec*1000:.2f} ms")
    print(f"  Total AADC Py evaluate() calls:   {economics.total_aadc_evals}")
    print(f"    of which kernel reuses:         {economics.total_aadc_kernel_reuses}")
    print(f"  Total GPU kernel launches:        {economics.total_gpu_evals}")
    if economics.total_bf_evals > 0:
        print(f"  Total BF (forward-only) evals:    {economics.total_bf_evals}")
    print(f"  Amortized recording cost/eval:    {economics.amortized_recording_ms:.2f} ms")
    print(f"  Cumulative AADC Py eval time:     {_fmt_time(economics.cumulative_aadc_time)}")
    print(f"  Cumulative GPU eval time:         {_fmt_time(economics.cumulative_gpu_time)}")
    if economics.cumulative_bf_time > 0:
        print(f"  Cumulative BF eval time:          {_fmt_time(economics.cumulative_bf_time)}")
    aadc_total = economics.recording_time_sec + economics.cumulative_aadc_time
    print(f"  AADC Py total (rec + eval):       {_fmt_time(aadc_total)}")
    if economics.cumulative_gpu_time > 0 and economics.cumulative_aadc_time > 0:
        print(f"  GPU/AADC Py speedup (eval only):  {economics.cumulative_aadc_time/economics.cumulative_gpu_time:.1f}x")
        print(f"  GPU/AADC Py speedup (inc. rec):   {aadc_total/economics.cumulative_gpu_time:.1f}x")
    if economics.cumulative_cpp_time > 0:
        print(f"  --- C++ AADC ---")
        print(f"  C++ AADC recording (1-time):      {economics.cpp_recording_time_sec*1000:.2f} ms")
        print(f"  Total C++ AADC evals:             {economics.total_cpp_evals}")
        print(f"  Cumulative C++ AADC eval time:    {_fmt_time(economics.cumulative_cpp_time)}")
        cpp_total = economics.cpp_recording_time_sec + economics.cumulative_cpp_time
        print(f"  C++ AADC total (rec + eval):      {_fmt_time(cpp_total)}")
        if economics.cumulative_aadc_time > 0:
            print(f"  C++/Py AADC speedup (eval):       {economics.cumulative_aadc_time/economics.cumulative_cpp_time:.1f}x")
    print("=" * 70)


def print_step_summary_table(steps):
    """Print condensed per-step summary table."""
    has_cpp = any(s.cpp_time_sec is not None for s in steps)
    has_bf = any(s.bf_time_sec is not None for s in steps)
    w = 97 if has_bf else 85
    print("\n" + "=" * w)
    print("  PER-STEP SUMMARY (Full ISDA SIMM v2.6)")
    print("=" * w)
    hdr = f"  {'Step':<30} {'AADC Py':>12} {'GPU':>12}"
    sep = f"  {'-'*30} {'-'*12} {'-'*12}"
    if has_bf:
        hdr += f" {'BF':>12}"
        sep += f" {'-'*12}"
    if has_cpp:
        hdr += f" {'C++ AADC':>12}"
        sep += f" {'-'*12}"
    hdr += f" {'Evals':>6} {'Reuse':>6}"
    sep += f" {'-'*6} {'-'*6}"
    print(hdr)
    print(sep)
    for s in steps:
        evals = max(s.aadc_evals, s.gpu_evals, s.cpp_evals, s.bf_evals)
        reuse = s.aadc_kernel_reuses
        line = (f"  {s.step_name:<30} {_fmt_time(s.aadc_time_sec):>12} "
                f"{_fmt_time(s.gpu_time_sec):>12}")
        if has_bf:
            line += f" {_fmt_time(s.bf_time_sec):>12}"
        if has_cpp:
            line += f" {_fmt_time(s.cpp_time_sec):>12}"
        line += f" {evals:>6} {reuse:>6}"
        print(line)
    print("=" * w)


# =============================================================================
# Markdown Report
# =============================================================================

def write_workflow_markdown(steps, economics, config, md_path=None):
    """Append markdown report to benchmark_trading_workflow.md."""
    if md_path is None:
        md_path = MD_PATH

    file_exists = os.path.exists(md_path) and os.path.getsize(md_path) > 0
    lines = []

    if not file_exists:
        lines.append("# Trading Day Workflow Benchmark (Full ISDA SIMM v2.6)\n")
        lines.append("Auto-generated by `benchmark_trading_workflow.py`. "
                      "Each run is appended below.\n")

    lines.append("---\n")
    lines.append(f"## Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("### Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| SIMM formula | Full ISDA v2.6 (correlations + concentration) |")
    lines.append(f"| Trades | {config['num_trades']:,} |")
    lines.append(f"| Portfolios | {config['num_portfolios']} |")
    lines.append(f"| Trade types | {config['trade_types']} |")
    lines.append(f"| Risk factors (K) | {config['K']} |")
    lines.append(f"| Intra-bucket correlations | {config.get('active_corrs', 'N/A')} pairs |")
    lines.append(f"| New trades (intraday) | {config['num_new_trades']} |")
    lines.append(f"| Optimize iterations | {config['optimize_iters']} |")
    lines.append(f"| Threads | {config['num_threads']} |")
    lines.append(f"| AADC available | {AADC_AVAILABLE} |")
    lines.append(f"| CUDA available | {CUDA_AVAILABLE}{' (simulator)' if CUDA_SIMULATOR else ''} |")
    lines.append(f"| C++ AADC available | {CPP_AVAILABLE} |")
    lines.append("")

    has_cpp = any(s.cpp_time_sec is not None for s in steps)
    has_bf = any(s.bf_time_sec is not None for s in steps)

    lines.append("### Per-Step Results\n")
    # Build header dynamically based on available backends
    hdr_cols = ["Step", "AADC Py Time", "GPU Time"]
    if has_bf:
        hdr_cols.append("BF Time")
    if has_cpp:
        hdr_cols.append("C++ AADC Time")
    hdr_cols += ["Evals", "Kernel Reuses"]
    lines.append("| " + " | ".join(hdr_cols) + " |")
    lines.append("|" + "|".join(["------"] * len(hdr_cols)) + "|")
    for s in steps:
        evals = max(s.aadc_evals, s.gpu_evals, s.cpp_evals, s.bf_evals)
        row = [f"{s.step_time} {s.step_name}",
               _fmt_time(s.aadc_time_sec), _fmt_time(s.gpu_time_sec)]
        if has_bf:
            row.append(_fmt_time(s.bf_time_sec))
        if has_cpp:
            row.append(_fmt_time(s.cpp_time_sec))
        row += [str(evals), str(s.aadc_kernel_reuses)]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Kernel Economics
    aadc_total = economics.recording_time_sec + economics.cumulative_aadc_time
    lines.append("### Kernel Economics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Kernel recordings | {economics.total_recordings} |")
    lines.append(f"| AADC Py recording (1-time) | {economics.recording_time_sec*1000:.2f} ms |")
    lines.append(f"| Total AADC Py evals | {economics.total_aadc_evals} |")
    lines.append(f"| Total kernel reuses | {economics.total_aadc_kernel_reuses} |")
    lines.append(f"| Total GPU evals | {economics.total_gpu_evals} |")
    if economics.total_bf_evals > 0:
        lines.append(f"| Total BF (forward-only) evals | {economics.total_bf_evals} |")
    lines.append(f"| Amortized recording/eval | {economics.amortized_recording_ms:.2f} ms |")
    lines.append(f"| Cumulative AADC Py time | {_fmt_time(economics.cumulative_aadc_time)} |")
    lines.append(f"| Cumulative GPU time | {_fmt_time(economics.cumulative_gpu_time)} |")
    if economics.cumulative_bf_time > 0:
        lines.append(f"| Cumulative BF time | {_fmt_time(economics.cumulative_bf_time)} |")
    lines.append(f"| AADC Py total (rec + eval) | {_fmt_time(aadc_total)} |")
    if economics.cumulative_gpu_time > 0 and economics.cumulative_aadc_time > 0:
        lines.append(f"| GPU speedup (eval only) | {economics.cumulative_aadc_time/economics.cumulative_gpu_time:.1f}x |")
        lines.append(f"| GPU speedup (inc. recording) | {aadc_total/economics.cumulative_gpu_time:.1f}x |")
    if economics.cumulative_cpp_time > 0:
        cpp_total = economics.cpp_recording_time_sec + economics.cumulative_cpp_time
        lines.append(f"| C++ AADC recording (1-time) | {economics.cpp_recording_time_sec*1000:.2f} ms |")
        lines.append(f"| Total C++ AADC evals | {economics.total_cpp_evals} |")
        lines.append(f"| Cumulative C++ AADC time | {_fmt_time(economics.cumulative_cpp_time)} |")
        lines.append(f"| C++ AADC total (rec + eval) | {_fmt_time(cpp_total)} |")
        if economics.cumulative_aadc_time > 0:
            lines.append(f"| C++/Py AADC speedup (eval) | {economics.cumulative_aadc_time/economics.cumulative_cpp_time:.1f}x |")
    lines.append("")

    # Step details
    for s in steps:
        d = s.details
        if s.step_name == "Margin Attribution" and "euler_error_pct" in d:
            lines.append(f"### {s.step_time} {s.step_name}\n")
            lines.append(f"- Euler decomposition error: {d['euler_error_pct']:.4f}%")
            lines.append("")
        elif s.step_name == "What-If Scenarios" and "scenarios" in d and d["scenarios"]:
            sc = d["scenarios"]
            lines.append(f"### {s.step_time} {s.step_name}\n")
            lines.append("| Scenario | IM | Change |")
            lines.append("|----------|-------|--------|")
            lines.append(f"| Baseline | ${d['base_im']:,.0f} | - |")
            if "stress_50bp" in sc:
                lines.append(f"| Rates +50bp | ${sc['stress_50bp']['im']:,.0f} | {sc['stress_50bp']['change_pct']:+.1f}% |")
            if "unwind_top5" in sc:
                lines.append(f"| Unwind top 5 | ${sc['unwind_top5']['im']:,.0f} | {sc['unwind_top5']['change_pct']:+.1f}% |")
            if "add_hedge" in sc:
                lines.append(f"| Add hedge | ${sc['add_hedge']['im']:,.0f} | {sc['add_hedge']['change_pct']:+.1f}% |")
            lines.append("")
            if "im_ladder" in sc:
                ladder = sc["im_ladder"]
                lines.append("**IM Ladder:** " + ", ".join(
                    f"{sh}x: ${v:,.0f}" for sh, v in zip(ladder["shock_levels"], ladder["im_values"])
                ))
                lines.append("")
        elif s.step_name.startswith("EOD:") and "initial_im" in d:
            lines.append(f"### {s.step_time} {s.step_name}\n")
            lines.append(f"- Initial IM: ${d['initial_im']:,.0f}")
            lines.append(f"- Final IM: ${d['final_im']:,.0f} (reduction: {d['reduction_pct']:.1f}%)")
            lines.append(f"- Trades moved: {d['trades_moved']}, Iterations: {d['iterations']}")
            lines.append("")

    lines.append("")

    with open(md_path, 'a') as f:
        f.write('\n'.join(lines))

    print(f"\n  Results appended to {md_path}")


# =============================================================================
# CSV Logging
# =============================================================================

def log_workflow_results(steps, economics, config):
    """Log workflow results to execution_log_portfolio.csv."""
    timestamp = datetime.now().isoformat()
    trade_types_str = config["trade_types"]
    log_rows = []
    max_iters = config.get("optimize_iters", 100)

    for s in steps:
        common = dict(
            timestamp=timestamp,
            trade_types_str=trade_types_str,
            num_trades=config["num_trades"],
            num_simm_buckets=config.get("num_simm_buckets", 3),
            num_portfolios=config["num_portfolios"],
            num_risk_factors=config["K"],
            crif_time_sec=config.get("crif_time", 0),
        )
        im_result = s.details.get("total_im", s.details.get("initial_im", 0))
        is_eod = s.step_name.startswith("EOD:")
        method = s.details.get("method", "gradient_descent") if is_eod else None

        # Build optimize_result dicts for each backend
        def _py_opt(key):
            """Build optimize_result from a Python backend (aadc_opt / gpu_opt)."""
            opt = s.details.get(key) if is_eod else None
            if opt:
                opt["max_iters"] = max_iters
                opt["method"] = method
            return opt

        aadc_opt = _py_opt("aadc_opt")
        gpu_opt = _py_opt("gpu_opt")

        # C++ optimize_result (from parsed C++ output)
        cpp_opt = None
        if is_eod:
            cpp_data = s.details.get("cpp", {})
            if "initial_im" in cpp_data and "final_im" in cpp_data:
                wall_ms = cpp_data.get("optimization_wall_ms",
                          cpp_data.get("optimization_eval_ms", 0))
                cpp_opt = {
                    "initial_im": cpp_data["initial_im"],
                    "final_im": cpp_data["final_im"],
                    "eval_time": wall_ms / 1000.0,
                    "trades_moved": cpp_data.get("trades_moved", 0),
                    "num_iterations": cpp_data.get("iterations", 0),
                    "max_iters": max_iters,
                    "method": method,
                }

        # Canonical opt: first available, used for backends without their own
        canonical_opt = aadc_opt or gpu_opt or cpp_opt
        if canonical_opt and "eval_time" not in canonical_opt:
            # aadc_opt/gpu_opt have 'eval_time' key from optimize_allocation
            pass

        # BF optimize_result — gpu_opt holds brute_force_gpu_search result for BF method
        bf_opt = gpu_opt if method == "gpu_brute_force" else canonical_opt
        if bf_opt and "method" not in bf_opt:
            bf_opt = dict(bf_opt, method=method)

        step_name = s.step_name.lower().replace(' ', '_')

        # Emit a row per backend that actually ran (has timing or evals)
        if AADC_AVAILABLE and (s.aadc_time_sec or s.aadc_evals):
            log_rows.append(_build_benchmark_log_row(
                model_name=f"workflow_{step_name}_aadc_full",
                model_version=MODEL_VERSION,
                num_threads=config["num_threads"],
                im_result=im_result,
                eval_time_sec=s.aadc_time_sec or 0,
                kernel_recording_sec=economics.recording_time_sec if s.step_name == "Portfolio Setup" else None,
                optimize_result=aadc_opt or (canonical_opt if is_eod else None),
                num_simm_evals=s.aadc_evals,
                **common,
            ))

        if CUDA_AVAILABLE and (s.gpu_time_sec or s.gpu_evals):
            log_rows.append(_build_benchmark_log_row(
                model_name=f"workflow_{step_name}_gpu_full",
                model_version=MODEL_VERSION,
                num_threads=1,
                im_result=im_result,
                eval_time_sec=s.gpu_time_sec or 0,
                optimize_result=gpu_opt or (canonical_opt if is_eod else None),
                num_simm_evals=s.gpu_evals,
                **common,
            ))

        if CUDA_AVAILABLE and (s.bf_time_sec or s.bf_evals):
            log_rows.append(_build_benchmark_log_row(
                model_name=f"workflow_{step_name}_bf_gpu",
                model_version=MODEL_VERSION,
                num_threads=1,
                im_result=im_result,
                eval_time_sec=s.bf_time_sec or 0,
                optimize_result=bf_opt if is_eod else None,
                num_simm_evals=s.bf_evals,
                **common,
            ))

        if CPP_AVAILABLE and (s.cpp_time_sec or s.cpp_evals):
            log_rows.append(_build_benchmark_log_row(
                model_name=f"workflow_{step_name}_cpp_aadc",
                model_version=MODEL_VERSION,
                num_threads=config["num_threads"],
                im_result=im_result,
                eval_time_sec=s.cpp_time_sec or 0,
                kernel_recording_sec=economics.cpp_recording_time_sec if s.step_name == "Portfolio Setup" else None,
                optimize_result=cpp_opt or (canonical_opt if is_eod else None),
                num_simm_evals=s.cpp_evals,
                **common,
            ))

    if log_rows:
        write_log(log_rows)
        print(f"  Logged {len(log_rows)} rows to data/execution_log_portfolio.csv")


# =============================================================================
# Main Workflow
# =============================================================================

def run_trading_workflow(
    num_trades=1000, num_portfolios=5, trade_types=None, num_threads=8,
    num_new_trades=50, optimize_iters=100, num_simm_buckets=3,
    refresh_interval=10, verbose=True, command_str="",
    exclude=None, output_file=None,
):
    if trade_types is None:
        trade_types = ["ir_swap"]

    # Tee output to file if requested
    tee = None
    if output_file:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tee = TeeWriter(output_path)
        sys.stdout = tee

    print("=" * 70)
    print("  Trading Day Workflow: AADC Py vs GPU vs C++ AADC (Full ISDA SIMM v2.6)")
    print("=" * 70)
    print(f"  Trades: {num_trades:<12} Portfolios: {num_portfolios}")
    print(f"  Types:  {','.join(trade_types):<12} Threads: {num_threads}")
    excluded = exclude or []
    methods_run = [m for m in ['gradient_descent', 'adam', 'gpu_brute_force'] if m not in excluded]
    print(f"  New trades: {num_new_trades:<8} Optimize iters: {optimize_iters}  Methods: {', '.join(methods_run)}")
    print(f"  AADC: {'Available' if AADC_AVAILABLE else 'NOT available':<12} "
          f"CUDA: {'Available' if CUDA_AVAILABLE else 'NOT available':<12} "
          f"C++: {'Available' if CPP_AVAILABLE else 'NOT available'}")
    if CUDA_AVAILABLE and CUDA_SIMULATOR:
        print(f"  CUDA mode: Simulator (timings not meaningful)")
    print(f"  Formula: Full ISDA v2.6 (correlations + concentration)")

    if not AADC_AVAILABLE and not CUDA_AVAILABLE and not CPP_AVAILABLE:
        print("\nERROR: No backends available. Cannot run benchmark.")
        return None

    # Setup (generates portfolio, CRIFs, builds corr/conc, records AADC kernel)
    print("\n  Setting up portfolio, correlations, kernel...")
    ctx = setup_portfolio_and_kernel(
        num_trades, num_portfolios, trade_types, num_simm_buckets, num_threads
    )
    print(f"  S matrix: {ctx['T']} trades x {ctx['K']} risk factors, {ctx['B']} buckets")
    print(f"  Intra-bucket correlations: {ctx['active_corrs']} non-trivial pairs")

    economics = KernelEconomics(recording_time_sec=ctx["rec_time"])

    config = {
        "num_trades": ctx["T"], "num_portfolios": ctx["P"],
        "trade_types": ",".join(trade_types), "num_threads": num_threads,
        "K": ctx["K"], "num_new_trades": num_new_trades,
        "optimize_iters": optimize_iters, "num_simm_buckets": num_simm_buckets,
        "crif_time": ctx["crif_time"], "command": command_str,
        "active_corrs": ctx["active_corrs"], "exclude": excluded,
    }

    # Run steps
    # When C++ is available, suppress per-step verbose output — it will be
    # reprinted after C++ results are merged (so all 3 backends appear together).
    step_verbose = verbose and not CPP_AVAILABLE
    steps = []

    s1 = step1_portfolio_setup(ctx, step_verbose)
    economics.update(s1)
    steps.append(s1)
    if CPP_AVAILABLE and verbose:
        print(f"  Step 1 done: AADC Py {_fmt_time(s1.aadc_time_sec)}, "
              f"GPU {_fmt_time(s1.gpu_time_sec)}")

    s2 = step2_margin_attribution(ctx, s1, step_verbose)
    economics.update(s2)
    steps.append(s2)
    if CPP_AVAILABLE and verbose:
        print(f"  Step 2 done: AADC Py {_fmt_time(s2.aadc_time_sec)}, "
              f"GPU {_fmt_time(s2.gpu_time_sec)}")

    s3 = step3_intraday_trading(ctx, s1, num_new_trades, refresh_interval, step_verbose)
    economics.update(s3)
    steps.append(s3)
    if CPP_AVAILABLE and verbose:
        print(f"  Step 3 done: AADC Py {_fmt_time(s3.aadc_time_sec)}, "
              f"GPU {_fmt_time(s3.gpu_time_sec)}")

    s4 = step4_whatif_scenarios(ctx, s1, s2, step_verbose)
    economics.update(s4)
    steps.append(s4)
    if CPP_AVAILABLE and verbose:
        print(f"  Step 4 done: AADC Py {_fmt_time(s4.aadc_time_sec)}, "
              f"GPU {_fmt_time(s4.gpu_time_sec)}")

    s5_list = step5_eod_optimization(ctx, optimize_iters, step_verbose, exclude=excluded)
    for s5 in s5_list:
        economics.update(s5)
        steps.append(s5)
        if CPP_AVAILABLE and verbose:
            print(f"  Step 5 ({s5.step_name}) done: AADC Py {_fmt_time(s5.aadc_time_sec)}, "
                  f"GPU {_fmt_time(s5.gpu_time_sec)}")

    # C++ AADC backend (runs all modes on shared data for apples-to-apples)
    if CPP_AVAILABLE:
        shared_data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "shared_benchmark_data"
        )
        _export_shared_data(ctx, shared_data_dir)
        print(f"\n  Exported shared data to {shared_data_dir} for C++ backend")

        cpp_results = _run_all_cpp_modes(
            num_trades, num_portfolios, num_threads, optimize_iters,
            seed=42, input_dir=shared_data_dir
        )
        _apply_cpp_results(steps, economics, cpp_results)
        if economics.cpp_recording_time_sec > 0:
            print(f"  C++ AADC kernel recording:      {economics.cpp_recording_time_sec*1000:.2f} ms")

    # Reprint step results with all backends (including C++)
    if CPP_AVAILABLE and verbose:
        print_cpp_vs_python_comparison(steps, ctx)

    # Summary
    print_step_summary_table(steps)
    print_workflow_summary(steps, economics, config)

    # Logging
    log_workflow_results(steps, economics, config)
    write_workflow_markdown(steps, economics, config)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)

    if tee:
        print(f"\n  Console output saved to {output_path}")
        tee.close()

    return {"steps": steps, "economics": economics, "config": config}


def main():
    parser = argparse.ArgumentParser(
        description="Trading Day Workflow Benchmark: AADC Python vs GPU vs C++ AADC (Full ISDA SIMM v2.6)"
    )
    parser.add_argument('--trades', '-t', type=int, default=1000)
    parser.add_argument('--portfolios', '-p', type=int, default=5)
    parser.add_argument('--trade-types', type=str, default='ir_swap')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--new-trades', type=int, default=50)
    parser.add_argument('--optimize-iters', type=int, default=100)
    parser.add_argument('--simm-buckets', type=int, default=3)
    parser.add_argument('--refresh-interval', type=int, default=10)
    parser.add_argument('--exclude', nargs='*',
                        choices=['gradient_descent', 'adam', 'gpu_brute_force'],
                        default=[],
                        help='Optimization methods to skip in EOD step (default: run all)')
    parser.add_argument('--output', '-o', type=str, default='auto',
                        help='Save console output to file (default: auto-generated in data/). Use "none" to disable.')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()
    trade_types = [t.strip() for t in args.trade_types.split(',')]

    # Resolve output file
    if args.output == 'none':
        output_file = None
    elif args.output == 'auto':
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"data/benchmark_workflow_{ts}.txt"
    else:
        output_file = args.output

    cmd_parts = ["python benchmark_trading_workflow.py"]
    cmd_parts.append(f"--trades {args.trades}")
    cmd_parts.append(f"--portfolios {args.portfolios}")
    if args.trade_types != 'ir_swap':
        cmd_parts.append(f"--trade-types {args.trade_types}")
    cmd_parts.append(f"--threads {args.threads}")
    cmd_parts.append(f"--new-trades {args.new_trades}")
    cmd_parts.append(f"--optimize-iters {args.optimize_iters}")
    if args.exclude:
        cmd_parts.append(f"--exclude {' '.join(args.exclude)}")
    command_str = ' '.join(cmd_parts)

    run_trading_workflow(
        num_trades=args.trades,
        num_portfolios=args.portfolios,
        trade_types=trade_types,
        num_threads=args.threads,
        num_new_trades=args.new_trades,
        optimize_iters=args.optimize_iters,
        num_simm_buckets=args.simm_buckets,
        refresh_interval=args.refresh_interval,
        verbose=not args.quiet,
        command_str=command_str,
        exclude=args.exclude,
        output_file=output_file,
    )


if __name__ == '__main__':
    main()
