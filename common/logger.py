"""
ISDA-SIMM Execution Logger

Logs benchmark results to data/execution_log.csv for performance tracking.
Enhanced to match the Asian Options Benchmark format for consistent comparison.
"""

import os
import csv
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path


@dataclass
class OperationCounts:
    """Mathematical operation counts for performance analysis."""
    total_generic_ops: int = 0    # +, -, *, /
    total_exp_ops: int = 0        # exp()
    total_sqrt_ops: int = 0       # sqrt()
    total_log_ops: int = 0        # log()
    total_comparisons: int = 0    # max(), min(), conditionals

    @property
    def total_math_ops(self) -> int:
        """Total of all mathematical operations."""
        return (self.total_generic_ops + self.total_exp_ops +
                self.total_sqrt_ops + self.total_log_ops + self.total_comparisons)


@dataclass
class SIMMExecutionRecord:
    """Record for a single SIMM benchmark execution.

    Enhanced to match Asian Options Benchmark format for consistent comparison.
    """
    # Model identification
    model_name: str
    model_version: str
    mode: str  # "price_only", "price_with_greeks", "margin_with_greeks"

    # Configuration
    num_trades: int
    num_risk_factors: int      # Number of risk factors (tenors × currencies)
    num_sensitivities: int     # Total sensitivities computed
    num_threads: int

    # Pricing results
    portfolio_value: float = 0.0
    avg_trade_value: float = 0.0
    min_trade_value: float = 0.0
    max_trade_value: float = 0.0

    # Greeks (averages)
    avg_delta: float = 0.0     # Average IR delta per trade

    # SIMM specific
    simm_total: float = 0.0
    crif_rows: int = 0

    # Timing
    eval_time_sec: float = 0.0
    first_run_time_sec: float = 0.0
    steady_state_time_sec: float = 0.0
    recording_time_sec: float = 0.0       # AADC kernel recording (if applicable)
    kernel_execution_time_sec: float = 0.0  # Per-evaluation time

    # Performance
    num_evals: int = 0
    threads_used: int = 1
    memory_mb: float = 0.0
    data_memory_mb: float = 0.0
    kernel_memory_mb: float = 0.0
    batch_size: int = 0

    # Operation counts
    operation_counts: Optional[OperationCounts] = None

    # Code metrics
    model_total_lines: int = 0
    model_math_lines: int = 0

    # Model metadata
    language: str = "Python"
    uses_aadc: bool = False

    # Status
    status: str = "success"
    error_message: str = ""

    @property
    def total_eval_time_sec(self) -> float:
        """Total evaluation time including recording overhead."""
        return self.eval_time_sec + self.recording_time_sec


class SIMMLogger:
    """Logger for SIMM benchmark executions.

    Enhanced to match Asian Options Benchmark format.
    """

    # CSV column headers - matching Asian Options format where applicable
    HEADER = [
        # Timestamp and model info
        "timestamp",
        "model_name",
        "model_version",
        "mode",

        # Configuration
        "num_trades",
        "num_risk_factors",
        "num_sensitivities",
        "num_threads",

        # Pricing results
        "portfolio_value",
        "avg_trade_value",
        "min_trade_value",
        "max_trade_value",

        # Greeks
        "avg_delta",

        # SIMM specific
        "simm_total",
        "crif_rows",

        # Timing
        "eval_time_sec",
        "first_run_time_sec",
        "steady_state_time_sec",
        "recording_time_sec",
        "kernel_execution_time_sec",
        "total_eval_time_sec",

        # Performance
        "num_evals",
        "threads_used",
        "memory_mb",
        "data_memory_mb",
        "kernel_memory_mb",
        "batch_size",

        # Operation counts
        "total_generic_ops",
        "total_exp_ops",
        "total_sqrt_ops",
        "total_log_ops",
        "total_comparisons",
        "total_math_ops",

        # Code metrics
        "model_total_lines",
        "model_math_lines",

        # Model metadata
        "language",
        "uses_aadc",

        # Status
        "status",
        "error_message",
    ]

    def __init__(self, log_dir: Optional[str] = None):
        """Initialize logger with optional custom log directory."""
        if log_dir is None:
            # Default to data/ directory relative to ISDA-SIMM root
            log_dir = Path(__file__).parent.parent / "data"
        self.log_path = Path(log_dir) / "execution_log.csv"
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        """Create log file with header if it doesn't exist."""
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADER)

    def log(self, record: SIMMExecutionRecord):
        """Log a single execution record."""
        ops = record.operation_counts or OperationCounts()

        row = [
            datetime.now().isoformat(),
            record.model_name,
            record.model_version,
            record.mode,

            # Configuration
            record.num_trades,
            record.num_risk_factors,
            record.num_sensitivities,
            record.num_threads,

            # Pricing results
            record.portfolio_value,
            record.avg_trade_value,
            record.min_trade_value,
            record.max_trade_value,

            # Greeks
            record.avg_delta,

            # SIMM specific
            record.simm_total,
            record.crif_rows,

            # Timing
            record.eval_time_sec,
            record.first_run_time_sec,
            record.steady_state_time_sec,
            record.recording_time_sec,
            record.kernel_execution_time_sec,
            record.total_eval_time_sec,

            # Performance
            record.num_evals,
            record.threads_used,
            record.memory_mb,
            record.data_memory_mb if record.data_memory_mb else "",
            record.kernel_memory_mb if record.kernel_memory_mb else "",
            record.batch_size if record.batch_size else "",

            # Operation counts
            ops.total_generic_ops,
            ops.total_exp_ops,
            ops.total_sqrt_ops,
            ops.total_log_ops,
            ops.total_comparisons,
            ops.total_math_ops,

            # Code metrics
            record.model_total_lines,
            record.model_math_lines,

            # Model metadata
            record.language,
            "yes" if record.uses_aadc else "no",

            # Status
            record.status,
            record.error_message,
        ]

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_execution(
        self,
        model_name: str,
        model_version: str,
        mode: str,
        num_trades: int,
        num_risk_factors: int,
        num_sensitivities: int,
        num_threads: int,
        eval_time_sec: float,
        memory_mb: float,
        language: str,
        uses_aadc: bool,
        # Optional fields
        portfolio_value: float = 0.0,
        avg_trade_value: float = 0.0,
        min_trade_value: float = 0.0,
        max_trade_value: float = 0.0,
        avg_delta: float = 0.0,
        simm_total: float = 0.0,
        crif_rows: int = 0,
        first_run_time_sec: float = 0.0,
        steady_state_time_sec: float = 0.0,
        recording_time_sec: float = 0.0,
        kernel_execution_time_sec: float = 0.0,
        num_evals: int = 0,
        threads_used: int = 1,
        data_memory_mb: float = 0.0,
        kernel_memory_mb: float = 0.0,
        batch_size: int = 0,
        operation_counts: Optional[OperationCounts] = None,
        model_total_lines: int = 0,
        model_math_lines: int = 0,
        status: str = "success",
        error_message: str = ""
    ):
        """Convenience method to log execution with individual parameters."""
        record = SIMMExecutionRecord(
            model_name=model_name,
            model_version=model_version,
            mode=mode,
            num_trades=num_trades,
            num_risk_factors=num_risk_factors,
            num_sensitivities=num_sensitivities,
            num_threads=num_threads,
            portfolio_value=portfolio_value,
            avg_trade_value=avg_trade_value,
            min_trade_value=min_trade_value,
            max_trade_value=max_trade_value,
            avg_delta=avg_delta,
            simm_total=simm_total,
            crif_rows=crif_rows,
            eval_time_sec=eval_time_sec,
            first_run_time_sec=first_run_time_sec,
            steady_state_time_sec=steady_state_time_sec,
            recording_time_sec=recording_time_sec,
            kernel_execution_time_sec=kernel_execution_time_sec,
            num_evals=num_evals,
            threads_used=threads_used,
            memory_mb=memory_mb,
            data_memory_mb=data_memory_mb,
            kernel_memory_mb=kernel_memory_mb,
            batch_size=batch_size,
            operation_counts=operation_counts,
            model_total_lines=model_total_lines,
            model_math_lines=model_math_lines,
            language=language,
            uses_aadc=uses_aadc,
            status=status,
            error_message=error_message,
        )
        self.log(record)


# Global logger instance
_logger = None


def get_logger() -> SIMMLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = SIMMLogger()
    return _logger


def log_execution(*args, **kwargs):
    """Convenience function to log execution using global logger."""
    get_logger().log_execution(*args, **kwargs)
