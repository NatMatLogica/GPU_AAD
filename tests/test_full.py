#!/usr/bin/env python3
"""Full test of IR swap pricer with enhanced logging."""
import sys
sys.path.insert(0, '/home/natashamanito/ISDA-SIMM')

import numpy as np
import tracemalloc
from pathlib import Path

from model.ir_swap_pricer import (
    MODEL_NAME, MODEL_VERSION, TENOR_LABELS,
    generate_trades, generate_market_data,
    price_with_greeks, generate_crif,
    measure_memory, count_operations
)
from common.logger import get_logger, SIMMExecutionRecord
from common.utils import count_code_lines

# Config
num_trades = 100
currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']

print('='*70)
print('IR Swap Pricer - Baseline v1.1.0')
print('='*70)

# Code metrics
model_file = '/home/natashamanito/ISDA-SIMM/model/ir_swap_pricer.py'
model_total_lines, model_math_lines = count_code_lines(model_file)
print(f'Code Lines: {model_total_lines} total, {model_math_lines} math')

# Memory
tracemalloc.start()
mem_before = measure_memory()

# Generate data
market_data = {c: generate_market_data(c, seed=42+i) for i, c in enumerate(currencies)}
trades = generate_trades(num_trades, currencies, seed=42)
data_mem = measure_memory() - mem_before
print(f'Data Memory: {data_mem:.1f} MB')

# Run pricing
result = price_with_greeks(trades, market_data)
print(f'\nResults:')
print(f'  Portfolio PV:   ${result.prices.sum():,.2f}')
print(f'  Eval Time:      {result.eval_time:.3f} s')
print(f'  Num Evals:      {result.num_evals:,}')
print(f'  Sensitivities:  {result.num_sensitivities:,}')
print(f'  First Run:      {result.first_run_time*1000:.2f} ms')

# CRIF
crif = generate_crif(trades, result)
print(f'  CRIF Rows:      {len(crif):,}')

# SIMM
try:
    from src.agg_margins import SIMM
    portfolio = SIMM(crif, 'USD', 1)
    simm_total = portfolio.simm
    print(f'  SIMM Total:     ${simm_total:,.2f}')
except Exception as e:
    simm_total = 0.0
    print(f'  SIMM Error:     {e}')

# Ops
ops = count_operations(num_trades, len(currencies), len(TENOR_LABELS), result.num_evals)
print(f'\nOperation Counts: {ops.total_math_ops:,} total math ops')

# Log
mem_total = measure_memory() - mem_before + data_mem
avg_delta = np.mean(np.abs(result.ir_delta).sum(axis=(1, 2)))

logger = get_logger()
record = SIMMExecutionRecord(
    model_name=MODEL_NAME,
    model_version=MODEL_VERSION,
    mode='price_with_greeks',
    num_trades=num_trades,
    num_risk_factors=len(TENOR_LABELS) * len(currencies),
    num_sensitivities=result.num_sensitivities,
    num_threads=1,
    portfolio_value=float(result.prices.sum()),
    avg_trade_value=float(result.prices.mean()),
    min_trade_value=float(result.prices.min()),
    max_trade_value=float(result.prices.max()),
    avg_delta=avg_delta,
    simm_total=simm_total,
    crif_rows=len(crif),
    eval_time_sec=result.eval_time,
    first_run_time_sec=result.first_run_time,
    steady_state_time_sec=result.steady_state_time,
    num_evals=result.num_evals,
    memory_mb=mem_total,
    data_memory_mb=data_mem,
    operation_counts=ops,
    model_total_lines=model_total_lines,
    model_math_lines=model_math_lines,
    language='Python',
    uses_aadc=False,
)
logger.log(record)
print(f'\nLogged to {logger.log_path}')
tracemalloc.stop()
print('Test complete!')
