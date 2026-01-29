#!/usr/bin/env python
"""Test script for allocation optimization."""

import sys
sys.path.insert(0, '.')

import numpy as np
from model.trade_types import generate_market_environment, generate_trades_by_type
from model.simm_allocation_optimizer import reallocate_trades_optimal

# Generate trades
currencies = ['USD', 'EUR']
print('Generating market and trades...')
market = generate_market_environment(currencies, seed=42)

trades = []
for tt in ['ir_swap', 'equity_option']:
    tt_trades = generate_trades_by_type(tt, 50, currencies, seed=42)
    trades.extend(tt_trades)

T = len(trades)
P = 5
print(f'Total trades: {T}, portfolios: {P}')

# Initial random allocation
np.random.seed(42)
group_ids = np.random.randint(0, P, T)
initial_allocation = np.zeros((T, P))
for t, g in enumerate(group_ids):
    initial_allocation[t, g] = 1.0

print('Running optimization...')
result = reallocate_trades_optimal(
    trades, market, P,
    initial_allocation=initial_allocation,
    num_threads=4,
    allow_partial=False,
    method='gradient_descent',
    max_iters=100,
    verbose=True,
)

print()
print('=' * 60)
print('OPTIMIZATION RESULTS')
print('=' * 60)
print(f'Initial IM:     ${result["initial_im"]:>18,.2f}')
print(f'Final IM:       ${result["final_im"]:>18,.2f}')
reduction = (1 - result['final_im'] / result['initial_im']) * 100 if result['initial_im'] > 0 else 0
print(f'Reduction:      {reduction:.1f}%')
print(f'Trades moved:   {result["trades_moved"]} of {T}')
print(f'Iterations:     {result["num_iterations"]}')
print(f'Converged:      {result["converged"]}')
print(f'Time:           {result["elapsed_time"]:.2f}s')
