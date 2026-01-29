#!/usr/bin/env python
"""Minimal kernel reuse benchmark."""
import sys
sys.path.insert(0, '.')
import time
import numpy as np
import aadc
from model.trade_types import IRSwapTrade, generate_market_environment

IR_TENORS = np.array([0.04, 0.08, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30])
NUM_IR_TENORS = 12

def interp_aadc(times, tenors, rates_aadc):
    results = []
    for t in times:
        if t <= tenors[0]:
            results.append(rates_aadc[0])
        elif t >= tenors[-1]:
            results.append(rates_aadc[-1])
        else:
            for i in range(len(tenors) - 1):
                if tenors[i] <= t <= tenors[i + 1]:
                    w = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
                    results.append(rates_aadc[i] * (1 - w) + rates_aadc[i + 1] * w)
                    break
    return results

def discount_factors_aadc(times, rates_aadc):
    return [np.exp(-r * t) for r, t in zip(interp_aadc(times, IR_TENORS, rates_aadc), times)]

def forward_rates_aadc(start_times, end_times, rates_aadc):
    df_starts = discount_factors_aadc(start_times, rates_aadc)
    df_ends = discount_factors_aadc(end_times, rates_aadc)
    return [(df_starts[i] / df_ends[i] - 1.0) / (end_times[i] - start_times[i])
            if end_times[i] > start_times[i] else df_starts[i] * 0.0
            for i in range(len(start_times))]

def price_irs(notional, fixed_rate, maturity, frequency, is_payer, rates_aadc):
    dt = 1.0 / frequency
    num_periods = int(maturity * frequency)
    payment_times = np.array([(i + 1) * dt for i in range(num_periods)])
    start_times = np.array([i * dt for i in range(num_periods)])
    dfs = discount_factors_aadc(payment_times, rates_aadc)
    fwd_rates = forward_rates_aadc(start_times, payment_times, rates_aadc)
    fixed_leg = notional * fixed_rate * dt * dfs[0]
    floating_leg = notional * fwd_rates[0] * dt * dfs[0]
    for i in range(1, num_periods):
        fixed_leg = fixed_leg + notional * fixed_rate * dt * dfs[i]
        floating_leg = floating_leg + notional * fwd_rates[i] * dt * dfs[i]
    return floating_leg - fixed_leg if is_payer else fixed_leg - floating_leg

def record_kernel(maturity, frequency, is_payer, currency, market):
    curve = market.curves[currency]
    nodiff_inputs = {}
    with aadc.record_kernel() as funcs:
        rates_aadc, rate_handles = [], []
        for i in range(NUM_IR_TENORS):
            r = aadc.idouble(float(curve.zero_rates[i]))
            rate_handles.append(r.mark_as_input())
            rates_aadc.append(r)
        notional = aadc.idouble(1e6)
        nodiff_inputs[notional.mark_as_input_no_diff()] = 1e6
        fixed_rate = aadc.idouble(0.03)
        nodiff_inputs[fixed_rate.mark_as_input_no_diff()] = 0.03
        pv = price_irs(notional, fixed_rate, maturity, frequency, is_payer, rates_aadc)
        pv_output = pv.mark_as_output()
    return funcs, rate_handles, pv_output, nodiff_inputs

print('='*70)
print('KERNEL REUSE BENCHMARK')
print('='*70)

currencies = ['USD', 'EUR', 'GBP', 'JPY']
market = generate_market_environment(currencies, seed=42)
workers = aadc.ThreadPool(4)

# Test 1: Many unique maturities (worst case for hard-coded)
print('\n>>> Scenario 1: 500 trades, ALL unique maturities')
maturities = np.linspace(1, 20, 500)
trades = [IRSwapTrade(trade_id=f'IRS_{i}', notional=1e6, currency=currencies[i%4],
                      maturity=maturities[i], fixed_rate=0.03, frequency=2, payer=i%2==0)
          for i in range(500)]

kernel_cache = {}
rec, eval_t = 0.0, 0.0
for trade in trades:
    key = (trade.currency, trade.maturity, trade.frequency, trade.payer)
    if key not in kernel_cache:
        t0 = time.perf_counter()
        kernel_cache[key] = record_kernel(trade.maturity, trade.frequency, trade.payer, trade.currency, market)
        rec += time.perf_counter() - t0
    funcs, rate_handles, pv_output, nodiff_inputs = kernel_cache[key]
    curve = market.curves[trade.currency]
    inputs = {rate_handles[i]: np.array([float(curve.zero_rates[i])]) for i in range(NUM_IR_TENORS)}
    inputs.update({h: np.array([v]) for h, v in nodiff_inputs.items()})
    t0 = time.perf_counter()
    aadc.evaluate(funcs, {pv_output: []}, inputs, workers)
    eval_t += time.perf_counter() - t0

print(f'Kernels: {len(kernel_cache)}')
print(f'Recording: {rec*1000:.1f} ms ({rec*1000/len(kernel_cache):.2f} ms/kernel)')
print(f'Evaluation: {eval_t*1000:.1f} ms ({eval_t*1000/500:.3f} ms/trade)')
print(f'Total: {(rec+eval_t)*1000:.1f} ms')

# Test 2: Few unique maturities (best case for hard-coded)
print('\n>>> Scenario 2: 500 trades, 5 unique maturities')
maturities = [1, 2, 5, 10, 20]
trades = [IRSwapTrade(trade_id=f'IRS_{i}', notional=1e6, currency=currencies[i%4],
                      maturity=maturities[i%5], fixed_rate=0.03, frequency=2, payer=i%2==0)
          for i in range(500)]

kernel_cache = {}
rec, eval_t = 0.0, 0.0
for trade in trades:
    key = (trade.currency, trade.maturity, trade.frequency, trade.payer)
    if key not in kernel_cache:
        t0 = time.perf_counter()
        kernel_cache[key] = record_kernel(trade.maturity, trade.frequency, trade.payer, trade.currency, market)
        rec += time.perf_counter() - t0
    funcs, rate_handles, pv_output, nodiff_inputs = kernel_cache[key]
    curve = market.curves[trade.currency]
    inputs = {rate_handles[i]: np.array([float(curve.zero_rates[i])]) for i in range(NUM_IR_TENORS)}
    inputs.update({h: np.array([v]) for h, v in nodiff_inputs.items()})
    t0 = time.perf_counter()
    aadc.evaluate(funcs, {pv_output: []}, inputs, workers)
    eval_t += time.perf_counter() - t0

print(f'Kernels: {len(kernel_cache)}')
print(f'Recording: {rec*1000:.1f} ms ({rec*1000/max(1,len(kernel_cache)):.2f} ms/kernel)')
print(f'Evaluation: {eval_t*1000:.1f} ms ({eval_t*1000/500:.3f} ms/trade)')
print(f'Total: {(rec+eval_t)*1000:.1f} ms')

# Test 3: 1000 trades with 20 unique maturities (typical)
print('\n>>> Scenario 3: 1000 trades, 20 unique maturities (typical)')
maturities = list(range(1, 21))
trades = [IRSwapTrade(trade_id=f'IRS_{i}', notional=1e6, currency=currencies[i%4],
                      maturity=maturities[i%20], fixed_rate=0.03, frequency=2, payer=i%2==0)
          for i in range(1000)]

kernel_cache = {}
rec, eval_t = 0.0, 0.0
for trade in trades:
    key = (trade.currency, trade.maturity, trade.frequency, trade.payer)
    if key not in kernel_cache:
        t0 = time.perf_counter()
        kernel_cache[key] = record_kernel(trade.maturity, trade.frequency, trade.payer, trade.currency, market)
        rec += time.perf_counter() - t0
    funcs, rate_handles, pv_output, nodiff_inputs = kernel_cache[key]
    curve = market.curves[trade.currency]
    inputs = {rate_handles[i]: np.array([float(curve.zero_rates[i])]) for i in range(NUM_IR_TENORS)}
    inputs.update({h: np.array([v]) for h, v in nodiff_inputs.items()})
    t0 = time.perf_counter()
    aadc.evaluate(funcs, {pv_output: []}, inputs, workers)
    eval_t += time.perf_counter() - t0

print(f'Kernels: {len(kernel_cache)}')
print(f'Recording: {rec*1000:.1f} ms ({rec*1000/max(1,len(kernel_cache)):.2f} ms/kernel)')
print(f'Evaluation: {eval_t*1000:.1f} ms ({eval_t*1000/1000:.3f} ms/trade)')
print(f'Total: {(rec+eval_t)*1000:.1f} ms')
print(f'Amortized total: {(rec+eval_t)*1000/1000:.3f} ms/trade')
