#!/usr/bin/env python
"""Wrapper to run CUDA SIMM benchmark with simulator enabled."""
import os
import sys

# Enable CUDA simulator BEFORE importing numba
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

# Now run the module with command line args
sys.argv = ['simm_portfolio_cuda', '--trades', '100', '--portfolios', '3', '--trade-types', 'ir_swap', '--optimize']

# Import and run
from model.simm_portfolio_cuda import main
main()
