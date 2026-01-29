#!/usr/bin/env python
"""Quick test for v2 kernel refactor."""
import sys
import os
os.chdir('/home/natashamanito/ISDA-SIMM')
sys.argv = ['test', '--trades', '10', '--portfolios', '3', '--threads', '8', '--trade-types', 'ir_swap']
from model.simm_portfolio_aadc import main
main()
