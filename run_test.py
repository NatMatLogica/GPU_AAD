#!/usr/bin/env python3
"""Test script for the updated IR swap pricer."""
import sys
sys.path.insert(0, '/home/natashamanito/ISDA-SIMM')

# Override argv for argparse
sys.argv = ['ir_swap_pricer', '--trades', '100', '--mode', 'greeks', '--run-simm']

try:
    from model.ir_swap_pricer import main
    result = main()
    print("\n\nTest completed successfully!")
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
