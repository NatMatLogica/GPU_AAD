#!/usr/bin/env python3
"""Test script for the updated IR swap pricer - outputs to file."""
import sys
sys.path.insert(0, '/home/natashamanito/ISDA-SIMM')

# Redirect stdout to file
output_file = '/home/natashamanito/ISDA-SIMM/test_result.txt'
with open(output_file, 'w') as f:
    old_stdout = sys.stdout
    sys.stdout = f

    try:
        # Override argv for argparse
        sys.argv = ['ir_swap_pricer', '--trades', '100', '--mode', 'greeks', '--run-simm']

        from model.ir_swap_pricer import main
        result = main()
        print("\n\nTest completed successfully!")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        sys.stdout = old_stdout

print(f"Output written to {output_file}")
