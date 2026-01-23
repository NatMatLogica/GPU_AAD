import sys, os
os.chdir('/home/natashamanito/ISDA-SIMM')
sys.path.insert(0, '/home/natashamanito/ISDA-SIMM')
sys.argv = ['t', '--trades', '3', '--simm-buckets', '2', '--portfolios', '2', '--threads', '4']
from model.simm_portfolio_baseline import main
main()
