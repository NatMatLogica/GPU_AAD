#!/usr/bin/env python3
import sys; sys.path.insert(0, '/home/natashamanito/ISDA-SIMM')
import pandas as pd; import numpy as np; import time
from src.wnc import set_simm_version
from src.agg_margins import SIMM

# Use v2.5 parameters to match Acadia
set_simm_version('2.5')

S_IR = [('RatesFX','Risk_IRCurve','USD','1','2w','OIS',4000000),('RatesFX','Risk_IRCurve','USD','1','3m','Municipal',-3000000),('RatesFX','Risk_IRCurve','USD','1','1y','Municipal',2000000),('RatesFX','Risk_IRCurve','USD','1','1y','Prime',3000000),('RatesFX','Risk_IRCurve','USD','1','1y','Prime',-1000000),('RatesFX','Risk_IRCurve','EUR','1','3y','Libor3m',-2000000),('RatesFX','Risk_IRCurve','EUR','1','3y','Libor6m',5000000),('RatesFX','Risk_IRCurve','EUR','1','5y','Libor12m',10000000),('RatesFX','Risk_IRCurve','EUR','1','5y','Libor12m',25000000),('RatesFX','Risk_IRCurve','EUR','1','10y','Libor12m',35000000),('RatesFX','Risk_IRCurve','AUD','1','1m','Libor3m',2000000),('RatesFX','Risk_IRCurve','AUD','1','6m','Libor3m',3000000),('RatesFX','Risk_IRCurve','AUD','1','2y','Libor3m',-2000000),('RatesFX','Risk_IRCurve','CHF','1','15y','Libor6m',-4000000),('RatesFX','Risk_IRCurve','CHF','1','20y','Libor6m',10000000),('RatesFX','Risk_IRCurve','CHF','1','30y','Libor6m',18000000),('RatesFX','Risk_Inflation','CHF','','','',-10000000),('RatesFX','Risk_XCcyBasis','CHF','','','',30000000),('RatesFX','Risk_IRCurve','JPY','2','2w','Libor1m',-1000000),('RatesFX','Risk_IRCurve','JPY','2','1m','Libor1m',-1500000),('RatesFX','Risk_IRCurve','JPY','2','3m','Libor3m',1500000),('RatesFX','Risk_IRCurve','JPY','2','6m','Libor3m',2000000),('RatesFX','Risk_IRCurve','JPY','2','1y','Libor6m',3000000),('RatesFX','Risk_IRCurve','JPY','2','2y','Libor6m',4000000),('RatesFX','Risk_IRCurve','JPY','2','3y','Libor6m',5000000),('RatesFX','Risk_IRCurve','JPY','2','5y','Libor12m',20000000),('RatesFX','Risk_IRCurve','JPY','2','10y','Libor12m',30000000),('RatesFX','Risk_IRCurve','JPY','2','15y','Libor12m',-1000000),('RatesFX','Risk_IRCurve','JPY','2','20y','Libor12m',-2000000),('RatesFX','Risk_IRCurve','JPY','2','30y','Libor12m',3000000),('RatesFX','Risk_Inflation','JPY','','','',5000000),('RatesFX','Risk_XCcyBasis','JPY','','','',500000),('RatesFX','Risk_IRCurve','CNY','3','2w','OIS',1000000),('RatesFX','Risk_IRCurve','CNY','3','1m','OIS',1500000),('RatesFX','Risk_IRCurve','CNY','3','3m','Libor1m',-500000),('RatesFX','Risk_IRCurve','CNY','3','6m','Libor3m',-1000000),('RatesFX','Risk_IRCurve','MXN','3','1y','Libor6m',9000000),('RatesFX','Risk_IRCurve','MXN','3','2y','Libor12m',10000000),('RatesFX','Risk_IRCurve','MXN','3','3y','OIS',-500000),('RatesFX','Risk_IRCurve','MXN','3','5y','OIS',-1000000),('Credit','Risk_IRCurve','BRL','3','10y','Libor6m',14000000),('Credit','Risk_IRCurve','BRL','3','15y','Libor6m',30000000),('Credit','Risk_IRCurve','BRL','3','20y','Libor12m',-800000),('Credit','Risk_IRCurve','BRL','3','30y','Libor12m',-800000),('Credit','Risk_Inflation','BRL','','','',2000000),('Credit','Risk_XCcyBasis','BRL','','','',-1000000)]
S_FX = [('RatesFX','Risk_FX','EUR','','','',50000000),('RatesFX','Risk_FX','EUR','','','',-50000000),('RatesFX','Risk_FX','EUR','','','',-5000000000),('RatesFX','Risk_FX','USD','','','',610000000),('RatesFX','Risk_FX','GBP','','','',910000000),('RatesFX','Risk_FX','EUR','','','',-900000000),('RatesFX','Risk_FX','CNY','','','',-200000000),('RatesFX','Risk_FX','KRW','','','',210000000),('RatesFX','Risk_FX','TRY','','','',80000000),('RatesFX','Risk_FX','BRL','','','',-300000000),('Credit','Risk_FX','BRL','','','',41000000),('Credit','Risk_FX','QAR','','','',-40000000)]

def to_crif(sens): return pd.DataFrame([{'ProductClass':s[0],'RiskType':s[1],'Qualifier':s[2],'Bucket':s[3],'Label1':s[4],'Label2':s[5],'Amount':s[6],'AmountCurrency':'USD','AmountUSD':s[6]} for s in sens])

def bench(crif, n=50):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sc = SIMM(crif, 'USD', 1.0)
        times.append((time.perf_counter()-t0)*1000)
    return np.mean(times), np.std(times), sc.simm

c_ir = to_crif(S_IR); c_fx = to_crif(S_FX); c_all = to_crif(S_IR + S_FX)
m_ir,s_ir,r_ir = bench(c_ir); m_fx,s_fx,r_fx = bench(c_fx); m_all,s_all,r_all = bench(c_all)
print('='*74)
print('COMPARISON: Our SIMM v2.5 vs AcadiaSoft SIMM v2.5 (apples-to-apples)')
print('='*74)
print(f'| Test   | Sens | Acadia 10d       | Ours             | Diff%  | Time    |')
print(f'|--------|------|------------------|------------------|--------|---------|')
print(f'| All_IR | {len(S_IR):4} | $11,126,437,227  | ${r_ir:15,.0f} | {(r_ir/11126437227-1)*100:+5.1f}% | {m_ir:5.1f}ms |')
print(f'| All_FX | {len(S_FX):4} | $45,609,126,471  | ${r_fx:15,.0f} | {(r_fx/45609126471-1)*100:+5.1f}% | {m_fx:5.1f}ms |')
print(f'| IR+FX  | {len(S_IR)+len(S_FX):4} | N/A              | ${r_all:15,.0f} |   N/A  | {m_all:5.1f}ms |')
print(f'\nPerformance (50 iterations):')
print(f'  All_IR: {m_ir:.1f}ms +/- {s_ir:.1f}ms')
print(f'  All_FX: {m_fx:.1f}ms +/- {s_fx:.1f}ms')
print(f'  Combined: {m_all:.1f}ms +/- {s_all:.1f}ms')
print(f'  Throughput: {(len(S_IR)+len(S_FX))/(m_all/1000):.0f} sens/sec')
