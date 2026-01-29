#!/usr/bin/env python3
import sys; sys.path.insert(0, '.')
from src.wnc import set_simm_version
from src.agg_margins import SIMM
import pandas as pd, numpy as np, time

set_simm_version('2.5')

# Acadia test data
S_IR = [('RatesFX','Risk_IRCurve','USD','1','2w','OIS',4e6),('RatesFX','Risk_IRCurve','USD','1','3m','Municipal',-3e6),('RatesFX','Risk_IRCurve','USD','1','1y','Municipal',2e6),('RatesFX','Risk_IRCurve','USD','1','1y','Prime',3e6),('RatesFX','Risk_IRCurve','USD','1','1y','Prime',-1e6),('RatesFX','Risk_IRCurve','EUR','1','3y','Libor3m',-2e6),('RatesFX','Risk_IRCurve','EUR','1','3y','Libor6m',5e6),('RatesFX','Risk_IRCurve','EUR','1','5y','Libor12m',10e6),('RatesFX','Risk_IRCurve','EUR','1','5y','Libor12m',25e6),('RatesFX','Risk_IRCurve','EUR','1','10y','Libor12m',35e6),('RatesFX','Risk_IRCurve','AUD','1','1m','Libor3m',2e6),('RatesFX','Risk_IRCurve','AUD','1','6m','Libor3m',3e6),('RatesFX','Risk_IRCurve','AUD','1','2y','Libor3m',-2e6),('RatesFX','Risk_IRCurve','CHF','1','15y','Libor6m',-4e6),('RatesFX','Risk_IRCurve','CHF','1','20y','Libor6m',10e6),('RatesFX','Risk_IRCurve','CHF','1','30y','Libor6m',18e6),('RatesFX','Risk_Inflation','CHF','','','',-10e6),('RatesFX','Risk_XCcyBasis','CHF','','','',30e6),('RatesFX','Risk_IRCurve','JPY','2','2w','Libor1m',-1e6),('RatesFX','Risk_IRCurve','JPY','2','1m','Libor1m',-1.5e6),('RatesFX','Risk_IRCurve','JPY','2','3m','Libor3m',1.5e6),('RatesFX','Risk_IRCurve','JPY','2','6m','Libor3m',2e6),('RatesFX','Risk_IRCurve','JPY','2','1y','Libor6m',3e6),('RatesFX','Risk_IRCurve','JPY','2','2y','Libor6m',4e6),('RatesFX','Risk_IRCurve','JPY','2','3y','Libor6m',5e6),('RatesFX','Risk_IRCurve','JPY','2','5y','Libor12m',20e6),('RatesFX','Risk_IRCurve','JPY','2','10y','Libor12m',30e6),('RatesFX','Risk_IRCurve','JPY','2','15y','Libor12m',-1e6),('RatesFX','Risk_IRCurve','JPY','2','20y','Libor12m',-2e6),('RatesFX','Risk_IRCurve','JPY','2','30y','Libor12m',3e6),('RatesFX','Risk_Inflation','JPY','','','',5e6),('RatesFX','Risk_XCcyBasis','JPY','','','',0.5e6),('RatesFX','Risk_IRCurve','CNY','3','2w','OIS',1e6),('RatesFX','Risk_IRCurve','CNY','3','1m','OIS',1.5e6),('RatesFX','Risk_IRCurve','CNY','3','3m','Libor1m',-0.5e6),('RatesFX','Risk_IRCurve','CNY','3','6m','Libor3m',-1e6),('RatesFX','Risk_IRCurve','MXN','3','1y','Libor6m',9e6),('RatesFX','Risk_IRCurve','MXN','3','2y','Libor12m',10e6),('RatesFX','Risk_IRCurve','MXN','3','3y','OIS',-0.5e6),('RatesFX','Risk_IRCurve','MXN','3','5y','OIS',-1e6),('Credit','Risk_IRCurve','BRL','3','10y','Libor6m',14e6),('Credit','Risk_IRCurve','BRL','3','15y','Libor6m',30e6),('Credit','Risk_IRCurve','BRL','3','20y','Libor12m',-0.8e6),('Credit','Risk_IRCurve','BRL','3','30y','Libor12m',-0.8e6),('Credit','Risk_Inflation','BRL','','','',2e6),('Credit','Risk_XCcyBasis','BRL','','','',-1e6)]

S_FX = [('RatesFX','Risk_FX','EUR','','','',50e6),('RatesFX','Risk_FX','EUR','','','',-50e6),('RatesFX','Risk_FX','EUR','','','',-5e9),('RatesFX','Risk_FX','USD','','','',610e6),('RatesFX','Risk_FX','GBP','','','',910e6),('RatesFX','Risk_FX','EUR','','','',-900e6),('RatesFX','Risk_FX','CNY','','','',-200e6),('RatesFX','Risk_FX','KRW','','','',210e6),('RatesFX','Risk_FX','TRY','','','',80e6),('RatesFX','Risk_FX','BRL','','','',-300e6),('Credit','Risk_FX','BRL','','','',41e6),('Credit','Risk_FX','QAR','','','',-40e6)]

def crif(s):
    return pd.DataFrame([{'ProductClass':x[0],'RiskType':x[1],'Qualifier':x[2],'Bucket':x[3],'Label1':x[4],'Label2':x[5],'Amount':x[6],'AmountCurrency':'USD','AmountUSD':x[6]} for x in s])

def bench(c,n=50):
    t=[]
    for _ in range(n):
        t0=time.perf_counter()
        r=SIMM(c,'USD',1.0)
        t.append((time.perf_counter()-t0)*1000)
    return np.mean(t),np.std(t),r.simm

c_ir=crif(S_IR); c_fx=crif(S_FX); c_all=crif(S_IR+S_FX)
m_ir,s_ir,r_ir=bench(c_ir)
m_fx,s_fx,r_fx=bench(c_fx)
m_all,s_all,r_all=bench(c_all)

print('='*76)
print('COMPARISON: Our SIMM v2.5 vs AcadiaSoft SIMM v2.5 (apples-to-apples)')
print('='*76)
print(f'| Test   | Sens | Acadia 10d       | Ours             | Diff%   | Time     |')
print(f'|--------|------|------------------|------------------|---------|----------|')
print(f'| All_IR | {len(S_IR):4} | $11,126,437,227  | ${r_ir:15,.0f} | {(r_ir/11126437227-1)*100:+6.2f}% | {m_ir:6.1f}ms |')
print(f'| All_FX | {len(S_FX):4} | $45,609,126,471  | ${r_fx:15,.0f} | {(r_fx/45609126471-1)*100:+6.2f}% | {m_fx:6.1f}ms |')
print(f'| IR+FX  | {len(S_IR)+len(S_FX):4} | N/A              | ${r_all:15,.0f} |    N/A  | {m_all:6.1f}ms |')
print()
print('Performance (50 iterations):')
print(f'  All_IR:   {m_ir:6.2f}ms +/- {s_ir:5.2f}ms  ({46/(m_ir/1000):,.0f} sens/sec)')
print(f'  All_FX:   {m_fx:6.2f}ms +/- {s_fx:5.2f}ms  ({12/(m_fx/1000):,.0f} sens/sec)')
print(f'  Combined: {m_all:6.2f}ms +/- {s_all:5.2f}ms  ({58/(m_all/1000):,.0f} sens/sec)')
