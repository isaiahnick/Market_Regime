import sqlite3
import pandas as pd

con = sqlite3.connect("/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db")

factors = pd.read_sql("SELECT date, proxy, value FROM factors_monthly ORDER BY date, proxy", con, parse_dates=['date'])
etfs = pd.read_sql("SELECT p.date, i.proxy, p.value FROM prices p JOIN instruments i ON p.instrument_id = i.instrument_id WHERE i.data_type = 'ETF' ORDER BY p.date, i.proxy", con, parse_dates=['date'])

con.close()

all_factors = pd.concat([factors, etfs], ignore_index=True)
factors_wide = all_factors.pivot(index='date', columns='proxy', values='value')

print("Index dtype:", factors_wide.index.dtype)
print("First date:", factors_wide.index[0])
print("First date type:", type(factors_wide.index[0]))

start_date = pd.to_datetime('2000-01-01')
print("start_date:", start_date)

try:
    result = factors_wide.index >= start_date
    print("Comparison worked!")
except Exception as e:
    print("Comparison failed:", e)