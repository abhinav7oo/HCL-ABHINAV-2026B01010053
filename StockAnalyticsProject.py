"""
=====================================================================
 STOCK MARKET ANALYTICS & STRATEGY BACK-TESTING  (NumPy + Pandas only)
=====================================================================
Dataset : cleaned_stock_data.csv  (30 stocks x 1000 trading days)

Only two libraries are imported: NumPy and Pandas.
No machine learning, no other module.

What this project does
  1.  Data quality check
  2.  Stock performance & risk report (return, Sharpe, Sortino, drawdown, VaR...)
  3.  Stock scorecard (overall ranking of all 30 stocks)
  4.  Sector analysis (risk, return and year-wise returns)
  5.  Correlation & diversification analysis
  6.  Portfolio comparison (equal-weight vs inverse-volatility)
  7.  Time patterns (weekday effect, yearly returns, best/worst days)
  8.  Market drawdown analysis (peak, trough, recovery)
  9.  Do technical indicators really work? (hit-rate test)
  10. Strategy back-testing (MA crossover, RSI, MACD vs Buy & Hold)
  11. Key findings + result CSV files

Run:  python StockAnalyticsProject.py
(keep cleaned_stock_data.csv in the same folder)
"""

import numpy as np
import pandas as pd

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)

# ---------------------------------------------------------------------
# SETTINGS (assumptions - change them if you want)
# ---------------------------------------------------------------------
INPUT_FILE = 'cleaned_stock_data.csv'
TRADING_DAYS = 252        # trading days in one year
RISK_FREE = 0.06          # 6% yearly risk-free rate (used in Sharpe / Sortino)
COST = 0.001              # 0.1% cost every time we buy or sell


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------
# 0. LOAD DATA
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# "Wide" tables: rows = dates, columns = stocks
close = df.pivot(index='Date', columns='Ticker', values='Close')
rsi = df.pivot(index='Date', columns='Ticker', values='RSI_14')
ma_fast = df.pivot(index='Date', columns='Ticker', values='MA_5')
ma_slow = df.pivot(index='Date', columns='Ticker', values='MA_20')
macd = df.pivot(index='Date', columns='Ticker', values='MACD')
boll_up = df.pivot(index='Date', columns='Ticker', values='Bollinger_Upper')
boll_low = df.pivot(index='Date', columns='Ticker', values='Bollinger_Lower')

ret_full = close.pct_change()              # daily return as a fraction (0.01 = 1%)
ret = ret_full.iloc[1:]                    # first day has no "yesterday", so skip it
sector_of = df.drop_duplicates('Ticker').set_index('Ticker')['Sector']


def perf_metrics(r):
    """Return and risk table for a DataFrame of daily returns (columns = assets)."""
    n = len(r)
    growth = (1 + r.fillna(0)).cumprod()                         # growth of Rs.1
    total = growth.iloc[-1] - 1
    ann_ret = growth.iloc[-1] ** (TRADING_DAYS / n) - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS - RISK_FREE) / ann_vol

    # Sortino: like Sharpe, but only counts the BAD (negative) days as risk
    downside = np.sqrt((r.clip(upper=0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = (r.mean() * TRADING_DAYS - RISK_FREE) / downside

    max_dd = (growth / growth.cummax() - 1).min()                # worst fall from a peak
    calmar = ann_ret / max_dd.abs()                              # return per unit of drawdown

    var95 = r.quantile(0.05)                                     # "bad day" level (worst 5%)
    cvar95 = r.where(r <= var95).mean()                          # average of those bad days
    win_rate = (r > 0).mean()

    return pd.DataFrame({
        'Total_Return_%': total * 100,
        'Annual_Return_%': ann_ret * 100,
        'Annual_Volatility_%': ann_vol * 100,
        'Sharpe_Ratio': sharpe,
        'Sortino_Ratio': sortino,
        'Max_Drawdown_%': max_dd * 100,
        'Calmar_Ratio': calmar,
        'VaR_95_%': var95 * 100,
        'CVaR_95_%': cvar95 * 100,
        'Win_Rate_%': win_rate * 100,
    })


# ---------------------------------------------------------------------
# 1. DATA QUALITY CHECK
# ---------------------------------------------------------------------
section("1. DATA QUALITY CHECK")
print("Rows x Columns            :", df.shape)
print("Stocks / Sectors          :", df['Ticker'].nunique(), "/", df['Sector'].nunique())
print("Date range                :", df['Date'].min().date(), "to", df['Date'].max().date())
print("Missing values            :", int(df.isnull().sum().sum()))
print("Duplicate (stock, date)   :", int(df.duplicated(['Ticker', 'Date']).sum()))
print("Rows where High < Low     :", int((df['High'] < df['Low']).sum()))
counts = df.groupby('Ticker').size()
print("Rows per stock (min, max) :", counts.min(), ",", counts.max())
print("\nSummary of key columns:")
print(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe().round(2))

# ---------------------------------------------------------------------
# 2. STOCK PERFORMANCE & RISK REPORT
# ---------------------------------------------------------------------
section("2. STOCK PERFORMANCE & RISK REPORT")
perf = perf_metrics(ret)
perf.insert(0, 'Sector', sector_of)
perf['Best_Day_%'] = ret.max() * 100
perf['Worst_Day_%'] = ret.min() * 100
perf['Avg_Daily_Volume'] = df.groupby('Ticker')['Volume'].mean().round(0)
perf = perf.round(2).sort_values('Sharpe_Ratio', ascending=False)

show = ['Sector', 'Total_Return_%', 'Annual_Volatility_%', 'Sharpe_Ratio', 'Sortino_Ratio',
        'Max_Drawdown_%', 'VaR_95_%']
print("Top 5 stocks (by Sharpe ratio):")
print(perf.head(5)[show])
print("\nBottom 5 stocks (by Sharpe ratio):")
print(perf.tail(5)[show])
print("\nHow to read: Sharpe/Sortino above 1 is good. Max drawdown is the biggest fall from a peak.")
print("VaR 95% = on the worst 5% of days, the stock lost at least this much in one day.")

# ---------------------------------------------------------------------
# 3. STOCK SCORECARD (overall ranking)
# ---------------------------------------------------------------------
section("3. STOCK SCORECARD (rank 1 = best on that measure)")
score = pd.DataFrame({
    'Return_Rank': perf['Annual_Return_%'].rank(ascending=False),
    'Sharpe_Rank': perf['Sharpe_Ratio'].rank(ascending=False),
    'Drawdown_Rank': perf['Max_Drawdown_%'].rank(ascending=False),   # smaller fall = better
    'Volatility_Rank': perf['Annual_Volatility_%'].rank(ascending=True),   # calmer = better
})
score['Avg_Rank'] = score.mean(axis=1)
score['Overall_Position'] = score['Avg_Rank'].rank(method='first').astype(int)
score.insert(0, 'Sector', sector_of)
score = score.sort_values('Overall_Position')
print(score.head(5))
print("...")
print(score.tail(3))

# ---------------------------------------------------------------------
# 4. SECTOR ANALYSIS
# ---------------------------------------------------------------------
section("4. SECTOR ANALYSIS (equal-weighted sector portfolios)")
sector_ret = pd.DataFrame({
    sec: ret[sector_of[sector_of == sec].index].mean(axis=1)
    for sec in sorted(sector_of.unique())
})
sector_perf = perf_metrics(sector_ret).round(2).sort_values('Sharpe_Ratio', ascending=False)
print(sector_perf[['Total_Return_%', 'Annual_Return_%', 'Annual_Volatility_%',
                   'Sharpe_Ratio', 'Max_Drawdown_%']])

sector_year = ((1 + sector_ret).groupby(sector_ret.index.year).prod() - 1) * 100
sector_year.index.name = 'Year'
sector_year = sector_year.round(2)
print("\nYear-wise sector returns (%):")
print(sector_year)

# ---------------------------------------------------------------------
# 5. CORRELATION & DIVERSIFICATION
# ---------------------------------------------------------------------
section("5. CORRELATION & DIVERSIFICATION")
corr = ret.corr()
rows, cols = np.triu_indices_from(corr.values, k=1)      # each pair once
pairs = pd.DataFrame({
    'Stock_A': corr.index[rows],
    'Stock_B': corr.columns[cols],
    'Correlation': corr.values[rows, cols],
}).sort_values('Correlation', ascending=False)
print("Most correlated pairs:")
print(pairs.head(5).round(3).to_string(index=False))
print("\nLeast correlated pairs:")
print(pairs.tail(5).round(3).to_string(index=False))

same = np.array([sector_of[a] == sector_of[b] for a, b in zip(pairs.Stock_A, pairs.Stock_B)])
print(f"\nAverage correlation, same-sector pairs     : {pairs.Correlation[same].mean():.3f}")
print(f"Average correlation, different-sector pairs: {pairs.Correlation[~same].mean():.3f}")

avg_stock_vol = perf['Annual_Volatility_%'].mean()
port_vol = perf_metrics(ret.mean(axis=1).to_frame('EW'))['Annual_Volatility_%'].iloc[0]
print(f"\nAverage single-stock volatility : {avg_stock_vol:.2f}%")
print(f"Equal-weight portfolio volatility: {port_vol:.2f}%")
print(f"Risk removed by diversification  : {(1 - port_vol / avg_stock_vol) * 100:.1f}%")

# ---------------------------------------------------------------------
# 6. PORTFOLIO COMPARISON
# ---------------------------------------------------------------------
section("6. PORTFOLIO COMPARISON")
vol = ret.std()
w_inv = (1 / vol) / (1 / vol).sum()          # calmer stocks get a bigger share
best_stock, worst_stock = perf.index[0], perf.index[-1]
portfolios = pd.DataFrame({
    'Equal-weight (all 30)': ret.mean(axis=1),
    'Inverse-volatility (all 30)': (ret * w_inv).sum(axis=1),
    f'Best stock ({best_stock})': ret[best_stock],
    f'Worst stock ({worst_stock})': ret[worst_stock],
})
portfolio_perf = perf_metrics(portfolios).round(2)
print(portfolio_perf[['Total_Return_%', 'Annual_Return_%', 'Annual_Volatility_%',
                      'Sharpe_Ratio', 'Max_Drawdown_%']])
print("\nNote: inverse-volatility weights use the whole history, so this is a study of")
print("past data, not a rule that could have been followed in real time.")

# ---------------------------------------------------------------------
# 7. TIME PATTERNS
# ---------------------------------------------------------------------
section("7. TIME PATTERNS (average of all 30 stocks)")
market = ret.mean(axis=1)

day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
by_day = market.groupby(market.index.dayofweek).mean() * 100
by_day.index = [day_names[i] for i in by_day.index]
print("Average daily return by weekday (%):")
print(by_day.round(4).to_string())

by_year = ((1 + market).groupby(market.index.year).prod() - 1) * 100
print("\nMarket return by year (%):")
print(by_year.round(2).to_string())

print("\nBest 3 market days (%):")
print((market.nlargest(3) * 100).round(2).to_string())
print("\nWorst 3 market days (%):")
print((market.nsmallest(3) * 100).round(2).to_string())

# ---------------------------------------------------------------------
# 8. MARKET DRAWDOWN ANALYSIS
# ---------------------------------------------------------------------
section("8. MARKET DRAWDOWN ANALYSIS (equal-weight market)")
growth = (1 + market).cumprod()
drawdown = growth / growth.cummax() - 1
trough_date = drawdown.idxmin()
peak_date = growth.loc[:trough_date].idxmax()
after = growth.loc[trough_date:]
recovered = after[after >= growth.loc[peak_date]]
print(f"Worst fall         : {drawdown.min() * 100:.2f}%")
print(f"Peak before fall   : {peak_date.date()}")
print(f"Lowest point       : {trough_date.date()}")
print(f"Recovered on       : {recovered.index[0].date() if len(recovered) else 'Not recovered yet'}")
print(f"Days spent falling : {np.busday_count(peak_date.date(), trough_date.date())} trading days")

# ---------------------------------------------------------------------
# 9. DO TECHNICAL INDICATORS REALLY WORK? (hit-rate test)
# ---------------------------------------------------------------------
section("9. DO TECHNICAL INDICATORS REALLY WORK?")
print("For every time an indicator gave a signal, did the NEXT day move the way it promised?\n")

next_ret = ret_full.shift(-1).to_numpy()          # tomorrow's return, same shape as the tables
valid = ~np.isnan(next_ret)
base_up = (next_ret[valid] > 0).mean() * 100       # chance a random day goes up
base_down = (next_ret[valid] < 0).mean() * 100

signals = [
    ('RSI below 30 (oversold)', 'Up', rsi < 30),
    ('RSI above 70 (overbought)', 'Down', rsi > 70),
    ('MA5 above MA20', 'Up', ma_fast > ma_slow),
    ('MACD above 0', 'Up', macd > 0),
    ('Close below lower Bollinger band', 'Up', close < boll_low),
    ('Close above upper Bollinger band', 'Down', close > boll_up),
]
sig_rows = []
for name, expected, sig in signals:
    mask = sig.to_numpy() & valid
    vals = next_ret[mask]
    if len(vals) == 0:
        continue
    hit = ((vals > 0).mean() if expected == 'Up' else (vals < 0).mean()) * 100
    base = base_up if expected == 'Up' else base_down
    sig_rows.append({
        'Signal': name,
        'Promises': expected,
        'Times_Fired': len(vals),
        'Hit_Rate_%': hit,
        'Normal_Day_Hit_Rate_%': base,
        'Edge_%pts': hit - base,
        'Avg_Next_Day_Return_%': vals.mean() * 100,
    })
signal_table = pd.DataFrame(sig_rows).round(2)
print(signal_table.to_string(index=False))
print("\nEdge = hit rate minus the hit rate of an ordinary day. An edge of only 1 or 2")
print("percentage points is usually just chance, not a real pattern.")

# ---------------------------------------------------------------------
# 10. STRATEGY BACK-TESTING
# ---------------------------------------------------------------------
section("10. STRATEGY BACK-TESTING (with 0.1% cost per trade)")


def run_strategy(position):
    """
    position: 1 = holding the stock, 0 = in cash (decided at the END of each day).
    We shift by 1 day, so today's signal earns TOMORROW's return (no cheating).
    """
    pos = position.shift(1).fillna(0).loc[ret.index]
    trades = pos.diff().abs()
    trades.iloc[0] = pos.iloc[0].abs()
    return pos * ret - trades * COST


pos_ma = (ma_fast > ma_slow).astype(float)                       # MA crossover
rsi_signal = pd.DataFrame(np.where(rsi < 30, 1.0, np.where(rsi > 70, 0.0, np.nan)),
                          index=rsi.index, columns=rsi.columns)
pos_rsi = rsi_signal.ffill().fillna(0.0)                         # RSI: keep last decision
pos_macd = (macd > 0).astype(float)                              # MACD momentum

strategies = {
    'Buy & Hold': ret,
    'MA Crossover (5/20)': run_strategy(pos_ma),
    'RSI (30/70)': run_strategy(pos_rsi),
    'MACD > 0': run_strategy(pos_macd),
}

buyhold_total = perf_metrics(ret)['Total_Return_%']
rows_out = []
for name, r in strategies.items():
    m = perf_metrics(r)
    p = perf_metrics(r.mean(axis=1).to_frame('P')).iloc[0]       # equal-weight portfolio of it
    rows_out.append({
        'Strategy': name,
        'Avg_Stock_Return_%': m['Total_Return_%'].mean(),
        'Avg_Sharpe': m['Sharpe_Ratio'].mean(),
        'Avg_Max_Drawdown_%': m['Max_Drawdown_%'].mean(),
        'Portfolio_Return_%': p['Total_Return_%'],
        'Portfolio_Sharpe': p['Sharpe_Ratio'],
        'Stocks_Beating_BuyHold': int((m['Total_Return_%'] > buyhold_total).sum())
        if name != 'Buy & Hold' else '-',
    })
strategy_table = pd.DataFrame(rows_out).round(2)
print(strategy_table.to_string(index=False))

# ---------------------------------------------------------------------
# 11. KEY FINDINGS + SAVE RESULTS
# ---------------------------------------------------------------------
section("11. KEY FINDINGS")
best_signal = signal_table.sort_values('Edge_%pts', ascending=False).iloc[0]
print(f"* Best stock by Sharpe ratio : {perf.index[0]} (Sharpe {perf.iloc[0]['Sharpe_Ratio']})")
print(f"* Weakest stock              : {perf.index[-1]} (Sharpe {perf.iloc[-1]['Sharpe_Ratio']})")
print(f"* Best overall scorecard     : {score.index[0]}")
print(f"* Best sector                : {sector_perf.index[0]}   |   Riskiest sector: "
      f"{sector_perf['Annual_Volatility_%'].idxmax()}")
print(f"* Diversification removed {(1 - port_vol / avg_stock_vol) * 100:.0f}% of the average stock's risk")
print(f"* Biggest indicator edge     : '{best_signal['Signal']}' "
      f"({best_signal['Edge_%pts']} percentage points)")
print(f"* Best back-tested strategy  : "
      f"{strategy_table.sort_values('Avg_Sharpe', ascending=False).iloc[0]['Strategy']}")

print("\nNOTE: This dataset is SIMULATED, so it has no real predictable pattern. That is")
print("why indicator edges are tiny and Buy & Hold is hard to beat. The value of the")
print("project is the complete analysis method, which works the same on real data.")

# Save every result table (files are created in the same folder as the script)
perf.to_csv('result_stock_performance.csv')
score.to_csv('result_stock_scorecard.csv')
sector_perf.to_csv('result_sector_summary.csv')
sector_year.to_csv('result_sector_yearly_returns.csv')
corr.round(3).to_csv('result_correlation_matrix.csv')
portfolio_perf.to_csv('result_portfolio_comparison.csv')
signal_table.to_csv('result_indicator_signal_test.csv', index=False)
strategy_table.to_csv('result_strategy_comparison.csv', index=False)
print("\n8 result files saved (names start with 'result_').")
