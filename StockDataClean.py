import numpy as np
import pandas as pd

# Load the stock price dataset
df = pd.read_csv('stock_price_data_30000.csv')
print(df.head(10))
print("\nShape before cleaning:", df.shape)

# 1. Remove duplicate records (same stock on the same date)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.drop_duplicates(subset=['Ticker', 'Date'], keep='first')

# 2. Parse and standardize Date (drop rows where the date is invalid)
df = df.dropna(subset=['Date'])
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

# 3. Clean Ticker column (Strip whitespace & Uppercase)
df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()

# 4. Clean & Standardize Sector
df['Sector'] = df['Sector'].astype(str).str.strip().str.title()
sector_map = {
    'It': 'IT',
    'Information Technology': 'IT',
    'Bank': 'Banking',
    'Banks': 'Banking',
    'Banking': 'Banking',
    'Energy': 'Energy',
    'Auto': 'Auto',
    'Automobile': 'Auto',
    'Fmcg': 'FMCG',
    'Pharma': 'Pharma',
    'Pharmaceutical': 'Pharma',
}
df['Sector'] = df['Sector'].map(sector_map).fillna('Unknown')

# 5. Clean numeric price & volume columns (strip ₹, $, commas; convert to numbers)
numeric_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Daily_Return_Pct', 'MA_5', 'MA_10', 'MA_20',
    'EMA_12', 'EMA_26', 'MACD', 'RSI_14', 'Volatility_10',
    'Bollinger_Upper', 'Bollinger_Lower', 'Volume_MA_10', 'Next_Day_Close',
]
for col in numeric_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace('₹', '', regex=False)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        .replace({'nan': np.nan, '': np.nan})
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 6. Handle invalid values (prices and volume must be positive)
price_cols = ['Open', 'High', 'Low', 'Close', 'Next_Day_Close']
for col in price_cols:
    df.loc[df[col] <= 0, col] = np.nan
df.loc[df['Volume'] < 0, 'Volume'] = np.nan

# RSI must stay between 0 and 100
df.loc[(df['RSI_14'] < 0) | (df['RSI_14'] > 100), 'RSI_14'] = np.nan

# 7. Sort by stock and date (needed for time-series filling)
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# 8. Impute missing values per stock
#    Prices: forward-fill then back-fill within each ticker (last known price)
for col in price_cols:
    df[col] = df.groupby('Ticker')[col].transform(lambda s: s.ffill().bfill())

#    Volume & indicators: fill with the median of that ticker
fill_median_cols = [c for c in numeric_cols if c not in price_cols]
for col in fill_median_cols:
    df[col] = df.groupby('Ticker')[col].transform(lambda s: s.fillna(s.median()))

# 9. Fix High/Low consistency (High must be the max, Low must be the min of the day)
df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)

# 10. Handle extreme outliers in daily return (cap at 1st and 99th percentile per stock)
def cap_outliers(s):
    return s.clip(s.quantile(0.01), s.quantile(0.99))

df['Daily_Return_Pct'] = df.groupby('Ticker')['Daily_Return_Pct'].transform(cap_outliers)

# 11. Clean Volume (whole numbers only)
df['Volume'] = df['Volume'].round().astype(int)

# 12. Standardize the target column Next_Day_Trend (Up / Down)
df['Next_Day_Trend'] = df['Next_Day_Trend'].astype(str).str.strip().str.title()
trend_map = {'Up': 'Up', 'U': 'Up', '1': 'Up', 'Down': 'Down', 'D': 'Down', '0': 'Down'}
df['Next_Day_Trend'] = df['Next_Day_Trend'].map(trend_map)
# Rebuild missing/invalid labels from the prices themselves
missing_trend = df['Next_Day_Trend'].isna()
df.loc[missing_trend, 'Next_Day_Trend'] = np.where(
    df.loc[missing_trend, 'Next_Day_Close'] > df.loc[missing_trend, 'Close'], 'Up', 'Down'
)

# 13. Round decimals to 2 places
float_cols = df.select_dtypes(include='float').columns
df[float_cols] = df[float_cols].round(2)

# Summary report
print("\nShape after cleaning:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nTrend distribution:\n", df['Next_Day_Trend'].value_counts())
print("\nSectors:", sorted(df['Sector'].unique()))

# Export cleaned dataset
df.to_csv('cleaned_stock_data.csv', index=False)
print("\nCleaned file saved as cleaned_stock_data.csv")
