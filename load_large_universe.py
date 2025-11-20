# load_large_universe.py
#
# Download and load large universe of tradeable assets
# Organized by what PC factors they can represent
#
# Usage: python load_large_universe.py --start 1995-01-01

import argparse
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"

# ============================================================================
# UNIVERSE DEFINITION - Organized by PC Factor Coverage
# ============================================================================

# Assets that represent EQUITY exposure (Equity_PC1, Local Equity_PC1, etc.)
EQUITY_UNIVERSE = {
    # US Broad Market
    'SPY': 'SPDR S&P 500',
    'IVV': 'iShares S&P 500',
    'VOO': 'Vanguard S&P 500',
    'VTI': 'Vanguard Total Stock Market',
    'ITOT': 'iShares Total Stock Market',
    
    # US Large Cap
    'QQQ': 'Nasdaq 100',
    'DIA': 'Dow Jones',
    'IWB': 'Russell 1000',
    'SCHX': 'US Large Cap',
    
    # US Mid/Small Cap
    'IWM': 'Russell 2000',
    'IJH': 'S&P Mid Cap',
    'IJR': 'S&P Small Cap',
    'VB': 'Small Cap',
    'VO': 'Mid Cap',
    
    # US Sectors
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLY': 'Consumer Discretionary',
    'XLB': 'Materials',
    'XLC': 'Communications',
    
    # International Developed
    'EFA': 'MSCI EAFE',
    'VEA': 'FTSE Developed',
    'IEFA': 'Core MSCI EAFE',
    'EWJ': 'Japan',
    'EWG': 'Germany',
    'EWU': 'UK',
    'EWC': 'Canada',
    'EWA': 'Australia',
    'EWL': 'Switzerland',
    'EWQ': 'France',
    
    # Emerging Markets
    'EEM': 'MSCI Emerging Markets',
    'VWO': 'FTSE Emerging',
    'IEMG': 'Core MSCI Emerging',
    'FXI': 'China Large Cap',
    'MCHI': 'China',
    'EWZ': 'Brazil',
    'EWY': 'South Korea',
    'EWT': 'Taiwan',
    'INDA': 'India',
    'RSX': 'Russia',
    'EWW': 'Mexico',
    'EZA': 'South Africa',
    'THD': 'Thailand',
    
    # Style/Factor
    'MTUM': 'Momentum',
    'QUAL': 'Quality',
    'SIZE': 'Size',
    'VLUE': 'Value',
    'USMV': 'Min Volatility',
    'VTV': 'Value',
    'VUG': 'Growth',
    'VBK': 'Small Cap Growth',
    'VBR': 'Small Cap Value',
    'SPLV': 'Low Volatility',
    'SPHB': 'High Beta',
    'SPHD': 'High Dividend',
    'DGRO': 'Dividend Growth',
}

# Assets that represent CREDIT exposure (Credit_PC1)
CREDIT_UNIVERSE = {
    # Investment Grade Corporate
    'LQD': 'Investment Grade Corporate',
    'VCIT': 'Intermediate Corporate',
    'VCLT': 'Long Corporate',
    'USIG': 'Investment Grade',
    'CORP': 'Corporate Bond',
    'IGIB': 'Intermediate Corporate',
    
    # High Yield
    'HYG': 'High Yield Corporate',
    'JNK': 'High Yield',
    'HYGH': 'High Yield 0-5 Year',
    'SJNK': 'Short High Yield',
    'ANGL': 'Fallen Angels',
    'FALN': 'Fallen Angel',
    
    # Bank Loans / Floating Rate
    'BKLN': 'Bank Loans',
    'SRLN': 'Senior Loan',
    'FLOT': 'Floating Rate',
    
    # Mortgage Backed
    'MBB': 'Mortgage Backed',
    'VMBS': 'Mortgage Backed',
    'CMBS': 'Commercial Mortgage',
}

# Assets that represent INTEREST RATES / DURATION (Interest Rates_PC1)
RATES_UNIVERSE = {
    # Short Duration Treasuries
    'SHY': 'Treasury 1-3 Year',
    'SHV': 'Treasury 0-1 Year',
    'VGSH': 'Short-Term Treasury',
    'SCHO': 'Short-Term Treasury',
    'BIL': 'T-Bills',
    
    # Intermediate Duration
    'IEF': 'Treasury 7-10 Year',
    'VGIT': 'Intermediate Treasury',
    'IEI': 'Treasury 3-7 Year',
    'SCHR': 'Intermediate Treasury',
    
    # Long Duration
    'TLT': 'Treasury 20+ Year',
    'VGLT': 'Long-Term Treasury',
    'TLH': 'Treasury 10-20 Year',
    'EDV': 'Extended Duration Treasury',
    'ZROZ': 'Zero Coupon 25+ Year',
    
    # TIPS (Inflation Protected)
    'TIP': 'TIPS',
    'VTIP': 'Short-Term TIPS',
    'SCHP': 'TIPS',
    'LTPZ': 'Long-Term TIPS',
    
    # Municipal
    'MUB': 'Municipal',
    'VTEB': 'Tax-Exempt',
    'HYD': 'High Yield Municipal',
}

# Assets that represent COMMODITIES (Commodities_PC1)
COMMODITY_UNIVERSE = {
    # Precious Metals
    'GLD': 'Gold',
    'IAU': 'Gold',
    'GLDM': 'Gold',
    'SLV': 'Silver',
    'PPLT': 'Platinum',
    'PALL': 'Palladium',
    'GDX': 'Gold Miners',
    'GDXJ': 'Junior Gold Miners',
    
    # Energy
    'USO': 'Crude Oil',
    'BNO': 'Brent Oil',
    'UNG': 'Natural Gas',
    'UCO': '2x Oil',
    'XLE': 'Energy Sector',
    'XOP': 'Oil & Gas Exploration',
    'OIH': 'Oil Services',
    
    # Broad Commodities
    'DBC': 'Commodities',
    'GSG': 'Commodities',
    'PDBC': 'Optimized Commodities',
    'COMT': 'Commodities',
    
    # Agriculture
    'DBA': 'Agriculture',
    'CORN': 'Corn',
    'WEAT': 'Wheat',
    'SOYB': 'Soybeans',
    'COW': 'Livestock',
    
    # Industrial Metals
    'CPER': 'Copper',
    'DBB': 'Base Metals',
}

# Assets that represent CURRENCIES / FX CARRY (Foreign Exchange Carry_PC1)
CURRENCY_UNIVERSE = {
    # Dollar
    'UUP': 'Dollar Bull',
    'UDN': 'Dollar Bear',
    
    # Major Currencies
    'FXE': 'Euro',
    'FXY': 'Yen',
    'FXB': 'British Pound',
    'FXA': 'Australian Dollar',
    'FXC': 'Canadian Dollar',
    'FXF': 'Swiss Franc',
    
    # Emerging Market FX
    'CEW': 'Brazil Real',
    'CYB': 'Chinese Yuan',
    'CNY': 'Chinese Yuan',
}

# Assets that represent VOLATILITY (Equity Short Volatility_PC1)
VOLATILITY_UNIVERSE = {
    'VXX': 'VIX Short-Term Futures',
    'VIXY': 'VIX Short-Term',
    'UVXY': 'VIX 2x',
    'SVXY': 'Short VIX',
    'VIXM': 'VIX Mid-Term',
    'VXZ': 'VIX Mid-Term Futures',
}

# Assets that represent TREND FOLLOWING (Trend Following_PC1)
TREND_UNIVERSE = {
    'DBMF': 'Managed Futures',
    'KMLM': 'Managed Futures',
    'CTA': 'Managed Futures',
}

# Assets that represent REAL ESTATE
REAL_ESTATE_UNIVERSE = {
    'VNQ': 'Real Estate',
    'IYR': 'Real Estate',
    'REET': 'Real Estate',
    'SCHH': 'Real Estate',
    'XLRE': 'Real Estate Sector',
    'RWR': 'Real Estate',
}

# Alternative Assets
ALTERNATIVE_UNIVERSE = {
    'TAN': 'Solar Energy',
    'ICLN': 'Clean Energy',
    'LIT': 'Lithium',
    'ARKG': 'Genomics',
    'ARKK': 'Innovation',
    'ARKQ': 'Autonomous Tech',
    'ARKW': 'Internet',
    'BOTZ': 'Robotics',
    'HACK': 'Cybersecurity',
    'FINX': 'Fintech',
}

# Original tradeable assets from your factor research
# These are already in your database - DO NOT try to download from Yahoo
ORIGINAL_FACTORS_IN_DB = {
    'MXWO_Index', 'MXWOU_Index', 'SXXP_Index', 'NKY_Index', 'SHCOMP_Index',
    'MXWD_Index', 'MXEA_Index', 'SPX_Index', 'SPXT_Index', 'SP5LVI_Index',
    'M1WOMVOL_Index', 'MXEF_Index', 'MXEFLC_Index', 'BCOM_Index', 
    'SPGSCI_Index', 'DXY_Curncy', 'EURUSD_Curncy', 'USDJPY_Curncy',
    'FXCARRSP_Index', 'FXCTG10_Index', 'NEIXCTAT_Index', 'NEIXBTRND_Index',
    'FF_UMD', 'FF_SMB', 'FF_HML', 'FF_RMW', 'MXWO000V_Index', 
    'MXUS000V_Index', 'MXEF000V_Index', 'MXWO000G_Index', 'MXUS000G_Index',
    'MXEF000G_Index', 'BXM_Index'
}

# Combine all NEW tickers (only Yahoo Finance ETFs)
ALL_TICKERS = {}
ALL_TICKERS.update(EQUITY_UNIVERSE)
ALL_TICKERS.update(CREDIT_UNIVERSE)
ALL_TICKERS.update(RATES_UNIVERSE)
ALL_TICKERS.update(COMMODITY_UNIVERSE)
ALL_TICKERS.update(CURRENCY_UNIVERSE)
ALL_TICKERS.update(VOLATILITY_UNIVERSE)
ALL_TICKERS.update(TREND_UNIVERSE)
ALL_TICKERS.update(REAL_ESTATE_UNIVERSE)
ALL_TICKERS.update(ALTERNATIVE_UNIVERSE)

print(f"Total NEW tickers to download: {len(ALL_TICKERS)}")
print(f"  (excludes {len(ORIGINAL_FACTORS_IN_DB)} factors already in database)")

# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_ticker_data(ticker, start_date, end_date, retries=3):
    """Download monthly price data for a single ticker with retry logic"""
    for attempt in range(retries):
        try:
            # Use Ticker().history() method (works with newer yfinance)
            print(f"      Attempt {attempt+1}: Creating ticker object for {ticker}...")
            ticker_obj = yf.Ticker(ticker)
            
            print(f"      Downloading history...")
            data = ticker_obj.history(start=start_date, end=end_date)
            
            print(f"      Got {len(data)} rows")
            
            if data.empty:
                print(f"      Empty data")
                return None
            
            # Use 'Close' column (not 'Adj Close' since history() returns different format)
            if 'Close' in data.columns:
                price_col = 'Close'
            elif 'Adj Close' in data.columns:
                price_col = 'Adj Close'
            else:
                print(f"      No price column found. Columns: {list(data.columns)}")
                return None
            
            print(f"      Converting to monthly...")
            # Convert to monthly (end of month)
            # Use 'M' for older pandas, 'ME' for newer
            try:
                monthly = data[price_col].resample('ME').last()
            except ValueError:
                monthly = data[price_col].resample('M').last()
            
            # Calculate returns
            returns = monthly.pct_change()
            
            print(f"      Success: {len(returns)} monthly returns")
            return returns
        
        except Exception as e:
            print(f"      Exception on attempt {attempt+1}: {type(e).__name__}: {e}")
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"      Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return None
    
    return None

def download_universe(tickers, start_date, end_date='2025-12-31', batch_size=10, delay=1):
    """
    Download data for entire universe
    Skips tickers that already have data in database
    
    Returns DataFrame with columns = tickers, index = dates, values = returns
    """
    # Check which tickers already have data
    con = sqlite3.connect(DB_PATH)
    existing = pd.read_sql("""
        SELECT DISTINCT i.proxy
        FROM instruments i
        JOIN prices p ON i.instrument_id = p.instrument_id
    """, con)
    con.close()
    
    existing_tickers = set(existing['proxy'].tolist())
    
    # Filter to only download new tickers
    tickers_to_download = {k: v for k, v in tickers.items() if k not in existing_tickers}
    tickers_already_have = {k: v for k, v in tickers.items() if k in existing_tickers}
    
    print(f"\nTicker status:")
    print(f"  Already in database: {len(tickers_already_have)} tickers")
    print(f"  Need to download: {len(tickers_to_download)} tickers")
    
    if len(tickers_to_download) == 0:
        print("\n  All tickers already in database. Nothing to download.")
        return pd.DataFrame()
    
    print(f"\nDownloading {len(tickers_to_download)} new tickers...")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Batch size: {batch_size} (delay {delay}s between batches)")
    
    all_returns = {}
    ticker_list = list(tickers_to_download.keys())
    
    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i:i+batch_size]
        
        print(f"\n  Batch {i//batch_size + 1}/{(len(ticker_list)-1)//batch_size + 1}")
        
        for ticker in batch:
            returns = download_ticker_data(ticker, start_date, end_date)
            
            if returns is not None and len(returns) > 0:
                all_returns[ticker] = returns
                print(f"    ✓ {ticker}: {len(returns)} months")
            else:
                print(f"    ✗ {ticker}: Failed")
        
        # Rate limiting - longer delay to avoid Yahoo Finance blocks
        if i + batch_size < len(ticker_list):
            print(f"    Waiting {delay}s...")
            time.sleep(delay)
    
    # Combine into DataFrame
    returns_df = pd.DataFrame(all_returns)
    
    print(f"\n  Successfully downloaded: {len(returns_df.columns)}/{len(tickers_to_download)} tickers")
    if len(returns_df) > 0:
        print(f"  Date range: {returns_df.index.min().strftime('%Y-%m')} to {returns_df.index.max().strftime('%Y-%m')}")
    
    return returns_df

# ============================================================================
# DATABASE INSERTION
# ============================================================================

def add_instruments_to_db(tickers_dict):
    """Add new instruments to database"""
    con = sqlite3.connect(DB_PATH)
    
    added = 0
    
    for ticker, name in tickers_dict.items():
        try:
            # Determine category based on which universe it came from
            if ticker in EQUITY_UNIVERSE:
                category = 'Equity'
            elif ticker in CREDIT_UNIVERSE:
                category = 'Credit'
            elif ticker in RATES_UNIVERSE:
                category = 'Interest Rates'
            elif ticker in COMMODITY_UNIVERSE:
                category = 'Commodities'
            elif ticker in CURRENCY_UNIVERSE:
                category = 'Currencies'
            elif ticker in VOLATILITY_UNIVERSE:
                category = 'Volatility'
            elif ticker in TREND_UNIVERSE:
                category = 'Trend Following'
            elif ticker in REAL_ESTATE_UNIVERSE:
                category = 'Real Estate'
            else:
                category = 'Alternative'
            
            # Insert or update
            con.execute("""
                INSERT OR IGNORE INTO instruments (proxy, name, category, data_type)
                VALUES (?, ?, ?, ?)
            """, (ticker, name, category, 'ETF'))
            
            added += 1
            
        except Exception as e:
            print(f"    Error adding {ticker}: {e}")
    
    con.commit()
    con.close()
    
    print(f"  Added {added} instruments to database")

def add_returns_to_db(returns_df):
    """Add return data to database"""
    con = sqlite3.connect(DB_PATH)
    
    # Get instrument IDs
    instruments = pd.read_sql("""
        SELECT instrument_id, proxy
        FROM instruments
    """, con)
    
    proxy_to_id = dict(zip(instruments['proxy'], instruments['instrument_id']))
    
    total_rows = 0
    
    for ticker in returns_df.columns:
        if ticker not in proxy_to_id:
            print(f"    Warning: {ticker} not in instruments table")
            continue
        
        instrument_id = proxy_to_id[ticker]
        ticker_returns = returns_df[ticker].dropna()
        
        if len(ticker_returns) == 0:
            continue
        
        # Prepare data
        data = []
        for date, value in ticker_returns.items():
            data.append({
                'instrument_id': instrument_id,
                'date': date,
                'value': value,
                'currency': 'USD'
            })
        
        # Insert
        df = pd.DataFrame(data)
        df.to_sql('temp_load', con, if_exists='replace', index=False)
        
        con.execute("""
            INSERT OR REPLACE INTO prices (instrument_id, date, value, currency)
            SELECT instrument_id, date, value, currency
            FROM temp_load
        """)
        
        total_rows += len(df)
    
    con.execute("DROP TABLE IF EXISTS temp_load")
    con.commit()
    con.close()
    
    print(f"  Added {total_rows:,} price records to database")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Download and load large universe of tradeable assets'
    )
    parser.add_argument('--start', type=str, default='1995-01-01',
                       help='Start date for data download')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Number of tickers to download per batch')
    parser.add_argument('--delay', type=float, default=3.0,
                       help='Delay between batches (seconds)')
    args = parser.parse_args()
    
    print("="*80)
    print("  LARGE UNIVERSE LOADER")
    print("  Downloading tradeable assets to represent all PC factors")
    print("="*80)
    
    print(f"\nUniverse breakdown:")
    print(f"  Equity:        {len(EQUITY_UNIVERSE)} tickers")
    print(f"  Credit:        {len(CREDIT_UNIVERSE)} tickers")
    print(f"  Rates:         {len(RATES_UNIVERSE)} tickers")
    print(f"  Commodities:   {len(COMMODITY_UNIVERSE)} tickers")
    print(f"  Currencies:    {len(CURRENCY_UNIVERSE)} tickers")
    print(f"  Volatility:    {len(VOLATILITY_UNIVERSE)} tickers")
    print(f"  Trend:         {len(TREND_UNIVERSE)} tickers")
    print(f"  Real Estate:   {len(REAL_ESTATE_UNIVERSE)} tickers")
    print(f"  Alternative:   {len(ALTERNATIVE_UNIVERSE)} tickers")
    print(f"  TOTAL:         {len(ALL_TICKERS)} tickers")
    
    # Download data
    returns_df = download_universe(
        ALL_TICKERS,
        start_date=args.start,
        batch_size=args.batch_size,
        delay=args.delay
    )
    
    if returns_df.empty:
        print("\nNo data downloaded. Exiting.")
        return
    
    # Add to database
    print("\n" + "="*80)
    print("  ADDING TO DATABASE")
    print("="*80)
    
    print("\nAdding instruments...")
    add_instruments_to_db(ALL_TICKERS)
    
    print("\nAdding return data...")
    add_returns_to_db(returns_df)
    
    # Summary
    con = sqlite3.connect(DB_PATH)
    
    instrument_count = pd.read_sql("SELECT COUNT(*) as count FROM instruments", con).iloc[0, 0]
    price_count = pd.read_sql("SELECT COUNT(*) as count FROM prices", con).iloc[0, 0]
    
    print("\n" + "="*80)
    print("  COMPLETE")
    print("="*80)
    
    print(f"\nDatabase now contains:")
    print(f"  Instruments: {instrument_count:,}")
    print(f"  Price records: {price_count:,}")
    
    # Show coverage
    coverage = pd.read_sql("""
        SELECT i.category, COUNT(DISTINCT i.proxy) as n_assets
        FROM instruments i
        JOIN prices p ON i.instrument_id = p.instrument_id
        GROUP BY i.category
        ORDER BY n_assets DESC
    """, con)
    
    print(f"\n  Coverage by category:")
    for _, row in coverage.iterrows():
        print(f"    {row['category']:20s}: {row['n_assets']:3d} assets")
    
    con.close()
    
    print()

if __name__ == "__main__":
    main()