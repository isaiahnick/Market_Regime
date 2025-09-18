import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

# Configuration
DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"
DATA_DIR = "/Users/isaiahnick/Desktop/Market Regime PCA/data"

def get_instruments():
    """Get Bloomberg instruments that need data"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT instrument_id, proxy, name 
        FROM instruments 
        WHERE data_type = 'BBG'
    """, conn)
    conn.close()
    return df

def load_csv(file_path):
    """Load Bloomberg CSV file - only use Last Price"""
    df = pd.read_csv(file_path)
    
    # Debug: print column names
    print(f"    Columns found: {list(df.columns)}")
    
    # Parse date with proper 2-digit year handling
    # Assume years 00-25 are 2000-2025, years 26-99 are 1926-1999
    def fix_year(date_str):
        parts = date_str.split('/')
        if len(parts[2]) == 2:  # 2-digit year
            year = int(parts[2])
            if year <= 25:  # 00-25 -> 2000-2025
                parts[2] = str(2000 + year)
            else:  # 26-99 -> 1926-1999
                parts[2] = str(1900 + year)
        return '/'.join(parts)
    
    df['Date'] = df['Date'].apply(fix_year)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # Find the price column - try different possible names
    price_column = None
    possible_names = ['Last Price', 'PX_LAST', 'Close', 'Price', 'Value']
    
    for col_name in possible_names:
        if col_name in df.columns:
            price_column = col_name
            break
    
    if price_column is None:
        # If no standard column found, use the first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            price_column = numeric_cols[0]
            print(f"    Using first numeric column: {price_column}")
        else:
            raise ValueError(f"No suitable price column found in {file_path}")
    else:
        print(f"    Using price column: {price_column}")
    
    # Rename to 'close' and convert to numeric
    df = df.rename(columns={price_column: 'close'})
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Keep only the close price column and remove rows with missing data
    df = df[['close']].dropna()
    
    return df

def save_to_db(instrument_id, df):
    """Save price data to database (simplified schema)"""
    conn = sqlite3.connect(DB_PATH)
    
    # Prepare data for database
    db_df = df.copy()
    db_df['instrument_id'] = instrument_id
    db_df['value'] = df['close']  # For consistency
    db_df['currency'] = 'USD'
    db_df = db_df.reset_index()  # Date becomes a column
    
    # Save to database - only insert close, value, currency (no open column)
    db_df.to_sql('temp_prices', conn, if_exists='replace', index=False)
    
    conn.execute("""
        INSERT OR REPLACE INTO prices 
        (instrument_id, date, close, value, currency)
        SELECT instrument_id, Date, close, value, currency 
        FROM temp_prices
    """)
    
    conn.execute("DROP TABLE temp_prices")
    conn.commit()
    conn.close()
    
    return len(db_df)

def calculate_realized_vol(price_series, window=22):
    """Calculate rolling realized volatility"""
    returns = np.log(price_series / price_series.shift(1))
    realized_vol = returns.rolling(window).std() * np.sqrt(252)
    return realized_vol

def main():
    """Load all Bloomberg data"""
    instruments = get_instruments()
    print(f"Loading {len(instruments)} instruments...")
    
    total_loaded = 0
    spx_data = None
    loaded_proxies = []
    missing_proxies = []
    
    for _, row in instruments.iterrows():
        instrument_id = row['instrument_id']
        proxy = row['proxy']
        
        # Find CSV file
        csv_file = os.path.join(DATA_DIR, f"{proxy}.csv")
        
        if os.path.exists(csv_file):
            print(f"Loading {proxy}...")
            
            # Load and save data
            df = load_csv(csv_file)
            count = save_to_db(instrument_id, df)
            total_loaded += count
            loaded_proxies.append(proxy)
            
            # Store SPX data for volatility calculation
            if proxy == "SPX_Index":
                spx_data = df
            
            print(f"  ✓ {count:,} records")
        else:
            print(f"  ✗ File not found: {proxy}.csv")
            missing_proxies.append(proxy)
    
    # Calculate SPX realized volatility if we have the data
    if spx_data is not None:
        print("\nCalculating SPX realized volatility...")
        
        # Get instrument ID for realized vol
        conn = sqlite3.connect(DB_PATH)
        result = pd.read_sql("""
            SELECT instrument_id FROM instruments 
            WHERE proxy = 'SPX_RealizedVol'
        """, conn)
        conn.close()
        
        if not result.empty:
            rv_instrument_id = result.iloc[0, 0]
            realized_vol = calculate_realized_vol(spx_data['close'])
            
            # Save realized vol as time series
            rv_df = pd.DataFrame({'close': realized_vol}).dropna()
            count = save_to_db(rv_instrument_id, rv_df)
            total_loaded += count
            loaded_proxies.append('SPX_RealizedVol')
            print(f"  ✓ {count:,} realized vol records")
        else:
            print(f"  ✗ SPX_RealizedVol instrument not found in database")
            missing_proxies.append('SPX_RealizedVol')
    
    # Summary report
    print(f"\n" + "="*50)
    print(f"LOADING SUMMARY")
    print(f"="*50)
    print(f"Total records loaded: {total_loaded:,}")
    print(f"Successfully loaded: {len(loaded_proxies)} factors")
    print(f"Missing files: {len(missing_proxies)} factors")
    
    if missing_proxies:
        print(f"\nFactors that did NOT get loaded (missing CSV files):")
        for proxy in sorted(missing_proxies):
            print(f"  - {proxy}")
    
    if loaded_proxies:
        print(f"\nFactors successfully loaded:")
        for proxy in sorted(loaded_proxies):
            print(f"  - {proxy}")

if __name__ == "__main__":
    main()