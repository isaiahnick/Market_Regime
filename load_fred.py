import os
import pandas as pd
import sqlite3
from fredapi import Fred

# Configuration
DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"

def get_fred_instruments():
    """Get FRED instruments that need data"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT instrument_id, proxy 
        FROM instruments 
        WHERE data_type = 'FRED'
    """, conn)
    conn.close()
    return df

def save_to_db(instrument_id, series):
    """Save FRED series to database"""
    conn = sqlite3.connect(DB_PATH)
    
    # Prepare data for database
    df = series.dropna().reset_index()
    df.columns = ['date', 'value']
    df['instrument_id'] = instrument_id
    df['currency'] = 'USD'
    
    # Save to database
    df.to_sql('temp_fred', conn, if_exists='replace', index=False)
    
    conn.execute("""
        INSERT OR REPLACE INTO prices 
        (instrument_id, date, value, currency)
        SELECT instrument_id, date, value, currency 
        FROM temp_fred
    """)
    
    conn.execute("DROP TABLE temp_fred")
    conn.commit()
    conn.close()
    
    return len(df)

def main():
    """Load all FRED data"""
    # Check for API key
    api_key = '3208e5d6a5cff74ab3954a73f1f12e5b'
    if not api_key:
        print("Error: FRED_API_KEY environment variable not set")
        return
    
    fred = Fred(api_key=api_key)
    instruments = get_fred_instruments()
    
    print(f"Loading {len(instruments)} FRED instruments...")
    
    total_loaded = 0
    loaded_proxies = []
    failed_proxies = []
    
    # Store series for calculated spreads
    stored_series = {}
    
    for _, row in instruments.iterrows():
        instrument_id = row['instrument_id']
        series_id = row['proxy']
        
        # Handle calculated series
        if series_id == "DGS30_minus_DGS5":
            print(f"Calculating {series_id}...")
            try:
                # Load required series
                if 'DGS30' not in stored_series:
                    stored_series['DGS30'] = fred.get_series('DGS30')
                if 'DGS5' not in stored_series:
                    stored_series['DGS5'] = fred.get_series('DGS5')
                
                # Calculate spread
                spread_series = stored_series['DGS30'] - stored_series['DGS5']
                spread_series = spread_series.dropna()
                
                count = save_to_db(instrument_id, spread_series)
                total_loaded += count
                loaded_proxies.append(series_id)
                print(f"  ✓ {count:,} records")
                
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                failed_proxies.append(series_id)
        else:
            # Handle regular FRED series
            try:
                print(f"Loading {series_id}...")
                series = fred.get_series(series_id)
                
                # Store for potential use in calculated series
                stored_series[series_id] = series
                
                count = save_to_db(instrument_id, series)
                total_loaded += count
                loaded_proxies.append(series_id)
                print(f"  ✓ {count:,} records")
                
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                failed_proxies.append(series_id)
    
    # Summary report
    print(f"\n" + "="*50)
    print(f"FRED LOADING SUMMARY")
    print(f"="*50)
    print(f"Total records loaded: {total_loaded:,}")
    print(f"Successfully loaded: {len(loaded_proxies)} series")
    print(f"Failed to load: {len(failed_proxies)} series")
    
    if failed_proxies:
        print(f"\nSeries that FAILED to load:")
        for proxy in sorted(failed_proxies):
            print(f"  - {proxy}")
    
    if loaded_proxies:
        print(f"\nSeries successfully loaded:")
        for proxy in sorted(loaded_proxies):
            print(f"  - {proxy}")

if __name__ == "__main__":
    main()