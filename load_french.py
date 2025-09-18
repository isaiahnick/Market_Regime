import pandas as pd
import sqlite3
from pandas.tseries.offsets import MonthEnd

# Configuration
DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"
DATA_DIR = "/Users/isaiahnick/Desktop/Market Regime PCA/data"

def load_french_csv(filename):
    """Load and parse French factor CSV file"""
    file_path = f"{DATA_DIR}/{filename}"
    df = pd.read_csv(file_path)
    
    # Filter out non-date rows (keep only valid dates)
    df = df[pd.to_datetime(df['Date'], errors='coerce').notna()]
    
    # Parse date and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    return df

def save_to_db(factor_name, series):
    """Save factor series to database"""
    conn = sqlite3.connect(DB_PATH)
    
    # Get instrument ID
    result = pd.read_sql("""
        SELECT instrument_id FROM instruments 
        WHERE proxy = ?
    """, conn, params=[factor_name])
    
    if result.empty:
        print(f"  ✗ {factor_name} not found in instruments table")
        conn.close()
        return 0
    
    instrument_id = result.iloc[0, 0]
    
    # Prepare data
    df = series.dropna().reset_index()
    df.columns = ['date', 'value']
    df['instrument_id'] = instrument_id
    df['currency'] = 'USD'
    
    # Save to database
    df.to_sql('temp_french', conn, if_exists='replace', index=False)
    
    conn.execute("""
        INSERT OR REPLACE INTO prices 
        (instrument_id, date, value, currency)
        SELECT instrument_id, date, value, currency 
        FROM temp_french
    """)
    
    conn.execute("DROP TABLE temp_french")
    conn.commit()
    conn.close()
    
    return len(df)

def main():
    """Load French factor data"""
    print("Loading French factor data...")
    
    # Load the CSV files
    print("Loading French_Factors.csv...")
    factors_data = load_french_csv("French_Factors.csv")
    
    print("Loading French_MOM.csv...")
    momentum_data = load_french_csv("French_MOM.csv")
    
    # Load each factor
    factors = [
        ('FF_SMB', factors_data['SMB']),
        ('FF_HML', factors_data['HML']),
        ('FF_RMW', factors_data['RMW']),
        ('FF_UMD', momentum_data['Mom']),
    ]
    
    total_loaded = 0
    
    for factor_name, factor_series in factors:
        print(f"Loading {factor_name}...")
        count = save_to_db(factor_name, factor_series)
        if count > 0:
            total_loaded += count
            start_date = factor_series.dropna().index.min().strftime('%Y-%m')
            end_date = factor_series.dropna().index.max().strftime('%Y-%m')
            print(f"Loaded {count:,} records ({start_date} to {end_date})")
        else:
            print(f"FAILED to load")
    
    print(f"\nTotal records loaded: {total_loaded:,}")

if __name__ == "__main__":
    main()