# seed_instruments.py
import pandas as pd
from sqlalchemy import create_engine, text

DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"
CSV_PATH = "factors.csv"

def seed_instruments():
    """Load instrument data from CSV into SQLite database."""
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    
    # Create table if it doesn't exist (removed source_hint column)
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS instruments (
        instrument_id INTEGER PRIMARY KEY,
        proxy         TEXT NOT NULL,
        name          TEXT,
        category      TEXT,
        data_type     TEXT,
        UNIQUE (proxy, data_type)
    );
    """
    
    upsert_sql = """
    INSERT INTO instruments (proxy, name, category, data_type)
    VALUES (:proxy, :name, :category, :data_type)
    ON CONFLICT(proxy, data_type) DO UPDATE SET
        name=excluded.name,
        category=excluded.category;
    """
    
    # Read CSV with better error handling
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Successfully loaded CSV with {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {CSV_PATH}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Validate CSV columns
    expected_columns = {"proxy", "name", "category", "data_type"}
    actual_columns = set(df.columns)
    
    missing_columns = expected_columns - actual_columns
    if missing_columns:
        print(f"Error: CSV missing required columns: {missing_columns}")
        print(f"Found columns: {list(actual_columns)}")
        return
    
    # Clean data
    df = df.dropna(subset=['proxy'])  # Remove rows with missing proxy
    df = df.fillna('')  # Fill other NaN values with empty strings
    
    print(f"Processing {len(df)} valid rows...")
    
    # Insert data
    try:
        with engine.begin() as conn:
            conn.execute(text(create_table_sql))
            
            rows_inserted = 0
            for _, row in df.iterrows():
                conn.execute(text(upsert_sql), {
                    "proxy": row["proxy"],
                    "name": row["name"],
                    "category": row["category"],
                    "data_type": row["data_type"],
                })
                rows_inserted += 1
            
            print(f"Successfully loaded {rows_inserted} instruments into database")
            
    except Exception as e:
        print(f"Error inserting data into database: {e}")
        return

if __name__ == "__main__":
    seed_instruments()