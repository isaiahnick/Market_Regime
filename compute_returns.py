import pandas as pd
import numpy as np
import sqlite3

# Configuration
DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"
FACTORS_CSV = "factors.csv"

import pandas as pd
import numpy as np
import sqlite3

# Configuration
DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"
FACTORS_CSV = "factors.csv"

def load_factor_mapping():
    """Load factor-category mapping and transformations from CSV"""
    df = pd.read_csv(FACTORS_CSV)
    df = df.dropna(subset=['category', 'proxy'])
    df = df[df['category'].str.strip() != '']
    df = df[df['proxy'].str.strip() != '']
    
    # Create category mapping
    category_mapping = {}
    for category, group in df.groupby('category'):
        category_mapping[category] = group['proxy'].tolist()
    
    # Create transformation mapping
    transformation_mapping = {}
    for _, row in df.iterrows():
        transformation_mapping[row['proxy']] = row.get('transformation', 'levels')  # Default to levels
    
    print(f"Found {len(category_mapping)} categories:")
    for cat, proxies in category_mapping.items():
        print(f"  {cat}: {len(proxies)} proxies")
    
    return category_mapping, transformation_mapping

def get_price_data():
    """Load all price data from database"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT i.proxy, i.category, p.date, 
               COALESCE(p.value, p.close) as price
        FROM prices p
        JOIN instruments i ON p.instrument_id = i.instrument_id
        ORDER BY i.proxy, p.date
    """, conn, parse_dates=['date'])
    conn.close()
    return df

def calculate_returns(group, proxy, transformation_mapping):
    """Calculate appropriate returns/levels for each instrument based on CSV transformation"""
    daily_prices = group.set_index('date')['price'].dropna()
    
    # Remove any NaT values from the index
    daily_prices = daily_prices[daily_prices.index.notna()]
    
    if daily_prices.empty:
        return pd.Series(dtype=float)
    
    transformation = transformation_mapping.get(proxy, 'levels')
    
    # Check if data is already monthly (median gap > 20 days)
    if len(daily_prices) > 1:
        date_gaps = daily_prices.index.to_series().diff()
        median_gap = date_gaps.median()
        is_monthly = median_gap > pd.Timedelta(days=20)
    else:
        is_monthly = False
    
    if transformation == 'log_returns':
        monthly_prices = daily_prices.resample('ME').last()
        monthly_returns = np.log(monthly_prices / monthly_prices.shift(1))
        return monthly_returns
        
    elif transformation == 'levels':
        monthly_levels = daily_prices.resample('ME').last()
        return monthly_levels
        
    elif transformation in ['factor_returns', 'use_directly']:
        # French factors: already monthly, don't resample
        if is_monthly:
            return daily_prices
        else:
            # Shouldn't happen, but handle it
            return daily_prices.resample('ME').last()
        
    elif transformation in ['first_differences', 'first differences']:
        monthly_levels = daily_prices.resample('ME').last()
        first_diff = monthly_levels.diff()
        return first_diff
        
    elif transformation in ['log_differences', 'log differences']:
        monthly_levels = daily_prices.resample('ME').last()
        log_diff = np.log(monthly_levels).diff()
        return log_diff
        
    elif transformation == 'yoy_change':
        monthly_levels = daily_prices.resample('ME').last()
        yoy_change = monthly_levels.pct_change(12) * 100
        return yoy_change
        
    elif transformation == 'mom_change':
        monthly_levels = daily_prices.resample('ME').last()
        mom_change = monthly_levels.pct_change(1) * 100
        return mom_change
        
    elif transformation == 'simple_returns':
        monthly_prices = daily_prices.resample('ME').last()
        simple_returns = monthly_prices.pct_change(1)
        return simple_returns
        
    else:
        print(f"  Warning: Unknown transformation '{transformation}' for {proxy}, using levels")
        monthly_levels = daily_prices.resample('ME').last()
        return monthly_levels

def save_to_db(monthly_data, category_mapping):
    """Save monthly data to database"""
    conn = sqlite3.connect(DB_PATH)
    
    # Save long format with categories
    monthly_data_with_cat = monthly_data.copy()
    proxy_to_category = {}
    for cat, proxies in category_mapping.items():
        for proxy in proxies:
            proxy_to_category[proxy] = cat
    
    monthly_data_with_cat['category'] = monthly_data_with_cat['proxy'].map(proxy_to_category)
    monthly_data_with_cat.to_sql('factors_monthly', conn, if_exists='replace', index=False)
    
    # Create and save wide format  
    wide = monthly_data.pivot(index='date', columns='proxy', values='value')
    wide.reset_index().to_sql('factors_monthly_raw', conn, if_exists='replace', index=False)
    
    # Create standardized version
    standardized = wide.copy()
    for col in standardized.columns:
        if col != 'date' and standardized[col].notna().sum() > 10:
            # Winsorize at 1%/99%
            q01, q99 = standardized[col].quantile([0.01, 0.99])
            standardized[col] = standardized[col].clip(q01, q99)
            # Z-score standardize
            standardized[col] = (standardized[col] - standardized[col].mean()) / standardized[col].std()
    
    standardized.reset_index().to_sql('factors_monthly_z', conn, if_exists='replace', index=False)
    
    # Create category-wise matrices for PCA
    for category, proxies in category_mapping.items():
        available_proxies = [p for p in proxies if p in wide.columns]
        if available_proxies:
            cat_data = wide[available_proxies].copy()
            # Drop rows with all NaN for this category
            cat_data = cat_data.dropna(how='all')
            cat_data.reset_index().to_sql(f'category_{category.lower().replace(" ", "_")}', 
                                        conn, if_exists='replace', index=False)
    
    conn.commit()
    conn.close()

def main():
    """Calculate consistent monthly factor data organized by categories"""
    print("Loading factor-category mapping and transformations...")
    category_mapping, transformation_mapping = load_factor_mapping()
    
    print("\nLoading price data...")
    df = get_price_data()
    
    if df.empty:
        print("No price data found. Run data loaders first.")
        return
    
    print(f"Processing {df['proxy'].nunique()} instruments...")
    
    monthly_data = []
    
    for proxy, group in df.groupby('proxy'):
        transformation = transformation_mapping.get(proxy, 'levels')
        print(f"Processing {proxy} (transformation: {transformation})...")
        
        values = calculate_returns(group, proxy, transformation_mapping)
        
        if values.empty:
            print(f"  ✗ No data")
            continue
        
        # Create monthly data frame
        proxy_df = pd.DataFrame({
            'date': values.index,
            'proxy': proxy,
            'value': values.values
        })
        
        monthly_data.append(proxy_df)
        valid_count = proxy_df['value'].notna().sum()
        print(f"  ✓ {valid_count:,} monthly observations")
    
    if not monthly_data:
        print("No monthly data created.")
        return
    
    # Combine all data
    all_monthly = pd.concat(monthly_data, ignore_index=True)
    
    print(f"\nSaving results...")
    save_to_db(all_monthly, category_mapping)
    
    total_obs = len(all_monthly)
    valid_obs = all_monthly['value'].notna().sum()
    print(f"Total observations: {total_obs:,}")
    print(f"Valid values: {valid_obs:,} ({valid_obs/total_obs*100:.1f}%)")
    
    print(f"\nData ready for PCA analysis by category:")
    for category in category_mapping.keys():
        print(f"  Table: category_{category.lower().replace(' ', '_')}")

if __name__ == "__main__":
    main()