# generate_regime_volatility_table.py
# Generate clean, professional academic-style table showing factor volatility indices by regime
# Uses standardized PC1 factors to create comparable volatility measures across categories
#
# Usage:
#   python generate_regime_volatility_table.py
#
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"
FACTORS_CSV = "factors.csv"

def load_factor_mapping():
    """Load factor-category mapping from CSV"""
    df = pd.read_csv(FACTORS_CSV)
    df = df.dropna(subset=['category', 'proxy'])
    df = df[df['category'].str.strip() != '']
    df = df[df['proxy'].str.strip() != '']
    
    # Create category mapping
    category_mapping = {}
    for category, group in df.groupby('category'):
        category_mapping[category] = group['proxy'].tolist()
    
    return category_mapping

def load_all_data():
    """Load GMM regimes and PCA factors"""
    con = sqlite3.connect(DB_PATH)
    
    try:
        # Load regime assignments
        regimes = pd.read_sql("SELECT * FROM gmm_regimes", con)
        regimes['date'] = pd.to_datetime(regimes['date'])
        
        # Load PCA factors (PC1 for each category)
        pca_wide = pd.read_sql("SELECT * FROM pca_factors_wide", con)
        
        # Normalize date column
        if 'date' not in pca_wide.columns:
            if 'index' in pca_wide.columns:
                pca_wide = pca_wide.rename(columns={'index': 'date'})
            else:
                for c in pca_wide.columns:
                    if 'date' in c.lower():
                        pca_wide = pca_wide.rename(columns={c: 'date'})
                        break
        
        pca_wide['date'] = pd.to_datetime(pca_wide['date'])
        
        # Get metadata
        meta = pd.read_sql("SELECT * FROM gmm_meta", con).iloc[0]
        k_star = int(meta['chosen_k'])
        
        return regimes, pca_wide, k_star
        
    finally:
        con.close()

def calculate_volatility_index(regimes, pca_wide, k_star):
    """
    Calculate volatility index for each category by regime using PC1 factors.
    
    Index = (std dev in regime) / (overall std dev)
    - Index > 1.0 means higher volatility than average
    - Index < 1.0 means lower volatility than average
    - Index = 1.0 means average volatility
    """
    
    # Merge regimes with PCA data
    merged = pd.merge(regimes[['date', 'regime']], pca_wide, on='date', how='inner')
    
    print(f"\nMerged data shape: {merged.shape}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    
    # Get all PC1 columns
    pc1_cols = [col for col in merged.columns if col.endswith('_PC1') and col not in ['date', 'regime']]
    pc1_cols = sorted(pc1_cols)
    
    print(f"\nFound {len(pc1_cols)} PC1 factors")
    
    # Exclude Equity Short Volatility
    pc1_cols = [col for col in pc1_cols if 'Equity Short Volatility' not in col]
    
    # Calculate volatility indices
    results = []
    
    for pc1_col in pc1_cols:
        category = pc1_col.replace('_PC1', '').replace('_', ' ')
        
        print(f"\nProcessing {category}...")
        
        # Calculate overall standard deviation
        overall_values = merged[pc1_col].dropna()
        overall_std = overall_values.std()
        
        if overall_std == 0 or np.isnan(overall_std):
            print(f"  ✗ No variation in data")
            continue
        
        row_data = {'Factor': category}
        
        for regime in range(k_star):
            regime_data = merged[merged['regime'] == regime]
            regime_values = regime_data[pc1_col].dropna()
            
            if len(regime_values) > 1:
                regime_std = regime_values.std()
                # Volatility index: regime std / overall std
                vol_index = regime_std / overall_std
                
                row_data[f'Market Regime {regime + 1}'] = vol_index
                print(f"  Regime {regime + 1}: {len(regime_values)} obs, vol index={vol_index:.3f}")
            else:
                row_data[f'Market Regime {regime + 1}'] = np.nan
                print(f"  Regime {regime + 1}: Insufficient data")
        
        results.append(row_data)
    
    # Convert to DataFrame
    volatility_df = pd.DataFrame(results)
    volatility_df = volatility_df.set_index('Factor')
    
    return volatility_df

def get_volatility_color(vol, vmin, vmax):
    """
    Return color based on volatility level using 4-color scheme:
    - Green: Very low volatility (0-25th percentile)
    - Yellow: Medium-low volatility (25-50th percentile)
    - Orange: Medium-high volatility (50-75th percentile)
    - Red: Very high volatility (75-100th percentile)
    """
    if np.isnan(vol):
        return 'white'
    
    # Normalize to 0-1 range
    normalized = (vol - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    
    # Define colors
    if normalized < 0.25:
        return '#90EE90'  # Light green (very low vol)
    elif normalized < 0.50:
        return '#FFFF99'  # Light yellow (medium-low vol)
    elif normalized < 0.75:
        return '#FFB366'  # Light orange (medium-high vol)
    else:
        return '#FFB6C6'  # Light red (very high vol)

def create_clean_volatility_table(volatility_df, k_star):
    """Create clean, professional table for volatility indices"""
    
    output_dir = "gmm_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up clean, professional styling
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 16,
    })
    
    # Prepare data
    regime_cols = [f'Market Regime {i+1}' for i in range(k_star)]
    table_data = volatility_df[regime_cols].copy()
    
    # Get min and max for color scaling
    all_values = table_data.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    vmin, vmax = all_values.min(), all_values.max()
    
    # Create figure with appropriate size
    n_rows = len(table_data)
    n_cols = k_star
    fig_width = 10
    fig_height = n_rows * 0.4 + 3
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare cell data and colors
    cell_text = []
    cell_colors = []
    
    for idx, row in table_data.iterrows():
        formatted_row = []
        row_colors = []
        
        for val in row.values:
            if np.isnan(val):
                formatted_row.append('N/A')
                row_colors.append('white')
            else:
                # Format as index (e.g., 0.85, 1.23)
                formatted_row.append(f'{val:.2f}')
                row_colors.append(get_volatility_color(val, vmin, vmax))
        
        cell_text.append(formatted_row)
        cell_colors.append(row_colors)
    
    # Row labels
    row_labels = list(table_data.index)
    
    # Column headers
    col_labels = regime_cols
    
    # Create table
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.2)
    
    # Apply colors to cells
    for i in range(len(cell_text)):
        for j in range(len(cell_text[i])):
            cell = table[(i+1, j)]
            cell.set_facecolor(cell_colors[i][j])
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)
            cell.set_text_props(weight='bold', color='black', fontsize=12)
    
    # Style column headers
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(weight='bold', color='black', fontsize=13)
        cell.set_edgecolor('black')
        cell.set_linewidth(1.0)
    
    # Style row labels
    for i in range(1, len(cell_text) + 1):
        cell = table[(i, -1)]
        cell.set_facecolor('white')
        cell.set_text_props(weight='bold', color='black', ha='left', fontsize=12)
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
    
    # Style corner cell (if it exists)
    try:
        table[(0, -1)].set_facecolor('#E8E8E8')
        table[(0, -1)].set_edgecolor('black')
        table[(0, -1)].set_linewidth(1.0)
    except KeyError:
        pass
    
    # Add title
    title_text = 'Factor Volatility Index by Market Regime'
    fig.suptitle(title_text, fontsize=18, fontweight='bold', y=0.97)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.02)
    
    filename = f"{output_dir}/table_regime_volatilities_clean.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.3)
    print(f"\n✓ Table saved to: {filename}")
    plt.show()
    
    return table_data

def save_to_database(volatility_df):
    """Save the volatility indices to database"""
    con = sqlite3.connect(DB_PATH)
    
    try:
        volatility_df.to_sql('gmm_regime_volatility_index', con, if_exists='replace', index=True)
        print(f"✓ Volatility indices saved to: gmm_regime_volatility_index")
        
    finally:
        con.close()

def print_summary_statistics(volatility_df, k_star):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print("\nVolatility Index Interpretation:")
    print("  < 0.80 = Low volatility regime")
    print("  0.80-1.20 = Normal volatility regime")
    print("  > 1.20 = High volatility regime")
    
    for regime in range(k_star):
        regime_col = f'Market Regime {regime + 1}'
        regime_data = volatility_df[regime_col].dropna()
        
        print(f"\n{regime_col}:")
        print(f"  Number of factors: {len(regime_data)}")
        print(f"  Mean index:        {regime_data.mean():>8.3f}")
        print(f"  Median index:      {regime_data.median():>8.3f}")
        print(f"  Std deviation:     {regime_data.std():>8.3f}")
        print(f"  Min index:         {regime_data.min():>8.3f}")
        print(f"  Max index:         {regime_data.max():>8.3f}")
        
        low_vol = (regime_data < 0.80).sum()
        high_vol = (regime_data > 1.20).sum()
        print(f"  Low volatility:    {low_vol} factors ({low_vol/len(regime_data)*100:.1f}%)")
        print(f"  High volatility:   {high_vol} factors ({high_vol/len(regime_data)*100:.1f}%)")

def main():
    """Generate volatility index table"""
    print("="*70)
    print("GENERATING FACTOR VOLATILITY INDEX TABLE")
    print("="*70)
    
    print("\nLoading factor mapping...")
    category_mapping = load_factor_mapping()
    print(f"✓ Found {len(category_mapping)} categories")
    
    print("\nLoading data from database...")
    regimes, pca_wide, k_star = load_all_data()
    print(f"✓ Loaded {len(regimes)} regime observations")
    print(f"✓ Number of regimes: {k_star}")
    
    print("\nCalculating volatility indices by regime...")
    print("Index = (regime std dev) / (overall std dev)")
    volatility_df = calculate_volatility_index(regimes, pca_wide, k_star)
    
    print("\nCreating clean academic table...")
    table_data = create_clean_volatility_table(volatility_df, k_star)
    
    print("\nSaving results to database...")
    save_to_database(volatility_df)
    
    print_summary_statistics(volatility_df, k_star)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()