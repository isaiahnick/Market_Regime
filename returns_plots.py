# generate_regime_returns_table.py
# Generate clean, professional academic-style table showing annualized factor mean returns by regime
# CORRECT CALCULATION: Uses actual factor returns, not PC1 z-scores
#
# Usage:
#   python generate_regime_returns_table.py
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
    """Load GMM regimes, PCA factors, and original factor data"""
    con = sqlite3.connect(DB_PATH)
    
    try:
        # Load regime assignments
        regimes = pd.read_sql("SELECT * FROM gmm_regimes", con)
        regimes['date'] = pd.to_datetime(regimes['date'])
        
        # Load original factor data (factors_monthly_raw has the actual returns/values)
        factors = pd.read_sql("SELECT * FROM factors_monthly_raw", con)
        
        # Normalize date column
        if 'date' not in factors.columns:
            if 'index' in factors.columns:
                factors = factors.rename(columns={'index': 'date'})
            else:
                for c in factors.columns:
                    if 'date' in c.lower():
                        factors = factors.rename(columns={c: 'date'})
                        break
        
        factors['date'] = pd.to_datetime(factors['date'])
        
        # Get metadata
        meta = pd.read_sql("SELECT * FROM gmm_meta", con).iloc[0]
        k_star = int(meta['chosen_k'])
        
        return regimes, factors, k_star
        
    finally:
        con.close()

def calculate_annualized_returns_correct(regimes, factors, k_star, category_mapping):
    """
    Calculate ACTUAL annualized mean returns for each category by regime.
    
    For each category:
    1. Get all proxies in that category
    2. Average their returns by month (equal weight)
    3. Calculate mean return by regime
    4. Annualize by multiplying by 12
    """
    
    # Merge regimes with factor data
    merged = pd.merge(regimes[['date', 'regime']], factors, on='date', how='inner')
    
    print(f"\nMerged data shape: {merged.shape}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    
    # Calculate category-level returns
    results = []
    
    # Exclude Equity Short Volatility
    categories_to_process = {k: v for k, v in category_mapping.items() if k != 'Equity Short Volatility'}
    
    for category, proxies in sorted(categories_to_process.items()):
        print(f"\nProcessing {category}...")
        
        # Get available proxies for this category
        available_proxies = [p for p in proxies if p in merged.columns]
        
        if len(available_proxies) == 0:
            print(f"  ✗ No proxies found in data")
            continue
        
        print(f"  Found {len(available_proxies)} proxies: {available_proxies[:3]}{'...' if len(available_proxies) > 3 else ''}")
        
        row_data = {'Factor': category}
        
        for regime in range(k_star):
            regime_data = merged[merged['regime'] == regime]
            
            # Get values for all proxies in this category
            category_values = regime_data[available_proxies]
            
            # Average across proxies (equal weight)
            avg_values = category_values.mean(axis=1).dropna()
            
            if len(avg_values) > 0:
                # Calculate mean return for this regime
                mean_monthly_return = avg_values.mean()
                
                # Annualize: multiply by 12 (simple annualization)
                annualized_return = mean_monthly_return * 12
                
                row_data[f'Market Regime {regime + 1}'] = annualized_return
                
                print(f"  Regime {regime + 1}: {len(avg_values)} months, mean={mean_monthly_return:.4f}, annualized={annualized_return:.4f}")
            else:
                row_data[f'Market Regime {regime + 1}'] = np.nan
                print(f"  Regime {regime + 1}: No data")
        
        results.append(row_data)
    
    # Convert to DataFrame
    returns_df = pd.DataFrame(results)
    returns_df = returns_df.set_index('Factor')
    
    return returns_df

def create_clean_academic_table(returns_df, k_star):
    """Create clean, professional table for academic publication"""
    
    output_dir = "gmm_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up clean, professional styling - match the visualization font
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 16,
    })
    
    # Define simple color scheme
    GREEN = '#90EE90'  # Light green for positive
    RED = '#FFB6C6'    # Light red for negative
    
    # Prepare data
    regime_cols = [f'Market Regime {i+1}' for i in range(k_star)]
    table_data = returns_df[regime_cols].copy()
    
    # Create figure with appropriate size and better spacing
    n_rows = len(table_data)
    n_cols = k_star
    fig_width = 10
    fig_height = n_rows * 0.4 + 3  # More space for title and legend
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare cell data and colors
    cell_text = []
    cell_colors = []
    
    for idx, row in table_data.iterrows():
        # Format values as percentages
        formatted_row = []
        row_colors = []
        
        for val in row.values:
            if np.isnan(val):
                formatted_row.append('N/A')
                row_colors.append('white')
            else:
                formatted_row.append(f'{val:.2%}')
                # Simple binary color: green if positive, red if negative
                row_colors.append(GREEN if val >= 0 else RED)
        
        cell_text.append(formatted_row)
        cell_colors.append(row_colors)
    
    # Row labels (factor names)
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
        pass  # Corner cell doesn't exist in some matplotlib versions
    
    # Add title with proper spacing
    title_text = 'Annualized Factor Mean Returns by Market Regime'
    fig.suptitle(title_text, fontsize=18, fontweight='bold', y=0.97)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.02)  # Minimal bottom spacing
    
    filename = f"{output_dir}/table_regime_returns_clean.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.3)
    print(f"\n✓ Table saved to: {filename}")
    plt.show()
    
    return table_data

def save_to_database(returns_df):
    """Save the regime returns table to database"""
    con = sqlite3.connect(DB_PATH)
    
    try:
        # Save mean returns
        returns_df.to_sql('gmm_regime_means', con, if_exists='replace', index=True)
        print(f"✓ Regime mean returns saved to: gmm_regime_means")
        
    finally:
        con.close()

def print_summary_statistics(returns_df, k_star):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for regime in range(k_star):
        regime_col = f'Market Regime {regime + 1}'
        regime_data = returns_df[regime_col].dropna()
        
        print(f"\n{regime_col}:")
        print(f"  Number of factors: {len(regime_data)}")
        print(f"  Mean return:       {regime_data.mean():>8.2%}")
        print(f"  Median return:     {regime_data.median():>8.2%}")
        print(f"  Std deviation:     {regime_data.std():>8.2%}")
        print(f"  Min return:        {regime_data.min():>8.2%}")
        print(f"  Max return:        {regime_data.max():>8.2%}")
        
        positive = (regime_data > 0).sum()
        negative = (regime_data < 0).sum()
        print(f"  Positive returns:  {positive} ({positive/len(regime_data)*100:.1f}%)")
        print(f"  Negative returns:  {negative} ({negative/len(regime_data)*100:.1f}%)")

def main():
    """Generate clean academic-style regime returns table"""
    print("="*70)
    print("GENERATING REGIME RETURNS TABLE (CORRECTED CALCULATION)")
    print("="*70)
    
    print("\nLoading factor mapping...")
    category_mapping = load_factor_mapping()
    print(f"✓ Found {len(category_mapping)} categories")
    
    print("\nLoading data from database...")
    regimes, factors, k_star = load_all_data()
    print(f"✓ Loaded {len(regimes)} regime observations")
    print(f"✓ Loaded {len(factors)} factor observations")
    print(f"✓ Number of regimes: {k_star}")
    
    print("\nCalculating annualized returns by regime (CORRECT METHOD)...")
    print("Method: Average proxies within category, calculate mean by regime, annualize")
    returns_df = calculate_annualized_returns_correct(regimes, factors, k_star, category_mapping)
    
    print("\nCreating clean academic table...")
    table_data = create_clean_academic_table(returns_df, k_star)
    
    print("\nSaving results to database...")
    save_to_database(returns_df)
    
    print_summary_statistics(returns_df, k_star)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()