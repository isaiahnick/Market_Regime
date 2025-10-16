# pca.py - MODIFIED VERSION
# Key changes:
# 1. Extract only PC1 per category (n_components=1)
# 2. Add imputation before PCA (dropna(how='all') + ffill + median)
# 3. Add diagnostic logging for sample size validation
#
# Usage:
#   python pca.py
#
import sqlite3
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"
FACTORS_CSV = "factors.csv"

def load_factor_mapping():
    df = pd.read_csv(FACTORS_CSV)
    df = df.dropna(subset=['category', 'proxy'])
    df = df[df['category'].astype(str).str.strip() != '']
    df = df[df['proxy'].astype(str).str.strip() != '']
    mapping = {cat: grp['proxy'].tolist() for cat, grp in df.groupby('category')}
    return mapping

def load_z_wide():
    con = sqlite3.connect(DB_PATH)
    try:
        sample = pd.read_sql("SELECT * FROM factors_monthly_z LIMIT 1", con)
        if 'index' in sample.columns:
            wide = pd.read_sql("SELECT * FROM factors_monthly_z", con, parse_dates=['index'])
        else:
            wide = pd.read_sql("SELECT * FROM factors_monthly_z", con)
    finally:
        con.close()

    # Normalize date column name to 'date'
    if 'date' not in wide.columns:
        if 'index' in wide.columns:
            wide = wide.rename(columns={'index': 'date'})
        else:
            for c in wide.columns:
                if 'date' in c.lower():
                    wide = wide.rename(columns={c: 'date'})
                    break
    wide['date'] = pd.to_datetime(wide['date'])
    wide = wide.sort_values('date').set_index('date')
    return wide

# ============================================================================
# MODIFIED FUNCTION - This is where the main changes are
# ============================================================================
def run_pca_for_category(cat: str, wide: pd.DataFrame, proxies):
    """
    Run PCA with only PC1 extraction and smart imputation
    
    CHANGES FROM ORIGINAL:
    1. Always extract only n_components=1 (PC1 only)
    2. Use dropna(how='all') instead of dropna(how='any')
    3. Add forward-fill imputation (limit=3 months)
    4. Add median imputation for remaining NaNs
    5. Add diagnostic logging
    """
    cols = [p for p in proxies if p in wide.columns]
    if len(cols) == 0:
        return None, None, None, None, None

    X = wide[cols].copy()
    
    # ========================================================================
    # CHANGE 1: Smart missing data handling (instead of dropna(how='any'))
    # ========================================================================
    
    # Count observations before and after for diagnostics
    total_periods = len(X)
    
    # Drop only rows where ALL factors are missing
    X = X.dropna(how='all')
    after_dropall = len(X)
    
    # Forward-fill short gaps (max 3 months for monthly data)
    X = X.fillna(method='ffill', limit=3)
    
    # Fill remaining NaNs with median (robust to outliers)
    X = X.fillna(X.median())
    
    # Count final observations
    final_periods = len(X)
    
    # Validate minimum sample size
    if X.empty or X.shape[0] < 3 or X.shape[1] < 1:
        return None, None, None, None, None

    # ========================================================================
    # CHANGE 2: Extract only PC1 (instead of multiple PCs with 90% threshold)
    # ========================================================================
    
    # Fit PCA with only 1 component
    pca = PCA(n_components=1, random_state=42)
    scores = pca.fit_transform(X.values)
    
    # Extract PC1
    pc = pd.Series(scores[:, 0], index=X.index, name=f"{cat}_PC1")
    
    # Standardize PC1
    std = pc.std()
    pc = (pc - pc.mean()) / (std if std and not np.isnan(std) and std != 0 else 1.0)
    
    # Store in dictionary format (keeping same structure as original)
    pc_dict = {"PC1": pc}
    
    # Loadings for PC1
    loadings = pd.Series(pca.components_[0], index=cols, name='loading_PC1')
    loadings_dict = {"PC1": loadings}
    
    # Variance explained by PC1
    exp_var = pca.explained_variance_ratio_[0]
    cumulative_var = exp_var  # Since we only have PC1, cumulative = PC1
    n_components = 1
    
    # ========================================================================
    # CHANGE 3: Diagnostic logging
    # ========================================================================
    obs_per_feature = final_periods / len(cols) if len(cols) > 0 else 0
    print(f"  Observations: {total_periods} → {after_dropall} (drop all NaN) → {final_periods} (after imputation)")
    print(f"  Ratio: {obs_per_feature:.1f}x observations per feature")
    print(f"  PC1 explains: {exp_var*100:.1f}% of variance")
    
    return pc_dict, loadings_dict, [exp_var], cumulative_var, n_components

# ============================================================================
# No changes needed below this line - save_outputs works with the new structure
# ============================================================================

def save_outputs(results_dict):
    all_series = []
    loadings_rows = []
    meta_rows = []
    
    for cat, result in results_dict.items():
        pc_dict, loadings_dict, exp_var, cum_var, n_comp = result
        
        # Collect all PC series
        for pc_name, series in pc_dict.items():
            series.name = f"{cat}_{pc_name}"
            all_series.append(series)
        
        # Collect loadings
        for pc_name, loadings in loadings_dict.items():
            for proxy, val in loadings.items():
                loadings_rows.append({
                    'category': cat, 
                    'pc': pc_name, 
                    'proxy': proxy, 
                    'loading': float(val)
                })
        
        # Collect metadata
        for i, ev in enumerate(exp_var):
            meta_rows.append({
                'category': cat,
                'pc': i+1,
                'explained_variance_ratio': float(ev),
                'cumulative_variance': float(np.sum(exp_var[:i+1]))
            })
    
    if not all_series:
        print("No PCA outputs to save.")
        return
    
    # Build wide and long tables
    wide = pd.concat(all_series, axis=1).sort_index()
    long = wide.reset_index().melt(id_vars=['date'], var_name='category_pc', value_name='value')
    long = long.dropna(subset=['value'])
    long['category'] = long['category_pc'].str.extract(r'(.+)_PC\d+')[0]
    long['pc'] = long['category_pc'].str.extract(r'PC(\d+)')[0].astype(int)
    
    loadings_df = pd.DataFrame(loadings_rows)
    meta_df = pd.DataFrame(meta_rows)
    
    con = sqlite3.connect(DB_PATH)
    try:
        long[['date','category','pc','value']].to_sql('pca_factors', con, if_exists='replace', index=False)
        wide.reset_index().to_sql('pca_factors_wide', con, if_exists='replace', index=False)
        meta_df.to_sql('pca_meta', con, if_exists='replace', index=False)
        loadings_df.to_sql('pca_loadings', con, if_exists='replace', index=False)
    finally:
        con.close()
    
    print("\nSaved tables: pca_factors, pca_factors_wide, pca_meta, pca_loadings")
    print(f"PCA factors: {wide.shape[1]} total PCs across categories")
    print(f"Date range: {wide.index.min().date()} to {wide.index.max().date()}")
    print(f"Total observations in wide format: {len(wide)}")

def main():
    print("="*80)
    print("PCA ANALYSIS - MODIFIED VERSION")
    print("Changes: PC1-only extraction + Smart imputation")
    print("="*80)
    
    print("\nLoading factor mapping...")
    mapping = load_factor_mapping()
    print(f"Found {len(mapping)} categories.")

    print("\nLoading z-scored wide data from SQLite...")
    wide = load_z_wide()
    print(f"Wide shape: {wide.shape}, dates: {wide.index.min().date()} to {wide.index.max().date()}")

    # ========================================================================
    # CHANGE 4: Add pre-PCA diagnostic summary
    # ========================================================================
    print("\n" + "="*80)
    print("SAMPLE SIZE VALIDATION")
    print("="*80)
    print(f"{'Category':<30} {'Features':<10} {'Complete':<12} {'After Impute':<15} {'Ratio':<10}")
    print("-"*80)
    
    for cat in sorted(mapping.keys()):
        proxies = mapping[cat]
        cols = [p for p in proxies if p in wide.columns]
        if len(cols) == 0:
            continue
            
        X_complete = wide[cols].dropna(how='any')
        X_imputed = wide[cols].dropna(how='all')
        X_imputed = X_imputed.fillna(method='ffill', limit=3).fillna(X_imputed.median())
        
        ratio = len(X_imputed) / len(cols) if len(cols) > 0 else 0
        
        print(f"{cat:<30} {len(cols):<10} {len(X_complete):<12} {len(X_imputed):<15} {ratio:>8.1f}x")
    
    print("="*80)

    # Run PCA for each category
    print("\n" + "="*80)
    print("RUNNING PCA (PC1 ONLY)")
    print("="*80)
    
    results_dict = {}

    for cat, proxies in sorted(mapping.items()):
        print(f"\nProcessing: {cat} ({len(proxies)} proxies)")
        result = run_pca_for_category(cat, wide, proxies)
        
        if result[0] is None:
            print(f"  ✗ Skipped (insufficient data)")
            continue
        
        results_dict[cat] = result
        print(f"  ✓ PC1 saved")

    # Save all results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    save_outputs(results_dict)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successfully processed {len(results_dict)} categories")
    print(f"Each category contributed 1 PC (PC1 only)")
    print(f"Total PCs for GMM: {len(results_dict)}")
    print("\nNext step: Run gmm.py with these PC1 features")

if __name__ == "__main__":
    main()