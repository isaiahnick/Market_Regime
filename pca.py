
# pca.py
# Compute a single PCA proxy (PC1) per category and store outputs to SQLite.
# Tables used/created:
#   - INPUT:  factors_monthly_z (wide, z-scored), factors.csv (for category -> proxy mapping)
#   - OUTPUT: pca_factors (long), pca_factors_wide (wide), pca_meta (per-category stats), pca_loadings (per-category loadings)
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

def run_pca_for_category(cat: str, wide: pd.DataFrame, proxies):
    cols = [p for p in proxies if p in wide.columns]
    if len(cols) == 0:
        return None, None, None

    X = wide[cols].copy()

    # Drop rows where all category columns are NaN; then drop remaining partial NaNs inside this category only.
    X = X.dropna(how='all')
    X = X.dropna(how='any')
    if X.empty or X.shape[0] < 3 or X.shape[1] < 1:
        return None, None, None

    pca = PCA(n_components=1, random_state=42)
    scores = pca.fit_transform(X.values)  # shape (T, 1)
    pc1 = pd.Series(scores[:, 0], index=X.index, name=f"{cat}_PC1")

    # Standardize PC1 to mean 0, std 1 for comparability
    std = pc1.std()
    pc1 = (pc1 - pc1.mean()) / (std if std and not np.isnan(std) and std != 0 else 1.0)

    # Loadings for PC1 mapped to proxies
    loadings = pd.Series(pca.components_[0], index=cols, name='loading')
    exp_var = float(pca.explained_variance_ratio_[0])

    return pc1, loadings, exp_var

def save_outputs(pc1_dict, loadings_dict, expvar_dict):
    # Build long and wide output tables
    all_series = []
    for cat, s in pc1_dict.items():
        ser = s.copy()
        ser.name = f"{cat}_PC1"
        all_series.append(ser)

    if not all_series:
        print("No PCA outputs to save.")
        return

    wide = pd.concat(all_series, axis=1).sort_index()
    long = wide.reset_index().melt(id_vars=['date'], var_name='category_pc', value_name='value')
    long['category'] = long['category_pc'].str.replace('_PC1$', '', regex=True)

    # Loadings table
    rows = []
    for cat, ld in loadings_dict.items():
        if ld is None: 
            continue
        for proxy, val in ld.items():
            rows.append({'category': cat, 'proxy': proxy, 'loading': float(val)})
    loadings_df = pd.DataFrame(rows)

    # Meta (explained variance)
    meta = pd.DataFrame([{'category': cat, 'pc': 1, 'explained_variance_ratio': float(ev)} 
                         for cat, ev in expvar_dict.items() if ev is not None])

    con = sqlite3.connect(DB_PATH)
    try:
        long[['date','category','value']].to_sql('pca_factors', con, if_exists='replace', index=False)
        wide.reset_index().rename(columns={'index': 'date'}).to_sql('pca_factors_wide', con, if_exists='replace', index=False)
        meta.to_sql('pca_meta', con, if_exists='replace', index=False)
        if not loadings_df.empty:
            loadings_df.to_sql('pca_loadings', con, if_exists='replace', index=False)
    finally:
        con.close()

    print("Saved tables: pca_factors, pca_factors_wide, pca_meta, pca_loadings")
    print(f"PCA factors: {wide.shape[1]} categories, date range: {wide.index.min().date()} to {wide.index.max().date()}, obs={wide.shape[0]}")

def main():
    print("Loading factor mapping...")
    mapping = load_factor_mapping()
    print(f"Found {len(mapping)} categories.")

    print("Loading z-scored wide data from SQLite...")
    wide = load_z_wide()
    print(f"Wide shape: {wide.shape}, dates: {wide.index.min().date()} to {wide.index.max().date()}")

    pc1_dict = {}
    loadings_dict = {}
    expvar_dict = {}

    for cat, proxies in mapping.items():
        print(f"Running PCA for category: {cat} ({len(proxies)} proxies; available: {sum(p in wide.columns for p in proxies)})")
        pc1, loads, ev = run_pca_for_category(cat, wide, proxies)
        if pc1 is None:
            print(f"  ✗ Skipped (insufficient data)")
            continue
        pc1_dict[cat] = pc1
        loadings_dict[cat] = loads
        expvar_dict[cat] = ev
        print(f"  ✓ PC1 saved, EVR={ev:.3f}, length={pc1.shape[0]}")

    save_outputs(pc1_dict, loadings_dict, expvar_dict)

if __name__ == "__main__":
    main()
