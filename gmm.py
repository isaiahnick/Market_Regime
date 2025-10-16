
# gmm.py
# Fit Gaussian Mixture Models on the 1st PC per category produced by pca.py
# Allows flexible start date and missing-value imputation so we don't require full coverage.
#
# Usage:
#   python gmm.py --start 1970-01-01 --kmin 2 --kmax 6 --cov full --impute median --standardize
#
import argparse
import json
import sqlite3
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"

def load_pca_wide():
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM pca_factors_wide", con)
    finally:
        con.close()
    # Normalize date col
    if 'date' not in df.columns:
        for c in df.columns:
            if 'date' in c.lower():
                df = df.rename(columns={c: 'date'})
                break
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    # Use only numeric columns
    feat_cols = [c for c in df.columns if c != 'date']
    return df[feat_cols]

@dataclass
class GMMDiagnostics:
    k: int
    aic: float
    bic: float
    silhouette: float

def fit_gmm_grid(X, kmin=2, kmax=6, covariance_type='full', random_state=42, n_init=10, max_iter=1000):
    results = []
    best_bic = np.inf
    best = None

    for k in range(kmin, kmax + 1):
        gm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=random_state, n_init=n_init, max_iter=max_iter)
        gm.fit(X)
        labels = gm.predict(X)
        aic = gm.aic(X)
        bic = gm.bic(X)
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = np.nan

        results.append(GMMDiagnostics(k, aic, bic, sil))
        if bic < best_bic:
            best_bic = bic
            best = (k, gm)

    return results, best

def impute_missing(X: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == 'none':
        return X
    if strategy in ('mean', 'median', 'most_frequent'):
        imp = SimpleImputer(strategy=strategy)
        X_imputed = imp.fit_transform(X.values)
        return pd.DataFrame(X_imputed, index=X.index, columns=X.columns)
    if strategy == 'ffill':
        return X.sort_index().ffill()
    if strategy == 'bfill':
        return X.sort_index().bfill()
    if strategy in ('ffill_bfill', 'ffbb'):
        return X.sort_index().ffill().bfill()
    # default to median
    imp = SimpleImputer(strategy='median')
    X_imputed = imp.fit_transform(X.values)
    return pd.DataFrame(X_imputed, index=X.index, columns=X.columns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=str, default='1995-01-01', help='Start date (inclusive), e.g., 1970-01-01')
    ap.add_argument('--kmin', type=int, default=2)
    ap.add_argument('--kmax', type=int, default=6)
    ap.add_argument('--cov', type=str, default='full', choices=['full', 'tied', 'diag', 'spherical'])
    ap.add_argument('--impute', type=str, default='median', choices=['none','mean','median','most_frequent','ffill','bfill','ffill_bfill','ffbb'])
    ap.add_argument('--standardize', action='store_true', help='Z-score the PCA features before GMM (recommended).')
    args = ap.parse_args()

    print("Loading PCA-wide data...")
    df = load_pca_wide()

    # Filter to start date (avoid truncating to last-common date)
    start = pd.to_datetime(args.start)
    if start is not None:
        df = df[df.index >= start]

    # Choose features (all numeric columns saved by PCA)
    feat_cols = [c for c in df.columns]  # keep all PC1s
    X = df[feat_cols].copy()

    # Impute missing values so we don't lose early history
    X = impute_missing(X, args.impute)

    # Drop any rows that are still NaN across the board (very rare after imputation)
    X = X.dropna(how='all')
    dates = X.index.to_list()

    # Standardize features (recommended even if PC1 is standardized)
    if args.standardize:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X.values)
    else:
        X_std = X.values

    print(f"Fitting GMM across K={args.kmin}..{args.kmax} on shape {X_std.shape} with cov='{args.cov}', start={start.date()}, impute='{args.impute}'")
    diagnostics, best = fit_gmm_grid(X_std, kmin=args.kmin, kmax=args.kmax, covariance_type=args.cov)

    # Save outputs
    k_star, gm = best
    # Posterior probabilities and labels
    probs = gm.predict_proba(X_std)  # shape (T, k_star)
    labels = probs.argmax(axis=1)

    # Build regimes table
    regimes = pd.DataFrame({'date': dates, 'regime': labels})
    for j in range(k_star):
        regimes[f'prob_{j}'] = probs[:, j]

    # Diagnostics table
    scores = pd.DataFrame([{'k': d.k, 'aic': d.aic, 'bic': d.bic, 'silhouette': d.silhouette} for d in diagnostics])

    # Meta / params
    meta = pd.DataFrame([{'chosen_k': k_star, 'covariance_type': gm.covariance_type, 'n_features': len(feat_cols), 'start_date': str(start.date()), 'impute': args.impute}])
    params = {
        'weights': gm.weights_.tolist(),
        'means': gm.means_.tolist(),
        'covariances': gm.covariances_.tolist(),
        'features': list(feat_cols),
    }
    params_df = pd.DataFrame([{'chosen_k': k_star, 'params_json': json.dumps(params)}])

    con = sqlite3.connect(DB_PATH)
    try:
        regimes.to_sql('gmm_regimes', con, if_exists='replace', index=False)
        scores.to_sql('gmm_scores', con, if_exists='replace', index=False)
        meta.to_sql('gmm_meta', con, if_exists='replace', index=False)
        params_df.to_sql('gmm_params', con, if_exists='replace', index=False)
    finally:
        con.close()

    print("Saved tables: gmm_regimes, gmm_scores, gmm_meta, gmm_params")
    print(f"Chosen K={k_star}; date range: {pd.to_datetime(dates).min().date()} to {pd.to_datetime(dates).max().date()}; obs={len(dates)}")

    # Print a quick summary
    print("\nModel selection diagnostics:")
    for d in diagnostics:
        print(f"  K={d.k} | AIC={d.aic:.1f} | BIC={d.bic:.1f} | Sil={d.silhouette:.3f}")

if __name__ == "__main__":
    main()
