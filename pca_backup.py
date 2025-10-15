# pca.py
# Compute a single PCA proxy (PC1) per category and store outputs to SQLite.
# Additionally: for each category, run FULL PCA and save a scree plot (EVR & cumulative EVR)
# showing how k is chosen (k* by EVR cutoff + an elbow heuristic).
#
# Tables used/created:
#   - INPUT:  factors_monthly_z (wide, z-scored), factors.csv (for category -> proxy mapping)
#   - OUTPUT: pca_factors (long), pca_factors_wide (wide), pca_meta (per-category stats), pca_loadings (per-category loadings)
#   - FILES : ./plots_pca/scree_<category>.png (one PNG per category)
#
# Usage:
#   python pca.py
#

import os
import sqlite3
import pandas as pd
import numpy as np

# Headless-friendly backend for saving figures on macOS/Linux servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# --- Paths ---
# DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"
DB_PATH = "./factor_lens.db"
FACTORS_CSV = "factors.csv"

# --- Scree plot & selection config ---
EVR_CUTOFF = 0.90           # choose k* = min k with cumulative EVR >= EVR_CUTOFF
PLOT_SCREES = True          # set False to disable plotting
PLOT_DIR = "./plots_pca"    # where scree PNGs will be saved


def load_factor_mapping():
    df = pd.read_csv(FACTORS_CSV)
    df = df.dropna(subset=["category", "proxy"])
    df = df[df["category"].astype(str).str.strip() != ""]
    df = df[df["proxy"].astype(str).str.strip() != ""]
    mapping = {cat: grp["proxy"].tolist() for cat, grp in df.groupby("category")}
    return mapping


def load_z_wide():
    con = sqlite3.connect(DB_PATH)
    try:
        sample = pd.read_sql("SELECT * FROM factors_monthly_z LIMIT 1", con)
        if "index" in sample.columns:
            wide = pd.read_sql("SELECT * FROM factors_monthly_z", con, parse_dates=["index"])
        else:
            wide = pd.read_sql("SELECT * FROM factors_monthly_z", con)
    finally:
        con.close()

    # Normalize date column name to 'date'
    if "date" not in wide.columns:
        if "index" in wide.columns:
            wide = wide.rename(columns={"index": "date"})
        else:
            for c in wide.columns:
                if "date" in c.lower():
                    wide = wide.rename(columns={c: "date"})
                    break
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.sort_values("date").set_index("date")
    return wide


def _safe_filename(s: str) -> str:
    """Make a string safe to use in filenames."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def plot_scree(evr: np.ndarray, n_keep: int, cat: str,
               cutoff: float = EVR_CUTOFF, out_dir: str = PLOT_DIR) -> str:
    """
    Create and save a scree plot for a category.

    X-axis shows k = number of PCA components (full, not truncated).
    We annotate:
      - Cumulative EVR 0.90 and 0.95 horizontal lines
      - k* = argmin k s.t. cumEVR >= cutoff (vertical line, orange)
      - k_elbow by a simple max-distance-to-line heuristic (vertical line, blue)
    """
    os.makedirs(out_dir, exist_ok=True)
    ks = np.arange(1, len(evr) + 1)
    cum = np.cumsum(evr)

    # Elbow via max distance to the straight line from (1,cum1) to (n,cumN)
    x1, y1 = 1.0, cum[0]
    x2, y2 = float(len(evr)), cum[-1]
    dx, dy = x2 - x1, y2 - y1
    den = np.hypot(dx, dy) if (dx != 0 or dy != 0) else 1.0
    dist = np.abs(dy * (ks - x1) - dx * (cum - y1)) / den
    k_elbow = int(ks[np.argmax(dist)])

    plt.figure(figsize=(6.6, 4.2))
    plt.plot(ks, evr, marker="o", label="Explained variance ratio")
    plt.plot(ks, cum, marker="s", linestyle="--", label="Cumulative explained variance")
    plt.axhline(0.90, linestyle=":", linewidth=1)
    plt.axhline(0.95, linestyle=":", linewidth=1)
    plt.axvline(n_keep, linestyle=":", linewidth=1, color="tab:orange")
    if k_elbow != n_keep:
        plt.axvline(k_elbow, linestyle=":", linewidth=1, color="tab:blue")

    plt.title(f"Scree Plot - {cat}")
    plt.xlabel("k (number of components)")
    plt.ylabel("Explained variance")
    plt.xticks(ks)
    plt.xlim(1, len(evr))
    plt.ylim(0, 1.02)
    plt.legend()
    plt.text(n_keep + 0.1, 0.98, f"k*={n_keep} (EVR≥{cutoff:.2f})", va="top")
    if k_elbow != n_keep:
        plt.text(k_elbow + 0.1, 0.93, f"k_elbow={k_elbow}", va="top")
    plt.tight_layout()

    fname = f"scree_{_safe_filename(cat)}.png"
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def run_pca_for_category(cat: str, wide: pd.DataFrame, proxies):
    """
    Keep DB outputs backward-compatible (PC1 only),
    but compute FULL PCA first to (a) get EVR for scree, (b) derive PC1.
    """
    cols = [p for p in proxies if p in wide.columns]
    if len(cols) == 0:
        return None, None, None

    X = wide[cols].copy()

    # Strict missing policy as in original: drop any NaN row (within this category)
    X = X.dropna(how="all")
    X = X.dropna(how="any")
    if X.empty or X.shape[0] < 3 or X.shape[1] < 1:
        return None, None, None

    # FULL PCA: up to min(p, T-1)
    n_full = int(min(X.shape[1], max(1, X.shape[0] - 1)))
    pca_full = PCA(n_components=n_full, svd_solver="full")
    scores_full = pca_full.fit_transform(X.values)          # shape (T, n_full)
    evr_full = pca_full.explained_variance_ratio_

    # Choose k* by cumulative EVR threshold (for reporting/plotting only)
    n_keep = int(np.searchsorted(np.cumsum(evr_full), EVR_CUTOFF) + 1)
    n_keep = int(np.clip(n_keep, 1, n_full))

    # Plot scree
    if PLOT_SCREES:
        try:
            _ = plot_scree(evr_full, n_keep, cat, cutoff=EVR_CUTOFF, out_dir=PLOT_DIR)
        except Exception:
            # Plotting should never break the pipeline
            pass

    # Build PC1 series (standardized), loadings, and EVR for PC1
    pc1 = pd.Series(scores_full[:, 0], index=X.index, name=f"{cat}_PC1")
    sd = pc1.std()
    pc1 = (pc1 - pc1.mean()) / (sd if sd and not np.isnan(sd) and sd != 0 else 1.0)

    loadings = pd.Series(pca_full.components_[0], index=cols, name="loading")
    exp_var = float(evr_full[0])

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
    long = wide.reset_index().melt(id_vars=["date"], var_name="category_pc", value_name="value")
    long["category"] = long["category_pc"].str.replace("_PC1$", "", regex=True)

    # Loadings table
    rows = []
    for cat, ld in loadings_dict.items():
        if ld is None:
            continue
        for proxy, val in ld.items():
            rows.append({"category": cat, "proxy": proxy, "loading": float(val)})
    loadings_df = pd.DataFrame(rows)

    # Meta (explained variance)
    meta = pd.DataFrame(
        [{"category": cat, "pc": 1, "explained_variance_ratio": float(ev)}
         for cat, ev in expvar_dict.items() if ev is not None]
    )

    con = sqlite3.connect(DB_PATH)
    try:
        long[["date", "category", "value"]].to_sql("pca_factors", con, if_exists="replace", index=False)
        wide.reset_index().rename(columns={"index": "date"}).to_sql("pca_factors_wide", con, if_exists="replace", index=False)
        meta.to_sql("pca_meta", con, if_exists="replace", index=False)
        if not loadings_df.empty:
            loadings_df.to_sql("pca_loadings", con, if_exists="replace", index=False)
    finally:
        con.close()

    print("Saved tables: pca_factors, pca_factors_wide, pca_meta, pca_loadings")
    print(
        f"PCA factors: {wide.shape[1]} categories, date range: {wide.index.min().date()} "
        f"to {wide.index.max().date()}, obs={wide.shape[0]}"
    )


def main():
    print("Loading factor mapping...")
    mapping = load_factor_mapping()
    print(f"Found {len(mapping)} categories.")

    print("Loading z-scored wide data from SQLite...")
    wide = load_z_wide()
    print(f"Wide shape: {wide.shape}, dates: {wide.index.min().date()} to {wide.index.max().date()}")

    if PLOT_SCREES:
        os.makedirs(PLOT_DIR, exist_ok=True)

    pc1_dict = {}
    loadings_dict = {}
    expvar_dict = {}

    for cat, proxies in mapping.items():
        available = sum(p in wide.columns for p in proxies)
        print(f"Running PCA for category: {cat} ({len(proxies)} proxies; available: {available})")
        pc1, loads, ev = run_pca_for_category(cat, wide, proxies)
        if pc1 is None:
            print("  ✗ Skipped (insufficient data)")
            continue
        pc1_dict[cat] = pc1
        loadings_dict[cat] = loads
        expvar_dict[cat] = ev
        print(f"  ✓ PC1 saved, EVR={ev:.3f}, length={pc1.shape[0]}")

    save_outputs(pc1_dict, loadings_dict, expvar_dict)


if __name__ == "__main__":
    main()