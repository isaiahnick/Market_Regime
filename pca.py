# pca.py
# Compute a single PCA proxy (PC1) per category and store outputs to SQLite.
# Additionally:
#   * Run a GLOBAL PCA on the integrated z-scored table (factors_monthly_z)
#     and save ONE scree plot + ONE loadings heatmap for the whole table.
#   * (Optional) per-category scree/loadings plots can be toggled on if desired.
#
# INPUT  : SQLite table factors_monthly_z (wide, z-scored), CSV factors.csv (category, proxy)
# OUTPUT : pca_factors (long), pca_factors_wide (wide), pca_meta (stats), pca_loadings (PC1 loadings)
# FILES  : ./plots_pca/scree_GLOBAL.png, ./plots_pca/heatmap_GLOBAL.png
#          (per-category plots only if toggled on)
#
# Usage:
#   python pca.py

import os
import sqlite3
import pandas as pd
import numpy as np

# Headless backend for figure saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# --- Paths ---
DB_PATH = "./factor_lens.db"
FACTORS_CSV = "factors.csv"

# --- Global plotting config (INTEGRATED Z-SCORE) ---
PLOT_GLOBAL = True                 # draw GLOBAL scree + heatmap based on the whole table
GLOBAL_MIN_COVER_RATIO = 0.80      # keep a row if >=80% features present
GLOBAL_IMPUTE = "median"           # 'median' | 'ffill_bfill'
GLOBAL_HEATMAP_MAX_K = 10          # show at most K PCs in global heatmap
GLOBAL_HEATMAP_MAX_FEATURES = 80   # show top features by max |loading| (to keep the figure readable)

# --- Per-category plotting (usually OFF since you asked for integrated plots) ---
PLOT_SCREES_PER_CAT   = False
PLOT_HEATMAPS_PER_CAT = False

# --- Common plotting/config ---
EVR_CUTOFF = 0.90
PLOT_DIR   = "./plots_pca"
HEATMAP_ANNOTATE_TOP = 3
HEATMAP_CMAP = "viridis"


# -------------------- IO helpers --------------------
def load_factor_mapping():
    df = pd.read_csv(FACTORS_CSV)
    df = df.dropna(subset=["category", "proxy"])
    df = df[df["category"].astype(str).str.strip() != ""]
    df = df[df["proxy"].astype(str).str.strip() != ""]
    return {cat: g["proxy"].tolist() for cat, g in df.groupby("category")}

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
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


# -------------------- Plotting --------------------
def plot_scree(evr: np.ndarray, n_keep: int, cat: str, cutoff: float = EVR_CUTOFF, out_dir: str = PLOT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ks = np.arange(1, len(evr) + 1)
    cum = np.cumsum(evr)

    # Elbow: max distance to line from (1,cum1) to (n,cumN)
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

    path = os.path.join(out_dir, f"scree_{_safe_filename(cat)}.png")
    plt.savefig(path, dpi=160); plt.close()
    return path

def plot_loadings_heatmap(components: np.ndarray, feature_names: list[str], evr: np.ndarray, cat: str,
                          out_dir: str = PLOT_DIR, max_k: int = 5, max_features: int = 40,
                          topn_annot: int = HEATMAP_ANNOTATE_TOP, cmap: str = HEATMAP_CMAP) -> str:
    os.makedirs(out_dir, exist_ok=True)
    n_comp, n_feat = components.shape
    k = int(min(max_k, n_comp))
    L = components[:k, :].T  # (features × PCs)

    # pick top features by max |loading| over shown PCs
    if n_feat > max_features:
        maxabs = np.max(np.abs(L), axis=1)
        idx = np.argsort(-maxabs)[:max_features]
        L = L[idx, :]; feature_names = [feature_names[i] for i in idx]

    order = np.argsort(-np.max(np.abs(L), axis=1))
    L = L[order, :]; feature_names = [feature_names[i] for i in order]

    h = max(4.0, 0.35 * L.shape[0]); w = max(6.0, 1.1 * k)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(L, aspect="auto", cmap=cmap)

    ax.set_title(f"PC Loadings – {cat}")
    ax.set_xlabel("PCs"); ax.set_ylabel("Features")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([f"PC{i+1}\n({evr[i]*100:.0f}%)" for i in range(k)])
    ax.set_yticks(np.arange(L.shape[0])); ax.set_yticklabels(feature_names)

    ax.set_xticks(np.arange(-.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-.5, L.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.tick_params(axis="x", which="both", labelrotation=0)

    if topn_annot and topn_annot > 0:
        for j in range(k):
            col = L[:, j]
            top_idx = np.argsort(-np.abs(col))[:min(topn_annot, len(col))]
            for i in top_idx:
                val = col[i]
                color = "white" if abs(val) > 0.8*np.max(np.abs(L)) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("Loading")
    plt.tight_layout()
    path = os.path.join(out_dir, f"heatmap_{_safe_filename(cat)}.png")
    plt.savefig(path, dpi=160); plt.close()
    return path


# -------------------- GLOBAL PCA (integrated z-score) --------------------
def run_global_pca_and_plots(wide: pd.DataFrame):
    # Use ALL numeric columns from the integrated z-scored table
    all_cols = [c for c in wide.columns if wide[c].dtype.kind in "fc"]
    X = wide[all_cols].copy()

    # coverage filter by row; then impute remaining gaps for columns
    min_cov = int(np.ceil(GLOBAL_MIN_COVER_RATIO * len(all_cols)))
    X = X.dropna(thresh=min_cov)
    if GLOBAL_IMPUTE == "ffill_bfill":
        X = X.sort_index().ffill().bfill()
    else:
        X = X.fillna(X.median())

    if X.shape[0] < 3 or X.shape[1] < 2:
        print("[GLOBAL] Not enough data for PCA after coverage/imputation.")
        return

    n_full = int(min(X.shape[1], max(1, X.shape[0] - 1)))
    pca_full = PCA(n_components=n_full, svd_solver="full")
    _ = pca_full.fit_transform(X.values)
    evr_full = pca_full.explained_variance_ratio_

    n_keep = int(np.searchsorted(np.cumsum(evr_full), EVR_CUTOFF) + 1)
    n_keep = int(np.clip(n_keep, 1, n_full))

    # Plots for the integrated table
    plot_scree(evr_full, n_keep, cat="GLOBAL", cutoff=EVR_CUTOFF, out_dir=PLOT_DIR)
    plot_loadings_heatmap(
        components=pca_full.components_,
        feature_names=all_cols,
        evr=evr_full,
        cat="GLOBAL",
        out_dir=PLOT_DIR,
        max_k=GLOBAL_HEATMAP_MAX_K,
        max_features=GLOBAL_HEATMAP_MAX_FEATURES,
        topn_annot=HEATMAP_ANNOTATE_TOP,
        cmap=HEATMAP_CMAP,
    )
    print(f"[GLOBAL] PCA for plots only: rows={X.shape[0]}, features={X.shape[1]}, k*={n_keep}")


# -------------------- Per-category PCA (DB outputs stay PC1) --------------------
def run_pca_for_category(cat: str, wide: pd.DataFrame, proxies):
    cols = [p for p in proxies if p in wide.columns]
    if len(cols) == 0:
        return None, None, None

    X = wide[cols].copy()
    X = X.dropna(how="all").dropna(how="any")
    if X.empty or X.shape[0] < 3 or X.shape[1] < 1:
        return None, None, None

    n_full = int(min(X.shape[1], max(1, X.shape[0] - 1)))
    pca_full = PCA(n_components=n_full, svd_solver="full")
    scores_full = pca_full.fit_transform(X.values)
    evr_full = pca_full.explained_variance_ratio_

    n_keep = int(np.searchsorted(np.cumsum(evr_full), EVR_CUTOFF) + 1)
    n_keep = int(np.clip(n_keep, 1, n_full))

    if PLOT_SCREES_PER_CAT:
        try: plot_scree(evr_full, n_keep, cat, cutoff=EVR_CUTOFF, out_dir=PLOT_DIR)
        except Exception: pass
    if PLOT_HEATMAPS_PER_CAT:
        try:
            plot_loadings_heatmap(pca_full.components_, cols, evr_full, cat,
                                  out_dir=PLOT_DIR, max_k=5, max_features=40,
                                  topn_annot=HEATMAP_ANNOTATE_TOP, cmap=HEATMAP_CMAP)
        except Exception: pass

    pc1 = pd.Series(scores_full[:, 0], index=X.index, name=f"{cat}_PC1")
    sd = pc1.std()
    pc1 = (pc1 - pc1.mean()) / (sd if sd and not np.isnan(sd) and sd != 0 else 1.0)

    loadings = pd.Series(pca_full.components_[0], index=cols, name="loading")
    exp_var = float(evr_full[0])
    return pc1, loadings, exp_var


# -------------------- Save DB outputs --------------------
def save_outputs(pc1_dict, loadings_dict, expvar_dict):
    all_series = []
    for cat, s in pc1_dict.items():
        ser = s.copy(); ser.name = f"{cat}_PC1"; all_series.append(ser)

    if not all_series:
        print("No PCA outputs to save."); return

    wide = pd.concat(all_series, axis=1).sort_index()
    long = wide.reset_index().melt(id_vars=["date"], var_name="category_pc", value_name="value")
    long["category"] = long["category_pc"].str.replace("_PC1$", "", regex=True)

    rows = []
    for cat, ld in loadings_dict.items():
        if ld is None: continue
        for proxy, val in ld.items():
            rows.append({"category": cat, "proxy": proxy, "loading": float(val)})
    loadings_df = pd.DataFrame(rows)

    meta = pd.DataFrame(
        [{"category": cat, "pc": 1, "explained_variance_ratio": float(ev)}
         for cat, ev in expvar_dict.items() if ev is not None]
    )

    con = sqlite3.connect(DB_PATH)
    try:
        long[["date","category","value"]].to_sql("pca_factors", con, if_exists="replace", index=False)
        wide.reset_index().rename(columns={"index":"date"}).to_sql("pca_factors_wide", con, if_exists="replace", index=False)
        meta.to_sql("pca_meta", con, if_exists="replace", index=False)
        if not loadings_df.empty:
            loadings_df.to_sql("pca_loadings", con, if_exists="replace", index=False)
    finally:
        con.close()

    print("Saved tables: pca_factors, pca_factors_wide, pca_meta, pca_loadings")
    print(f"PCA factors: {wide.shape[1]} categories, date range: {wide.index.min().date()} to {wide.index.max().date()}, obs={wide.shape[0]}")


# -------------------- Main --------------------
def main():
    print("Loading factor mapping...")
    mapping = load_factor_mapping()
    print(f"Found {len(mapping)} categories.")

    print("Loading z-scored wide data from SQLite...")
    wide = load_z_wide()
    print(f"Wide shape: {wide.shape}, dates: {wide.index.min().date()} to {wide.index.max().date()}")

    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1) GLOBAL plots on the integrated z-score table
    if PLOT_GLOBAL:
        run_global_pca_and_plots(wide)

    # 2) Per-category PC1 (unchanged DB outputs)
    pc1_dict, loadings_dict, expvar_dict = {}, {}, {}
    for cat, proxies in mapping.items():
        available = sum(p in wide.columns for p in proxies)
        print(f"Running PCA for category: {cat} ({len(proxies)} proxies; available: {available})")
        pc1, loads, ev = run_pca_for_category(cat, wide, proxies)
        if pc1 is None:
            print("  ✗ Skipped (insufficient data)"); continue
        pc1_dict[cat] = pc1; loadings_dict[cat] = loads; expvar_dict[cat] = ev
        print(f"  ✓ PC1 saved, EVR={ev:.3f}, length={pc1.shape[0]}")

    save_outputs(pc1_dict, loadings_dict, expvar_dict)


if __name__ == "__main__":
    main()