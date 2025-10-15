import os
import sqlite3
import textwrap
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA


# ----------------------------- Config ----------------------------------------

DB_PATH = "./factor_lens.db"            # <- change if needed
TABLE = "factors_monthly_z"             # z-scored wide table
PLOT_DIR = "./plots_pca"

# PCA / selection
K_PC = 10                               # use first K PCs
PER_PC = 1                              # pick up to PER_PC features per PC (|loading|)
ROW_COVER = 0.80                        # keep row if >= 80% columns present
IMPUTE_METHOD = "median"                # 'median' | 'ffill_bfill'
DEDUP_CORR_CUT = 0.90                   # drop candidate if corr > cut with already chosen

# Main figure (exact size for PPT placeholder)
FIG_W, FIG_H, DPI = 11.51, 5.26, 100    # 1151 x 526 px
LEFT_W_FRAC = 0.48                      # width fraction for left text panel
FONT_FAMILY = "Arial"
SMALL, MED, LARGE = 9, 11, 14
LEFT_TEXT_TOP  = 0.8
LEFT_TEXT_STEP = 0.1
LEFT_TITLE     = "Selected features from first 10 PCs"
LEFT_TITLE_Y    = 0.95
SUPTITLE_Y     = 0.97

# Heatmap annotations
ANNOTATE_MODE = 'top_plus_rep'            # 'none' | 'top_per_pc' | 'all'
TOPN = 1                                # used when 'top_per_pc'

# Heatmap x-axis label density (avoid crowding)
X_TICK_LABEL_MAX = 10                    # show at most N labels on x-axis
SHOW_EVR_IN_TICKS = True                # include EVR % under PC label
ROTATE_X_TICKS = 0                      # degree rotation (0/30/45)

# Ranked card (like your sample)
MAKE_RANKED_CARD = True
RANK_MODE = "dominant"                  # 'dominant' (strongest PC per feature) or 'pc'
RANK_PC_INDEX = 0                       # only used when RANK_MODE == 'pc' (0=PC1)
RANK_TOP = 12                           # single-card: show top-N rows by |value|
RANK_FIG_W, RANK_FIG_H, RANK_DPI = 11.51, 5.26, 100
RANK_CMAP = "Reds"

# Multi-page ranked cards (show ALL selected features across pages)
RANK_SHOW_ALL_PAGES = True              # also export paged cards covering ALL selected features
RANK_PAGE_SIZE = 7                      # e.g., 7 per image -> page1: 1-7, page2: 8-15
RANK_ORDER = "pc"                       # 'pc' = keep PC order, 'strength' = sort by |loading|


# --------------------------- Data utilities ----------------------------------

def load_z_wide(db_path: str, table: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", con)
    finally:
        con.close()
    # normalize date col
    if "date" not in df.columns:
        for c in df.columns:
            if c.lower() == "index" or "date" in c.lower():
                df = df.rename(columns={c: "date"})
                break
    df["date"] = pd.to_datetime(df["date"])  # parse
    df = df.sort_values("date").set_index("date")
    return df


def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df[cols]


def preprocess_global(X: pd.DataFrame, row_cover: float, impute: str) -> pd.DataFrame:
    # coverage by row
    thresh = int(np.ceil(row_cover * X.shape[1]))
    X = X.dropna(thresh=thresh)
    # impute
    if impute == "ffill_bfill":
        X = X.sort_index().ffill().bfill()
    else:
        X = X.fillna(X.median())
    return X


# ------------------------ Feature selection ----------------------------------

def pick_sparse_features(components: np.ndarray,
                         feature_names: List[str],
                         X_df: pd.DataFrame,
                         k_pc: int = 15,
                         per_pc: int = 1,
                         corr_cut: float = 0.90
                         ) -> Tuple[List[str], Dict[int, List[Tuple[str, float]]]]:
    """
    Greedy: for each of the first k_pc PCs, take up to per_pc features by
    absolute loading, skipping any that are > corr_cut correlated with already
    chosen ones. Returns (chosen, picked_per_pc[j] -> [(feat, loading), ...]).
    """
    n_comp, _ = components.shape
    k = int(min(k_pc, n_comp))

    # correlation matrix on the imputed X
    corr = X_df[feature_names].corr().abs()

    chosen: List[str] = []
    picked_per_pc: Dict[int, List[Tuple[str, float]]] = {}
    used = set()

    for j in range(k):
        v = components[j, :]  # loadings of PC j
        order = np.argsort(-np.abs(v))  # by |loading|
        cnt = 0
        picked_per_pc[j] = []
        for idx in order:
            fname = feature_names[idx]
            if fname in used:
                continue
            if chosen:
                # corr with any chosen so far
                if corr.loc[fname, chosen].max() >= corr_cut:
                    continue
            chosen.append(fname)
            used.add(fname)
            picked_per_pc[j].append((fname, float(v[idx])))
            cnt += 1
            if cnt >= per_pc:
                break
    return chosen, picked_per_pc


# --------------------------- Plotting helpers --------------------------------

def wrap_label(s: str, width: int = 60) -> str:
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False))


def render_ranked_feature_card(chosen_sorted: List[str],
                               L: np.ndarray,
                               evr: np.ndarray,
                               mode: str = "dominant",
                               pc_index: int = 0,
                               top: int = 12,
                               fig_w: float = 11.51,
                               fig_h: float = 5.26,
                               dpi: int = 100,
                               cmap: str = "Reds",
                               out_path: Optional[str] = None) -> Optional[str]:
    """
    Render a single-column 'feature card' figure:
    left = selected features; right = colored value block with the loading.
    mode='dominant': for each feature, use its strongest PC (by |loading|), and display which PC it is.
    mode='pc'      : use loadings from a specific PC (pc_index).
    """
    n_rows, k = L.shape
    if n_rows == 0 or k == 0:
        return None

    if mode == "pc":
        j = int(np.clip(pc_index, 0, k - 1))
        vals = L[:, j]
        pc_idx = np.full(n_rows, j, dtype=int)
        title = f"Selected features — loadings on PC{j+1} ({evr[j]*100:.0f}%)"
    else:
        pc_idx = np.argmax(np.abs(L), axis=1)
        vals = L[np.arange(n_rows), pc_idx]
        title = "Selected features — dominant loading (PC listed on the right)"

    # Rank rows by absolute value and take top N
    order = np.argsort(-np.abs(vals))[:min(top, n_rows)]
    names = [chosen_sorted[i] for i in order]
    v = vals[order]
    p = pc_idx[order]

    vmax = float(np.max(np.abs(v))) if np.isfinite(np.max(np.abs(v))) and np.max(np.abs(v)) > 0 else 1.0
    cmap_obj = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    colors = [cmap_obj(norm(abs(x))) for x in v]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(names) - 0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=8)

    # layout
    x_name = 0.02
    x_box = 0.64
    x_pc = 0.98

    for i, (nm, val, col, pcj) in enumerate(zip(names, v, colors, p)):
        y = i
        ax.text(x_name, y, nm, ha="left", va="center", fontsize=11)
        # colored rectangle + value
        ax.add_patch(plt.Rectangle((x_box - 0.22, y - 0.34), 0.44, 0.68, color=col, ec="none"))
        ax.text(x_box, y, f"{val:+.2f}", ha="center", va="center", fontsize=11, color="black")
        ax.text(x_pc, y, f"PC{pcj+1}", ha="right", va="center", fontsize=10, color="#333")

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ------------------------------ Main -----------------------------------------

def main():
    # fonts & dirs
    plt.rcParams["font.family"] = FONT_FAMILY
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1) load integrated z-scored table
    wide = load_z_wide(DB_PATH, TABLE)
    X = select_numeric(wide)
    X = preprocess_global(X, ROW_COVER, IMPUTE_METHOD)

    # 2) PCA (full number of comps allowed by data)
    n_full = int(min(X.shape[1], max(1, X.shape[0] - 1)))
    pca = PCA(n_components=n_full, svd_solver="full")
    _ = pca.fit_transform(X.values)
    evr = pca.explained_variance_ratio_
    comps = pca.components_                     # (n_comp, n_feat)
    feat_names = list(X.columns)

    # 3) Select features from first K PCs
    chosen, picked_per_pc = pick_sparse_features(
        components=comps,
        feature_names=feat_names,
        X_df=X,
        k_pc=K_PC,
        per_pc=PER_PC,
        corr_cut=DEDUP_CORR_CUT,
    )

    # 4) Build loadings matrix L for the selected features (features × PCs)
    k = min(K_PC, comps.shape[0])
    if len(chosen) == 0 or k == 0:
        L = np.empty((0, 0))
        chosen_sorted: List[str] = []
    else:
        idx_map = {f: feat_names.index(f) for f in chosen}
        L = np.zeros((len(chosen), k))
        for j in range(k):
            for r, f in enumerate(chosen):
                L[r, j] = comps[j, idx_map[f]]
        # order rows by the PC where |loading| is maximal (nice reading)
        order = np.argsort(-np.max(np.abs(L), axis=1))
        L = L[order, :]
        chosen_sorted = [chosen[i] for i in order]

    # 5) Compose the PPT slide: left bullets + right mini heatmap
    NUDGE_HEATMAP_RIGHT = -0.035  
    CB_PAD   = 0.014             
    CB_WIDTH = 0.014             
    RIGHT_MARGIN_SAFETY = 0.002  
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    left = fig.add_axes([0.06, 0.10, LEFT_W_FRAC, 0.82])
    right = fig.add_axes([0.06 + LEFT_W_FRAC + 0.04, 0.16, 1 - 0.06 - (LEFT_W_FRAC + 0.04) - 0.06, 0.74])
    rp = right.get_position()
    right.set_position([rp.x0 + NUDGE_HEATMAP_RIGHT, rp.y0, rp.width, rp.height])
    HEATMAP_USE_ABS = True               
    HEATMAP_CMAP = LinearSegmentedColormap.from_list(
        "pink_red_brown",
        [("#f0b3b3"), ("#ce0000"), ("#670000")],
        N=256
    )
    HEATMAP_NORM = None

    # left panel
    left.axis("off")
    left.set_title(LEFT_TITLE, fontsize=LARGE, y=LEFT_TITLE_Y, loc="left", pad=0)

    lines = []
    cum = 0.0
    k_used = min(k, len(evr))  
    for j in range(k_used):
        cum += evr[j]
        picks = picked_per_pc.get(j, [])
        parts = [f"{nm} ({val:+.2f})" for (nm, val) in picks]
        if parts:
            line = f"PC{j+1} ({evr[j]*100:.0f}%, cum {cum*100:.0f}%): " + ",  ".join(parts)
        else:
            line = f"PC{j+1} ({evr[j]*100:.0f}%, cum {cum*100:.0f}%)"
        lines.append(line)

    y_top, y_bottom = 0.94, 0.10                  
    step_auto = (y_top - y_bottom) / (len(lines) + 0.2)
    step = min(LEFT_TEXT_STEP, step_auto)         
    y = y_top
    for line in lines:
        wrapped = wrap_label(line, width=64)
        left.text(0.0, y, wrapped, transform=left.transAxes,
                va="top", ha="left", fontsize=MED)
        y -= step

    # right panel: mini heatmap
    if L.size:
        vmax = float(np.nanmax(np.abs(L))) if L.size else 1.0
        data_for_color = np.abs(L) if HEATMAP_USE_ABS else L.clip(min=0)

        im = right.imshow(
            data_for_color,
            aspect="auto",
            cmap=HEATMAP_CMAP,
            vmin=0.0,
            vmax=vmax,
            norm=HEATMAP_NORM
        )

        right.set_title(f"Loadings (selected features × first {k_used} PCs)", fontsize=LARGE, pad=6)
        right.set_xlabel("PCs", fontsize=MED)
        right.set_ylabel("Features", fontsize=MED)

        # ---- x-axis tick density control (avoid crowding) ----
        max_labels = X_TICK_LABEL_MAX
        step = int(np.ceil(k_used / max(1, max_labels)))  # k_used=10, max_labels=10 -> step=1
        tick_idx = np.arange(0, k_used, max(1, step))
        right.set_xticks(tick_idx)
        if SHOW_EVR_IN_TICKS:
            right.set_xticklabels(
                [f"PC{i+1}\n({evr[i]*100:.0f}%)" for i in tick_idx],
                fontsize=SMALL, rotation=ROTATE_X_TICKS
            )
        else:
            right.set_xticklabels([f"PC{i+1}" for i in tick_idx],
                                fontsize=SMALL, rotation=ROTATE_X_TICKS)

        # y ticks (all features)
        right.set_yticks(np.arange(len(chosen_sorted)))
        right.set_yticklabels([wrap_label(f, 28) for f in chosen_sorted], fontsize=SMALL)

        # minor grid
        right.set_xticks(np.arange(-0.5, k_used, 1), minor=True)
        right.set_yticks(np.arange(-0.5, len(chosen_sorted), 1), minor=True)
        right.grid(which="minor", color="w", linestyle=":", linewidth=0.5, alpha=0.5)

        # ---- annotate numbers (sparse) ----
        vmax = np.nanmax(np.abs(L)) if L.size else 1.0

        if ANNOTATE_MODE == 'all':
            for r in range(L.shape[0]):
                for c in range(L.shape[1]):
                    val = L[r, c]
                    color = 'white' if abs(val) > 0.7 * vmax else 'black'
                    right.text(c, r, f"{val:.2f}", ha='center', va='center',
                            fontsize=SMALL, color=color)

        elif ANNOTATE_MODE == 'top_per_pc':
            for c in range(L.shape[1]):
                col = L[:, c]
                r = int(np.nanargmax(np.abs(col)))
                val = col[r]
                color = 'white' if abs(val) > 0.7 * vmax else 'black'
                right.text(c, r, f"{val:.2f}", ha='center', va='center',
                        fontsize=SMALL, color=color)

        elif ANNOTATE_MODE == 'picked_per_pc':
            name_to_row = {nm: i for i, nm in enumerate(chosen_sorted)}
            for c in range(L.shape[1]):
                picks = picked_per_pc.get(c, [])
                if not picks:
                    continue
                rep_name, _ = picks[0]
                r = name_to_row.get(rep_name)
                if r is None:
                    continue
                val = L[r, c]
                color = 'white' if abs(val) > 0.7 * vmax else 'black'
                right.text(c, r, f"{val:.2f}", ha='center', va='center',
                        fontsize=SMALL, color=color)

        elif ANNOTATE_MODE == 'top_plus_rep':
            name_to_row = {nm: i for i, nm in enumerate(chosen_sorted)}
            for c in range(L.shape[1]):
                rep_r = None
                picks = picked_per_pc.get(c, [])
                if picks:
                    rep_name, _ = picks[0]
                    rep_r = name_to_row.get(rep_name)
                    if rep_r is not None:
                        v_rep = L[rep_r, c]
                        col = 'white' if abs(v_rep) > 0.7 * vmax else 'black'
                        right.text(c, rep_r, f"{v_rep:.2f}", ha='center', va='center',
                                fontsize=SMALL, color=col)

                r_top = int(np.nanargmax(np.abs(L[:, c])))
                if rep_r is None or r_top != rep_r:
                    v_top = L[r_top, c]
                    col2 = 'white' if abs(v_top) > 0.7 * vmax else 'black'
                    right.text(c, r_top, f"{v_top:.2f}", ha='center', va='center',
                            fontsize=SMALL, color=col2)

        # colorbar
        rpos = right.get_position()
        cb_x0 = min(rpos.x1 + CB_PAD, 1.0 - CB_WIDTH - RIGHT_MARGIN_SAFETY)
        cax = fig.add_axes([cb_x0, rpos.y0, CB_WIDTH, rpos.height])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Loading", fontsize=MED)

    else:
        right.axis("off")
        right.text(
            0.5,
            0.5,
            "No features selected after de-dup.\nTry lowering correlation cut or using PER_PC > 1.",
            ha="center",
            va="center",
            fontsize=12,
        )



    # save exact-size PNG + PDF
    out_png = os.path.join(PLOT_DIR, "ppt_pca_selected_features.png")
    out_pdf = os.path.join(PLOT_DIR, "ppt_pca_selected_features.pdf")
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved: {out_png}\nSaved: {out_pdf}")

    # 6) Ranked feature card(s)
    if MAKE_RANKED_CARD and L.size:
        # Single ranked card (top-N by |value| among the selected rows)
        out_rank = os.path.join(PLOT_DIR, "ppt_ranked_selected_features.png")
        render_ranked_feature_card(
            chosen_sorted=chosen_sorted,
            L=L,
            evr=evr,
            mode=RANK_MODE,
            pc_index=RANK_PC_INDEX,
            top=RANK_TOP,
            fig_w=RANK_FIG_W,
            fig_h=RANK_FIG_H,
            dpi=RANK_DPI,
            cmap=RANK_CMAP,
            out_path=out_rank,
        )
        print(f"Saved: {out_rank}")

        # Multi-page cards to show ALL selected features in pages (e.g., 1-7, 8-15)
        if RANK_SHOW_ALL_PAGES:
            # Decide row order for paging
            if RANK_ORDER == "pc":
                # preserve PC order: flatten picks per PC (PER_PC may be >1)
                names_pc: List[str] = []
                for j in range(k):
                    for (fname, _val) in picked_per_pc.get(j, []):
                        if fname in chosen:  # keep only those finally selected after de-dup
                            names_pc.append(fname)
                # map feature -> row index in the heatmap matrix L (which is in chosen_sorted order)
                row_map = {nm: i for i, nm in enumerate(chosen_sorted)}
                rows_idx = [row_map[nm] for nm in names_pc if nm in row_map]
                names_order = [chosen_sorted[i] for i in rows_idx]
                L_order = L[rows_idx, :] if rows_idx else np.empty((0, 0))
            else:
                # strength order (already used by chosen_sorted / L)
                names_order = list(chosen_sorted)
                L_order = L

            n = len(names_order)
            if n > 0 and L_order.size:
                page = 1
                for start in range(0, n, RANK_PAGE_SIZE):
                    end = min(start + RANK_PAGE_SIZE, n)
                    names_slice = names_order[start:end]
                    L_slice = L_order[start:end, :]
                    out_paged = os.path.join(PLOT_DIR, f"ppt_ranked_selected_features_p{page}.png")
                    render_ranked_feature_card(
                        chosen_sorted=names_slice,
                        L=L_slice,
                        evr=evr,
                        mode="dominant",       # show each row's strongest PC
                        pc_index=RANK_PC_INDEX,
                        top=len(names_slice),  # show all rows in this slice
                        fig_w=RANK_FIG_W,
                        fig_h=RANK_FIG_H,
                        dpi=RANK_DPI,
                        cmap=RANK_CMAP,
                        out_path=out_paged,
                    )
                    print(f"Saved: {out_paged} (rows {start+1}-{end})")
                    page += 1


if __name__ == "__main__":
    main()