# simple_diagnostics.py
# Clear, simple visualizations to understand what's happening
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"

def load_data():
    con = sqlite3.connect(DB_PATH)
    pca_wide = pd.read_sql("SELECT * FROM pca_factors_wide", con)
    if 'date' not in pca_wide.columns:
        for c in pca_wide.columns:
            if 'date' in c.lower():
                pca_wide = pca_wide.rename(columns={c: 'date'})
                break
    pca_wide['date'] = pd.to_datetime(pca_wide['date'])
    pca_wide = pca_wide.set_index('date')
    
    regimes = pd.read_sql("SELECT * FROM gmm_regimes", con)
    regimes['date'] = pd.to_datetime(regimes['date'])
    con.close()
    
    return pca_wide, regimes

def plot_top_pcs_over_time(pca_wide, regimes):
    """Plot top 4 PCs over time with regime backgrounds"""
    
    # Filter to GMM period (1995 onwards)
    gmm_start = pd.Timestamp('1995-01-01')
    pca_wide = pca_wide[pca_wide.index >= gmm_start]
    
    pc_cols = [c for c in pca_wide.columns if 'PC' in c][:12]
    
    fig, axes = plt.subplots(len(pc_cols), 1, figsize=(20, 3*len(pc_cols)))
    
    colors = ['#4ECDC4', '#FF6B6B']  # 0=Teal (Steady), 1=Red (Crisis)
    
    for idx, pc in enumerate(pc_cols):
        ax = axes[idx]
        
        # Plot PC values
        pc_data = pca_wide[pc].replace({None: np.nan})
        ax.plot(pca_wide.index, pc_data, linewidth=1.5, color='black', alpha=0.8, zorder=2)
        
        # Add scatter to show actual data points
        valid_mask = pc_data.notna()
        ax.scatter(pca_wide.index[valid_mask], pc_data[valid_mask], 
                  s=3, color='black', alpha=0.3, zorder=3)
        
        # Color background by regime - use thicker bars for visibility
        regime_data = regimes.set_index('date')['regime']
        
        # Create continuous regime periods
        current_regime = None
        start_date = None
        
        for date in regime_data.index:
            regime = regime_data.loc[date]
            if pd.notna(regime):
                regime = int(regime)
                
                if regime != current_regime:
                    # End previous regime
                    if current_regime is not None and start_date is not None:
                        ax.axvspan(start_date, date, 
                                  alpha=0.25, color=colors[current_regime], 
                                  linewidth=0, zorder=1)
                    # Start new regime
                    current_regime = regime
                    start_date = date
        
        # Close last regime
        if current_regime is not None and start_date is not None:
            ax.axvspan(start_date, regime_data.index[-1], 
                      alpha=0.25, color=colors[current_regime], 
                      linewidth=0, zorder=1)
        
        ax.set_ylabel(pc.replace('_PC1', ''), fontweight='bold', fontsize=12)
        ax.set_title(f'{pc.replace("_PC1", "")} Over Time\n(Background: Teal=Steady State, Red=Crisis)', 
                    fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(gmm_start, pca_wide.index.max())
        
        # Add regime legend on first plot
        if idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#4ECDC4', alpha=0.25, label='Regime 0 (Steady State)'),
                Patch(facecolor='#FF6B6B', alpha=0.25, label='Regime 1 (Crisis)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        if idx == len(pc_cols) - 1:
            ax.set_xlabel('Date', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('gmm_plots/pcs_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: pcs_over_time.png")

def plot_regime_means(pca_wide, regimes):
    """Show mean value of each PC in each regime"""
    
    pc_cols = [c for c in pca_wide.columns if 'PC' in c]
    
    # Align dates properly
    df = pca_wide[pc_cols].copy()
    regime_series = regimes.set_index('date')['regime']
    
    # Only keep dates that exist in both
    common_dates = df.index.intersection(regime_series.index)
    df = df.loc[common_dates]
    df['regime'] = regime_series.loc[common_dates]
    
    # Convert None to NaN
    df = df.replace({None: np.nan})
    
    # Only drop regime NaNs, keep PCs with some NaNs
    df = df.dropna(subset=['regime'])
    df['regime'] = df['regime'].astype(int)
    
    print(f"Data shape for analysis: {df.shape}")
    print(f"Regimes present: {sorted(df['regime'].unique())}")
    print(f"Non-null counts per PC:")
    for pc in pc_cols:
        print(f"  {pc.replace('_PC1', '')}: {df[pc].notna().sum()}")
    
    if len(df) == 0:
        print("ERROR: No overlapping data between PCA and regimes!")
        return
    
    # Calculate means by regime (ignoring NaNs)
    means = df.groupby('regime')[pc_cols].mean()
    stds = df.groupby('regime')[pc_cols].std()
    
    # Remove columns that are all NaN
    means = means.dropna(axis=1, how='all')
    stds = stds.dropna(axis=1, how='all')
    
    if means.empty:
        print("ERROR: All PCs are NaN!")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Heatmap of means
    sns.heatmap(means.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax1, cbar_kws={'label': 'Mean PC Value'})
    ax1.set_xlabel('Regime')
    ax1.set_ylabel('Principal Component')
    ax1.set_title('Mean PC Values by Regime\n(Shows what defines each regime)', fontweight='bold')
    
    # Heatmap of std devs
    sns.heatmap(stds.T, annot=True, fmt='.2f', cmap='Reds',
                ax=ax2, cbar_kws={'label': 'Std Dev'})
    ax2.set_xlabel('Regime')
    ax2.set_ylabel('Principal Component')
    ax2.set_title('Within-Regime Volatility\n(High values = regime is internally noisy)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('gmm_plots/regime_means.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: regime_means.png")
    
    # Print the actual numbers
    print("\n" + "="*80)
    print("REGIME CHARACTERISTICS")
    print("="*80)
    for regime in sorted(means.index):
        print(f"\nREGIME {regime}:")
        regime_means = means.loc[regime].sort_values(ascending=False)
        print("  Highest PCs:")
        for pc, val in regime_means.head(3).items():
            print(f"    {pc.replace('_PC1', '')}: {val:+.2f}")
        print("  Lowest PCs:")
        for pc, val in regime_means.tail(3).items():
            print(f"    {pc.replace('_PC1', '')}: {val:+.2f}")

def plot_2d_view(pca_wide, regimes):
    """Simple 2D projection"""
    
    pc_cols = [c for c in pca_wide.columns if 'PC' in c]
    
    # Align dates properly
    df = pca_wide[pc_cols].copy()
    regime_series = regimes.set_index('date')['regime']
    
    common_dates = df.index.intersection(regime_series.index)
    df = df.loc[common_dates]
    df['regime'] = regime_series.loc[common_dates]
    
    # Convert None to NaN
    df = df.replace({None: np.nan})
    df = df.dropna(subset=['regime'])
    
    # Drop PCs that are all NaN
    df = df.dropna(axis=1, how='all')
    
    # Now drop rows with any NaN in remaining PCs
    df = df.dropna()
    df['regime'] = df['regime'].astype(int)
    
    pc_cols = [c for c in df.columns if c != 'regime']
    
    if len(df) == 0 or len(pc_cols) == 0:
        print("ERROR: No complete data!")
        return
    
    X = df[pc_cols].values
    y = df['regime'].values
    
    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#4ECDC4', '#FF6B6B']  # 0=Teal (Steady), 1=Red (Crisis)
    
    for regime in sorted(np.unique(y)):
        mask = y == regime
        regime_label = 'Steady State' if regime == 0 else 'Crisis'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   alpha=0.5, s=30, color=colors[regime],
                   label=f'Regime {regime} ({regime_label}, n={mask.sum()})',
                   edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title('2D Projection of All PCs\n(If clusters are clear, you\'ll see separation)', 
                 fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gmm_plots/2d_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 2d_simple.png")
    print(f"\nTotal variance captured: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    print("If points are mixed together → weak separation")
    print("If clear clusters → good separation")

def calculate_overlap(pca_wide, regimes):
    """Quantify regime overlap"""
    
    pc_cols = [c for c in pca_wide.columns if 'PC' in c]
    
    # Align dates properly
    df = pca_wide[pc_cols].copy()
    regime_series = regimes.set_index('date')['regime']
    
    common_dates = df.index.intersection(regime_series.index)
    df = df.loc[common_dates]
    df['regime'] = regime_series.loc[common_dates]
    
    # Convert None to NaN
    df = df.replace({None: np.nan})
    df = df.dropna(subset=['regime'])
    df['regime'] = df['regime'].astype(int)
    
    # Only analyze PCs with data
    pc_cols = [c for c in pc_cols if df[c].notna().sum() > 0]
    
    if len(df) == 0 or len(pc_cols) == 0:
        print("ERROR: No overlapping data!")
        return
    
    print("\n" + "="*80)
    print("REGIME OVERLAP ANALYSIS")
    print("="*80)
    
    # For each PC, show the range of values in each regime
    for pc in pc_cols[:8]:  # Top 8 PCs
        print(f"\n{pc.replace('_PC1', '')}:")
        for regime in sorted(df['regime'].unique()):
            vals = df[df['regime'] == regime][pc]
            print(f"  Regime {regime}: [{vals.min():+.2f}, {vals.max():+.2f}]  "
                  f"mean={vals.mean():+.2f}, std={vals.std():.2f}")

def main():
    print("="*80)
    print("SIMPLE GMM DIAGNOSTICS")
    print("="*80)
    
    pca_wide, regimes = load_data()
    
    print("\n1. Plotting PCs over time with regime colors...")
    plot_top_pcs_over_time(pca_wide, regimes)
    
    print("\n2. Calculating regime mean characteristics...")
    plot_regime_means(pca_wide, regimes)
    
    print("\n3. Creating 2D projection...")
    plot_2d_view(pca_wide, regimes)
    
    print("\n4. Analyzing overlap...")
    calculate_overlap(pca_wide, regimes)
    
    print("\n" + "="*80)
    print("DONE - Check the plots to understand your regimes")
    print("="*80)

if __name__ == "__main__":
    main()