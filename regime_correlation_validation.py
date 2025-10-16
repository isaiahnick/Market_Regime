# regime_correlation_validation.py
# Test if PC correlations actually differ across regimes
# If regimes are real, correlations should be different in each regime
# If it's noise, correlations will be similar across regimes

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def calculate_regime_correlations(pca_wide, regimes):
    """Calculate correlation matrices for each regime"""
    
    pc_cols = [c for c in pca_wide.columns if 'PC' in c]
    
    # Align data
    df = pca_wide[pc_cols].copy()
    regime_series = regimes.set_index('date')['regime']
    
    common_dates = df.index.intersection(regime_series.index)
    df = df.loc[common_dates]
    df['regime'] = regime_series.loc[common_dates]
    
    df = df.replace({None: np.nan})
    df = df.dropna(subset=['regime'])
    df['regime'] = df['regime'].astype(int)
    
    # Drop PCs with all NaN
    df = df.dropna(axis=1, how='all')
    pc_cols = [c for c in df.columns if c != 'regime']
    
    print(f"Analyzing {len(pc_cols)} PCs across {df['regime'].nunique()} regimes")
    print(f"Total observations: {len(df)}\n")
    
    # Calculate correlations for each regime
    regime_corrs = {}
    regime_counts = {}
    
    for regime in sorted(df['regime'].unique()):
        regime_data = df[df['regime'] == regime][pc_cols]
        
        # Need at least 30 observations for stable correlation
        if len(regime_data) < 30:
            print(f"Warning: Regime {regime} only has {len(regime_data)} observations")
            continue
        
        corr_matrix = regime_data.corr()
        regime_corrs[regime] = corr_matrix
        regime_counts[regime] = len(regime_data)
        
        print(f"Regime {regime}: {len(regime_data)} observations")
    
    # Overall correlation (for comparison)
    overall_corr = df[pc_cols].corr()
    
    return regime_corrs, overall_corr, regime_counts, pc_cols

def plot_correlation_comparison(regime_corrs, overall_corr, regime_counts, pc_cols):
    """Plot correlation matrices side by side"""
    
    n_regimes = len(regime_corrs)
    fig, axes = plt.subplots(1, n_regimes + 1, figsize=(6*(n_regimes+1), 5))
    
    # Clean PC names for display
    clean_names = [c.replace('_PC1', '') for c in pc_cols]
    
    # Plot each regime
    for idx, (regime, corr) in enumerate(sorted(regime_corrs.items())):
        ax = axes[idx]
        
        # Reindex with clean names
        corr_clean = corr.copy()
        corr_clean.index = clean_names
        corr_clean.columns = clean_names
        
        sns.heatmap(corr_clean, annot=False, cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, ax=ax, square=True,
                   cbar_kws={'label': 'Correlation'})
        ax.set_title(f'Regime {regime}\n({regime_counts[regime]} obs)', 
                    fontweight='bold', fontsize=12)
    
    # Plot overall correlation
    ax = axes[-1]
    overall_clean = overall_corr.copy()
    overall_clean.index = clean_names
    overall_clean.columns = clean_names
    
    sns.heatmap(overall_clean, annot=False, cmap='RdBu_r', center=0,
               vmin=-1, vmax=1, ax=ax, square=True,
               cbar_kws={'label': 'Correlation'})
    ax.set_title('Overall\n(All periods)', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('gmm_plots/correlation_by_regime.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Saved: correlation_by_regime.png")

def calculate_correlation_differences(regime_corrs, overall_corr, pc_cols):
    """Quantify how much correlations differ across regimes"""
    
    print("\n" + "="*80)
    print("CORRELATION DIFFERENCE ANALYSIS")
    print("="*80)
    
    if len(regime_corrs) < 2:
        print("Need at least 2 regimes to compare")
        return
    
    # Calculate average absolute difference in correlations
    regime_list = sorted(regime_corrs.keys())
    
    # For each pair of regimes, calculate difference
    differences = []
    
    for i, r1 in enumerate(regime_list):
        for r2 in regime_list[i+1:]:
            corr1 = regime_corrs[r1]
            corr2 = regime_corrs[r2]
            
            # Get upper triangle (avoid diagonal and duplicates)
            mask = np.triu(np.ones_like(corr1, dtype=bool), k=1)
            
            diff = np.abs(corr1.values - corr2.values)
            avg_diff = diff[mask].mean()
            max_diff = diff[mask].max()
            
            differences.append({
                'regime_pair': f'{r1} vs {r2}',
                'avg_abs_diff': avg_diff,
                'max_diff': max_diff
            })
            
            print(f"\nRegime {r1} vs Regime {r2}:")
            print(f"  Average absolute correlation difference: {avg_diff:.3f}")
            print(f"  Maximum correlation difference: {max_diff:.3f}")
    
    # Overall interpretation
    avg_of_avgs = np.mean([d['avg_abs_diff'] for d in differences])
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print(f"Average correlation difference across regimes: {avg_of_avgs:.3f}")
    print()
    
    if avg_of_avgs > 0.15:
        print("✓ STRONG EVIDENCE: Correlations differ substantially across regimes")
        print("  This suggests regimes capture real market structure changes")
    elif avg_of_avgs > 0.08:
        print("✓ MODERATE EVIDENCE: Correlations differ moderately across regimes")
        print("  Regimes capture some real structure, but may have overlap")
    else:
        print("✗ WEAK EVIDENCE: Correlations are similar across regimes")
        print("  Regimes may be capturing noise rather than real structure")
    
    print()
    print("Rule of thumb:")
    print("  > 0.15: Strong regime differences (good)")
    print("  0.08-0.15: Moderate differences (acceptable)")
    print("  < 0.08: Weak differences (questionable)")

def find_key_correlation_changes(regime_corrs, pc_cols):
    """Find which PC pairs change most across regimes"""
    
    print("\n" + "="*80)
    print("KEY CORRELATION CHANGES")
    print("="*80)
    
    if len(regime_corrs) < 2:
        return
    
    regime_list = sorted(regime_corrs.keys())
    
    # Calculate correlation change for each PC pair
    changes = []
    
    for i, pc1 in enumerate(pc_cols):
        for j, pc2 in enumerate(pc_cols):
            if i >= j:  # Skip diagonal and lower triangle
                continue
            
            # Get correlations across regimes
            corrs_across_regimes = []
            for regime in regime_list:
                if pc1 in regime_corrs[regime].columns and pc2 in regime_corrs[regime].columns:
                    corrs_across_regimes.append(regime_corrs[regime].loc[pc1, pc2])
            
            if len(corrs_across_regimes) >= 2:
                # Calculate range (max - min)
                corr_range = max(corrs_across_regimes) - min(corrs_across_regimes)
                
                changes.append({
                    'pc1': pc1.replace('_PC1', ''),
                    'pc2': pc2.replace('_PC1', ''),
                    'range': corr_range,
                    'min_corr': min(corrs_across_regimes),
                    'max_corr': max(corrs_across_regimes)
                })
    
    # Sort by range
    changes_df = pd.DataFrame(changes).sort_values('range', ascending=False)
    
    print("\nTop 10 PC pairs with biggest correlation changes across regimes:")
    print()
    for idx, row in changes_df.head(10).iterrows():
        print(f"{row['pc1']:25} <-> {row['pc2']:25}")
        print(f"  Range: {row['range']:.3f}  (from {row['min_corr']:+.3f} to {row['max_corr']:+.3f})")
    
    print("\nThese pairs change most across regimes - key regime drivers!")

def main():
    print("="*80)
    print("REGIME CORRELATION VALIDATION")
    print("Testing if regimes show real structural differences")
    print("="*80)
    
    pca_wide, regimes = load_data()
    
    print("\nCalculating correlations by regime...")
    regime_corrs, overall_corr, regime_counts, pc_cols = calculate_regime_correlations(pca_wide, regimes)
    
    print("\nCreating correlation comparison plots...")
    plot_correlation_comparison(regime_corrs, overall_corr, regime_counts, pc_cols)
    
    calculate_correlation_differences(regime_corrs, overall_corr, pc_cols)
    
    find_key_correlation_changes(regime_corrs, pc_cols)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("If correlations differ significantly across regimes (>0.15 avg diff),")
    print("your regimes are capturing real market structure changes, not noise.")
    print("Check the heatmaps to see which correlations flip across regimes.")

if __name__ == "__main__":
    main()