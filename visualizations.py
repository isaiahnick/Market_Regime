# visualizations.py
# Generate visualizations for GMM results stored in the database
# This script reads from the database tables created by gmm.py and generates all charts
#
# Usage:
#   python visualizations.py --start 1970-01-01 --kmin 2 --kmax 6 --cov full --impute median --standardize
#
import argparse
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from dataclasses import dataclass

DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"

@dataclass
class GMMDiagnostics:
    k: int
    aic: float
    bic: float
    silhouette: float

def load_pca_wide():
    """Load PCA data - same as gmm.py"""
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

def impute_missing(X: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Impute missing values - same as gmm.py"""
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

def load_gmm_results():
    """Load GMM results from database"""
    con = sqlite3.connect(DB_PATH)
    try:
        # Load regime assignments and probabilities
        regimes = pd.read_sql("SELECT * FROM gmm_regimes", con)
        regimes['date'] = pd.to_datetime(regimes['date'])
        
        # Load diagnostics
        scores = pd.read_sql("SELECT * FROM gmm_scores", con)
        diagnostics = [GMMDiagnostics(row['k'], row['aic'], row['bic'], row['silhouette']) 
                      for _, row in scores.iterrows()]
        
        # Load metadata
        meta = pd.read_sql("SELECT * FROM gmm_meta", con).iloc[0]
        k_star = int(meta['chosen_k'])
        
        return regimes, diagnostics, k_star
    finally:
        con.close()

def create_professional_regime_charts(regimes, k_star, dates, args):
    """Create professional regime charts matching the exhibit style"""
    
    # Set up professional plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 16,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.edgecolor': 'black'
    })
    
    # Define professional color scheme matching your exhibits
    regime_colors = {
        0: '#4ECDC4',  # Teal/Cyan - Steady State
        1: '#FF6B6B',  # Red/Pink - Crisis
        2: '#45B7D1',  # Blue - Another state
        3: '#96CEB4',  # Light Green - Another state
        4: '#FFEAA7',  # Yellow - WOI
        5: '#DDA0DD'   # Purple - Another state
    }
    
    # Create generic regime labels
    regime_labels = {
        0: 'Regime 0 (Steady State)',
        1: 'Regime 1 (Crisis)'
    }
    # Add generic labels for K>2 (shouldn't happen with your model)
    for i in range(2, k_star):
        regime_labels[i] = f'Regime {i}'
    
    # Create output directory
    output_dir = "gmm_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert dates to datetime
    plot_dates = pd.to_datetime(dates)
    
    # 1. Historical Timeline Chart (like Exhibit 4)
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Calculate regime frequencies for legend
    regime_counts = regimes['regime'].value_counts().sort_index()
    total_obs = len(regimes)
    
    # Create the timeline visualization
    y_pos = 0.5
    bar_height = 0.8
    
    for i, (date, regime) in enumerate(zip(plot_dates, regimes['regime'])):
        color = regime_colors.get(regime, '#CCCCCC')
        ax.barh(y_pos, 1, height=bar_height, left=i, color=color, edgecolor='none')
    
    # Format x-axis with years
    years = pd.date_range(start=plot_dates.min(), end=plot_dates.max(), freq='YS')
    year_positions = []
    year_labels = []
    
    for year in years:
        # Find position of this year in our data
        try:
            pos = np.where(plot_dates >= year)[0][0]
            year_positions.append(pos)
            year_labels.append(str(year.year))
        except IndexError:
            continue
    
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_labels, rotation=45, ha='right')
    ax.set_xlim(0, len(plot_dates))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    
    # Create legend with frequencies
    legend_elements = []
    for regime in sorted(regime_counts.index):
        count = regime_counts[regime]
        pct = (count / total_obs) * 100
        label = f"{regime_labels[regime]}, {pct:.0f}%"
        color = regime_colors.get(regime, '#CCCCCC')
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label))
    
    ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.15), 
              ncol=min(4, len(legend_elements)), frameon=True)
    
    ax.set_title('Exhibit 4: Highest Probability Market Conditions\nThroughout History', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Remove spines except bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"{output_dir}/exhibit_4_historical_timeline.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 2. Recent Period Timeline (like Exhibit 5)
    # Filter for recent period (last 2 years or so)
    recent_cutoff = plot_dates.max() - pd.DateOffset(years=2)
    recent_mask = plot_dates >= recent_cutoff
    recent_dates = plot_dates[recent_mask]
    recent_regimes = regimes[recent_mask]
    
    if len(recent_dates) > 0:
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Create timeline for recent period
        for i, (date, regime) in enumerate(zip(recent_dates, recent_regimes['regime'])):
            color = regime_colors.get(regime, '#CCCCCC')
            ax.barh(0.5, 1, height=0.8, left=i, color=color, edgecolor='none')
        
        # Format x-axis with months for recent period
        if len(recent_dates) > 50:  # If more than ~50 observations, show quarterly
            freq = 'QS'
            date_format = '%m/%Y'
        else:  # Show monthly
            freq = 'MS'
            date_format = '%m/%Y'
            
        date_ticks = pd.date_range(start=recent_dates.min(), end=recent_dates.max(), freq=freq)
        tick_positions = []
        tick_labels = []
        
        for tick_date in date_ticks:
            try:
                pos = np.where(recent_dates >= tick_date)[0][0]
                tick_positions.append(pos)
                tick_labels.append(tick_date.strftime(date_format))
            except IndexError:
                continue
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_xlim(0, len(recent_dates))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        
        # Legend
        legend_elements = []
        for regime in sorted(recent_regimes['regime'].unique()):
            label = regime_labels[regime]
            color = regime_colors.get(regime, '#CCCCCC')
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label))
        
        ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.25), 
                  ncol=len(legend_elements), frameon=True)
        
        start_date = recent_dates.min().strftime('%B %d, %Y')
        end_date = recent_dates.max().strftime('%B %d, %Y')
        
        ax.set_title(f'Exhibit 5: Highest Probability Market Conditions Since the\nBeginning of 2020\n\n'
                     f'Period: {start_date} - {end_date}.', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Remove spines except bottom
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(f"{output_dir}/exhibit_5_recent_timeline.png", dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.show()
    
    # 3. Stacked Probability Chart (like Exhibit 6)
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare probability data for stacking
    prob_cols = [f'prob_{j}' for j in range(k_star)]
    prob_matrix = regimes[prob_cols].values.T  # Transpose for stacking
    
    # Create stacked area plot
    colors_list = [regime_colors.get(i, '#CCCCCC') for i in range(k_star)]
    labels_list = [regime_labels[i] for i in range(k_star)]
    
    # Use recent period for this chart too
    recent_prob_matrix = prob_matrix[:, recent_mask] if len(recent_dates) > 0 else prob_matrix
    recent_plot_dates = recent_dates if len(recent_dates) > 0 else plot_dates
    
    ax.stackplot(range(len(recent_plot_dates)), *recent_prob_matrix, 
                 labels=labels_list, colors=colors_list, alpha=0.8)
    
    # Format axes
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(recent_plot_dates)-1)
    
    # Y-axis as percentages
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_ylabel('Market Condition Probability', fontweight='bold')
    
    # X-axis formatting
    if len(recent_plot_dates) > 50:
        freq = 'QS'
        date_format = '%m/%Y'
    else:
        freq = 'MS' 
        date_format = '%m/%Y'
        
    date_ticks = pd.date_range(start=recent_plot_dates.min(), end=recent_plot_dates.max(), freq=freq)
    tick_positions = []
    tick_labels = []
    
    for tick_date in date_ticks:
        try:
            pos = np.where(recent_plot_dates >= tick_date)[0][0]
            tick_positions.append(pos)
            tick_labels.append(tick_date.strftime(date_format))
        except IndexError:
            continue
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Legend
    ax.legend(loc='center', bbox_to_anchor=(0.5, -0.2), ncol=min(4, k_star), frameon=True)
    
    start_date = recent_plot_dates.min().strftime('%B %d, %Y')
    end_date = recent_plot_dates.max().strftime('%B %d, %Y')
    
    ax.set_title(f'Exhibit 6: Market Condition Probabilities Since the\nBeginning of 2020\n\n'
                 f'Period: {start_date} - {end_date}.', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(f"{output_dir}/exhibit_6_probabilities.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 4. Regime Classification Confidence Chart - Full Timeline
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Calculate confidence as the maximum probability for each time period
    prob_cols = [f'prob_{j}' for j in range(k_star)]
    confidence = regimes[prob_cols].max(axis=1)
    
    # Use full timeline instead of recent period
    full_plot_dates = plot_dates
    full_confidence = confidence
    
    # Plot confidence as a line chart
    ax.plot(range(len(full_plot_dates)), full_confidence, 
            color='#2E86AB', linewidth=1.2, alpha=0.8)
    ax.fill_between(range(len(full_plot_dates)), full_confidence, 
                    alpha=0.3, color='#2E86AB')
    
    # Format axes
    ax.set_ylim(0.5, 1.05)  # Start at 0.5 since we expect reasonable confidence
    ax.set_xlim(0, len(full_plot_dates)-1)
    
    # Y-axis formatting
    ax.set_ylabel('Max Probability', fontweight='bold')
    ax.set_title('Regime Classification Confidence', fontsize=16, fontweight='bold', pad=20)
    
    # X-axis formatting with years for full timeline
    years = pd.date_range(start=full_plot_dates.min(), end=full_plot_dates.max(), freq='YS')
    year_positions = []
    year_labels = []
    
    for year in years:
        try:
            pos = np.where(full_plot_dates >= year)[0][0]
            year_positions.append(pos)
            year_labels.append(str(year.year))
        except IndexError:
            continue
    
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_labels, rotation=45, ha='right')
    ax.set_xlabel('Date', fontweight='bold')
    
    # Add horizontal reference lines
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='High Confidence (95%)')
    ax.axhline(y=0.80, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence (80%)')
    ax.axhline(y=0.60, color='red', linestyle='--', alpha=0.7, label='Low Confidence (60%)')
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=True)
    
    # Add period info for full timeline
    start_date = full_plot_dates.min().strftime('%B %d, %Y')
    end_date = full_plot_dates.max().strftime('%B %d, %Y')
    
    # Add subtitle with period
    fig.suptitle(f'Regime Classification Confidence - Full Timeline\n\nPeriod: {start_date} - {end_date}', 
                 fontsize=16, fontweight='bold', y=0.95)
    ax.set_title('')  # Remove the original title since we're using suptitle
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{output_dir}/regime_classification_confidence_full.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"\nProfessional regime charts saved to '{output_dir}' directory:")
    print("- exhibit_4_historical_timeline.png: Complete historical timeline")
    print("- exhibit_5_recent_timeline.png: Recent period timeline")  
    print("- exhibit_6_probabilities.png: Recent period probabilities")
    print("- regime_classification_confidence_full.png: Classification confidence - full timeline")


def create_table_visualizations(mean_returns, volatilities, regime_stats, k_star):
    """Create professional PNG table visualizations"""
    
    output_dir = "gmm_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set professional styling
    plt.rcParams.update({
        'font.size': 10,
        'font.weight': 'normal',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 14,
    })
    
    # 1. Mean Returns Table (like Exhibit 2 style)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    returns_data = mean_returns.round(4)
    
    # Create color-coded table
    # Determine color scale based on data range
    vmin, vmax = returns_data.values.min(), returns_data.values.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn  # Red for negative, green for positive
    
    # Convert to list format for table
    table_data = []
    row_labels = []
    
    for idx, row in returns_data.iterrows():
        formatted_row = [f"{val:.2%}" for val in row.values]
        table_data.append(formatted_row)
        row_labels.append(f"Regime {idx}")
    
    # Column headers (factor names)
    col_labels = [col.replace('_PC1', '') for col in returns_data.columns]
    
    # Create table
    table = ax.table(cellText=table_data,
                    rowLabels=row_labels,
                    colLabels=col_labels,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Color coding for cells
    for i in range(len(table_data)):
        for j in range(len(table_data[i])):
            val = returns_data.iloc[i, j]
            color = cmap(norm(val))
            # Make text white if background is dark
            text_color = 'white' if val < (vmin + vmax) / 2 else 'black'
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_text_props(color=text_color, weight='bold')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Header styling
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#404040')
        table[(0, j)].set_text_props(color='white', weight='bold')
    
    # Row label styling  
    for i in range(1, len(table_data) + 1):
        table[(i, -1)].set_facecolor('#404040')
        table[(i, -1)].set_text_props(color='white', weight='bold')
    
    ax.set_title('Exhibit 2: Annualized Factor Mean Returns in the Market Conditions', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exhibit_2_mean_returns.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 2. Volatilities Table (like Exhibit 3 style)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare volatility data
    vol_data = volatilities.round(4)
    
    # Color scale for volatilities (higher = more red)
    vol_vmin, vol_vmax = vol_data.values.min(), vol_data.values.max()
    vol_norm = plt.Normalize(vmin=vol_vmin, vmax=vol_vmax)
    vol_cmap = plt.cm.Reds  # White to red scale
    
    # Convert to table format
    vol_table_data = []
    vol_row_labels = []
    
    for idx, row in vol_data.iterrows():
        formatted_row = [f"{val:.2%}" for val in row.values]
        vol_table_data.append(formatted_row)
        vol_row_labels.append(f"Regime {idx}")
    
    # Create volatility table
    vol_table = ax.table(cellText=vol_table_data,
                        rowLabels=vol_row_labels,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
    
    # Color coding for volatility cells
    for i in range(len(vol_table_data)):
        for j in range(len(vol_table_data[i])):
            val = vol_data.iloc[i, j]
            color = vol_cmap(vol_norm(val))
            # Make text black on light backgrounds, white on dark
            text_color = 'white' if val > (vol_vmin + vol_vmax) * 0.7 else 'black'
            vol_table[(i+1, j)].set_facecolor(color)
            vol_table[(i+1, j)].set_text_props(color=text_color, weight='bold')
    
    # Style the volatility table
    vol_table.auto_set_font_size(False)
    vol_table.set_fontsize(9)
    vol_table.scale(1, 2)
    
    # Header styling
    for j in range(len(col_labels)):
        vol_table[(0, j)].set_facecolor('#404040')
        vol_table[(0, j)].set_text_props(color='white', weight='bold')
    
    # Row label styling
    for i in range(1, len(vol_table_data) + 1):
        vol_table[(i, -1)].set_facecolor('#404040')
        vol_table[(i, -1)].set_text_props(color='white', weight='bold')
    
    ax.set_title('Exhibit 3: Factor Volatilities in the Market Conditions', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exhibit_3_volatilities.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 3. Regime Statistics Summary Table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare stats data
    stats_data = []
    for _, row in regime_stats.iterrows():
        stats_data.append([row['Regime'], row['Periods'], row['Frequency (%)'], row['Avg Probability']])
    
    stats_table = ax.table(cellText=stats_data,
                          colLabels=['Regime', 'Periods', 'Frequency', 'Avg Probability'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
    
    # Style the stats table
    stats_table.auto_set_font_size(False)
    stats_table.set_fontsize(11)
    stats_table.scale(1, 2)
    
    # Header styling
    for j in range(4):
        stats_table[(0, j)].set_facecolor('#404040')
        stats_table[(0, j)].set_text_props(color='white', weight='bold')
    
    # Alternate row colors
    for i in range(1, len(stats_data) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(4):
            stats_table[(i, j)].set_facecolor(color)
    
    ax.set_title('Regime Summary Statistics', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/regime_summary_statistics.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"\nTable visualizations saved to '{output_dir}' directory:")
    print("- exhibit_2_mean_returns.png: Color-coded mean returns by regime")
    print("- exhibit_3_volatilities.png: Color-coded volatilities by regime") 
    print("- regime_summary_statistics.png: Regime statistics summary")


def create_visualizations(regimes, diagnostics, X, dates, k_star, args):
    """Create comprehensive visualizations for GMM results"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory for plots
    output_dir = "gmm_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and generate summary tables for visualizations
    print("Loading regime summary tables from database...")
    con = sqlite3.connect(DB_PATH)
    try:
        mean_returns = pd.read_sql("SELECT * FROM gmm_regime_means", con)
        mean_returns.set_index('regime', inplace=True)
        volatilities = pd.read_sql("SELECT * FROM gmm_regime_volatilities", con)
        volatilities.set_index('regime', inplace=True)
        regime_stats = pd.read_sql("SELECT * FROM gmm_regime_statistics", con)
    except Exception as e:
        print(f"Warning: Could not load summary tables from database: {e}")
        mean_returns = pd.DataFrame()
        volatilities = pd.DataFrame()
        regime_stats = pd.DataFrame()
    finally:
        con.close()
    
    # First create the professional regime charts
    print("\nCreating professional regime charts...")
    create_professional_regime_charts(regimes, k_star, dates, args)
    
    # Create table visualizations if data is available
    if not mean_returns.empty and not volatilities.empty and not regime_stats.empty:
        print("\nCreating table visualizations...")
        create_table_visualizations(mean_returns, volatilities, regime_stats, k_star)
    
    # 1. Model Selection Diagnostics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    k_values = [d.k for d in diagnostics]
    aic_values = [d.aic for d in diagnostics]
    bic_values = [d.bic for d in diagnostics]
    sil_values = [d.silhouette for d in diagnostics]
    
    # AIC/BIC plot
    ax1.plot(k_values, aic_values, 'o-', label='AIC', linewidth=2, markersize=8)
    ax1.plot(k_values, bic_values, 's-', label='BIC', linewidth=2, markersize=8)
    ax1.axvline(x=k_star, color='red', linestyle='--', alpha=0.7, label=f'Selected K={k_star}')
    ax1.set_xlabel('Number of Components (K)')
    ax1.set_ylabel('Information Criterion')
    ax1.set_title('Model Selection: AIC vs BIC')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Silhouette scores
    ax2.plot(k_values, sil_values, 'o-', color='green', linewidth=2, markersize=8)
    ax2.axvline(x=k_star, color='red', linestyle='--', alpha=0.7, label=f'Selected K={k_star}')
    ax2.set_xlabel('Number of Components (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # BIC improvement
    bic_improvements = [bic_values[0] - bic for bic in bic_values]
    ax3.bar(k_values, bic_improvements, alpha=0.7, color='skyblue')
    ax3.axvline(x=k_star, color='red', linestyle='--', alpha=0.7, label=f'Selected K={k_star}')
    ax3.set_xlabel('Number of Components (K)')
    ax3.set_ylabel('BIC Improvement from K=2')
    ax3.set_title('BIC Improvement')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Model comparison table
    ax4.axis('tight')
    ax4.axis('off')
    table_data = []
    for d in diagnostics:
        table_data.append([d.k, f"{d.aic:.1f}", f"{d.bic:.1f}", f"{d.silhouette:.3f}"])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['K', 'AIC', 'BIC', 'Silhouette'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Model Diagnostics Summary')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_selection.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Regime Time Series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Convert dates to datetime if they aren't already
    plot_dates = pd.to_datetime(dates)

    # Define colors for K=2 model
    regime_colors_list = ['#4ECDC4', '#FF6B6B']  # 0=Teal, 1=Red
    regime_names = ['Steady State', 'Crisis']

    # Regime assignments over time
    for regime in range(k_star):
        mask = regimes['regime'] == regime
        color = regime_colors_list[regime] if regime < len(regime_colors_list) else plt.cm.Set1(regime/k_star)
        label = f'Regime {regime} ({regime_names[regime]})' if regime < len(regime_names) else f'Regime {regime}'
        ax1.scatter(plot_dates[mask], [regime] * sum(mask), 
                c=color, alpha=0.6, s=10, label=label)
    
    ax1.set_ylabel('Regime')
    ax1.set_title(f'Market Regimes Over Time (K={k_star})')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yticks(range(k_star))
    
    # Regime probabilities as stacked area
    prob_cols = [f'prob_{j}' for j in range(k_star)]
    prob_data = regimes[prob_cols].values.T

    ax2.stackplot(plot_dates, *prob_data, 
                labels=[f'Regime {i} ({regime_names[i]})' if i < len(regime_names) else f'Regime {i}' for i in range(k_star)], 
                colors=regime_colors_list[:k_star], alpha=0.7)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Probability')
    ax2.set_title('Regime Probabilities Over Time')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/regime_timeseries.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Feature Space Visualization (if 2D or can be reduced to 2D)
    if X.shape[1] >= 2:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Use first two principal components for visualization
        pc1, pc2 = X[:, 0], X[:, 1]
        
        # Scatter plot colored by regime
        scatter = ax1.scatter(pc1, pc2, c=regimes['regime'], cmap='Set1', alpha=0.6, s=20)
        ax1.set_xlabel('First Feature (PC1)')
        ax1.set_ylabel('Second Feature (PC2)')
        ax1.set_title('Data Points Colored by Regime')
        plt.colorbar(scatter, ax=ax1)
        ax1.grid(True, alpha=0.3)
        
        # Regime centroids
        for regime in range(k_star):
            mask = regimes['regime'] == regime
            if sum(mask) > 0:
                centroid_x = np.mean(pc1[mask])
                centroid_y = np.mean(pc2[mask])
                ax1.scatter(centroid_x, centroid_y, c='black', s=200, marker='x', linewidth=3)
                ax1.annotate(f'R{regime}', (centroid_x, centroid_y), 
                           xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        # Feature distributions by regime
        for i, regime in enumerate(range(k_star)):
            mask = regimes['regime'] == regime
            if sum(mask) > 0:
                ax2.hist(pc1[mask], bins=30, alpha=0.6, label=f'Regime {regime}', density=True)
        ax2.set_xlabel('First Feature (PC1)')
        ax2.set_ylabel('Density')
        ax2.set_title('Feature 1 Distribution by Regime')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        for i, regime in enumerate(range(k_star)):
            mask = regimes['regime'] == regime
            if sum(mask) > 0:
                ax3.hist(pc2[mask], bins=30, alpha=0.6, label=f'Regime {regime}', density=True)
        ax3.set_xlabel('Second Feature (PC2)')
        ax3.set_ylabel('Density')
        ax3.set_title('Feature 2 Distribution by Regime')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Regime transition matrix heatmap
        transitions = np.zeros((k_star, k_star))
        prev_regime = regimes['regime'].iloc[0]
        for curr_regime in regimes['regime'].iloc[1:]:
            transitions[prev_regime, curr_regime] += 1
            prev_regime = curr_regime
        
        # Normalize to probabilities
        row_sums = transitions.sum(axis=1)
        transitions_norm = transitions / row_sums[:, np.newaxis]
        transitions_norm = np.nan_to_num(transitions_norm)
        
        im = ax4.imshow(transitions_norm, cmap='Blues', aspect='auto')
        ax4.set_xlabel('To Regime')
        ax4.set_ylabel('From Regime')
        ax4.set_title('Regime Transition Probabilities')
        ax4.set_xticks(range(k_star))
        ax4.set_yticks(range(k_star))
        
        # Add text annotations
        for i in range(k_star):
            for j in range(k_star):
                text = ax4.text(j, i, f'{transitions_norm[i, j]:.2f}',
                              ha="center", va="center", color="black" if transitions_norm[i, j] < 0.5 else "white")
        
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Regime Statistics Summary
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Regime duration statistics
    regime_changes = regimes['regime'].diff().fillna(0) != 0
    regime_periods = []
    current_regime = regimes['regime'].iloc[0]
    current_duration = 1
    
    for i in range(1, len(regimes)):
        if regimes['regime'].iloc[i] == current_regime:
            current_duration += 1
        else:
            regime_periods.append((current_regime, current_duration))
            current_regime = regimes['regime'].iloc[i]
            current_duration = 1
    regime_periods.append((current_regime, current_duration))
    
    # Average duration by regime
    regime_durations = {}
    for regime, duration in regime_periods:
        if regime not in regime_durations:
            regime_durations[regime] = []
        regime_durations[regime].append(duration)
    
    colors = plt.cm.Set1(np.linspace(0, 1, k_star))
    avg_durations = [np.mean(regime_durations.get(i, [0])) for i in range(k_star)]
    ax1.bar(range(k_star), avg_durations, alpha=0.7, color=colors)
    ax1.set_xlabel('Regime')
    ax1.set_ylabel('Average Duration (periods)')
    ax1.set_title('Average Regime Duration')
    ax1.set_xticks(range(k_star))
    ax1.grid(True, alpha=0.3)
    
    # Regime frequency
    regime_counts = regimes['regime'].value_counts().sort_index()
    ax2.pie(regime_counts.values, labels=[f'Regime {i}' for i in regime_counts.index], 
            autopct='%1.1f%%', colors=colors[:len(regime_counts)])
    ax2.set_title('Regime Frequency Distribution')
    
    # Regime probability statistics
    prob_stats = []
    for regime in range(k_star):
        prob_col = f'prob_{regime}'
        stats = {
            'regime': regime,
            'mean_prob': regimes[prob_col].mean(),
            'max_prob': regimes[prob_col].max(),
            'std_prob': regimes[prob_col].std()
        }
        prob_stats.append(stats)
    
    prob_df = pd.DataFrame(prob_stats)
    x_pos = np.arange(len(prob_df))
    
    ax3.bar(x_pos - 0.2, prob_df['mean_prob'], 0.4, label='Mean', alpha=0.7, color='skyblue')
    ax3.bar(x_pos + 0.2, prob_df['std_prob'], 0.4, label='Std Dev', alpha=0.7, color='lightcoral')
    ax3.set_xlabel('Regime')
    ax3.set_ylabel('Probability')
    ax3.set_title('Regime Probability Statistics')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Regime {i}' for i in range(k_star)])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Summary statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    summary_data = []
    for regime in range(k_star):
        count = sum(regimes['regime'] == regime)
        pct = count / len(regimes) * 100
        avg_dur = avg_durations[regime]
        max_prob = regimes[f'prob_{regime}'].max()
        summary_data.append([f'Regime {regime}', count, f'{pct:.1f}%', f'{avg_dur:.1f}', f'{max_prob:.3f}'])
    
    summary_table = ax4.table(cellText=summary_data,
                             colLabels=['Regime', 'Count', 'Frequency', 'Avg Duration', 'Max Prob'],
                             cellLoc='center',
                             loc='center')
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(10)
    summary_table.scale(1.2, 1.5)
    ax4.set_title('Regime Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/regime_statistics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nAll visualizations saved to '{output_dir}' directory:")
    print("- exhibit_4_historical_timeline.png: Professional historical timeline")
    print("- exhibit_5_recent_timeline.png: Professional recent timeline")  
    print("- exhibit_6_probabilities.png: Professional probability chart")
    print("- model_selection.png: Model selection diagnostics")
    print("- regime_timeseries.png: Regime assignments and probabilities over time")
    print("- feature_analysis.png: Feature space analysis and transitions")
    print("- regime_statistics.png: Comprehensive regime statistics")


def main():
    ap = argparse.ArgumentParser(description="Generate visualizations for GMM results from database")
    ap.add_argument('--start', type=str, default='1995-01-01', help='Start date (inclusive), e.g., 1970-01-01')
    ap.add_argument('--kmin', type=int, default=2)
    ap.add_argument('--kmax', type=int, default=6)
    ap.add_argument('--cov', type=str, default='full', choices=['full', 'tied', 'diag', 'spherical'])
    ap.add_argument('--impute', type=str, default='median', choices=['none','mean','median','most_frequent','ffill','bfill','ffill_bfill','ffbb'])
    ap.add_argument('--standardize', action='store_true', help='Z-score the PCA features before GMM (recommended).')
    args = ap.parse_args()

    print("Loading GMM results from database...")
    try:
        regimes, diagnostics, k_star = load_gmm_results()
    except Exception as e:
        print(f"Error loading GMM results: {e}")
        print("Make sure you've run gmm.py first to generate the results.")
        return

    print(f"Loaded GMM results: K={k_star}, {len(regimes)} observations")

    # Recreate the data processing pipeline to match what was used in gmm.py
    print("Recreating data processing pipeline...")
    df = load_pca_wide()

    # Filter to start date
    start = pd.to_datetime(args.start)
    if start is not None:
        df = df[df.index >= start]

    # Choose features (all numeric columns saved by PCA)
    feat_cols = [c for c in df.columns]
    X = df[feat_cols].copy()

    # Impute missing values so we don't lose early history
    X = impute_missing(X, args.impute)

    # Drop any rows that are still NaN across the board
    X = X.dropna(how='all')
    dates = X.index.to_list()

    # Standardize features if requested
    if args.standardize:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X.values)
    else:
        X_std = X.values

    # Filter regimes to match the processed data
    regime_dates = pd.to_datetime(regimes['date'])
    processed_dates = pd.to_datetime(dates)
    
    # Ensure we have matching dates between regimes and processed data
    regimes_filtered = regimes[regime_dates.isin(processed_dates)].copy()
    regimes_filtered = regimes_filtered.sort_values('date').reset_index(drop=True)
    
    print(f"Generating visualizations for {len(regimes_filtered)} observations...")
    
    # Generate all visualizations
    create_visualizations(regimes_filtered, diagnostics, X_std, dates, k_star, args)

if __name__ == "__main__":
    main()