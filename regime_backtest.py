# regime_backtest_simple.py
#
# Simplified backtest using crisis predictions
# Matches regime_transition_predictor.py

import argparse
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"

TRADEABLE_FACTORS = [
    'MXWO_Index', 'MXWOU_Index', 'SXXP_Index', 'NKY_Index', 
    'SHCOMP_Index', 'MXWD_Index', 'MXEA_Index',
    'SPX_Index', 'SPXT_Index',
    'SP5LVI_Index', 'M1WOMVOL_Index',
    'MXEF_Index', 'MXEFLC_Index',
    'BCOM_Index', 'SPGSCI_Index',
    'DXY_Curncy', 'EURUSD_Curncy', 'USDJPY_Curncy',
    'FXCARRSP_Index', 'FXCTG10_Index',
    'NEIXCTAT_Index', 'NEIXBTRND_Index',
    'FF_UMD', 'FF_SMB', 'FF_HML', 'FF_RMW',
    'MXWO000V_Index', 'MXUS000V_Index', 'MXEF000V_Index',
    'MXWO000G_Index', 'MXUS000G_Index', 'MXEF000G_Index',
    'BXM_Index'
]

EQUITY_PROXY = 'SPX_Index'

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    con = sqlite3.connect(DB_PATH)
    
    # Regimes
    regimes = pd.read_sql("SELECT date, regime FROM gmm_regimes", con, parse_dates=['date'])
    regimes = regimes.sort_values('date').set_index('date')
    
    # PCA factors
    pca = pd.read_sql("SELECT * FROM pca_factors_wide", con, parse_dates=['date'])
    pca = pca.sort_values('date').set_index('date')
    
    # Factor returns
    factors = pd.read_sql("""
        SELECT date, proxy, value
        FROM factors_monthly
        ORDER BY date, proxy
    """, con, parse_dates=['date'])
    
    con.close()
    
    factors_wide = factors.pivot(index='date', columns='proxy', values='value')
    tradeable = [c for c in factors_wide.columns if c in TRADEABLE_FACTORS]
    factors_wide = factors_wide[tradeable]
    
    return regimes, pca, factors_wide

# ============================================================================
# FEATURE ENGINEERING (same as predictor)
# ============================================================================

def create_simple_features(pca_factors):
    features = pd.DataFrame(index=pca_factors.index)
    
    key_cols = [
        'Equity_PC1', 'Credit_PC1', 'Interest Rates_PC1',
        'Equity Short Volatility_PC1', 'Foreign Exchange Carry_PC1'
    ]
    key_cols = [c for c in key_cols if c in pca_factors.columns]
    
    for col in key_cols:
        features[f'{col}_level'] = pca_factors[col]
        features[f'{col}_mom1'] = pca_factors[col].diff(1)
        features[f'{col}_mom3'] = pca_factors[col].diff(3)
        features[f'{col}_vol3m'] = pca_factors[col].rolling(3).std()
    
    features['factor_dispersion'] = pca_factors.std(axis=1)
    features['avg_abs_level'] = np.abs(pca_factors).mean(axis=1)
    features['max_abs_level'] = np.abs(pca_factors).max(axis=1)
    features['n_extreme'] = (np.abs(pca_factors) > 2).sum(axis=1)
    
    return features

def add_regime_features(features, regimes):
    regime_series = regimes['regime']
    
    durations = []
    current_duration = 0
    prev_regime = None
    
    for date, regime in regime_series.items():
        if regime == prev_regime:
            current_duration += 1
        else:
            current_duration = 1
        durations.append(current_duration)
        prev_regime = regime
    
    duration_series = pd.Series(durations, index=regime_series.index)
    
    features_with_regime = features.copy()
    features_with_regime['regime_duration'] = duration_series
    features_with_regime['regime_duration_sq'] = duration_series ** 2
    features_with_regime['current_regime'] = regime_series
    
    return features_with_regime

# ============================================================================
# CRISIS PREDICTOR
# ============================================================================

def train_crisis_model(features, regimes, train_end):
    regime_series = regimes['regime']
    next_regime = regime_series.shift(-1)
    
    crisis_regime = 1
    labels = (next_regime == crisis_regime).astype(int)
    labels = labels.iloc[:-1]
    
    # Align and filter to training period
    common = features.index.intersection(labels.index)
    X = features.loc[common]
    y = labels.loc[common]
    
    train_mask = X.index <= train_end
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    # Remove NaN
    valid = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train = X_train[valid]
    y_train = y_train[valid]
    
    if len(X_train) < 24:
        return None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_scaled, y_train)
    
    return model, scaler

def predict_crisis_prob(model, scaler, features, date):
    if model is None or scaler is None:
        return None
    
    X = features.loc[[date]]
    if X.isna().any().any():
        return None
    
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]
    
    return prob

# ============================================================================
# FACTOR SELECTION
# ============================================================================

def select_defensive_factors(returns, equity, train_end, n=10):
    train = returns[returns.index <= train_end]
    train_eq = equity[equity.index <= train_end]
    
    if len(train) < 24:
        return []
    
    common = train.index.intersection(train_eq.index)
    train = train.loc[common]
    train_eq = train_eq.loc[common]
    
    correlations = {}
    for col in train.columns:
        data = train[col].dropna()
        if len(data) < 12:
            continue
        
        common_dates = data.index.intersection(train_eq.index)
        if len(common_dates) < 12:
            continue
        
        corr = data.loc[common_dates].corr(train_eq.loc[common_dates])
        if not np.isnan(corr):
            correlations[col] = abs(corr)
    
    # Lowest correlation = most defensive
    sorted_factors = sorted(correlations.items(), key=lambda x: x[1])
    return [f[0] for f in sorted_factors[:n]]

def select_growth_factors(returns, train_end, n=10):
    train = returns[returns.index <= train_end]
    
    if len(train) < 24:
        return []
    
    sharpes = {}
    for col in train.columns:
        data = train[col].dropna()
        if len(data) < 12:
            continue
        
        mean_ret = data.mean() * 12
        vol = data.std() * np.sqrt(12)
        
        if vol > 0:
            sharpes[col] = mean_ret / vol
    
    # Highest Sharpe = best growth
    sorted_factors = sorted(sharpes.items(), key=lambda x: x[1], reverse=True)
    return [f[0] for f in sorted_factors[:n]]

# ============================================================================
# DYNAMIC ALLOCATION
# ============================================================================

def compute_allocation(crisis_prob, growth_factors, defensive_factors):
    """
    Allocate based on crisis probability
    
    crisis_prob < 0.3:  100% growth
    0.3 - 0.7:          Blend
    crisis_prob > 0.7:  100% defensive
    """
    if crisis_prob is None:
        crisis_prob = 0.5  # Fallback
    
    if crisis_prob < 0.3:
        defensive_weight = 0.0
    elif crisis_prob > 0.7:
        defensive_weight = 1.0
    else:
        # Linear blend
        defensive_weight = (crisis_prob - 0.3) / 0.4
    
    growth_weight = 1.0 - defensive_weight
    
    allocations = {}
    
    if len(growth_factors) > 0 and growth_weight > 0:
        for f in growth_factors:
            allocations[f] = growth_weight / len(growth_factors)
    
    if len(defensive_factors) > 0 and defensive_weight > 0:
        for f in defensive_factors:
            allocations[f] = defensive_weight / len(defensive_factors)
    
    return allocations, defensive_weight

# ============================================================================
# BACKTEST
# ============================================================================

def run_backtest(regimes, pca, factor_returns, start='2000-01-01', retrain_freq=12, n_factors=10):
    print(f"\nRunning backtest from {start}...")
    
    equity_returns = factor_returns[EQUITY_PROXY] if EQUITY_PROXY in factor_returns.columns else None
    
    # Engineer features
    print("  Engineering features...")
    features = create_simple_features(pca)
    features = add_regime_features(features, regimes)
    
    # Walk forward
    test_dates = factor_returns.loc[start:].index
    results = []
    
    model = None
    scaler = None
    growth_factors = []
    defensive_factors = []
    
    for i, date in enumerate(test_dates):
        # Retrain
        if i % retrain_freq == 0:
            train_end = date - pd.DateOffset(months=1)
            
            print(f"  Training ({i+1}/{len(test_dates)}): up to {train_end.strftime('%Y-%m')}")
            
            model, scaler = train_crisis_model(features, regimes, train_end)
            
            if equity_returns is not None:
                defensive_factors = select_defensive_factors(
                    factor_returns, equity_returns, train_end, n_factors
                )
            
            growth_factors = select_growth_factors(factor_returns, train_end, n_factors)
            
            print(f"    Growth: {len(growth_factors)} factors | Defensive: {len(defensive_factors)} factors")
        
        # Predict crisis probability
        crisis_prob = predict_crisis_prob(model, scaler, features, date)
        
        # Allocate
        allocations, defensive_weight = compute_allocation(
            crisis_prob, growth_factors, defensive_factors
        )
        
        # Calculate return
        if len(allocations) == 0:
            continue
        
        portfolio_return = 0
        for factor, weight in allocations.items():
            if factor in factor_returns.columns and date in factor_returns.index:
                ret = factor_returns.loc[date, factor]
                if not np.isnan(ret):
                    portfolio_return += weight * ret
        
        equity_return = equity_returns.loc[date] if equity_returns is not None and date in equity_returns.index else None
        
        current_regime = regimes.loc[date, 'regime'] if date in regimes.index else None
        
        results.append({
            'date': date,
            'regime': current_regime,
            'crisis_prob': crisis_prob,
            'defensive_weight': defensive_weight,
            'portfolio_return': portfolio_return,
            'equity_return': equity_return
        })
    
    return pd.DataFrame(results).set_index('date')

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_performance(results):
    print("\n" + "="*80)
    print("  BACKTEST RESULTS")
    print("="*80)
    
    n_months = len(results)
    n_years = n_months / 12
    
    # Strategy
    results['portfolio_cumret'] = (1 + results['portfolio_return']).cumprod()
    results['equity_cumret'] = (1 + results['equity_return']).cumprod()
    
    strat_total = results['portfolio_cumret'].iloc[-1] - 1
    strat_annual = (1 + strat_total) ** (1 / n_years) - 1
    strat_vol = results['portfolio_return'].std() * np.sqrt(12)
    strat_sharpe = strat_annual / strat_vol if strat_vol > 0 else 0
    
    cummax = results['portfolio_cumret'].cummax()
    dd = (results['portfolio_cumret'] - cummax) / cummax
    strat_max_dd = dd.min()
    
    # Equity - handle NaN properly
    valid_equity = results['equity_return'].dropna()
    
    if len(valid_equity) > 0:
        eq_cumret = (1 + valid_equity).cumprod()
        eq_total = eq_cumret.iloc[-1] - 1
        eq_years = len(valid_equity) / 12
        eq_annual = (1 + eq_total) ** (1 / eq_years) - 1
        eq_vol = valid_equity.std() * np.sqrt(12)
        eq_sharpe = eq_annual / eq_vol if eq_vol > 0 else 0
        
        eq_cumret_series = (1 + results['equity_return'].fillna(0)).cumprod()
        cummax_eq = eq_cumret_series.cummax()
        dd_eq = (eq_cumret_series - cummax_eq) / cummax_eq
        eq_max_dd = dd_eq.min()
    else:
        eq_annual = np.nan
        eq_vol = np.nan
        eq_sharpe = np.nan
        eq_max_dd = np.nan
    
    print(f"\nPeriod: {n_years:.1f} years")
    print(f"\nStrategy:")
    print(f"  Annual Return:  {strat_annual:7.2%}")
    print(f"  Volatility:     {strat_vol:7.2%}")
    print(f"  Sharpe:         {strat_sharpe:7.2f}")
    print(f"  Max Drawdown:   {strat_max_dd:7.2%}")
    
    print(f"\nEquity Benchmark:")
    if not np.isnan(eq_annual):
        print(f"  Annual Return:  {eq_annual:7.2%}")
        print(f"  Volatility:     {eq_vol:7.2%}")
        print(f"  Sharpe:         {eq_sharpe:7.2f}")
        print(f"  Max Drawdown:   {eq_max_dd:7.2%}")
        
        print(f"\nImprovement:")
        print(f"  Return:         {strat_annual - eq_annual:+7.2%}")
        print(f"  Volatility:     {strat_vol - eq_vol:+7.2%}")
        print(f"  Max DD:         {strat_max_dd - eq_max_dd:+7.2%}")
    else:
        print(f"  Warning: Equity data incomplete")
        print(f"  Volatility:     {eq_vol:7.2%}")
        print(f"  Max Drawdown:   {eq_max_dd:7.2%}")
        
        print(f"\nImprovement:")
        print(f"  Return:         N/A (incomplete equity data)")
        print(f"  Volatility:     {strat_vol - eq_vol:+7.2%}")
        if not np.isnan(eq_max_dd):
            print(f"  Max DD:         {strat_max_dd - eq_max_dd:+7.2%}")
    
    # By crisis probability
    print("\n" + "="*80)
    print("  PERFORMANCE BY CRISIS PROBABILITY")
    print("="*80)
    
    results['prob_bucket'] = pd.cut(
        results['crisis_prob'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low (0-30%)', 'Med (30-50%)', 'High (50-70%)', 'Very High (70-100%)']
    )
    
    bucket_stats = results.groupby('prob_bucket').agg({
        'portfolio_return': ['mean', 'count'],
        'equity_return': 'mean',
        'defensive_weight': 'mean'
    }).round(4)
    
    print("\n", bucket_stats)
    
    print("\nInterpretation:")
    print("  - Low prob: Should match or lag equity (taking growth risk)")
    print("  - High prob: Should outperform equity (defensive protection)")
    print("  - Very High: Should significantly outperform (max defense)")
    
    return results

def create_charts(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Cumulative returns
    ax = axes[0, 0]
    ax.plot(results.index, results['portfolio_cumret'], label='Strategy', linewidth=2, color='#2E86AB')
    ax.plot(results.index, results['equity_cumret'], label='Equity', linewidth=2, color='#E74C3C', alpha=0.7)
    ax.set_ylabel('Cumulative Return', fontweight='bold')
    ax.set_title('Cumulative Performance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Crisis probability
    ax = axes[0, 1]
    ax.plot(results.index, results['crisis_prob'], color='#9B59B6', linewidth=1.5)
    ax.fill_between(results.index, 0.7, 1.0, alpha=0.2, color='red', label='Very High Risk')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Crisis Probability', fontweight='bold')
    ax.set_title('Predicted Crisis Risk', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 3. Defensive allocation
    ax = axes[1, 0]
    ax.fill_between(results.index, 0, results['defensive_weight'], alpha=0.6, color='#3498DB', label='Defensive')
    ax.fill_between(results.index, results['defensive_weight'], 1, alpha=0.6, color='#2ECC71', label='Growth')
    ax.set_ylabel('Allocation', fontweight='bold')
    ax.set_title('Dynamic Allocation', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 4. Returns by bucket
    ax = axes[1, 1]
    
    bucket_returns = results.groupby('prob_bucket').agg({
        'portfolio_return': 'mean',
        'equity_return': 'mean'
    }) * 12 * 100
    
    x = range(len(bucket_returns))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], bucket_returns['portfolio_return'],
           width, label='Strategy', alpha=0.7, color='#2E86AB')
    ax.bar([i + width/2 for i in x], bucket_returns['equity_return'],
           width, label='Equity', alpha=0.7, color='#E74C3C')
    
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_returns.index, rotation=45, ha='right')
    ax.set_ylabel('Annualized Return (%)', fontweight='bold')
    ax.set_title('Returns by Crisis Probability', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=1)
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='2000-01-01')
    parser.add_argument('--retrain_freq', type=int, default=12)
    parser.add_argument('--n_factors', type=int, default=10)
    args = parser.parse_args()
    
    print("="*80)
    print("  LEARNED CRISIS PROBABILITY BACKTEST")
    print("  Dynamic allocation based on predicted crisis risk")
    print("="*80)
    
    print("\nLoading data...")
    regimes, pca, factor_returns = load_data()
    
    print(f"  Regimes: {regimes.shape}")
    print(f"  PCA: {pca.shape}")
    print(f"  Factors: {factor_returns.shape}")
    
    results = run_backtest(
        regimes, pca, factor_returns,
        start=args.start,
        retrain_freq=args.retrain_freq,
        n_factors=args.n_factors
    )
    
    if len(results) == 0:
        print("\nNo results.")
        return
    
    results = analyze_performance(results)
    
    print("\nGenerating charts...")
    fig = create_charts(results)
    
    import os
    output_dir = "crisis_backtest_results"
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(f"{output_dir}/crisis_backtest.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    results.to_csv(f"{output_dir}/crisis_backtest_detail.csv")
    
    print(f"\n{'='*80}")
    print("  COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to {output_dir}/")
    print("\nThe strategy dynamically adjusts allocation based on LEARNED")
    print("crisis probability, not fixed historical transition rates.")
    print()

if __name__ == "__main__":
    main()