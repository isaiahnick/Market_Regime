# regime_transition_predictor.py
#
# Simplified transition predictor that focuses on what matters
# Predicts: "Will we be in crisis regime next month?"
#
# Key improvements:
# 1. Predict crisis vs not-crisis (more actionable)
# 2. Fewer, more predictive features
# 3. Better probability calibration
# 4. Handles class imbalance properly

import argparse
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "/Users/isaiahnick/Desktop/Market Regime PCA/factor_lens.db"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_gmm_regimes():
    con = sqlite3.connect(DB_PATH)
    regimes = pd.read_sql("SELECT date, regime FROM gmm_regimes", con, parse_dates=['date'])
    con.close()
    regimes = regimes.sort_values('date').set_index('date')
    return regimes

def load_pca_factors():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM pca_factors_wide", con, parse_dates=['date'])
    con.close()
    df = df.sort_values('date').set_index('date')
    return df

# ============================================================================
# SIMPLIFIED FEATURE ENGINEERING
# ============================================================================

def create_simple_features(pca_factors):
    """
    Create ONLY the most predictive features
    Based on importance analysis: regime duration, dispersion, levels, momentum
    """
    features = pd.DataFrame(index=pca_factors.index)
    
    # Key factors that matter most
    key_cols = [
        'Equity_PC1', 'Credit_PC1', 'Interest Rates_PC1',
        'Equity Short Volatility_PC1', 'Foreign Exchange Carry_PC1'
    ]
    
    key_cols = [c for c in key_cols if c in pca_factors.columns]
    
    # 1. Current levels (baseline)
    for col in key_cols:
        features[f'{col}_level'] = pca_factors[col]
    
    # 2. Momentum (1 and 3 month)
    for col in key_cols:
        features[f'{col}_mom1'] = pca_factors[col].diff(1)
        features[f'{col}_mom3'] = pca_factors[col].diff(3)
    
    # 3. Volatility (3-month rolling)
    for col in key_cols:
        features[f'{col}_vol3m'] = pca_factors[col].rolling(3).std()
    
    # 4. Cross-sectional (VERY important)
    features['factor_dispersion'] = pca_factors.std(axis=1)
    features['avg_abs_level'] = np.abs(pca_factors).mean(axis=1)
    features['max_abs_level'] = np.abs(pca_factors).max(axis=1)
    
    # 5. Extreme moves
    features['n_extreme'] = (np.abs(pca_factors) > 2).sum(axis=1)
    
    return features

def add_regime_features(features, regimes):
    """Add regime-specific features"""
    regime_series = regimes['regime']
    
    # Regime duration
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
# IMPROVED MODEL TRAINING
# ============================================================================

def train_crisis_predictor(X_train, y_train):
    """
    Train gradient boosting model
    Better for imbalanced data and probability calibration
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Gradient Boosting handles probabilities better than Random Forest
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,  # Shallow trees prevent overfit
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_scaled, y_train)
    
    return model, scaler

# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def walk_forward_predict(features, labels, regimes, 
                        train_start='1995-01-01',
                        test_start='1998-01-01',
                        retrain_freq=6):
    """Walk-forward with periodic retraining"""
    results = []
    
    # Align data
    common_dates = features.index.intersection(labels.index).intersection(regimes.index)
    features = features.loc[common_dates]
    labels = labels.loc[common_dates]
    regimes = regimes.loc[common_dates]
    
    test_dates = features.loc[test_start:].index
    
    print(f"\nWalk-forward validation from {test_start}")
    print(f"  Test periods: {len(test_dates)}")
    print(f"  Retrain every: {retrain_freq} months")
    
    model = None
    scaler = None
    
    for i, test_date in enumerate(test_dates):
        # Retrain periodically
        if i % retrain_freq == 0 or model is None:
            train_dates = features.index[(features.index >= train_start) & (features.index < test_date)]
            
            if len(train_dates) < 24:
                continue
            
            X_train = features.loc[train_dates]
            y_train = labels.loc[train_dates]
            
            # Remove NaN
            valid = ~(X_train.isna().any(axis=1) | y_train.isna())
            X_train = X_train[valid]
            y_train = y_train[valid]
            
            if len(X_train) < 24:
                continue
            
            print(f"  Training model ({i+1}/{len(test_dates)}): {len(X_train)} obs, up to {train_dates[-1].strftime('%Y-%m')}")
            model, scaler = train_crisis_predictor(X_train, y_train)
        
        # Predict
        X_test = features.loc[[test_date]]
        
        if X_test.isna().any().any():
            continue
        
        X_scaled = scaler.transform(X_test)
        crisis_prob = model.predict_proba(X_scaled)[0, 1]
        
        actual = labels.loc[test_date]
        current_regime = regimes.loc[test_date, 'regime']
        
        results.append({
            'date': test_date,
            'crisis_prob': crisis_prob,
            'actual_crisis': actual,
            'current_regime': current_regime
        })
    
    return pd.DataFrame(results).set_index('date')

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results(results):
    """Comprehensive analysis"""
    print("\n" + "="*80)
    print("  CRISIS PREDICTION RESULTS")
    print("="*80)
    
    # AUC
    auc = roc_auc_score(results['actual_crisis'], results['crisis_prob'])
    print(f"\nAUC-ROC: {auc:.3f}")
    
    if auc > 0.70:
        print("  ✓ EXCELLENT: Strong predictive power")
    elif auc > 0.65:
        print("  ✓ GOOD: Meaningful predictions")
    elif auc > 0.60:
        print("  ~ MODERATE: Some signal")
    else:
        print("  ✗ WEAK: Limited predictive ability")
    
    # Calibration by decile
    print("\n" + "="*80)
    print("  PROBABILITY CALIBRATION")
    print("="*80)
    
    results['prob_decile'] = pd.qcut(results['crisis_prob'], q=5, labels=False, duplicates='drop')
    
    calibration = results.groupby('prob_decile').agg({
        'crisis_prob': ['mean', 'min', 'max'],
        'actual_crisis': ['sum', 'count', 'mean']
    }).round(3)
    
    print("\nPredicted Probability vs Actual Crisis Rate:")
    print("\nDecile  Pred_Prob_Range  Actual_Rate  Count")
    print("-" * 50)
    
    for decile in calibration.index:
        pred_min = calibration.loc[decile, ('crisis_prob', 'min')]
        pred_max = calibration.loc[decile, ('crisis_prob', 'max')]
        actual_rate = calibration.loc[decile, ('actual_crisis', 'mean')]
        count = int(calibration.loc[decile, ('actual_crisis', 'count')])
        
        print(f"  {decile}     {pred_min:.2f}-{pred_max:.2f}        {actual_rate:.1%}      {count:3d}")
    
    # Thresholds
    print("\n" + "="*80)
    print("  DECISION THRESHOLDS")
    print("="*80)
    
    for thresh in [0.3, 0.5, 0.7]:
        predicted = (results['crisis_prob'] >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(results['actual_crisis'], predicted).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nThreshold: {thresh:.0%}")
        print(f"  Precision: {precision:.1%}  (accuracy when predicting crisis)")
        print(f"  Recall:    {recall:.1%}  (% of actual crises caught)")
        print(f"  TP: {tp:3d}  FP: {fp:3d}  TN: {tn:3d}  FN: {fn:3d}")
    
    return auc, calibration

def create_charts(results, feature_importance=None):
    """Create visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Probability over time
    ax = axes[0, 0]
    ax.plot(results.index, results['crisis_prob'], color='#9B59B6', linewidth=1.5, alpha=0.8)
    
    # Mark actual crises
    crises = results[results['actual_crisis'] == 1]
    ax.scatter(crises.index, crises['crisis_prob'], color='red', s=100, 
              alpha=0.7, marker='x', linewidth=2, label='Actual Crisis', zorder=5)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(results.index, 0.7, 1.0, alpha=0.2, color='red', label='High Risk')
    ax.set_ylabel('Crisis Probability', fontweight='bold')
    ax.set_title('Predicted Crisis Probability Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 2. Calibration plot
    ax = axes[0, 1]
    
    deciles = results.groupby('prob_decile').agg({
        'crisis_prob': 'mean',
        'actual_crisis': 'mean'
    }).reset_index()
    
    ax.scatter(deciles['crisis_prob'], deciles['actual_crisis'], s=100, alpha=0.7, color='#2E86AB')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Calibration')
    ax.set_xlabel('Predicted Probability', fontweight='bold')
    ax.set_ylabel('Actual Crisis Rate', fontweight='bold')
    ax.set_title('Calibration: Predicted vs Actual', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Distribution by outcome
    ax = axes[1, 0]
    
    crisis_probs = results[results['actual_crisis'] == 1]['crisis_prob']
    no_crisis_probs = results[results['actual_crisis'] == 0]['crisis_prob']
    
    ax.hist(no_crisis_probs, bins=30, alpha=0.6, color='green', label='No Crisis', density=True)
    ax.hist(crisis_probs, bins=30, alpha=0.6, color='red', label='Crisis', density=True)
    ax.set_xlabel('Predicted Probability', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Probability Distribution by Outcome', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Feature importance
    ax = axes[1, 1]
    
    if feature_importance is not None:
        top_features = feature_importance.head(15)
        y_pos = range(len(top_features))
        ax.barh(y_pos, top_features['importance'], alpha=0.7, color='#3498DB')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'], fontsize=8)
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title('Top Predictive Features', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_start', type=str, default='1995-01-01')
    parser.add_argument('--test_start', type=str, default='2000-01-01')
    parser.add_argument('--retrain_freq', type=int, default=12)
    args = parser.parse_args()
    
    print("="*80)
    print("  SIMPLIFIED CRISIS PREDICTION MODEL")
    print("  Predicts: Will we be in crisis regime next month?")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    regimes = load_gmm_regimes()
    pca_factors = load_pca_factors()
    
    print(f"  Regimes: {regimes.shape}")
    print(f"  PCA factors: {pca_factors.shape}")
    
    # Create labels: 1 if next month is crisis
    print("\nCreating labels...")
    regime_series = regimes['regime']
    next_regime = regime_series.shift(-1)
    
    crisis_regime = 1
    labels = (next_regime == crisis_regime).astype(int)
    labels = labels.iloc[:-1]
    
    crisis_count = labels.sum()
    print(f"  Crisis periods: {crisis_count} / {len(labels)} ({crisis_count/len(labels):.1%})")
    
    # Engineer features
    print("\nEngineering features...")
    features = create_simple_features(pca_factors)
    features = add_regime_features(features, regimes)
    
    print(f"  Features created: {features.shape[1]}")
    
    # Walk-forward
    print("\nRunning walk-forward validation...")
    results = walk_forward_predict(
        features, labels, regimes,
        train_start=args.train_start,
        test_start=args.test_start,
        retrain_freq=args.retrain_freq
    )
    
    if len(results) == 0:
        print("\nNo results generated.")
        return
    
    print(f"\n  Generated {len(results)} predictions")
    
    # Analyze
    auc, calibration = analyze_results(results)
    
    # Train final model for feature importance
    print("\nTraining final model...")
    common_dates = features.index.intersection(labels.index)
    X = features.loc[common_dates]
    y = labels.loc[common_dates]
    
    valid = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid]
    y = y[valid]
    
    final_model, final_scaler = train_crisis_predictor(X, y)
    
    # Feature importance
    importances = final_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Predictive Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Visualize
    print("\nCreating visualizations...")
    fig = create_charts(results, feature_importance)
    
    import os
    output_dir = "crisis_prediction_results"
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(f"{output_dir}/crisis_prediction_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    results.to_csv(f"{output_dir}/crisis_predictions.csv")
    feature_importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    print(f"\n{'='*80}")
    print("  COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to {output_dir}/")
    print("\nKey Metrics:")
    print(f"  AUC-ROC: {auc:.3f}")
    print(f"  Total predictions: {len(results)}")
    print(f"  Actual crises: {results['actual_crisis'].sum()}")
    
    # Actionable summary
    print("\n" + "="*80)
    print("  HOW TO USE THESE PREDICTIONS")
    print("="*80)
    
    high_risk_months = (results['crisis_prob'] > 0.5).sum()
    print(f"\nHigh risk periods (prob > 50%): {high_risk_months} months")
    print(f"  → Shift to defensive allocation")
    
    very_high = (results['crisis_prob'] > 0.7).sum()
    print(f"\nVery high risk (prob > 70%): {very_high} months")
    print(f"  → Maximum defensive position")
    
    print()

if __name__ == "__main__":
    main()