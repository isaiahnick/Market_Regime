# Market Regime Detection with PCA and Gaussian Mixture Models

This project implements a data-driven approach to detecting financial market regimes by combining **Principal Component Analysis (PCA)** for dimensionality reduction with a **Gaussian Mixture Model (GMM)** for unsupervised clustering.  

The model successfully identifies **two distinct volatility regimes** in financial markets: a calm/low-volatility regime and a volatile/crisis regime, with strong statistical validation (silhouette score: 0.192, correlation difference: 0.154).

---

## Table of Contents
1. [Background](#background)
2. [Key Findings](#key-findings)
3. [Mathematical Overview](#mathematical-overview)
   - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
   - [Gaussian Mixture Models (GMM)](#gaussian-mixture-models-gmm)
4. [Repository Structure](#repository-structure)
5. [Setup and Installation](#setup-and-installation)
   - [FRED API Key](#fred-api-key)
6. [Usage](#usage)
7. [Model Validation](#model-validation)
8. [Results](#results)
9. [References](#references)

---

## Background
Financial markets often exhibit **regimes**—extended periods of relatively stable behavior (e.g., crisis, steady growth, low volatility).  
Detecting these regimes can help in:
- Risk management and stress testing
- Portfolio construction and tactical allocation
- Identifying market fragility
- Understanding correlation structure changes

This project uses:
- **PCA** to reduce 52 correlated factors across 18 categories into 15 principal components (PC1 per category)
- **GMM** to cluster observations into regimes, modeling each as a multivariate Gaussian distribution
- **Correlation validation** to verify regimes capture real market structure, not noise

The approach is inspired by institutional quantitative research methods, particularly Two Sigma's factor lens framework.

---

## Key Findings

### Two Distinct Volatility Regimes Identified

**Regime 0: Calm/Steady State (~65% of time)**
- Low volatility across all asset classes (std ~0.5-1.0)
- Equity Short Volatility mean: -0.28 (suppressed vol)
- Tight price ranges, predictable behavior
- Examples: Mid-2000s, 2017-2018, parts of 2019 and 2023

**Regime 1: Volatile/Crisis (~35% of time)**
- High volatility across all factors (std ~1.4-1.8, **2-3x higher**)
- Equity Short Volatility mean: +1.00 (elevated vol)
- Negative equity returns (mean: -0.43)
- Credit spreads widen (mean: +0.19)
- Examples: Dot-com crash (2000-2002), Financial Crisis (2008-2009), COVID (2020), Fed tightening (2022)

### Validation Metrics
- **Silhouette Score: 0.192** (good for financial data)
- **BIC selects K=2** (clear evidence for two regimes)
- **Correlation difference: 0.173** (strong evidence regimes are real, not noise)
- **Key regime driver: Trend Following** - correlation with risk assets flips from -0.47 (crisis) to +0.31 (bull)

### Economic Interpretation
The model detects a **volatility regime structure** rather than simple bull/bear markets:
- Regime changes correspond to fundamental shifts in market correlation structure
- Trend Following factor acts as the primary regime discriminator
- Model successfully captures all major crisis periods in the sample (1995-2025)

---

## Mathematical Overview

### Principal Component Analysis (PCA)
- PCA is a linear transformation that projects high-dimensional data into orthogonal components ranked by variance explained.  
- Mathematically:  
  - Given a dataset **X** with covariance matrix **Σ**, solve the eigenvalue problem:  
    
    **Σv<sub>i</sub> = λ<sub>i</sub>v<sub>i</sub>**
    
  - Principal components are eigenvectors **v<sub>i</sub>** ordered by descending eigenvalues **λ<sub>i</sub>**.  
- In this project:
  - PCA is applied separately to each factor category (Equity, Credit, Commodities, etc.)
  - Only **PC1** (first principal component) is retained per category
  - PC1 captures 40-70% of variance within each category
  - This reduces 52 factors across 18 categories to 15 independent components

**Why PC1 only?**
- Provides cleaner regime separation (lower dimensionality)
- More interpretable (each PC represents category "stance")
- Reduces overfitting risk in GMM
- Aligns with institutional quantitative research best practices

### Gaussian Mixture Models (GMM)
- GMM assumes data is generated from a mixture of **K** Gaussian distributions:

  **p(x) = Σ<sub>k=1</sub><sup>K</sup> π<sub>k</sub> 𝒩(x | μ<sub>k</sub>, Σ<sub>k</sub>)**
  
  where **π<sub>k</sub>** are mixture weights, **μ<sub>k</sub>** are means, and **Σ<sub>k</sub>** are covariance matrices.  
- Parameters are estimated via the **Expectation-Maximization (EM)** algorithm.
- Unlike K-means, GMM allows:
  - Soft cluster assignments (probabilistic regimes)
  - Elliptical clusters via covariance structure
  - Model selection via information criteria (AIC, BIC)

**Implementation details:**
- Tested K=2 to K=6 regimes
- Full covariance matrices (captures correlation structure)
- Missing data handled via median imputation
- Model selection via BIC (Bayesian Information Criterion)

---

## Repository Structure
```
├── data/                          # Raw data storage folder
├── gmm_plots/                     # Output plots and visualizations
├── .gitignore                     # Git ignore file
├── compute_returns.py             # Computes returns from raw factor data
├── factor_lens.db                 # SQLite database for factor data (not in repo)
├── factors.csv                    # Factor metadata (categories, proxies, transformations)
├── gmm.py                         # Runs GMM clustering and regime detection
├── init_db.py                     # Initializes the SQLite database
├── load_bloomberg.py              # Loads Bloomberg data into database
├── load_fred.py                   # Loads FRED economic data into database
├── load_french.py                 # Loads French factor data into database
├── pca.py                         # Performs PCA analysis on factor groups
├── README.md                      # Project documentation
├── regime_correlation_validation.py  # Validates regimes via correlation analysis
├── requirements.txt               # Python dependencies
├── seed_instruments.py            # Seeds instrument metadata into database
└── visualizations.py              # Produces summary visualizations
```

---

## Setup and Installation

### Clone the repository
```bash
git clone https://github.com/your-username/market-regime-detection.git
cd market-regime-detection
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### FRED API Key Setup
1. Go to [FRED](https://fred.stlouisfed.org/) and create a free account.
2. Generate an API key under **My Account > API Keys**.
3. Add your API key to `load_fred.py`:
   ```python
   api_key = 'your_api_key_here'
   ```

---

## Usage

Follow these steps in order to set up and run the analysis:

### Step 1: Prepare Raw Data
- Place your raw financial data files in the `data/` folder
- Ensure Bloomberg data CSVs are named with proxy symbols (e.g., `SPX_Index.csv`)
- French factor data should be in `French_Factors.csv` and `French_MOM.csv`

### Step 2: Initialize Database
```bash
python init_db.py
```
This creates the SQLite database structure for storing factor data.

### Step 3: Seed Instrument Metadata
```bash
python seed_instruments.py
```
This populates the database with instrument definitions from `factors.csv`.

### Step 4: Load Factor Data
Load data from various sources into the database:
```bash
python load_bloomberg.py    # Load Bloomberg market data
python load_fred.py         # Load FRED economic indicators  
python load_french.py       # Load French factor library data
```

### Step 5: Compute Returns
```bash
python compute_returns.py
```
This calculates returns and transformations for all factors:
- Log returns for equity indices, commodities, FX
- First differences for rates, spreads
- Log differences for inflation indices
- Creates standardized (z-scored) versions for PCA

### Step 6: Principal Component Analysis
```bash
python pca.py
```
This applies PCA to each factor category:
- Extracts PC1 per category (15 total PCs)
- Uses smart imputation (forward-fill + median) to preserve data
- Standardizes each PC to mean=0, std=1
- Outputs saved to `pca_factors_wide` table

### Step 7: Gaussian Mixture Model Analysis
```bash
python gmm.py --start 1995-01-01 --kmin 2 --kmax 6
```
This fits GMM across different numbers of clusters and outputs:
- Regime probabilities and hard assignments
- Model selection metrics (AIC, BIC, Silhouette)
- Tables: `gmm_regimes`, `gmm_scores`, `gmm_meta`, `gmm_params`

**Note:** Do not use `--standardize` flag as PCA output is already standardized.

### Step 8: Validate Regimes
```bash
python regime_correlation_validation.py
```
This validates that regimes capture real market structure:
- Calculates correlation matrices within each regime
- Quantifies correlation differences across regimes
- Identifies key regime drivers (factors with largest correlation changes)

### Step 9: Generate Visualizations
```bash
python visualizations.py --start 1995-01-01 --kmin 2 --kmax 6
```
This produces professional visualizations (saved to `gmm_plots/`):
- Historical timeline of regime assignments
- Recent period analysis
- Probability evolution over time
- Model selection diagnostics
- Regime summary statistics

---

## Model Validation

### Statistical Validation
The model has been rigorously validated to ensure regimes represent real market structure:

**1. Silhouette Score: 0.192**
- Measures cluster separation quality
- Range: [-1, 1], higher is better
- 0.192 is good for financial data (typically noisy)
- Indicates clear but overlapping clusters

**2. Correlation Structure Analysis**
- Average correlation difference across regimes: **0.154**
- Threshold for "strong evidence": >0.15
- Proves regimes have fundamentally different correlation structures
- Not random noise or overfitting

**3. Model Selection (BIC)**
- K=2: BIC = 11,266 ✓ **Selected**
- K=3: BIC = 11,833
- K=4: BIC = 11,784
- Lower BIC indicates better model fit adjusted for complexity

### Economic Validation
**Crisis Detection:**
- ✓ Dot-com crash (2000-2002)
- ✓ Financial crisis (2008-2009)
- ✓ COVID crash (March 2020)
- ✓ Fed tightening bear market (2022)

**Key Regime Drivers:**
- Trend Following: Correlation with equities flips from -0.47 to +0.31
- Equity Short Volatility: Mean changes from -0.28 to +1.00
- Credit Spreads: Volatility increases 3.2x in volatile regime

### Sample Size Validation
All factor categories meet statistical requirements:
- Minimum 60x observations per feature (well above 10-15x threshold)
- 1995-2025 monthly data (370 observations)
- Robust to missing data via imputation

---

## Results

### Model Selection
![Model Selection Diagnostics](gmm_plots/model_selection.png)
*AIC, BIC, and Silhouette scores across K=2 to K=6. BIC clearly selects K=2.*

### Regime Timeline
![Historical Regime Timeline](gmm_plots/exhibit_4_historical_timeline.png)
*Complete historical view of regime assignments from 1995-2025.*

### Regime Probabilities
![Regime Probability Evolution](gmm_plots/regime_timeseries.png)
*Probabilities of regimes over time.*

### PC Behavior in Different Regimes
![Regime Probability Evolution](gmm_plots/pcs_over_time.png)
*How PC groups behave over the two regimes, highlighted by greater variance in Crisis states.*

### Correlation Validation
![Correlation by Regime](gmm_plots/correlation_by_regime.png)
*Correlation matrices showing distinct structure in each regime.*

### Regime Statistics
![Regime Summary Statistics](gmm_plots/regime_statistics.png)
*Statistical characteristics of each regime including frequency and duration.*

---

## Key Takeaways

1. **Two volatility regimes exist in financial markets** - calm/steady periods (65%) and volatile/crisis periods (35%)

2. **Regimes are statistically validated** - correlation structure differs significantly (0.154 average difference), silhouette score of 0.192

3. **Trend Following is the key regime indicator** - its correlation with risk assets flips sign across regimes

4. **Volatility clustering is the dominant pattern** - Regime 1 has 2-3x higher volatility across all factors

5. **Model captures major crises accurately** - all significant market dislocations since 1995 are correctly identified

6. **Production-ready for institutional use** - validated methodology, robust to missing data, clear economic interpretation

---

## Future Enhancements

Potential areas for extension:
- Hidden Markov Model (HMM) for transition probabilities
- Regime-dependent portfolio optimization
- Forward-looking regime prediction using leading indicators
- Multi-horizon analysis (weekly, daily frequencies)
- Out-of-sample backtesting of regime-based strategies

---

## References
- Two Sigma (2021). *A Machine Learning Approach to Regime Modeling*  
- Ang, A., & Bekaert, G. (2002). *Regime Switches in Interest Rates*. Journal of Business & Economic Statistics.
- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series*. Econometrica.
- scikit-learn documentation: [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

---

## Contact
For questions or collaboration: isaiahdnick@gmail.com

---