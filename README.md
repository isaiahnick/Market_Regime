# Market Regime Detection with PCA and Gaussian Mixture Models

This project implements a data-driven approach to detecting financial market regimes by combining **Principal Component Analysis (PCA)** for dimensionality reduction with a **Gaussian Mixture Model (GMM)** for unsupervised clustering.  

The goal is to identify persistent market conditions (regimes) using historical financial and macroeconomic factors, inspired by approaches such as Two Sigma’s factor lens research.

---

## Table of Contents
1. [Background](#background)
2. [Mathematical Overview](#mathematical-overview)
   - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
   - [Gaussian Mixture Models (GMM)](#gaussian-mixture-models-gmm)
3. [Repository Structure](#repository-structure)
4. [Setup and Installation](#setup-and-installation)
   - [FRED API Key](#fred-api-key)
5. [Usage](#usage)
6. [Results](#results)
7. [References](#references)

---

## Background
Financial markets often exhibit **regimes**—extended periods of relatively stable behavior (e.g., crisis, steady growth, inflationary periods).  
Detecting these regimes can help in:
- Risk management and stress testing
- Portfolio construction and tactical allocation
- Identifying market fragility

This project uses:
- **PCA** to reduce correlated factor data into orthogonal components.
- **GMM** to cluster observations into regimes, each modeled as a multivariate Gaussian distribution.

---

## Mathematical Overview

### Principal Component Analysis (PCA)
- PCA is a linear transformation that projects high-dimensional data into orthogonal components ranked by variance explained.  
- Mathematically:  
  - Given a dataset \( X \) with covariance matrix \( \Sigma \), solve the eigenvalue problem:  
    \[
    \Sigma v_i = \lambda_i v_i
    \]
  - Principal components are eigenvectors \( v_i \) ordered by descending eigenvalues \( \lambda_i \).  
- In this project, PCA reduces groups of factors (e.g., Equity, Credit, Commodities) into representative components.

### Gaussian Mixture Models (GMM)
- GMM assumes data is generated from a mixture of \( K \) Gaussian distributions:
  \[
  p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)
  \]
  where \( \pi_k \) are mixture weights, \( \mu_k \) are means, and \( \Sigma_k \) are covariance matrices.  
- Parameters are estimated via the **Expectation-Maximization (EM)** algorithm.
- Unlike K-means, GMM allows:
  - Soft cluster assignments (probabilistic regimes)
  - Elliptical clusters via covariance structure

---

## Repository Structure
```
├── compute_returns.py   # Computes returns from raw factor data
├── gmm.py               # Runs PCA + GMM clustering and regime detection
├── factors.csv          # Factor metadata (categories, proxies, names, data sources)
├── README.md            # Project documentation
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

### FRED API Key
1. Go to [FRED](https://fred.stlouisfed.org/) and create a free account.
2. Generate an API key under **My Account > API Keys**.
3. Export it in your shell (or add to `.bashrc`/`.zshrc`):
   ```bash
   export FRED_API_KEY="your_api_key_here"
   ```
4. The scripts will automatically use this key when downloading factor data.

---

## Usage

### Step 1: Compute Returns
```bash
python compute_returns.py
```

### Step 2: Run PCA + GMM
```bash
python pca.py
python gmm.py
```

- The script will:
  - Apply PCA to factor groups
  - Fit GMM across candidate numbers of clusters
  - Output regime probabilities and diagnostic metrics (AIC, BIC, Silhouette)

---

## Results
![GMM Cluster Selection Analysis](gmm_plots/model_selection.png)
![Regime Evolution](gmm_plots/regime_timeseries.png)
![Regime Summary Statistics](gmm_plots/regime_statistics.png)


Example:
```markdown
![Regime Probability Over Time](results/regime_probs.jpg)
![Cluster Factor Means](results/means.jpg)
![Cluster Factor Volatility](results/vols.jpg)
```

---

## References
- Two Sigma (2021). *A Machine Learning Approach to Regime Modeling*  
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.  
- scikit-learn documentation: [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

---
