# Project Outputs Guide

This guide explains the metrics, charts, tables, and files produced by the Portfolio Risk Analysis and Investment Recommendation System. The Streamlit app and command-line runner share the same analysis pipeline.

> This is a historical educational analysis, not personalized investment advice or a forecast. Synthetic demonstration prices are artificial and must not be read as actual market history.

## Data and run behavior

Choose Yahoo Finance for adjusted historical daily prices (requires internet and Yahoo availability), or Synthetic demonstration for deterministic artificial prices (seed 42, works offline). The end date is exclusive. Prices become simple daily percentage returns. The loader forward-fills missing prices and drops rows that remain incomplete across retained tickers; feature NaNs are mean-imputed before standardization.

## Dashboard

- **Stocks:** number of tickers in this run.
- **Period:** first and last available price dates; these can differ from requested dates.
- **PCA components / variance retained:** number of components and cumulative standardized feature variance; PCA keeps enough components to reach 90%.
- **Clusters:** selected K-means group count, automatic by silhouette or manually chosen.
- **Stock universe:** one row per ticker with annualized return, annualized volatility, Sharpe-like ratio, maximum drawdown, relative risk label, and cluster.
- **Source / provenance ID:** selected data source and identifier for the run metadata.

Cluster numbers are arbitrary IDs; a smaller number is not inherently safer or better.

## Risk Analysis

### Annualized return and volatility chart

Each point represents a ticker. The horizontal axis is annualized volatility (daily sample standard deviation × √252); farther right means more historical variability. The vertical axis is annualized arithmetic mean return (daily mean × 252); it is not a forecast. Color is the relative volatility label. The chart does not account for valuation, liquidity, taxes, or future uncertainty.

### Risk category distribution

Counts stocks in Low, Medium, and High relative volatility terciles. The labels are recalculated for the selected universe, so adding or removing tickers can change them. They are not absolute safety ratings.

### Per-stock metrics

| Metric | Interpretation |
| --- | --- |
| Annualized return (`mean_annual_return`) | Arithmetic average daily return × 252. |
| Annualized volatility (`annual_volatility`) | Sample standard deviation of daily returns × √252. |
| Sharpe-like ratio (`sharpe_like_ratio`) | Annualized arithmetic return divided by annualized volatility; no risk-free rate is subtracted. |
| Maximum drawdown (`max_drawdown`) | Largest peak-to-trough fall in the compounded wealth path; usually zero or negative. |
| Skewness (`skewness`) | Asymmetry of daily returns; sign indicates the longer tail's direction. |
| Kurtosis (`kurtosis`) | Pandas sample excess kurtosis; positive values indicate heavier tails than a normal distribution under this convention. |
| 5th percentile daily return (`var_95_daily`) | Historical 5th percentile daily return, typically negative. It is not a forecast or a positive loss amount. |

### Rolling risk table

Shows the same features for trailing 252-observation (one-year), 756-observation (three-year), and full available windows. A fixed window appears only when enough observations exist. Labels are recalculated within each window. “Stocks retaining the same category” is the share with matching one- and three-year labels, when both are available; it does not measure return stability.

## PCA & Clusters

- **Explained variance chart:** bars show the share explained by each PC; the line shows cumulative variance. The retained PCs reach the 90% target.
- **PCA stock scatter:** each ticker is located by PC1 and PC2 scores and colored by cluster. Nearby points have similar combinations of input features in this projection; later components may contain additional differences.
- **Feature loadings:** rows are the seven risk/return features and columns are PCs. Absolute loading size indicates contribution; sign indicates direction. A whole component's sign can flip without changing the solution, so interpret patterns rather than “good” signs.
- **Cluster evaluation:** silhouette is generally better when higher; Davies–Bouldin is generally better when lower. Automatic K-means tests k=2 through at most 6 and chooses the highest silhouette. Adjusted Rand compares assignments on an early 70% portion of ticker-ordered rows against full-sample assignments; it is not a time-series backtest. Hierarchical silhouette compares agglomerative clustering with the chosen cluster count. These are geometric diagnostics, not measures of investment quality.
- **Ticker component table:** ticker, cluster ID, and retained PCA scores (PC1, PC2, etc.). Scores are not prices or returns.

## Portfolio Examples

Separate candidate tables are shown for Low, Medium, and High relative risk groups. Candidates are selected using historical Sharpe-like ratios while favoring distinct clusters. The size control limits the candidates used in the displayed example.

- **Equal:** equal initial weights, capped and redistributed if needed.
- **Inverse volatility:** weights proportional to inverse historical volatility, with cap redistribution.
- **Minimum variance:** long-only covariance optimization based on historical returns, bounded by the per-stock maximum.

The weight column is the portfolio fraction and should sum to 1. Tables also show risk label, cluster, historical return/risk metrics, drawdown, and 5th percentile return. The examples use the full selected history, are not point-in-time recommendations, and omit taxes, liquidity, reliable sector constraints, and personal suitability. Minimum-variance estimates are not guaranteed to reduce future risk.

## Benchmark & Backtest

This page fetches NIFTY 50 (`^NSEI`) from Yahoo Finance even if the stock run used synthetic data. The backtest estimates weights using only the preceding training window, then applies them to the next rebalance interval. Transaction cost is charged on rebalance turnover; the initial move from zero holdings incurs cost.

| Summary output | Interpretation |
| --- | --- |
| Observations | Aligned portfolio and benchmark daily returns. |
| Portfolio / benchmark annualized return | Compounded daily return expressed as a one-year rate. |
| Excess return annualized | Mean daily portfolio-minus-benchmark return × 252. |
| Beta | Portfolio/benchmark sample covariance divided by benchmark sample variance; historical co-movement. |
| Jensen's alpha annualized | Annualized CAPM intercept using the configured risk-free rate (currently 0%). |
| Tracking error | Standard deviation of daily active returns × √252. |
| Information ratio | Annualized arithmetic active return divided by tracking error. |
| Sharpe ratio | Annualized excess arithmetic return divided by annualized volatility. |
| Annualized volatility | Portfolio daily sample standard deviation × √252. |
| Maximum drawdown | Largest peak-to-trough loss in backtest portfolio equity. |
| CAGR | Compounded growth rate using observed return count / 252 years. |
| Risk-free rate annual | Assumption used by Sharpe and alpha; currently 0%. |

The **Growth of 1 unit** chart compounds portfolio and benchmark returns from 1. **Drawdown** shows portfolio decline from its prior equity high. **Average one-way rebalance turnover** currently averages the full sum of absolute weight changes, which is two-sided turnover by common convention. Backtest outcomes depend on window, rebalance frequency, method, cost, data, and benchmark alignment; they do not predict future performance.

## Model Evaluation

Logistic Regression and Random Forest classify Low/Medium/High labels derived from volatility. Annualized volatility itself is removed from model inputs to reduce direct target leakage. Evaluation runs only if there are enough stocks and at least three examples per class.

- **Test accuracy:** fraction of holdout labels correct.
- **Balanced accuracy:** average recall across classes, weighting classes equally.
- **Weighted precision / recall / F1:** class metrics averaged by class support. In single-label classification, weighted recall equals accuracy.
- **CV accuracy mean / std:** average and standard deviation of stratified cross-validation accuracy on training data.
- **Classification report:** per-class precision, recall, F1, and support.
- **Confusion matrix:** actual classes are rows and predicted classes are columns, in the displayed label order.

With roughly two dozen stocks, these scores are highly sample-sensitive. They measure reproduction of derived labels, not forecasts of returns.

## Generated files

All paths are relative to the project root. Source-specific folders separate Yahoo and synthetic runs.

### Prices and processed data

| File | Contents |
| --- | --- |
| `data/yahoo_finance/raw_prices.csv` or `data/synthetic/raw_prices.csv` | Price snapshot used by the run. Despite “raw” in the directory name, Yahoo data is adjusted closing prices. |
| `data/processed/<source>/daily_returns.csv` | Simple daily percentage returns by ticker. |
| `data/processed/<source>/feature_matrix.csv` | One row per ticker with seven mean-imputed risk/return features. |
| `data/processed/<source>/stock_analysis.csv` | Features, risk labels, PCA coordinates, and cluster IDs. |
| `data/processed/<source>/pca_components.csv` | Retained PCA scores by ticker. |

### Reports (`reports/analysis/<source>/`)

| File | Contents |
| --- | --- |
| `pca_factor_loadings.csv` | Feature loadings by principal component. |
| `rolling_risk.csv` | Rolling/full-period risk features and labels where enough data exists. |
| `recommendations_low.csv`, `recommendations_medium.csv`, `recommendations_high.csv` | Candidate tables for each relative risk group; weights are calculated by the interactive app. |
| `provenance.json` | Run ID, source, requested/actual dates, tickers, missing-data handling, package versions, seed (synthetic), and PCA/clustering settings. |

After a successful backtest, `reports/analysis/<source>/backtesting/backtest_daily.csv` contains daily returns, rebalance turnover/cost, equity, and drawdown; `summary.json` records settings, benchmark, provenance ID, and summary metrics.

### Fitted models (`models/<source>/`)

`imputer.joblib`, `scaler.joblib`, `pca.joblib`, and `kmeans.joblib` save fitted preprocessing/transformation/clustering objects. `risk_classifier.joblib` saves the fitted Random Forest when evaluation ran. Each file includes a provenance ID. The app does not load these files to serve live ticker predictions.

### Optional static CLI charts

When a Random Forest is fitted, the CLI saves charts under `reports/analysis/<source>/`: `scree_plot.png` (cumulative variance and 90% target), `pca_clusters.png` (PC scores by cluster), `pca_risk_labels.png` (PC scores by risk group), `feature_correlation_heatmap.png` (feature correlations), `feature_importance.png` (Random Forest impurity split importance, not causality), and `recommendation_ranking.png` (top eight historical Sharpe-like ratios). The Streamlit app renders interactive charts in the browser; a normal app analysis does not create these six static images.

## Conventions and limits

- Returns are simple percentage changes, not log returns. Mean return is arithmetic annualization; compounded growth is used for equity and drawdown calculations.
- Risk categories, PCA, clusters, and weights change with the selected tickers and dates.
- A Yahoo Finance missing/delisted ticker can cause download failures or a smaller usable universe.
- Historical estimates can be noisy, particularly with short histories and small stock samples. No output establishes future performance.
