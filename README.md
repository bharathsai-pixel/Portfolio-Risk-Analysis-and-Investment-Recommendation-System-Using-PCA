# Portfolio Risk Analysis and Investment Recommendation System Using PCA

This is an end-to-end Machine Learning project that turns historical stock prices into risk/return features, identifies latent risk factors with PCA, clusters similar stocks, classifies risk buckets, and prepares diversified rule-based recommendations. It is an educational decision-support project, not financial advice.

## What it delivers

- Historical daily prices from Yahoo Finance for 24 NSE stocks, with reproducible offline demo data available.
- Missing-value imputation, standardisation, annual return/volatility, Sharpe-like ratio, maximum drawdown, skewness, kurtosis, and 95% historical VaR.
- PCA retaining at least 90% of feature variance and saved factor loadings.
- K-means cluster selection using silhouette scores.
- Logistic Regression baseline and Random Forest risk classifier, evaluated with accuracy, weighted precision, recall, F1 and stratified cross-validation.
- Low, medium and high risk profiles, favouring a spread across PCA clusters.
- CSV outputs, saved model artifacts and five report-ready plots.

## Run

Activate the existing environment, then run either command from the repository root:

```powershell
.\.MLvenv\Scripts\Activate.ps1
python main.py
python main.py --synthetic  # fully offline, deterministic demonstration
```

`main.py` uses Yahoo Finance by default. If a download is unavailable, use `--synthetic`; it exercises every pipeline stage without presenting the data as real market data.

## Outputs

| Location | Contents |
| --- | --- |
| `data/` | raw prices, returns, feature matrix, PCA coordinates and full stock analysis |
| `models/` | fitted imputer, scaler, PCA, K-means and Random Forest artifacts |
| `reports/` | scree plot, cluster/risk plots, correlation heatmap, feature importance, ranking chart, recommendations and JSON metrics |

## Report / viva points

Explain PC loadings as latent risk factors rather than direct causal factors. Discuss survivorship bias, historical data not guaranteeing future outcomes, changing market regimes, omitted transaction costs/taxes, and the small cross-sectional sample used for risk classification. The Random Forest is compared with a regularised logistic baseline; this supports a bias–variance comparison.
