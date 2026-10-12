# Portfolio Risk Analysis and Investment Recommendation System Using PCA

A local Python application for exploring risk and return characteristics of Indian NSE stocks. It transforms historical prices into stock-level features, uses PCA and clustering to summarize relationships, compares volatility-derived risk categories, and provides example portfolio weights and a chronological benchmark backtest.

This is an educational analysis tool, not personalized investment advice or a forecast. Historical performance can change and does not guarantee future results.

## Features

- Yahoo Finance adjusted daily prices for the default 24 NSE stocks, or deterministic synthetic prices for an offline demonstration.
- Annualized mean return and volatility, Sharpe-like ratio, maximum drawdown, skewness, kurtosis, and fifth-percentile daily return.
- Mean imputation and standardization, PCA retaining at least 90% of feature variance, and saved factor loadings.
- K-means cluster selection using silhouette scores, with Davies–Bouldin scores, hierarchical comparison, and early-period adjusted Rand comparison.
- Low/Medium/High relative risk labels based on volatility terciles and Logistic Regression / Random Forest evaluation.
- Example equal, inverse-volatility, and minimum-variance portfolio weights with long-only and maximum per-stock weight constraints.
- Rolling one-year, three-year, and full-period risk summaries where data allows.
- NIFTY 50 (`^NSEI`) comparison and a periodic, chronological backtest with turnover-based transaction costs.
- Streamlit web interface with dashboard, risk, PCA/clusters, portfolio, benchmark/backtest, and model evaluation pages.
- Source-separated raw prices, generated analysis tables, model artifacts, and provenance metadata.

## Requirements and installation

Python 3.10 or newer is recommended. Online mode and NIFTY benchmark analysis require internet access and Yahoo Finance availability. Synthetic analysis and the core tests do not need market data access.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

An included `.MLvenv` can also be used if already set up:

```powershell
.\.MLvenv\Scripts\Activate.ps1
```

### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the application

Start the web app from the repository root:

```powershell
streamlit run app.py
```

Use the **Run Analysis** page to choose Yahoo Finance or synthetic data, start and exclusive end dates, and stock tickers. Results are only recalculated after clicking **Run Analysis**. Yahoo Finance failures are displayed as errors; the app does not silently switch to synthetic data. The app does not currently cache analyses, so a submitted configuration is downloaded and processed anew.

The same command works from macOS/Linux after activating the environment:

```bash
streamlit run app.py
```

## Command-line pipeline

The existing batch workflow remains available:

```powershell
python main.py --synthetic
python main.py
```

The first command runs deterministic artificial prices without network access. The second downloads real prices. Optional date arguments use an exclusive end date:

```powershell
python main.py --synthetic --start 2021-01-01 --end 2025-01-01
```

macOS/Linux use the same commands with `python` after environment activation.

## Architecture

```text
app.py                       Streamlit user interface
main.py                      Command-line entry point
src/pipeline.py              Shared analysis workflow for both interfaces
src/data_loader.py           Yahoo Finance and deterministic sample prices
src/preprocess.py             Daily returns and stock-level features
src/pca_pipeline.py           PCA and feature loadings
src/clustering.py             K-means and cluster validation comparisons
src/classifier.py             Volatility risk labels and classifiers
src/evaluate.py               Holdout and stratified CV metrics
src/rolling_analysis.py       Rolling risk summaries and label consistency
src/portfolio_optimizer.py    Equal, inverse-volatility, minimum-variance weights
src/benchmark.py              NIFTY 50 retrieval and aligned metrics
src/backtesting.py            Chronological walk-forward backtest
src/provenance.py             Source paths and reproducibility metadata
src/recommend.py              Rule-based profile candidate selection
src/visualize.py              Static report charts
tests/                        Deterministic tests using generated data
```

## Course outcome map (CO1–CO6)

This table maps the attached ML course outcomes to the techniques and engineering practices actually present in this repository. It is a project-alignment guide, not a claim that the project covers every topic in the course outcome.

| Course outcome | Used in this project | Where it appears | Not implemented here |
| --- | --- | --- | --- |
| **CO1 — End-to-end ML lifecycle** | Yahoo Finance or synthetic prices → daily returns and engineered risk features → imputation/scaling → PCA and clusters → volatility-derived labels → classifier training/evaluation → provenance-tagged joblib artifacts and reports. The local Streamlit app lets a user submit an analysis and view its results. | `src/data_loader.py`, `src/preprocess.py`, `src/pipeline.py`, `src/provenance.py`, `app.py`, `main.py` | This is not a deployed prediction service: the app runs an analysis rather than accepting a ticker feature vector and serving a stored model prediction. There is no REST API, model registry, feature store, production monitoring, drift alert, or automated retraining loop. |
| **CO2 — Linear supervised learning** | Multiclass Logistic Regression classifies Low/Medium/High volatility-derived labels. Inputs are mean-imputed and standardized; the target volatility feature itself is excluded from classifier inputs to avoid direct target leakage. The model uses class balancing and regularization. | `src/classifier.py`, `src/preprocess.py`, `src/pipeline.py` | No linear/ridge/lasso/elastic-net regression or explicit regularization comparison is included. No separate binary logistic-regression lesson, coefficient interpretation page, or MinMax scaling comparison. |
| **CO3 — Tree-based supervised learning** | A class-balanced Random Forest is trained alongside Logistic Regression. A feature-importance chart uses the forest's impurity-based `feature_importances_`. | `src/classifier.py`, `src/evaluate.py`, `src/visualize.py`, `app.py` | No standalone decision tree, pruning study, out-of-bag evaluation, gradient boosting, XGBoost, LightGBM, permutation importance, or SHAP explanation. |
| **CO4 — Unsupervised learning** | Standardized features are reduced with PCA; K-means is selected using silhouette score. Davies–Bouldin scores, agglomerative hierarchical clustering comparison, and an adjusted Rand comparison across an early subperiod and the full sample are also calculated. Clusters are shown in the first two PCA dimensions. | `src/pca_pipeline.py`, `src/clustering.py`, `app.py` | No DBSCAN, t-SNE, UMAP, anomaly detection, or use of cluster IDs as supervised-model inputs. The early/full comparison is a small-sample stability check, not a point-in-time historical constituent study. |
| **CO5 — Evaluation and model selection** | A stratified holdout split and stratified cross-validation are used where sample counts allow. Reported classification metrics include accuracy, balanced accuracy, weighted precision, weighted recall, weighted F1, and a confusion matrix. | `src/pipeline.py`, `src/evaluate.py`, `app.py` | No separate train/validation/test three-way protocol, nested CV, ROC-AUC, PR-AUC, calibration, hyperparameter search, learning curves, or statistical-significance testing. With about 24 stocks, these scores are unstable and are not evidence of future predictive performance. |
| **CO6 — ML engineering** | The workflow is split into reusable modules and can run from the CLI or local Streamlit UI. Joblib artifacts are saved with provenance IDs; source-specific data paths and configuration metadata help keep analyses traceable. | `src/pipeline.py`, `src/provenance.py`, `main.py`, `app.py` | No online feature store, ONNX export, container, FastAPI/gRPC serving, experiment tracker, versioned dataset store, latency/throughput or drift monitoring, scheduled retraining, or production deployment. |

### One analysis traced through CO1

For one submitted analysis, `src/data_loader.py` obtains the selected source's prices and writes them beneath that source's directory. `src/preprocess.py` converts prices into daily returns and per-stock risk/return features. `src/pipeline.py` imputes and scales those features, fits PCA and K-means, derives volatility buckets, evaluates the classifiers, and builds example portfolios. It writes processed tables, source-specific reports, provenance metadata, and fitted joblib artifacts. The Streamlit UI reads the returned in-memory analysis to render its pages; it does not load the saved classifier to serve individual predictions. A later run is a new analysis, not an automated monitoring or retraining cycle.

## Methodology and metric conventions

Daily returns are simple percentage changes, not logarithmic returns. Annualized return in the stock feature table is arithmetic mean daily return multiplied by 252; annualized volatility is daily sample standard deviation multiplied by √252. The Sharpe-like feature is annualized mean return divided by annualized volatility and does not subtract a risk-free rate. Maximum drawdown is computed from the cumulative simple-return wealth path. `var_95_daily` is the historical 5th percentile daily return, not a forecast or a positive loss amount.

PCA operates on standardized features and selects enough components to retain at least 90% of observed feature variance. Loadings describe the feature combinations, not causal factors. K-means is evaluated on PCA coordinates for k from 2 through at most 6. Silhouette and Davies–Bouldin scores describe geometric cluster properties, not investment value. The hierarchical comparison and adjusted Rand index compare clustering algorithms/periods; temporal stability is approximate because this project has only a cross-section of stocks rather than point-in-time historical constituents.

Risk groups are relative terciles of annualized volatility across the selected stock set. Classifiers learn these derived labels from the same small cross-sectional feature set; the target volatility column is excluded from the classifier inputs to avoid direct target leakage. Test/CV scores can be unstable with about 24 stocks and do not establish reliable prediction.

### Portfolio weights

Portfolio examples first select up to five candidates per risk profile by historical Sharpe-like ratio, favoring distinct PCA clusters. The web app lets users compare:

- **Equal:** same starting weight for each candidate.
- **Inverse volatility:** weight proportional to the inverse of each candidate's historical volatility.
- **Minimum variance:** minimize estimated covariance-based variance with long-only SLSQP optimization.

The interface enforces a configurable maximum individual weight and reports infeasible constraints. Sector and liquidity constraints are not applied because the current pipeline has no reliable point-in-time sector classifications or liquidity series. Portfolio formation uses the full selected analysis period in the example page; these examples are descriptive and are not a point-in-time strategy.

### Benchmark and backtesting

The benchmark page downloads NIFTY 50 (`^NSEI`) separately and reports an error if no benchmark data is available. Portfolio and benchmark returns are aligned by date. Benchmark metrics use simple daily returns and a configurable-in-code annual risk-free-rate assumption currently set to 0%; beta is sample covariance divided by benchmark sample variance; Jensen's alpha is based on daily CAPM excess returns and annualized by 252; tracking error is annualized standard deviation of daily active returns; information ratio is arithmetic annualized active return divided by tracking error. Compounded annual growth is used for annualized portfolio/benchmark growth rates in the metrics output.

The walk-forward backtest uses a trailing training window (default 252 trading days) to estimate weights, applies them only to the following period (default 21 trading days), then rebalances. Turnover is the sum of absolute weight changes; the configured transaction cost rate is charged against that notional at rebalance. Equity and drawdown series are shown against NIFTY when available. Insufficient history produces an error rather than fabricated results.

The backtest avoids look-ahead in weight estimation, but it does not remove survivorship bias: the stock universe is selected from today's fixed ticker list and historical membership/delistings are not reconstructed. Yahoo Finance adjusted prices, missing dates, trading frictions, and model selection remain practical limitations.

## Data provenance and outputs

Raw data from each source is kept separately so a synthetic run cannot overwrite downloaded raw data (or vice versa):

```text
data/yahoo_finance/raw_prices.csv
data/synthetic/raw_prices.csv
data/processed/yahoo_finance/...
data/processed/synthetic/...
reports/analysis/yahoo_finance/...
reports/analysis/synthetic/...
models/yahoo_finance/...
models/synthetic/...
```

Each analysis report directory includes a `provenance.json` with a unique ID, source, UTC creation time, requested and observed dates, tickers, frequency, missing-data handling, synthetic seed, package versions, and key configuration. Model artifacts are wrapped with the same provenance ID. Synthetic mode is always explicitly labeled and uses seed 42. No artifacts from a different source are loaded into an analysis.

Generated data include raw prices, daily returns, feature matrix, PCA coordinates, and stock analysis. Reports include PCA loadings, rolling risk, three risk-profile recommendation CSVs, and provenance. The CLI also creates PNG figures when classifier fitting is possible. After a benchmark backtest, the app writes its daily result and a configuration/metrics summary under the source-specific report directory's `backtesting/` folder.

## Tests

Run deterministic automated tests from the repository root:

```powershell
python -m pytest -q
```

The tests cover synthetic reproducibility, data-source paths, feature/PCA output, cluster validation, benchmark metrics, rolling analysis, constrained weights, chronological backtesting, transaction costs, and provenance serialization. They do not require Yahoo Finance access.

## Known limitations and disclaimer

- Historical risk and return are not forecasts and do not guarantee future outcomes.
- The fixed 24-stock universe may contain selection and survivorship bias.
- Risk buckets are relative, not absolute safety ratings; model metrics are unstable at this sample size.
- Synthetic data exists only to exercise the code and must never be read as market history.
- Sector, liquidity, taxes, personal suitability, and reliable execution costs are not fully modeled.
- The backtest uses a fixed current universe and simplified transaction costs; it is not a production trading system.

This project is for education and research only. It is not financial advice. Consult a qualified professional before making investment decisions.
