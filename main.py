"""Run the end-to-end Portfolio Risk Analysis and Investment Recommendation System."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.classifier import build_models, label_risk
from src.clustering import cluster_stocks
from src.data_loader import TICKERS, load_prices
from src.evaluate import evaluate_models
from src.pca_pipeline import factor_loadings, run_pca
from src.preprocess import build_feature_matrix, compute_daily_returns, impute_and_scale
from src.recommend import recommend_portfolio
from src.visualize import create_visualizations

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", "{:.4f}".format)


def main(synthetic=False):
    data_dir, model_dir, report_dir = Path("data"), Path("models"), Path("reports")
    for directory in (data_dir, model_dir, report_dir): directory.mkdir(exist_ok=True)

    # ---- 1. Data ----
    source = "deterministic offline demo data" if synthetic else "Yahoo Finance"
    print(f"[1/7] Fetching price data from {source}...")
    prices = load_prices(data_dir / "raw_prices.csv", TICKERS, synthetic=synthetic)
    missing = sorted(set(TICKERS) - set(prices.columns))
    if missing:
        print(f"WARNING: {len(missing)} ticker(s) failed to download and were dropped: {', '.join(missing)}")
    print(f"Downloaded shape: {prices.shape}")
    print("Price preview (first 6 of " + str(prices.shape[1]) + " stocks):")
    print(prices.iloc[:5, :6].to_string(), "\n")

    # ---- 2. Preprocess ----
    print("[2/7] Computing daily returns and feature matrix...")
    returns = compute_daily_returns(prices)
    returns.to_csv(data_dir / "daily_returns.csv", index_label="Date")
    features = build_feature_matrix(returns)
    features, scaled, imputer, scaler = impute_and_scale(features)
    features.to_csv(data_dir / "feature_matrix.csv", index_label="Ticker")
    print(f"Feature matrix shape: {features.shape}")
    print(features.head().to_string(), "\n")

    # ---- 3. PCA ----
    print("[3/7] Scaling features and running PCA...")
    pca, components = run_pca(scaled)
    loadings = factor_loadings(pca, features.columns)
    components.to_csv(data_dir / "pca_components.csv", index_label="Ticker")
    loadings.to_csv(report_dir / "pca_factor_loadings.csv", index_label="Feature")
    explained = pca.explained_variance_ratio_
    print(f"Explained variance ratio (per component): {explained.round(3)}")
    print(f"Cumulative variance: {explained.cumsum().round(3)}")
    print(f"Selected components to retain 90% variance: {pca.n_components_}")
    print("Top feature loadings on PC1:")
    print(loadings["PC1"].sort_values(key=abs, ascending=False).head(4).round(3).to_string(), "\n")

    # ---- 4. Clustering ----
    print("[4/7] Clustering stocks...")
    labels, kmeans, silhouette_scores = cluster_stocks(components)
    print(f"Silhouette scores by k: { {k: round(v, 3) for k, v in silhouette_scores.items()} }")
    print(f"Selected k = {kmeans.n_clusters}")
    print("Cluster assignments:")
    for stock, label in zip(features.index, labels):
        print(f"  {stock:<15} -> Cluster {label}")

    # ---- 5. Classification ----
    print("\n[5/7] Labelling risk tiers and training classifiers...")
    labelled = label_risk(features); analysis = labelled.join(components); analysis["cluster"] = labels
    X, y = scaled.loc[analysis.index], analysis["risk_label"].astype(str)
    print(f"Risk label distribution: {y.value_counts().to_dict()}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, stratify=y, random_state=42)
    print(f"Train/test split: {len(X_train)} train, {len(X_test)} test")
    models = build_models()

    # ---- 6. Evaluate ----
    print("\n[6/7] Evaluating classifiers...")
    metrics = evaluate_models(models, X_train, X_test, y_train, y_test, verbose=True)
    forest = models["Random Forest"]
    importances = pd.Series(forest.feature_importances_, index=features.columns)
    print("\nFeature importances (Random Forest):")
    print(importances.sort_values(ascending=False).round(3).to_string(), "\n")

    # ---- 7. Recommend ----
    print("[7/7] Generating diversified recommendations...")
    recommendations = []
    for profile in ("Low", "Medium", "High"):
        portfolio = recommend_portfolio(analysis, profile)
        portfolio.to_csv(report_dir / f"recommendations_{profile.lower()}.csv", index_label="Ticker")
        recommendations.append({"profile": profile, "assets": portfolio.index.tolist()})
        print(f"\n{profile}-risk profile:")
        terminal_columns = ["cluster", "risk_label", "sharpe_like_ratio", "annual_volatility"]
        print(portfolio[terminal_columns].to_string())
        print(f"Full recommendation metrics: {report_dir / f'recommendations_{profile.lower()}.csv'}")
    analysis.to_csv(data_dir / "stock_analysis.csv", index_label="Ticker")
    create_visualizations(features, pca, components, analysis, forest, report_dir)
    joblib.dump(imputer, model_dir / "imputer.joblib"); joblib.dump(scaler, model_dir / "scaler.joblib"); joblib.dump(pca, model_dir / "pca.joblib"); joblib.dump(kmeans, model_dir / "kmeans.joblib"); joblib.dump(forest, model_dir / "risk_classifier.joblib")
    summary = {"data_source": "deterministic demo data" if synthetic else "Yahoo Finance", "stocks": len(analysis), "pca_components": int(pca.n_components_), "explained_variance": float(pca.explained_variance_ratio_.sum()), "silhouette_scores": silhouette_scores, "model_metrics": metrics, "recommendation_profiles": recommendations}
    with open(report_dir / "evaluation_summary.json", "w", encoding="utf-8") as file: json.dump(summary, file, indent=2)
    print(f"\nDone. PCA retained {summary['explained_variance']:.1%} variance across {pca.n_components_} components.")
    print(f"Artifacts saved to {data_dir}/, {model_dir}/, and {report_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--synthetic", action="store_true", help="run with deterministic offline sample prices")
    main(synthetic=parser.parse_args().synthetic)
