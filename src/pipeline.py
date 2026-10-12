"""Shared end-to-end analysis used by both CLI and Streamlit."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from .data_loader import TICKERS, load_prices
from .preprocess import compute_daily_returns, build_feature_matrix, impute_and_scale
from .pca_pipeline import run_pca, factor_loadings
from .clustering import cluster_stocks, temporal_stability
from .classifier import label_risk, build_models
from .evaluate import evaluate_models
from .recommend import recommend_portfolio
from .rolling_analysis import rolling_risk_analysis
from .provenance import source_paths, create_provenance, save_provenance


def run_analysis(source="synthetic", start="2019-01-01", end="2026-01-01", tickers=None, root=".", n_clusters=None):
    tickers = list(tickers or TICKERS)
    synthetic = source == "synthetic"
    raw_path, processed = source_paths(source, Path(root) / "data")
    prices = load_prices(raw_path, tickers, start=start, end=end, synthetic=synthetic)
    returns = compute_daily_returns(prices)
    features, scaled, imputer, scaler = impute_and_scale(build_feature_matrix(returns))
    pca, components = run_pca(scaled)
    labels, kmeans, cluster_scores = cluster_stocks(components, n_clusters=n_clusters)
    labelled = label_risk(features)
    analysis = labelled.join(components)
    analysis["cluster"] = labels
    metrics = {}
    classifier_features = scaled.drop(columns=["annual_volatility"], errors="ignore")
    if len(analysis) >= 9 and labelled.risk_label.value_counts().min() >= 3:
        X, y = classifier_features, labelled.risk_label.astype(str)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)
        models = build_models()
        metrics = evaluate_models(models, Xtr, Xte, ytr, yte)
        forest = models["Random Forest"]
    else:
        models = {}; forest = None
    analysis.index.name = "Ticker"
    processed.mkdir(parents=True, exist_ok=True)
    returns.to_csv(processed / "daily_returns.csv", index_label="Date")
    features.to_csv(processed / "feature_matrix.csv", index_label="Ticker")
    analysis.to_csv(processed / "stock_analysis.csv")
    components.to_csv(processed / "pca_components.csv", index_label="Ticker")
    loadings = factor_loadings(pca, features.columns)
    report = Path(root) / "reports" / "analysis" / source
    report.mkdir(parents=True, exist_ok=True)
    loadings.to_csv(report / "pca_factor_loadings.csv", index_label="Feature")
    rolling, rolling_consistency = rolling_risk_analysis(returns)
    rolling.to_csv(report / "rolling_risk.csv", index=False)
    recommendations = {profile: recommend_portfolio(analysis, profile) for profile in ("Low", "Medium", "High")}
    for profile, portfolio in recommendations.items():
        portfolio.to_csv(report / f"recommendations_{profile.lower()}.csv", index_label="Ticker")
    metadata = create_provenance(source, prices, start, end, prices.columns, seed=42 if synthetic else None,
        config={"pca_variance_target": .90, "cluster_method": "manual" if n_clusters else "KMeans silhouette selection",
                "requested_cluster_count": n_clusters, "selected_cluster_count": int(kmeans.n_clusters)})
    save_provenance(metadata, report / "provenance.json")
    for name, artifact in {"imputer": imputer, "scaler": scaler, "pca": pca, "kmeans": kmeans, "risk_classifier": forest}.items():
        if artifact is not None:
            path = Path(root) / "models" / source / f"{name}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"provenance_id": metadata["provenance_id"], "model": artifact}, path)
    return {"source": source, "prices": prices, "returns": returns, "features": features,
            "scaled": scaled, "pca": pca, "components": components, "loadings": loadings,
            "analysis": analysis, "cluster_model": kmeans, "cluster_scores": cluster_scores,
            "cluster_stability": temporal_stability(features, components, labels),
            "classifier_features": list(classifier_features.columns),
            "classifier_metrics": metrics, "models": models, "recommendations": recommendations,
            "rolling": rolling, "rolling_consistency": rolling_consistency,
            "metadata": metadata, "report_dir": report}
