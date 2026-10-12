"""All report figures produced by the pipeline."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def create_visualizations(features, pca, components, analysis, classifier, output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    cumulative = pca.explained_variance_ratio_.cumsum()
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(range(1, len(cumulative)+1), cumulative, marker="o")
    ax.axhline(.90, color="crimson", linestyle="--", label="90% target"); ax.set(xlabel="Principal component", ylabel="Cumulative explained variance", title="PCA Scree Plot"); ax.legend(); fig.tight_layout(); fig.savefig(output / "scree_plot.png", dpi=180); plt.close(fig)
    x, y = components.columns[:2]
    for column, filename, title in [("cluster", "pca_clusters.png", "Stocks grouped by PCA cluster"), ("risk_label", "pca_risk_labels.png", "Stocks by risk category")]:
        fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=analysis, x=x, y=y, hue=column, s=95, ax=ax)
        for name, row in analysis.iterrows(): ax.annotate(name.replace(".NS", ""), (row[x], row[y]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_title(title); fig.tight_layout(); fig.savefig(output / filename, dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6)); sns.heatmap(features.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax); ax.set_title("Risk and Return Feature Correlations"); fig.tight_layout(); fig.savefig(output / "feature_correlation_heatmap.png", dpi=180); plt.close(fig)
    importance = classifier.feature_importances_; names = getattr(classifier, "feature_names_in_", features.columns)
    order = importance.argsort()
    fig, ax = plt.subplots(figsize=(7, 4)); ax.barh([names[i] for i in order], importance[order], color="#2a9d8f"); ax.set(title="Random Forest Feature Importance", xlabel="Importance"); fig.tight_layout(); fig.savefig(output / "feature_importance.png", dpi=180); plt.close(fig)
    recommended = analysis.sort_values("sharpe_like_ratio", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(recommended.index.str.replace(".NS", "", regex=False), recommended["sharpe_like_ratio"], color="#457b9d"); ax.axhline(0, color="black", linewidth=.8); ax.set(title="Top risk-adjusted return candidates", ylabel="Sharpe-like ratio"); ax.tick_params(axis="x", rotation=35); fig.tight_layout(); fig.savefig(output / "recommendation_ranking.png", dpi=180); plt.close(fig)
