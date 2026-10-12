"""Command-line runner for the shared portfolio analysis pipeline."""
import argparse
from src.pipeline import run_analysis
from src.visualize import create_visualizations


def main(synthetic=False, start="2019-01-01", end="2026-01-01"):
    result = run_analysis("synthetic" if synthetic else "yahoo_finance", start, end)
    report = result["report_dir"]
    if result["models"].get("Random Forest") is not None:
        create_visualizations(result["features"], result["pca"], result["components"],
                              result["analysis"], result["models"]["Random Forest"], report)
    print(f"Source: {result['source']} | observations: {len(result['prices'])} | stocks: {len(result['analysis'])}")
    print(f"PCA retained {result['pca'].explained_variance_ratio_.sum():.1%} variance across {result['pca'].n_components_} components")
    print(f"Selected {result['cluster_model'].n_clusters} clusters; cluster scores: {result['cluster_scores']}")
    if result["classifier_metrics"]:
        print(f"Classifier metrics: {result['classifier_metrics']}")
    for profile, portfolio in result["recommendations"].items():
        print(f"\n{profile}-risk example portfolio:\n{portfolio.to_string()}")
    print(f"Reports: {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="use deterministic offline demonstration data")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-01-01", help="exclusive end date")
    args = parser.parse_args()
    main(args.synthetic, args.start, args.end)
