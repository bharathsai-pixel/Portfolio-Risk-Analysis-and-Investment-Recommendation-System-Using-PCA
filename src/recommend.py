"""Diversified, rule-based investment recommendations (not financial advice)."""
import pandas as pd


def recommend_portfolio(analysis, target_risk="Low", top_n=5):
    candidates = analysis[analysis["risk_label"].astype(str) == target_risk].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values("sharpe_like_ratio", ascending=False)
    # First pass favours one candidate from each cluster, then fills remaining places.
    diversified = candidates.drop_duplicates("cluster")
    selected = pd.concat([diversified, candidates]).loc[lambda d: ~d.index.duplicated()].head(top_n)
    return selected[["cluster", "risk_label", "mean_annual_return", "annual_volatility",
                     "sharpe_like_ratio", "max_drawdown", "var_95_daily"]]
