"""Full-period and trailing-window risk summaries."""
import numpy as np
import pandas as pd
from .preprocess import build_feature_matrix
from .classifier import label_risk


def rolling_risk_analysis(returns, windows=(252, 756)):
    records = []
    for name, window in [("1 year", windows[0]), ("3 years", windows[1]), ("Full period", None)]:
        sample = returns.tail(window) if window else returns
        if len(sample) < 3 or (window is not None and len(sample) < window):
            continue
        features = build_feature_matrix(sample)
        labels = label_risk(features)["risk_label"]
        for ticker, row in features.iterrows():
            records.append({"window": name, "ticker": ticker, **row.to_dict(), "risk_label": str(labels.loc[ticker])})
    table = pd.DataFrame(records)
    consistency = None
    pivot = table.pivot(index="ticker", columns="window", values="risk_label") if not table.empty else pd.DataFrame()
    if {"1 year", "3 years"}.issubset(pivot.columns):
        consistency = float((pivot["1 year"] == pivot["3 years"]).mean())
    return table, consistency
