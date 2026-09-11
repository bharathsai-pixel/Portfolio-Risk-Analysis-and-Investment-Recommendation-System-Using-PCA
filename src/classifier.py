"""Risk-bucket labelling and comparable supervised models."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def label_risk(feature_df):
    labelled = feature_df.copy()
    # Ranking avoids qcut failures if two assets have equal reported volatility.
    labelled["risk_label"] = pd.qcut(labelled["annual_volatility"].rank(method="first"), 3,
                                      labels=["Low", "Medium", "High"])
    return labelled


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=1,
                                                   class_weight="balanced", random_state=42),
    }
