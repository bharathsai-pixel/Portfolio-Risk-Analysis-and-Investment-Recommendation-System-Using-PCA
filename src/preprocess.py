"""Transform daily prices into interpretable per-stock risk features."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def compute_daily_returns(prices):
    return prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")


def _max_drawdown(series):
    wealth = (1 + series.fillna(0)).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def build_feature_matrix(returns, annual_trading_days=252):
    """Build one row per stock; returns and volatility are annualised."""
    features = pd.DataFrame(index=returns.columns)
    features["mean_annual_return"] = returns.mean() * annual_trading_days
    features["annual_volatility"] = returns.std() * np.sqrt(annual_trading_days)
    features["sharpe_like_ratio"] = features["mean_annual_return"] / features["annual_volatility"]
    features["max_drawdown"] = returns.apply(_max_drawdown)
    features["skewness"] = returns.skew()
    features["kurtosis"] = returns.kurt()
    features["var_95_daily"] = returns.quantile(0.05)
    return features.replace([np.inf, -np.inf], np.nan)


def impute_and_scale(feature_df):
    imputer = SimpleImputer(strategy="mean")
    filled = pd.DataFrame(imputer.fit_transform(feature_df), index=feature_df.index,
                          columns=feature_df.columns)
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(filled), index=filled.index,
                          columns=filled.columns)
    return filled, scaled, imputer, scaler
