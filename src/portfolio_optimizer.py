"""Long-only portfolio weighting with explicit max-weight constraints."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_weights(returns, method="equal", max_weight=1.0):
    if returns.empty or not 0 < max_weight <= 1:
        raise ValueError("Returns must be non-empty and max_weight must be in (0, 1].")
    assets = list(returns.columns)
    n = len(assets)
    if n * max_weight < 1 - 1e-10:
        raise ValueError("Infeasible maximum weight: eligible assets cannot sum to 100%.")
    if method == "equal":
        raw = np.ones(n)
    elif method == "inverse_volatility":
        vol = returns.std(ddof=1).replace(0, np.nan).to_numpy()
        raw = np.divide(1, vol, out=np.zeros_like(vol), where=np.isfinite(vol))
    elif method == "minimum_variance":
        cov = returns.cov().to_numpy(dtype=float) * 252
        cov = np.nan_to_num((cov + cov.T) / 2)
        cov += np.eye(n) * max(float(np.trace(cov)) / max(n, 1) * 1e-8, 1e-10)
        result = minimize(lambda w: float(w @ cov @ w), np.ones(n)/n, method="SLSQP",
                          bounds=[(0, max_weight)]*n, constraints=[{"type": "eq", "fun": lambda w: w.sum()-1}],
                          options={"maxiter": 1000, "ftol": 1e-12})
        if not result.success:
            raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
        raw = result.x
    else:
        raise ValueError("method must be equal, inverse_volatility, or minimum_variance")
    if not np.isfinite(raw).all() or raw.sum() <= 0:
        raise ValueError("Cannot calculate finite weights; check asset volatility and history.")
    weights = raw / raw.sum()
    # Capped proportional redistribution for methods with closed-form initial weights.
    if method != "minimum_variance":
        for _ in range(n + 1):
            over = weights > max_weight + 1e-12
            if not over.any(): break
            excess = (weights[over] - max_weight).sum(); weights[over] = max_weight
            free = ~over
            if not free.any(): raise ValueError("Infeasible maximum-weight constraint.")
            weights[free] += excess * weights[free] / weights[free].sum()
    return pd.Series(weights, index=assets, name="weight")
