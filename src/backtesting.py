"""Chronological periodic-rebalance backtest without look-ahead in weights."""
import numpy as np
import pandas as pd
from .portfolio_optimizer import portfolio_weights


def backtest(returns, benchmark_returns=None, training_window=252, rebalance_every=21,
             method="equal", transaction_cost=0.001, max_weight=0.25):
    if training_window < 20 or rebalance_every < 1 or transaction_cost < 0:
        raise ValueError("Use training_window >= 20, rebalance_every >= 1, and nonnegative transaction cost.")
    if len(returns) <= training_window:
        raise ValueError("Insufficient history: observations must exceed the training window.")
    rows, previous = [], pd.Series(0.0, index=returns.columns)
    # Only returns up to t-1 estimate weights; weights apply to subsequent observations.
    for start in range(training_window, len(returns), rebalance_every):
        train = returns.iloc[max(0, start-training_window):start].dropna(axis=1, how="any")
        if train.shape[1] == 0:
            continue
        weights = portfolio_weights(train, method, max_weight)
        held = returns.iloc[start:min(start+rebalance_every, len(returns))][weights.index]
        turnover = float((weights.reindex(previous.index, fill_value=0)-previous).abs().sum())
        cost = transaction_cost * turnover
        for offset, (date, daily) in enumerate(held.iterrows()):
            ret = float(daily @ weights)
            if offset == 0:
                ret -= cost
            row = {"date": date, "portfolio_return": ret, "turnover": turnover if offset == 0 else 0.0,
                   "transaction_cost": cost if offset == 0 else 0.0}
            if benchmark_returns is not None and date in benchmark_returns.index:
                row["benchmark_return"] = float(benchmark_returns.loc[date])
            rows.append(row)
        previous = weights.reindex(returns.columns, fill_value=0)
    result = pd.DataFrame(rows).set_index("date")
    result["equity"] = (1 + result.portfolio_return).cumprod()
    result["drawdown"] = result.equity / result.equity.cummax() - 1
    if "benchmark_return" in result:
        result["benchmark_equity"] = (1 + result.benchmark_return.fillna(0)).cumprod()
    return result
