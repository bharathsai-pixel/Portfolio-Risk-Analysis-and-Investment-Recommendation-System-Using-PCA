"""Benchmark return alignment and standard market-relative statistics."""
import numpy as np
import pandas as pd
import yfinance as yf


def fetch_benchmark(start, end, ticker="^NSEI"):
    prices = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if prices.empty:
        raise RuntimeError(f"Yahoo Finance returned no benchmark data for {ticker}.")
    close = prices["Close"] if "Close" in prices else prices
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename("benchmark_price").dropna()


def benchmark_metrics(portfolio_returns, benchmark_returns, risk_free_rate=0.0, periods=252):
    aligned = pd.concat([portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    if len(aligned) < 3:
        raise ValueError("At least three aligned daily return observations are required.")
    p, b = aligned.portfolio, aligned.benchmark
    excess = p - b
    variance = b.var(ddof=1)
    beta = float(p.cov(b) / variance) if variance > 0 else float("nan")
    daily_rf = (1 + risk_free_rate) ** (1 / periods) - 1
    alpha_daily = float((p - daily_rf).mean() - beta * (b - daily_rf).mean())
    te = float(excess.std(ddof=1) * np.sqrt(periods))
    wealth = (1 + p).cumprod()
    years = len(p) / periods
    cagr = float(wealth.iloc[-1] ** (1 / years) - 1) if years > 0 and wealth.iloc[-1] > 0 else float("nan")
    volatility = float(p.std(ddof=1) * np.sqrt(periods))
    sharpe = float((p.mean() - daily_rf) * periods / volatility) if volatility else float("nan")
    drawdown = float((wealth / wealth.cummax() - 1).min())
    return {"observations": len(aligned), "portfolio_return_annualized": float((1+p).prod() ** (periods/len(p))-1),
            "benchmark_return_annualized": float((1+b).prod() ** (periods/len(b))-1),
            "excess_return_annualized": float(excess.mean() * periods), "beta": beta,
            "jensens_alpha_annualized": alpha_daily * periods, "tracking_error": te,
            "information_ratio": float(excess.mean()*periods/te) if te else float("nan"),
            "sharpe_ratio": sharpe, "annualized_volatility": volatility,
            "maximum_drawdown": drawdown, "cagr": cagr,
            "risk_free_rate_annual": risk_free_rate, "return_convention": "simple daily returns; arithmetic annualization except CAGR fields"}
