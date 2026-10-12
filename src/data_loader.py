"""Historical-price acquisition and a deterministic offline demo dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "WIPRO.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "TATAMOTORS.NS",
    "ADANIPORTS.NS", "ASIANPAINT.NS", "BHARTIARTL.NS", "ULTRACEMCO.NS",
]


def fetch_price_data(tickers, start="2019-01-01", end="2026-01-01"):
    """Download adjusted closing prices and remove unavailable tickers."""
    # A serial request avoids yfinance's occasional concurrent SQLite-cache lock
    # on Windows. It is slower, but more reliable and reproducible for a project.
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=False)
    if raw.empty:
        raise RuntimeError("No prices were returned by Yahoo Finance.")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="any")
    if prices.shape[1] < 9:
        raise RuntimeError("Too few complete price series were downloaded.")
    return prices


def make_demo_prices(tickers=TICKERS, periods=1250, seed=42, start="2021-01-01"):
    """Create correlated, labelled-free sample prices when online data is unavailable."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    market = rng.normal(0.00035, 0.009, periods)
    sector = rng.normal(0, 0.006, (periods, 4))
    returns = {}
    for index, ticker in enumerate(tickers):
        group, volatility = index % 4, 0.006 + 0.0012 * (index % 6)
        beta = 0.55 + 0.08 * (index % 5)
        drift = 0.00015 + 0.00005 * (index % 5)
        daily = drift + beta * market + 0.35 * sector[:, group]
        returns[ticker] = daily + rng.normal(0, volatility, periods)
    return 100 * (1 + pd.DataFrame(returns, index=dates)).cumprod()


def load_prices(output_path, tickers=TICKERS, start="2019-01-01", end="2026-01-01",
                synthetic=False):
    if synthetic:
        periods = max(3, len(pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))))
        prices = make_demo_prices(tickers, periods=periods, seed=42, start=start)
    else:
        prices = fetch_price_data(tickers, start, end)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="Date")
    return prices
