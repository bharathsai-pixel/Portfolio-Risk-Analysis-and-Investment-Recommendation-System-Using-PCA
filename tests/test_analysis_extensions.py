import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import make_demo_prices
from src.preprocess import compute_daily_returns, build_feature_matrix
from src.pca_pipeline import run_pca
from src.clustering import cluster_stocks
from src.benchmark import benchmark_metrics
from src.rolling_analysis import rolling_risk_analysis
from src.portfolio_optimizer import portfolio_weights
from src.backtesting import backtest
from src.provenance import source_paths, create_provenance, save_provenance


def sample_returns(n=320, assets=6):
    rng = np.random.default_rng(11)
    return pd.DataFrame(rng.normal(.0002, .01, (n, assets)),
                        index=pd.bdate_range("2022-01-03", periods=n),
                        columns=[f"S{i}" for i in range(assets)])


def test_synthetic_is_reproducible():
    a = make_demo_prices(["A", "B"], periods=50)
    b = make_demo_prices(["A", "B"], periods=50)
    pd.testing.assert_frame_equal(a, b)


def test_source_paths_are_separate():
    real, _ = source_paths("yahoo_finance", "data")
    demo, _ = source_paths("synthetic", "data")
    assert real != demo and "yahoo_finance" in str(real) and "synthetic" in str(demo)


def test_features_and_pca():
    returns = sample_returns()
    features = build_feature_matrix(returns)
    assert features.shape == (6, 7)
    pca, coords = run_pca((features-features.mean())/features.std(ddof=0))
    assert coords.shape[0] == 6
    assert pca.explained_variance_ratio_.sum() >= .9 - 1e-9


def test_cluster_validation():
    rng=np.random.default_rng(2)
    points=pd.DataFrame(np.r_[rng.normal(-2,.2,(8,2)),rng.normal(2,.2,(8,2))])
    labels, model, scores=cluster_stocks(points)
    assert len(labels)==16 and model.n_clusters in scores
    assert all("davies_bouldin" in metric for metric in scores.values())


def test_benchmark_metrics_alignment():
    idx=pd.bdate_range("2024-01-01",periods=10)
    out=benchmark_metrics(pd.Series(.001,index=idx),pd.Series(.0005,index=idx))
    assert out["observations"]==10 and np.isfinite(out["beta"])


def test_rolling_full_period_available():
    table, _=rolling_risk_analysis(sample_returns(80,6))
    assert set(table.window)=={"Full period"}


@pytest.mark.parametrize("method",["equal","inverse_volatility","minimum_variance"])
def test_portfolio_weights_are_valid_and_capped(method):
    weights=portfolio_weights(sample_returns(100,6),method,max_weight=.25)
    assert np.isclose(weights.sum(),1)
    assert (weights>=0).all() and (weights<=.25+1e-7).all()


def test_backtest_is_out_of_sample_and_chronological():
    returns=sample_returns(100,4)
    result=backtest(returns,training_window=40,rebalance_every=10,transaction_cost=.001,max_weight=.25)
    assert result.index.min() > returns.index[39]
    assert result.index.is_monotonic_increasing
    assert result.transaction_cost.iloc[0] > 0


def test_provenance_serializes(tmp_path):
    prices=make_demo_prices(["A","B"],periods=5)
    metadata=create_provenance("synthetic",prices,"2021-01-01","2022-01-01",prices.columns,seed=42)
    path=tmp_path/"provenance.json"
    save_provenance(metadata,path)
    assert json.loads(path.read_text())["synthetic_random_seed"]==42


def test_streamlit_app_starts():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert any("Portfolio Risk Analysis" in str(element.value) for element in app.title)
    app.button[0].click().run()
    assert not app.exception
    assert any(radio.value == "Yahoo Finance" for radio in app.radio)
