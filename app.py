"""Streamlit interface for the portfolio risk analysis project."""
from datetime import date
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import TICKERS
from src.pipeline import run_analysis
from src.benchmark import fetch_benchmark, benchmark_metrics
from src.backtesting import backtest
from src.portfolio_optimizer import portfolio_weights

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Portfolio Risk Analysis", page_icon="◈", layout="wide")
st.markdown("""<style>
 .block-container{max-width:1250px;padding-top:2rem} [data-testid=stMetric]{background:#fff;border:1px solid #e5e7eb;padding:16px;border-radius:12px}
 h1,h2,h3{color:#14213d} .stCaption{color:#65718a}
</style>""", unsafe_allow_html=True)

st.sidebar.title("Portfolio Risk Analysis")
pages = ["Dashboard", "Run Analysis", "Risk Analysis", "PCA & Clusters", "Portfolio Examples", "Benchmark & Backtest", "Model Evaluation"]
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Dashboard"
page = st.sidebar.radio("Navigate", pages, key="nav_page")
result = st.session_state.get("analysis_result")

def need_result():
    if result is None:
        st.info("No analysis is available yet. Open **Run Analysis** to create one.")
        return False
    return True

def source_badge(data):
    if data["source"] == "synthetic":
        st.warning("SYNTHETIC DEMONSTRATION DATA — artificial prices; not actual market history.")
    else:
        st.success("REAL MARKET DATA — Yahoo Finance adjusted prices. Historical results are not forecasts.")

if page not in {"Dashboard", "Run Analysis"} and result is None:
    st.info("No analysis is available yet. Open **Run Analysis** to create one.")
    st.stop()

if page == "Dashboard":
    st.title("Portfolio Risk Analysis")
    st.write("Analyze historical stock performance, compare risk levels, discover stock relationships, and explore example portfolios using PCA and machine learning.")
    if not need_result():
        st.button("Begin analysis", type="primary",
                  on_click=lambda: st.session_state.update(nav_page="Run Analysis"))
    else:
        source_badge(result)
        a, b, c, d, e = st.columns(5)
        a.metric("Stocks", len(result["analysis"]))
        b.metric("Period", f"{result['prices'].index.min():%Y-%m} – {result['prices'].index.max():%Y-%m}")
        c.metric("PCA components", result["pca"].n_components_)
        d.metric("Variance retained", f"{result['pca'].explained_variance_ratio_.sum():.1%}")
        e.metric("Clusters", result["cluster_model"].n_clusters)
        st.subheader("Stock universe")
        cols=["mean_annual_return","annual_volatility","sharpe_like_ratio","max_drawdown","risk_label","cluster"]
        st.dataframe(result["analysis"][cols].style.format({c:"{:.2%}" for c in ["mean_annual_return","annual_volatility","max_drawdown"]}).format("{:.3f}", subset=["sharpe_like_ratio"]), use_container_width=True)
        st.caption(f"Source: {result['source']} · Provenance ID: {result['metadata']['provenance_id']}")

elif page == "Run Analysis":
    st.title("Run Analysis")
    st.caption("The analysis runs only when you submit this form. Yahoo Finance failures are shown as errors; data is never silently replaced.")
    with st.form("analysis_form"):
        source_choice=st.radio("Data source", ["Yahoo Finance", "Synthetic demonstration"], horizontal=True)
        c1,c2=st.columns(2)
        start=c1.date_input("Start date", date(2019,1,1))
        end=c2.date_input("End date (exclusive)", date(2026,1,1))
        selected=st.multiselect("NSE stocks", TICKERS, default=TICKERS)
        cluster_mode=st.selectbox("Clustering",["Automatic selection","Choose cluster count"])
        cluster_count=st.number_input("Number of clusters",min_value=2,max_value=6,value=4,disabled=cluster_mode=="Automatic selection")
        go_run=st.form_submit_button("Run Analysis", type="primary")
    if go_run:
        if start >= end: st.error("End date must be after start date.")
        elif len(selected) < 3: st.error("Select at least three stocks for PCA and clustering.")
        else:
            try:
                with st.spinner("Loading prices and running analysis…"):
                    st.session_state.analysis_result=run_analysis("synthetic" if source_choice.startswith("Synthetic") else "yahoo_finance", start.isoformat(), end.isoformat(), selected, n_clusters=int(cluster_count) if cluster_mode=="Choose cluster count" else None)
                st.success("Analysis complete. Open Dashboard or another page to explore the results.")
                st.rerun()
            except Exception as exc:
                logging.exception("Analysis failed")
                st.error(f"Analysis could not be completed: {exc}")

elif page == "Risk Analysis" and need_result():
    st.title("Risk Analysis"); source_badge(result)
    d=result["analysis"]
    ticker=st.selectbox("Select a stock", d.index)
    row=d.loc[ticker]
    metrics=[("Annualized return",row.mean_annual_return,"Approximate yearly return from the selected history."),("Annualized volatility",row.annual_volatility,"Measures how much daily returns fluctuate, scaled to a year."),("Sharpe-like ratio",row.sharpe_like_ratio,"Annualized return divided by volatility; no risk-free rate is subtracted."),("Maximum drawdown",row.max_drawdown,"Largest historical peak-to-trough decline."),("Skewness",row.skewness,"Indicates whether return outcomes are asymmetric."),("Kurtosis",row.kurtosis,"Describes tail heaviness relative to a normal distribution."),("5th percentile daily return",row.var_95_daily,"A historical lower-tail daily return quantile; not a forecast.")]
    cols=st.columns(4)
    for i,(label,value,help_text) in enumerate(metrics): cols[i%4].metric(label, f"{value:.2%}" if "return" in label.lower() or "drawdown" in label.lower() else f"{value:.3f}", help=help_text)
    st.subheader("Annualized return and volatility")
    fig=px.scatter(d,x="annual_volatility",y="mean_annual_return",color="risk_label",hover_name=d.index,labels={"annual_volatility":"Annualized volatility","mean_annual_return":"Annualized return"})
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("Risk category distribution")
    st.plotly_chart(px.histogram(d,x="risk_label",color="risk_label"),use_container_width=True)
    st.info("Risk categories are relative volatility terciles: stocks are ranked by annualized volatility and split into three similarly sized groups. They are not absolute safety ratings.")
    rolling=result["rolling"]
    if not rolling.empty:
        st.subheader("Risk across available history windows")
        st.dataframe(rolling[rolling.ticker==ticker].set_index("window"),use_container_width=True)
        if result["rolling_consistency"] is not None: st.caption(f"Stocks retaining the same Low/Medium/High category across 1-year and 3-year windows: {result['rolling_consistency']:.1%}.")
        else: st.caption("Not enough observations for both one-year and three-year comparisons.")

elif page == "PCA & Clusters" and need_result():
    st.title("PCA and Clustering"); source_badge(result)
    pca=result["pca"]
    ratios=pca.explained_variance_ratio_
    st.write("PCA compresses the selected risk and return features into fewer combined components while retaining as much measured variance as possible.")
    fig=go.Figure(); fig.add_bar(x=[f"PC{i+1}" for i in range(len(ratios))],y=ratios,name="Component variance"); fig.add_scatter(x=[f"PC{i+1}" for i in range(len(ratios))],y=np.cumsum(ratios),name="Cumulative",yaxis="y2")
    fig.update_layout(yaxis_title="Explained variance",yaxis2={"title":"Cumulative variance","overlaying":"y","side":"right","range":[0,1.05]})
    st.plotly_chart(fig,use_container_width=True)
    comp=result["components"].copy(); comp["cluster"]=result["analysis"].cluster
    if comp.shape[1] >= 2: st.plotly_chart(px.scatter(comp,x=comp.columns[0],y=comp.columns[1],color="cluster",hover_name=comp.index,labels={comp.columns[0]:"Principal component 1",comp.columns[1]:"Principal component 2"}),use_container_width=True)
    st.subheader("Feature loadings"); st.dataframe(result["loadings"],use_container_width=True)
    st.subheader("Cluster evaluation"); st.json({"scores_by_k":result["cluster_scores"],"temporal_and_hierarchical_comparison":result["cluster_stability"]})
    st.caption("Silhouette measures separation in the PCA representation; Davies–Bouldin summarizes within-cluster scatter. Neither proves financial usefulness. Adjusted Rand compares early-period and full-sample group assignments.")
    st.dataframe(result["analysis"][ ["cluster"] + list(result["components"].columns[:2]) ],use_container_width=True)

elif page == "Portfolio Examples" and need_result():
    st.title("Portfolio Examples"); source_badge(result)
    st.warning("Historical-analysis examples only, not personalized investment advice. These do not account for taxes, liquidity, or reliable sector constraints.")
    method=st.selectbox("Weighting method",["equal","inverse_volatility","minimum_variance"])
    max_weight=st.slider("Maximum weight per stock",0.05,1.0,0.25,0.05)
    size=st.slider("Maximum portfolio size",1,min(10,len(result["analysis"])),5)
    for profile,base in result["recommendations"].items():
        st.subheader(f"{profile} Risk")
        selected=base.head(size)
        if selected.empty: st.info("No eligible assets in this risk category."); continue
        try:
            history=result["returns"][selected.index].dropna()
            weights=portfolio_weights(history,method,max_weight)
            table=selected.copy(); table.insert(0,"weight",weights)
            st.dataframe(table,use_container_width=True)
            st.caption(f"Weights sum to {weights.sum():.6f}. Optimization uses the available historical returns; it is not a forecast.")
        except Exception as exc: st.error(f"Could not construct this portfolio with the selected constraints: {exc}")

elif page == "Benchmark & Backtest" and need_result():
    st.title("Benchmark and Backtest"); source_badge(result)
    st.caption("Backtest rebalances chronologically: each weight uses only the prior training window and applies to subsequent daily returns. Costs are charged on turnover at each rebalance.")
    cost=st.number_input("Transaction cost per traded notional",min_value=0.0,max_value=0.05,value=0.001,step=0.0005,format="%.4f")
    train=st.number_input("Training window (trading days)",min_value=20,max_value=2000,value=252)
    freq=st.number_input("Rebalance interval (trading days)",min_value=1,max_value=252,value=21)
    method=st.selectbox("Backtest weighting",["equal","inverse_volatility","minimum_variance"])
    if st.button("Run chronological backtest",type="primary"):
        try:
            with st.spinner("Fetching NIFTY 50 benchmark and evaluating out-of-sample periods…"):
                benchmark=fetch_benchmark(result["prices"].index.min().date().isoformat(), (result["prices"].index.max()+pd.Timedelta(days=1)).date().isoformat())
                bret=benchmark.pct_change(fill_method=None).dropna()
                bt=backtest(result["returns"],bret,int(train),int(freq),method,cost,max_weight=0.25)
            st.session_state.backtest_result=bt
            bm=benchmark_metrics(bt.portfolio_return,bt.benchmark_return.dropna())
            st.session_state.benchmark_result=bm
            out=Path(result["report_dir"])/"backtesting"
            out.mkdir(parents=True,exist_ok=True)
            bt.to_csv(out/"backtest_daily.csv",index_label="Date")
            (out/"summary.json").write_text(json.dumps({"provenance_id":result["metadata"]["provenance_id"],"weighting_method":method,"training_window_days":int(train),"rebalance_every_days":int(freq),"transaction_cost_rate":cost,"benchmark":"^NSEI","metrics":bm},indent=2,default=str),encoding="utf-8")
        except Exception as exc:
            logging.exception("Benchmark or backtest failed")
            st.error(f"Benchmark or backtest unavailable: {exc}")
    bt=st.session_state.get("backtest_result")
    if bt is not None:
        st.json(st.session_state.get("benchmark_result",{}))
        st.plotly_chart(px.line(bt,y=["equity","benchmark_equity"],labels={"value":"Growth of 1 unit","index":"Date"}),use_container_width=True)
        st.plotly_chart(px.area(bt,y="drawdown",labels={"drawdown":"Drawdown","index":"Date"}),use_container_width=True)
        st.metric("Average one-way rebalance turnover",f"{bt.turnover.mean():.1%}")
    st.caption("Simple daily returns are used. CAGR uses compounded growth over observed periods; excess return and Jensen alpha use annualized daily arithmetic means. Risk-free rate defaults to 0%. Benchmark results require actual ^NSEI data; unavailable data is reported as an error.")

elif page == "Model Evaluation" and need_result():
    st.title("Model Evaluation"); source_badge(result)
    st.warning("The labels are generated from annualized-volatility terciles and the models learn those derived labels. With roughly 24 stocks, test and cross-validation scores are unstable and do not show reliable future predictive performance.")
    if result["classifier_metrics"]:
        for name,metrics in result["classifier_metrics"].items():
            st.subheader(name)
            st.json({k:v for k,v in metrics.items() if k not in ("classification_report","confusion_matrix","confusion_matrix_labels")})
            st.write("Confusion matrix labels:",metrics["confusion_matrix_labels"])
            st.dataframe(pd.DataFrame(metrics["confusion_matrix"],index=metrics["confusion_matrix_labels"],columns=metrics["confusion_matrix_labels"]))
    else: st.info("This stock sample is too small for a defensible stratified evaluation.")
