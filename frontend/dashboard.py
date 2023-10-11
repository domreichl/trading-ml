import os
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# CONFIG
st.set_page_config(
    page_title="Stock Price Prediction",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# DATA
data_dir = os.path.join(os.path.dirname(__file__), "data")
predictions = pd.read_csv(os.path.join(data_dir, "predictions.csv"))
performance = pd.read_csv(os.path.join(data_dir, "performance.csv"))
trades = pd.read_csv(os.path.join(data_dir, "trades.csv")).drop(columns=["ID"])
trading_statistics = pd.read_csv(os.path.join(data_dir, "trading_statistics.csv"))
backtest = pd.read_csv(os.path.join(data_dir, "backtest.csv"))


# PREDICTIONS
st.title("Stock Price Predictions")
st.markdown(f"Last updated: {predictions['Date'].max()}")
model_selected = st.radio(
    "Choose prediction model",
    options=list(performance["Model"].unique()),
    horizontal=True,
)
ts_selected = st.selectbox("Select ISIN", list(predictions["ISIN"].unique()))
preds = predictions[predictions["Model"] == model_selected]
pr = preds[preds["ISIN"] == ts_selected].reset_index(drop=True)
st.subheader(f"Closing Prices for {ts_selected}")
fig_prices = go.Figure()
fig_prices.add_trace(
    go.Scatter(
        x=pr["Date"],
        y=pr["Price"],
        name="actual",
        line=dict(color="navy", width=5),
    )
)
fig_prices.add_trace(
    go.Scatter(
        x=pr["Date"],
        y=pr["PricePredicted"],
        name="predicted",
        line=dict(color="blueviolet", width=5),
    )
)
fig_prices.update_traces(marker={"size": 12})
fig_prices.update_layout(
    yaxis_title="Price [€]",
    xaxis=dict(title="Date", tickformat="%b %d"),
)
st.plotly_chart(fig_prices)


# MODELS
st.title("Model Performance")
st.subheader("Testing")
metric_selected = st.radio(
    "Select metric",
    options=list(performance["Metric"].unique()),
    index=3,
    horizontal=True,
)
st.bar_chart(
    performance[performance["Metric"] == metric_selected],
    x="Model",
    y="Score",
    color="#b35300",
)
st.subheader("Trading")
st.plotly_chart(
    px.box(x=trades["MODEL"], y=trades["GROSS_PROFIT"]).update_layout(
        xaxis_title="Model", yaxis_title="Gross Profit [€]"
    )
)


# TRADES
st.title("Trading Statistics")
ts = dict(zip(trading_statistics["Statistic"], trading_statistics["Value"]))
counts = pd.DataFrame(
    {
        "trades": [f"{int(ts['N_TRADES'])} Trades"] * 2,
        "count": [
            f"{int(ts['N_TRADES_WIN'])} Wins",
            f"{int(ts['N_TRADES_LOSS'])} Losses",
        ],
        "percentage": [ts["WIN_RATE"], 100 - ts["WIN_RATE"]],
    }
)
st.plotly_chart(
    px.sunburst(
        counts,
        path=["trades", "count"],
        values="percentage",
        color="count",
        color_discrete_map={
            "(?)": "#000000",
            counts["count"][0]: "#00b400",
            counts["count"][1]: "#d80000",
        },
    )
)
a1, a2, a3 = st.columns(3)
a1.metric("Trades", f"{int(ts['N_TRADES'])}")
a2.metric("Volume", f"{int(ts['TOTAL_VOLUME'])}€")
a3.metric("Gross Profit", f"{ts['TOTAL_GROSS_PROFIT']}€")
b1, b2, b3 = st.columns(3)
b1.metric("Highest Win", f"{ts['MAX_WIN']}€")
b2.metric("Highest Loss", f"{ts['MAX_LOSS']}€")
b3.metric("Net Profit", f"{ts['TOTAL_NET_PROFIT']}€")
c1, c2, c3 = st.columns(3)
c1.metric("SQN", f"{ts['SQN']}")
c2.metric("Fees", f"{ts['TOTAL_FEES']}€")
c3.metric("Average Net Profit per Trade", f"{ts['AVG_PROFIT']}€")
if st.button("Show details"):
    st.dataframe(trades)


# BACKTEST
st.title("General Backtest")
st.subheader(
    f"Expected Profits when Trading Top ATX Stocks Initially Worth 1000€ as a Function of Model Precision"
)
precision = float(st.slider("Model Precision", 0.0, 1.0, 0.5, 0.05))
backtest["Holding Weeks"] = backtest["Holding Weeks"].astype(str)
st.plotly_chart(
    px.bar(
        backtest[backtest["Model Precision"] == precision],
        x="Holding Weeks",
        y="Expected Monthly Profit [€]",
        color=["red", "darkgreen"],
    )
)
st.plotly_chart(
    px.bar(
        backtest[backtest["Model Precision"] == precision],
        x="Holding Weeks",
        y="Expected Profit per Trade [€]",
        color=["darkgreen", "red"],
    )
)
