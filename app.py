import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predictor import FEATURES, build_features, default_k, load_data, predict_next, train_models

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- caching
@st.cache_data(ttl=3600, show_spinner=False)
def get_raw(ticker: str, years: int) -> pd.DataFrame:
    return load_data(ticker, years)


@st.cache_resource(show_spinner=False)
def get_model(ticker: str, years: int, k: int, test_size: float):
    raw = get_raw(ticker, years)
    feats = build_features(raw)
    return train_models(feats, k=k or None, test_size=test_size)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📈 Stock Predictor")
ticker = st.sidebar.text_input(
    "Ticker symbol", value="AAPL", help="Indian stocks: RELIANCE.NS, TCS.NS, INFY.BO ..."
).strip().upper()
years = st.sidebar.slider("Years of history", 3, 15, 10)
test_pct = st.sidebar.slider("Test period (%)", 10, 40, 20, step=5)
auto_k = st.sidebar.checkbox("Auto-select K (√n rule)", value=True)
k_manual = 0
if not auto_k:
    k_manual = st.sidebar.slider("K (neighbours)", 3, 201, 51, step=2)
run = st.sidebar.button("Predict", type="primary", use_container_width=True)

st.title("Stock Price Predictor")
st.caption("KNN regression + classification on Yahoo Finance daily data. Educational project, not financial advice.")

if "active" not in st.session_state:
    st.session_state.active = None
if run:
    st.session_state.active = (ticker, years, k_manual, test_pct / 100)

if st.session_state.active is None:
    st.info("Enter a ticker in the sidebar and press **Predict**.")
    st.stop()

t, y, kk, ts = st.session_state.active

# ---------------------------------------------------------------- load + train
try:
    with st.spinner(f"Downloading {t} data and training models..."):
        raw = get_raw(t, y)
        bundle = get_model(t, y, kk, ts)
except Exception as e:  # network errors, bad ticker, too little data
    st.error(str(e))
    st.stop()

df = build_features(raw)
res = predict_next(bundle, df)
m = bundle.metrics

tab_pred, tab_perf, tab_whatif, tab_data = st.tabs(
    ["Prediction", "Model performance", "What-if", "Data"]
)

# ---------------------------------------------------------------- prediction
with tab_pred:
    st.subheader(f"{t}: next trading day after {res['as_of']:%d %b %Y}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"{res['last_close']:,.2f}")
    c2.metric(
        "Predicted next close",
        f"{res['pred_close']:,.2f}",
        f"{res['pred_return_pct']:+.2f}%",
    )
    c3.metric("Predicted direction", ("⬆ UP" if res["direction"] == "UP" else "⬇ DOWN"))
    c4.metric("Probability of up move", f"{res['prob_up'] * 100:.0f}%")

    recent = df.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent["Close"], name="Close", line=dict(width=2)))
    for ma in ("MA_5", "MA_20"):
        fig.add_trace(go.Scatter(x=recent.index, y=recent[ma], name=ma, line=dict(width=1, dash="dot")))
    next_day = res["as_of"] + pd.tseries.offsets.BDay(1)
    fig.add_trace(
        go.Scatter(
            x=[res["as_of"], next_day],
            y=[res["last_close"], res["pred_close"]],
            name="Prediction",
            mode="lines+markers",
            line=dict(color="#d62728", dash="dash"),
        )
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Daily stock moves are close to random. Check the **Model performance** tab: "
        "if direction accuracy is not clearly above the baseline, treat this output as a demo, not a signal."
    )

# ---------------------------------------------------------------- performance
with tab_perf:
    st.subheader("Held-out test results (most recent data the model never saw)")
    st.caption(
        f"Trained on {m['Train_Rows']:,} days from {m['Train_Start']:%b %Y}; "
        f"tested on {m['Test_Rows']:,} days ({m['Test_Start']:%d %b %Y} to {m['Test_End']:%d %b %Y}). K = {bundle.k}."
    )
    a, b, c, d = st.columns(4)
    a.metric("MAE", f"{m['MAE']:,.2f}", f"naive baseline {m['Baseline_MAE_naive']:,.2f}", delta_color="off")
    b.metric("RMSE", f"{m['RMSE']:,.2f}")
    c.metric("R² (price)", f"{m['R2']:.3f}")
    d.metric("MAPE", f"{m['MAPE_pct']:.2f}%")

    e, f = st.columns(2)
    e.metric(
        "Direction accuracy",
        f"{m['Direction_Accuracy_pct']:.1f}%",
        f"{m['Direction_Accuracy_pct'] - m['Baseline_Direction_pct']:+.1f} pts vs always-guess-majority",
        delta_color="normal",
    )
    f.metric("Majority-class baseline", f"{m['Baseline_Direction_pct']:.1f}%")

    tf = bundle.test_frame
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=tf.index, y=tf["Actual"], name="Actual next-day close"))
    fig2.add_trace(go.Scatter(x=tf.index, y=tf["Predicted"], name="Predicted", line=dict(dash="dot")))
    fig2.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Note: because price barely changes day to day, any model can look good on price error. "
        "The honest test is whether MAE beats the naive baseline and direction accuracy beats the majority baseline."
    )

# ---------------------------------------------------------------- what-if
with tab_whatif:
    st.subheader("What-if: enter your own latest-day values")
    st.caption("Replaces the most recent day's values (like the original manual-input mode) and re-runs the prediction.")
    last = raw.iloc[-1]
    with st.form("whatif"):
        c1, c2, c3 = st.columns(3)
        o = c1.number_input("Open", value=float(last["Open"]), format="%.2f")
        h = c2.number_input("High", value=float(last["High"]), format="%.2f")
        l = c3.number_input("Low", value=float(last["Low"]), format="%.2f")
        c4, c5 = st.columns(2)
        cl = c4.number_input("Close", value=float(last["Close"]), format="%.2f")
        vol = c5.number_input("Volume", value=float(last["Volume"]), step=1000.0, format="%.0f")
        go_btn = st.form_submit_button("Predict from these values")
    if go_btn:
        if not (l <= min(o, cl) and h >= max(o, cl)):
            st.error("High must be the highest and Low the lowest of the day's prices.")
        else:
            mod = raw.copy()
            mod.iloc[-1, mod.columns.get_loc("Open")] = o
            mod.iloc[-1, mod.columns.get_loc("High")] = h
            mod.iloc[-1, mod.columns.get_loc("Low")] = l
            mod.iloc[-1, mod.columns.get_loc("Close")] = cl
            mod.iloc[-1, mod.columns.get_loc("Volume")] = vol
            r2 = predict_next(bundle, build_features(mod))
            x1, x2, x3 = st.columns(3)
            x1.metric("Predicted next close", f"{r2['pred_close']:,.2f}", f"{r2['pred_return_pct']:+.2f}%")
            x2.metric("Direction", "⬆ UP" if r2["direction"] == "UP" else "⬇ DOWN")
            x3.metric("Probability of up move", f"{r2['prob_up'] * 100:.0f}%")

# ---------------------------------------------------------------- data
with tab_data:
    st.subheader("Latest data and features")
    show = df[["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "MA_20", "Daily_Return"]].tail(60)
    st.dataframe(show.sort_index(ascending=False), use_container_width=True)
    st.download_button(
        "Download full dataset (CSV)",
        df.to_csv().encode("utf-8"),
        file_name=f"{t}_features.csv",
        mime="text/csv",
    )
