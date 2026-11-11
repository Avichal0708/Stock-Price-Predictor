"""
Core logic for the Stock Price Predictor.

Converted from the original notebook (Stock_Price_Predictor.ipynb):
- same idea: KNN regressor (next-day close) + KNN classifier (up/down)
- same data source: yfinance daily data
Fixes applied are listed in README.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Scale-free features (ratios / returns) so the model works for any stock
# price level and never has to "extrapolate" beyond prices seen in training.
FEATURES = [
    "Open_Close",      # Open / Close - 1
    "High_Close",      # High / Close - 1
    "Low_Close",       # Low / Close - 1
    "Daily_Return",    # Close.pct_change()
    "Close_MA5",       # Close / MA_5 - 1
    "Close_MA10",      # Close / MA_10 - 1
    "Close_MA20",      # Close / MA_20 - 1
    "Volume_Ratio",    # Volume / 20-day avg volume
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(ticker: str, years: int = 10) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    import yfinance as yf  # imported here so the module can be tested offline

    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Please enter a ticker symbol.")

    end = datetime.today().date() + timedelta(days=1)  # end is exclusive
    start = end - timedelta(days=int(365.25 * years))

    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if data is None or data.empty:
        raise ValueError(
            f"No data found for '{ticker}'. Check the symbol "
            "(Indian stocks need a suffix, e.g. RELIANCE.NS or TCS.BO)."
        )

    # Newer yfinance versions return MultiIndex columns even for one ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")

    df = data[needed].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Add indicators, features and targets to raw OHLCV data."""
    df = raw.copy()

    # Drop rows with invalid prices instead of filling them with the mean.
    price_cols = ["Open", "High", "Low", "Close"]
    df = df[(df[price_cols] > 0).all(axis=1)]

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Daily_Return"] = df["Close"].pct_change()

    df["Open_Close"] = df["Open"] / df["Close"] - 1
    df["High_Close"] = df["High"] / df["Close"] - 1
    df["Low_Close"] = df["Low"] / df["Close"] - 1
    df["Close_MA5"] = df["Close"] / df["MA_5"] - 1
    df["Close_MA10"] = df["Close"] / df["MA_10"] - 1
    df["Close_MA20"] = df["Close"] / df["MA_20"] - 1
    vol_avg = df["Volume"].rolling(20).mean().replace(0, np.nan)
    df["Volume_Ratio"] = df["Volume"] / vol_avg

    # Targets (NaN on the very last row - that is the row we predict from)
    df["Next_Day_Close"] = df["Close"].shift(-1)
    df["Next_Return"] = df["Next_Day_Close"] / df["Close"] - 1
    df["Price_Direction"] = np.where(
        df["Next_Day_Close"].isna(), np.nan, (df["Next_Day_Close"] > df["Close"]).astype(float)
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class ModelBundle:
    scaler: StandardScaler
    regressor: KNeighborsRegressor
    classifier: KNeighborsClassifier
    k: int
    metrics: dict
    test_frame: pd.DataFrame  # actual vs predicted on the held-out period


def default_k(n_samples: int) -> int:
    """sqrt(n) rule from the notebook, forced odd (avoids classifier ties)."""
    k = max(3, int(np.sqrt(n_samples)))
    return k if k % 2 == 1 else k + 1


def train_models(df: pd.DataFrame, k: int | None = None, test_size: float = 0.2) -> ModelBundle:
    """Train on the older data, test on the most recent data (no shuffling)."""
    data = df.dropna(subset=FEATURES + ["Next_Return", "Price_Direction"])
    if len(data) < 200:
        raise ValueError("Not enough history to train (need at least ~200 trading days).")

    split = int(len(data) * (1 - test_size))
    train, test = data.iloc[:split], data.iloc[split:]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURES])
    x_test = scaler.transform(test[FEATURES])  # transform only - no leakage

    k = k or default_k(len(train))
    reg = KNeighborsRegressor(n_neighbors=k, weights="distance")
    clf = KNeighborsClassifier(n_neighbors=k, p=2, metric="euclidean")
    reg.fit(x_train, train["Next_Return"])
    clf.fit(x_train, train["Price_Direction"].astype(int))

    pred_ret = reg.predict(x_test)
    pred_price = test["Close"].to_numpy() * (1 + pred_ret)
    actual_price = test["Next_Day_Close"].to_numpy()
    pred_dir = clf.predict(x_test)
    actual_dir = test["Price_Direction"].astype(int).to_numpy()

    mse = mean_squared_error(actual_price, pred_price)
    naive_price = test["Close"].to_numpy()  # baseline: "tomorrow = today"
    majority = max(actual_dir.mean(), 1 - actual_dir.mean())

    metrics = {
        "MAE": mean_absolute_error(actual_price, pred_price),
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(actual_price, pred_price),
        "MAPE_pct": mean_absolute_percentage_error(actual_price, pred_price) * 100,
        "Direction_Accuracy_pct": float((pred_dir == actual_dir).mean() * 100),
        "Baseline_MAE_naive": mean_absolute_error(actual_price, naive_price),
        "Baseline_Direction_pct": float(majority * 100),
        "Train_Rows": len(train),
        "Test_Rows": len(test),
        "Train_Start": train.index[0],
        "Test_Start": test.index[0],
        "Test_End": test.index[-1],
    }

    test_frame = pd.DataFrame(
        {"Actual": actual_price, "Predicted": pred_price}, index=test.index
    )
    return ModelBundle(scaler, reg, clf, k, metrics, test_frame)


def predict_next(bundle: ModelBundle, df: pd.DataFrame) -> dict:
    """Predict the next trading day from the last row of `df`."""
    last = df.iloc[[-1]]
    if last[FEATURES].isna().any(axis=None):
        raise ValueError("Latest row has missing values; need more history.")

    x = bundle.scaler.transform(last[FEATURES])
    pred_ret = float(bundle.regressor.predict(x)[0])
    close = float(last["Close"].iloc[0])
    proba = bundle.classifier.predict_proba(x)[0]
    classes = list(bundle.classifier.classes_)
    p_up = float(proba[classes.index(1)]) if 1 in classes else 0.0

    return {
        "as_of": last.index[0],
        "last_close": close,
        "pred_close": close * (1 + pred_ret),
        "pred_return_pct": pred_ret * 100,
        "direction": "UP" if p_up >= 0.5 else "DOWN",
        "prob_up": p_up,
    }
