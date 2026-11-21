"""Realised volatility, the HAR forecast, and point-in-time (purged) walk-forward forecasting.

All volatilities are annualised decimals (0.15 = 15%). Models predict annualised *variance*
over the next `h` trading days; forecasts are returned as volatility (sqrt of variance).

Leakage rule enforced here: at day i the model may only train on samples j whose label
(returns j+1 .. j+h) is fully known, i.e. j + h <= i.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd

ANN = 252
VAR_FLOOR = 1e-4  # 1% vol


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def trailing_vol(returns: pd.Series, window: int) -> pd.Series:
    """Volatility over the last `window` days, known at the close of each day."""
    return np.sqrt(returns.pow(2).rolling(window).mean() * ANN)


def forward_realised_vol(returns: pd.Series, h: int) -> pd.Series:
    """Volatility over the NEXT h days (uses the future: a label, never an input)."""
    fwd_var = returns.pow(2)[::-1].rolling(h).mean()[::-1].shift(-1) * ANN
    return np.sqrt(fwd_var).rename(f"rv_fwd_{h}d")


def har_features(returns: pd.Series) -> pd.DataFrame:
    """Annualised variance over the last 1, 5 and 22 days (Corsi's HAR inputs)."""
    r2 = returns.pow(2)
    return pd.DataFrame({
        "var_d": r2 * ANN,
        "var_w": r2.rolling(5).mean() * ANN,
        "var_m": r2.rolling(22).mean() * ANN,
    })


class Forecaster(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class HARForecaster:
    """Ordinary least squares on [1, var_d, var_w, var_m] (plus any extra columns)."""

    def __init__(self):
        self.coef_: np.ndarray | None = None
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARForecaster":
        self.columns_ = list(X.columns)
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])
        self.coef_, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        A = np.column_stack([np.ones(len(X)), X[self.columns_].to_numpy()])
        return np.maximum(A @ self.coef_, VAR_FLOOR)


def build_dataset(close: pd.Series, h: int, extra_features: pd.DataFrame | None = None):
    """Features known at each close, plus the forward-variance label (NaN where unknown)."""
    r = log_returns(close)
    X = har_features(r)
    if extra_features is not None:
        X = X.join(extra_features, how="left")
    y = forward_realised_vol(r, h).pow(2)
    return X, y


def walk_forward_forecast(close: pd.Series, h: int = 21, min_train: int = 504, refit_every: int = 21,
                          model_factory: Callable[[], Forecaster] = HARForecaster,
                          extra_features: pd.DataFrame | None = None) -> pd.Series:
    """Point-in-time forecast of next-h-day volatility for every day with enough history."""
    X, y = build_dataset(close, h, extra_features)
    feat_ok = X.notna().all(axis=1).to_numpy()
    lab_ok = y.notna().to_numpy()
    n = len(close)
    out = np.full(n, np.nan)
    model = None
    last_fit = -10**9
    for i in range(n):
        if not feat_ok[i]:
            continue
        # samples whose label is fully observed by day i
        train_idx = np.flatnonzero(feat_ok[: max(i - h + 1, 0)] & lab_ok[: max(i - h + 1, 0)])
        if len(train_idx) < min_train:
            continue
        if model is None or i - last_fit >= refit_every:
            model = model_factory().fit(X.iloc[train_idx], y.iloc[train_idx])
            last_fit = i
        out[i] = model.predict(X.iloc[[i]])[0]
    return pd.Series(np.sqrt(out), index=close.index, name="rv_forecast")


def forecast_latest(close: pd.Series, h: int = 21, min_train: int = 504,
                    model_factory: Callable[[], Forecaster] = HARForecaster,
                    extra_features: pd.DataFrame | None = None) -> float:
    """Forecast for the last day only, trained on every label known by then (used live)."""
    X, y = build_dataset(close, h, extra_features)
    i = len(close) - 1
    if not X.iloc[i].notna().all():
        raise ValueError("not enough history for features on the latest day")
    ok = (X.notna().all(axis=1) & y.notna()).to_numpy()
    train_idx = np.flatnonzero(ok[: max(i - h + 1, 0)])
    if len(train_idx) < min_train:
        raise ValueError(f"only {len(train_idx)} labelled samples; need {min_train}")
    model = model_factory().fit(X.iloc[train_idx], y.iloc[train_idx])
    return float(np.sqrt(model.predict(X.iloc[[i]])[0]))
