"""Alternative forecasters that plug into walk_forward_forecast / forecast_latest via model_factory.

Starter for the team: a gradient-boosting model on the same HAR inputs. It must beat HAR
out of sample (see leakage.shuffled_label_test and the backtest) before it replaces it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .volatility import VAR_FLOOR


class GradientBoostingForecaster:
    def __init__(self, **params):
        from sklearn.ensemble import GradientBoostingRegressor
        defaults = dict(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
        self.model = GradientBoostingRegressor(**{**defaults, **params})
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingForecaster":
        self.columns_ = list(X.columns)
        self.model.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.model.predict(X[self.columns_].to_numpy()), VAR_FLOOR)

    def feature_importances(self) -> dict:
        return dict(zip(self.columns_, self.model.feature_importances_))


MODELS = {"har": "volarb.volatility:HARForecaster", "gbm": "volarb.forecasters:GradientBoostingForecaster"}


def model_factory(name: str):
    import importlib
    module, cls = MODELS[name].split(":")
    return getattr(importlib.import_module(module), cls)
