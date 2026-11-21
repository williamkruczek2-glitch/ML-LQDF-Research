"""Leakage checks: prove the forecast never sees the future, and that results survive delay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .volatility import HARForecaster, build_dataset, walk_forward_forecast


def truncation_test(close: pd.Series, cfg: StrategyConfig, n_cuts: int = 5, model_factory=HARForecaster) -> dict:
    """Forecasts made with data up to day c must equal those made with the full history."""
    full = walk_forward_forecast(close, cfg.horizon_days, cfg.har_min_train, cfg.har_refit_every, model_factory)
    valid = np.flatnonzero(full.notna().to_numpy())
    if len(valid) == 0:
        return {"passed": None, "detail": "no forecasts to test"}
    cuts = np.linspace(valid[0], len(close) - 1, n_cuts).astype(int)
    worst = 0.0
    for c in cuts:
        part = walk_forward_forecast(close.iloc[: c + 1], cfg.horizon_days, cfg.har_min_train, cfg.har_refit_every, model_factory)
        worst = max(worst, abs(float(part.iloc[-1]) - float(full.iloc[c])))
    return {"passed": bool(worst < 1e-10), "max_abs_diff": worst, "cuts_checked": len(cuts)}


def shuffled_label_test(close: pd.Series, cfg: StrategyConfig, seed: int = 0, n_shuffles: int = 20,
                        model_factory=HARForecaster) -> dict:
    """Train on real vs shuffled labels; shuffled skill should collapse towards zero."""
    X, y = build_dataset(close, cfg.horizon_days)
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    split = int(len(X) * 0.7)
    gap = cfg.horizon_days                            # purge: train labels end before test starts
    Xtr, ytr, Xte, yte = X.iloc[: split - gap], y.iloc[: split - gap], X.iloc[split:], y.iloc[split:]
    actual = np.sqrt(yte.to_numpy())
    benchmark = float(np.sqrt(ytr).mean())            # "always predict the average training vol"

    def r2(pred):
        return float(1 - np.sum((np.sqrt(pred) - actual) ** 2) / np.sum((benchmark - actual) ** 2))

    real = r2(model_factory().fit(Xtr, ytr).predict(Xte))
    rng = np.random.default_rng(seed)
    shuffled = [r2(model_factory().fit(Xtr, pd.Series(rng.permutation(ytr.to_numpy()), index=ytr.index)).predict(Xte))
                for _ in range(n_shuffles)]
    return {"real_r2": real, "shuffled_r2": float(np.mean(shuffled)), "shuffled_r2_max": float(np.max(shuffled)),
            "test_days": int(len(Xte)), "passed": bool(real > np.max(shuffled))}


def delay_test(close: pd.Series, iv: pd.Series, cfg: StrategyConfig, run, **kw) -> dict:
    """Run the backtest acting on today's signal vs yesterday's; a real edge should survive."""
    base = run(close, iv, cfg, signal_lag=0, **kw).metrics
    lagged = run(close, iv, cfg, signal_lag=1, **kw).metrics
    return {"sharpe_lag0": base["sharpe"], "sharpe_lag1": lagged["sharpe"],
            "return_lag0": base["total_return"], "return_lag1": lagged["total_return"]}
