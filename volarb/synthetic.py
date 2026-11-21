"""Synthetic market with a known variance risk premium, for tests and demos (no internet needed).

Volatility switches between a calm and a stressed regime. Implied vol is the expected
next-month vol given today's regime (knowable today) plus `premium_pts` vol points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_market(n_days: int = 2500, premium_pts: float = 3.0, seed: int = 7, calm: float = 0.12,
                stressed: float = 0.30, stay: float = 0.985, noise_pts: float = 0.5, start: str = "2016-01-04"):
    rng = np.random.default_rng(seed)
    P = np.array([[stay, 1 - stay], [1 - stay, stay]])
    sig = np.array([calm, stressed])
    # expected annualised variance over the next 21 days from each regime
    exp_var = np.zeros(2)
    Pk = np.eye(2)
    for _ in range(21):
        Pk = Pk @ P
        exp_var += Pk @ (sig ** 2)
    exp_vol = np.sqrt(exp_var / 21)

    state = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        state[t] = state[t - 1] if rng.random() < P[state[t - 1], state[t - 1]] else 1 - state[t - 1]
    # Student-t shocks (fat tails), scaled to unit variance
    shocks = rng.standard_t(5, n_days) / np.sqrt(5 / 3)
    rets = sig[state] / np.sqrt(252) * shocks
    rets[0] = 0.0
    idx = pd.bdate_range(start, periods=n_days)
    close = pd.Series(400 * np.exp(np.cumsum(rets)), index=idx, name="close")
    iv = pd.Series(exp_vol[state] + premium_pts / 100 + rng.normal(0, noise_pts / 100, n_days), index=idx, name="iv")
    return close, iv.clip(lower=0.05)
