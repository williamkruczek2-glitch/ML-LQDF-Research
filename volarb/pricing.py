"""Black-Scholes-Merton pricing, Greeks and implied volatility (continuous dividend yield q).

Conventions: vol and rates are decimals (0.15 = 15%), T in years, prices per share.
Vega is per 1.00 of vol (divide by 100 for per vol point); theta is per year.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _pdf(x):
    return np.exp(-0.5 * np.square(x)) / SQRT_2PI


def _d1d2(S, K, T, r, q, sigma):
    vol_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_t
    return d1, d1 - vol_t


def bs_price(S, K, T, r, q, sigma, right: str) -> float:
    """Price of a European call ('C') or put ('P'). At or after expiry returns intrinsic value."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if right == "C" else max(K - S, 0.0)
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    df_q, df_r = math.exp(-q * T), math.exp(-r * T)
    if right == "C":
        return float(S * df_q * ndtr(d1) - K * df_r * ndtr(d2))
    return float(K * df_r * ndtr(-d2) - S * df_q * ndtr(-d1))


def bs_greeks(S, K, T, r, q, sigma, right: str) -> dict:
    """delta, gamma, vega (per 1.00 vol), theta (per year)."""
    if T <= 0 or sigma <= 0:
        itm = (S > K) if right == "C" else (S < K)
        return {"delta": (1.0 if right == "C" else -1.0) if itm else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    df_q, df_r = math.exp(-q * T), math.exp(-r * T)
    pdf = float(_pdf(d1))
    gamma = df_q * pdf / (S * sigma * math.sqrt(T))
    vega = S * df_q * pdf * math.sqrt(T)
    if right == "C":
        delta = df_q * float(ndtr(d1))
        theta = -S * df_q * pdf * sigma / (2 * math.sqrt(T)) - r * K * df_r * float(ndtr(d2)) + q * S * df_q * float(ndtr(d1))
    else:
        delta = df_q * (float(ndtr(d1)) - 1.0)
        theta = -S * df_q * pdf * sigma / (2 * math.sqrt(T)) + r * K * df_r * float(ndtr(-d2)) - q * S * df_q * float(ndtr(-d1))
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def implied_vol(price, S, K, T, r, q, right: str, lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-7) -> float | None:
    """Implied volatility by bisection. None if the price is outside no-arbitrage bounds."""
    if T <= 0 or price <= 0:
        return None
    fwd_intrinsic = (S * math.exp(-q * T) - K * math.exp(-r * T)) if right == "C" else (K * math.exp(-r * T) - S * math.exp(-q * T))
    lower = max(fwd_intrinsic, 0.0)
    upper = S * math.exp(-q * T) if right == "C" else K * math.exp(-r * T)
    if not (lower - 1e-9 <= price <= upper + 1e-9):
        return None
    f_lo = bs_price(S, K, T, r, q, lo, right) - price
    f_hi = bs_price(S, K, T, r, q, hi, right) - price
    if f_lo > 0 or f_hi < 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(S, K, T, r, q, mid, right) - price
        if abs(f_mid) < tol:
            return mid
        if f_mid > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
