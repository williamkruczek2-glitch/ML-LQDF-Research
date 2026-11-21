"""Daily backtest of the strategy on SPY closes and a 30-day implied vol series (VIX / 100).

Each day at the close: mark the option structure to market with Black-Scholes at that day's
implied vol, apply the exit rules and the signal, open a new structure if flat, then
delta-hedge with SPY shares. Costs: option half-spread + fee per contract leg on entry and
exit, and basis points on hedge trades.

Known simplifications (see README): one flat implied vol for every strike (no skew), VIX as
the implied vol of the ATM options, an expiry exactly target_dte days out, fills at the close.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .metrics import summarise, variance_premium
from .risk import DrawdownGuard, exit_reason, hedge_trade, size_units
from .signals import Regime, SignalInputs, decide, in_blackout
from .structures import (Structure, long_straddle, max_loss_per_unit, short_iron_fly,
                         theoretical_delta, theoretical_value)
from .volatility import (HARForecaster, forward_realised_vol, log_returns, trailing_vol,
                         walk_forward_forecast)

KIND_TO_REGIME = {"short_iron_fly": Regime.SHORT_VOL, "long_straddle": Regime.LONG_VOL}


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


def _round(x: float, step: float) -> float:
    return round(round(x / step) * step, 6)


def build_structure(regime: Regime, spot: float, today: date, iv: float, equity: float,
                    cfg: StrategyConfig) -> Structure | None:
    expiry = today + timedelta(days=cfg.target_dte)
    atm = _round(spot, cfg.strike_step)
    if regime == Regime.SHORT_VOL:
        width = max(cfg.strike_step, _round(spot * cfg.wing_pct, cfg.strike_step))
        legs, kind = short_iron_fly(cfg.underlying, expiry, atm, atm - width, atm + width), "short_iron_fly"
    else:
        legs, kind = long_straddle(cfg.underlying, expiry, atm), "long_straddle"
    price = theoretical_value(legs, spot, today, iv, cfg)
    units = size_units(max_loss_per_unit(kind, legs, price), equity, cfg)
    if units < 1:
        return None
    return Structure(kind, legs, units, today, price)


def leg_costs(s: Structure, cfg: StrategyConfig) -> float:
    return s.units * len(s.legs) * (cfg.option_half_spread * 100 + cfg.option_fee_per_contract)


def run_backtest(close: pd.Series, iv: pd.Series, cfg: StrategyConfig | None = None, *,
                 forecast: pd.Series | None = None, model_factory=HARForecaster, signal_lag: int = 0,
                 events: list[date] | None = None, start=None, end=None) -> BacktestResult:
    cfg = cfg or StrategyConfig()
    df = pd.concat([close.rename("close"), iv.rename("iv")], axis=1).dropna()
    r = log_returns(df["close"])
    df["rv_5d"] = trailing_vol(r, 5)
    df["rv_forecast"] = forecast.reindex(df.index) if forecast is not None else walk_forward_forecast(
        df["close"], cfg.horizon_days, cfg.har_min_train, cfg.har_refit_every, model_factory)
    df["iv_5d_ago"] = df["iv"].shift(5)
    # what the signal is allowed to see (signal_lag > 0 = act on stale information)
    sig = df[["iv", "rv_forecast", "rv_5d", "iv_5d_ago"]].shift(signal_lag)
    if start is not None:
        df, sig = df.loc[start:], sig.loc[start:]
    if end is not None:
        df, sig = df.loc[:end], sig.loc[:end]
    events = events or []

    equity = cfg.initial_equity
    guard = DrawdownGuard(equity)
    s: Structure | None = None
    shares = 0
    prev_value = prev_spot = None
    rows, trades = [], []

    for ts, row in df.iterrows():
        d, spot, iv_now = ts.date(), float(row["close"]), float(row["iv"])
        x = sig.loc[ts]
        signal_ok = not x.isna()[["iv", "rv_forecast", "rv_5d"]].any()
        opt_pnl = hedge_pnl = costs = 0.0
        value = None
        note = ""

        if s is not None:
            value = theoretical_value(s.legs, spot, d, iv_now, cfg)
            opt_pnl = (value - prev_value) * 100 * s.units
        if shares:
            hedge_pnl = shares * (spot - prev_spot)
        equity_pre = equity + opt_pnl + hedge_pnl
        halted = guard.update(equity_pre, cfg)

        decision = None
        if signal_ok:
            current = KIND_TO_REGIME[s.kind] if s else Regime.FLAT
            decision = decide(SignalInputs(float(x["iv"]), float(x["rv_forecast"]), float(x["rv_5d"]),
                                           None if pd.isna(x["iv_5d_ago"]) else float(x["iv_5d_ago"]), current,
                                           in_blackout(d, events, cfg.event_blackout_days)), cfg)

        # exits
        if s is not None:
            reason = "drawdown halt" if halted else exit_reason(s, value, d, cfg)
            if reason is None and decision is not None and decision.target != KIND_TO_REGIME[s.kind]:
                reason = decision.reason
            if reason:
                c = leg_costs(s, cfg)
                costs += c
                trades.append({"entry_date": s.entry_date, "exit_date": d, "kind": s.kind, "units": s.units,
                               "entry_price": s.entry_price, "exit_value": value, "pnl": s.pnl(value) - 2 * c,
                               "reason": reason})
                note = f"close {s.kind}: {reason}"
                s, value = None, None
                if shares:
                    costs += abs(shares) * spot * cfg.stock_cost_bps / 1e4
                    shares = 0

        # entries (not on a day we just closed, and never after a halt)
        if s is None and not note and not halted and decision is not None and decision.target != Regime.FLAT:
            s = build_structure(decision.target, spot, d, iv_now, equity_pre - costs, cfg)
            if s is not None:
                s.meta = {"edge_pts": decision.edge_pts, "iv": iv_now}
                value = s.entry_price
                costs += leg_costs(s, cfg)
                note = f"open {s.kind} x{s.units}: {decision.reason}"

        # delta hedge
        if s is not None:
            opt_delta = theoretical_delta(s.legs, spot, d, iv_now, cfg) * 100 * s.units
            trade = hedge_trade(opt_delta, shares, s.units, cfg)
            if trade:
                costs += abs(trade) * spot * cfg.stock_cost_bps / 1e4
                shares += trade

        equity = equity_pre - costs
        rows.append({"date": ts, "close": spot, "iv": iv_now, "rv_forecast": row["rv_forecast"],
                     "edge_pts": decision.edge_pts if decision else np.nan,
                     "regime": KIND_TO_REGIME[s.kind].value if s else Regime.FLAT.value,
                     "units": s.units if s else 0, "hedge_shares": shares, "option_pnl": opt_pnl,
                     "hedge_pnl": hedge_pnl, "costs": costs, "equity": equity, "note": note})
        prev_value, prev_spot = value, spot

    daily = pd.DataFrame(rows).set_index("date")
    trades_df = pd.DataFrame(trades, columns=["entry_date", "exit_date", "kind", "units", "entry_price",
                                              "exit_value", "pnl", "reason"])
    metrics = summarise(daily["equity"], trades_df)
    metrics["total_costs"] = float(daily["costs"].sum())
    metrics["option_pnl"] = float(daily["option_pnl"].sum())
    metrics["hedge_pnl"] = float(daily["hedge_pnl"].sum())
    metrics["variance_premium"] = variance_premium(df["iv"], forward_realised_vol(r, cfg.horizon_days).reindex(df.index))
    return BacktestResult(daily, trades_df, metrics)
