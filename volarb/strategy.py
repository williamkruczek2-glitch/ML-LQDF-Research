"""The live strategy: run once a day near the close (and optionally re-hedge intraday).

Each daily run: forecast volatility from SPY history, read ~30-day ATM implied vol from the
option chain, apply the exit rules and the signal, open or close the structure with a
multi-leg order, then delta-hedge with SPY shares. State lives in state/state.json.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .broker import OptionQuote, OrderLeg, leg_prices
from .config import StrategyConfig
from .pricing import bs_greeks, implied_vol
from .risk import DrawdownGuard, exit_reason, hedge_trade, size_units
from .signals import Decision, Regime, SignalInputs, decide, in_blackout, load_events
from .structures import (Structure, choose_strikes, long_straddle, max_loss_per_unit, parse_occ,
                         pick_expiry, short_iron_fly)
from .volatility import HARForecaster, forecast_latest, log_returns, trailing_vol

log = logging.getLogger("volarb.strategy")
NY = ZoneInfo("America/New_York")
KIND_TO_REGIME = {"short_iron_fly": Regime.SHORT_VOL, "long_straddle": Regime.LONG_VOL}


class PositionMismatch(RuntimeError):
    pass


class VolArbStrategy:
    def __init__(self, broker, cfg: StrategyConfig, state_dir: str | Path = "state", model_factory=HARForecaster):
        self.broker, self.cfg, self.model_factory = broker, cfg, model_factory
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state.setdefault("structure", None)
        self.state.setdefault("iv_log", [])

    # ------------------------------------------------------------------------------- state
    def save(self) -> None:
        if getattr(self.broker, "dry_run", False):
            return                                    # a dry run never changes the record
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(self.state_path)

    def _append_csv(self, name: str, row: dict) -> None:
        if getattr(self.broker, "dry_run", False):
            return
        path = self.dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    @property
    def structure(self) -> Structure | None:
        return Structure.from_dict(self.state["structure"]) if self.state["structure"] else None

    def _set_structure(self, s: Structure | None) -> None:
        self.state["structure"] = s.to_dict() if s else None

    # --------------------------------------------------------------------------- helpers
    def reconcile(self, s: Structure | None) -> Structure | None:
        """Check the broker holds exactly what the state file says; refuse to trade otherwise."""
        held = {}
        for sym, qty in self.broker.positions().items():
            try:
                parse_occ(sym)
                held[sym] = qty
            except ValueError:
                pass
        expected = {l.symbol: l.qty * s.units for l in s.legs} if s else {}
        if held == expected:
            return s
        if s and not held:
            log.warning("state says %s is open but the account holds no options; clearing it "
                        "(closed manually or exercised?)", s.kind)
            self._set_structure(None)
            return None
        raise PositionMismatch(f"account options {held} do not match state {expected}; fix by hand before trading")

    def atm_iv(self, quotes: list[OptionQuote], atm: float, spot: float, today: date) -> float:
        ivs = []
        for q in quotes:
            if q.strike != atm:
                continue
            if q.iv:
                ivs.append(q.iv)
            elif q.mid:
                T = max((q.expiry - today).days, 1) / 365
                v = implied_vol(q.mid, spot, q.strike, T, self.cfg.rate, self.cfg.div_yield, q.right)
                if v:
                    ivs.append(v)
        if not ivs:
            raise ValueError(f"no implied vol available at strike {atm}")
        return sum(ivs) / len(ivs)

    def _iv_days_ago(self, n: int) -> float | None:
        logd = self.state["iv_log"]
        return float(logd[-n]["iv"]) if len(logd) >= n else None

    def _value(self, s: Structure, quotes: dict[str, OptionQuote]) -> float:
        return sum(l.qty * quotes[l.symbol].mid for l in s.legs)

    def _option_delta_shares(self, s: Structure, quotes: dict[str, OptionQuote], spot: float, today: date,
                             iv: float) -> float:
        total = 0.0
        for l in s.legs:
            q = quotes.get(l.symbol)
            d = q.delta if q and q.delta is not None else bs_greeks(
                spot, l.strike, max((l.expiry - today).days, 0) / 365, self.cfg.rate, self.cfg.div_yield,
                (q.iv if q and q.iv else iv), l.right)["delta"]
            total += l.qty * d
        return total * 100 * s.units

    # ---------------------------------------------------------------------------- actions
    def _open(self, decision: Decision, chain: dict[str, OptionQuote], expiry: date, atm: float, lower: float,
              upper: float, equity: float, today: date) -> Structure | None:
        u = self.cfg.underlying
        if decision.target == Regime.SHORT_VOL:
            kind, legs = "short_iron_fly", short_iron_fly(u, expiry, atm, lower, upper)
        else:
            kind, legs = "long_straddle", long_straddle(u, expiry, atm)
        missing = [l.symbol for l in legs if l.symbol not in chain]
        if missing:
            log.warning("contracts not in chain: %s", missing)
            return None
        order_legs = [OrderLeg(l.symbol, "buy" if l.qty > 0 else "sell", "buy_to_open" if l.qty > 0 else "sell_to_open")
                      for l in legs]
        mid, natural = leg_prices(order_legs, chain)
        units = size_units(max_loss_per_unit(kind, legs, mid), equity, self.cfg)
        if units < 1:
            log.info("position too small for the risk budget; skipping")
            return None
        res = self.broker.place_structure_order(order_legs, units, mid, natural, self.cfg.order_attempts,
                                                self.cfg.order_wait_seconds)
        if res.status not in ("filled", "dry_run"):
            log.warning("open order not filled")
            return None
        s = Structure(kind, legs, units, today, res.price, {"edge_pts": decision.edge_pts, "reason": decision.reason})
        self._append_csv("trades.csv", {"time": datetime.now(NY).isoformat(timespec="seconds"), "action": "open",
                                        "kind": kind, "units": units, "price": res.price, "pnl": "",
                                        "reason": decision.reason})
        return s

    def _close(self, s: Structure, quotes: dict[str, OptionQuote], reason: str) -> bool:
        order_legs = [OrderLeg(l.symbol, "sell" if l.qty > 0 else "buy", "sell_to_close" if l.qty > 0 else "buy_to_close")
                      for l in s.legs]
        mid, natural = leg_prices(order_legs, quotes)
        res = self.broker.place_structure_order(order_legs, s.units, mid, natural, self.cfg.order_attempts,
                                                self.cfg.order_wait_seconds)
        if res.status not in ("filled", "dry_run"):
            log.warning("close order not filled; will retry next run")
            return False
        pnl = s.pnl(-res.price)                       # closing price = -(value of the position)
        self._append_csv("trades.csv", {"time": datetime.now(NY).isoformat(timespec="seconds"), "action": "close",
                                        "kind": s.kind, "units": s.units, "price": res.price, "pnl": round(pnl, 2),
                                        "reason": reason})
        return True

    def _hedge(self, s: Structure | None, spot: float, today: date, iv: float) -> int:
        u = self.cfg.underlying
        stock = self.broker.positions().get(u, 0)
        if s is None:
            trade = -stock
        else:
            quotes = self.broker.option_quotes([l.symbol for l in s.legs])
            trade = hedge_trade(self._option_delta_shares(s, quotes, spot, today, iv), stock, s.units, self.cfg)
        if trade:
            self.broker.submit_stock(u, trade)
        return trade

    # -------------------------------------------------------------------------------- runs
    def run_daily(self, today: date | None = None) -> dict:
        cfg, b, u = self.cfg, self.broker, self.cfg.underlying
        today = today or datetime.now(NY).date()
        equity = b.account_equity()
        guard = DrawdownGuard(self.state.get("equity_peak", equity), self.state.get("halted", False))
        halted = guard.update(equity, cfg)
        self.state["equity_peak"], self.state["halted"] = guard.peak, guard.halted

        spot = b.spot(u)
        closes = b.daily_closes(u, cfg.history_days)
        if closes.index[-1].date() < today:          # today's price as the provisional close
            closes.loc[pd.Timestamp(today)] = spot
        forecast = forecast_latest(closes, cfg.horizon_days, cfg.har_min_train, self.model_factory)
        rv5 = float(trailing_vol(log_returns(closes), 5).iloc[-1])

        band = cfg.wing_pct + 0.03
        chain_list = b.option_chain(u, today + timedelta(days=cfg.min_dte), today + timedelta(days=cfg.max_dte),
                                    round(spot * (1 - band)), round(spot * (1 + band)))
        expiry = pick_expiry(sorted({q.expiry for q in chain_list}), today, cfg)
        if expiry is None:
            raise RuntimeError(f"no expiry between {cfg.min_dte} and {cfg.max_dte} days")
        chain = {q.symbol: q for q in chain_list if q.expiry == expiry}
        atm, lower, upper = choose_strikes(sorted({q.strike for q in chain.values()}), spot, cfg)
        iv = self.atm_iv(list(chain.values()), atm, spot, today)

        s = self.reconcile(self.structure)
        current = KIND_TO_REGIME[s.kind] if s else Regime.FLAT
        events = load_events(cfg.events_file)
        decision = decide(SignalInputs(iv, forecast, rv5, self._iv_days_ago(5), current,
                                       in_blackout(today, events, cfg.event_blackout_days)), cfg)
        actions = []

        if s is not None:
            quotes = b.option_quotes([l.symbol for l in s.legs])
            reason = "drawdown halt" if halted else exit_reason(s, self._value(s, quotes), today, cfg)
            if reason is None and decision.target != current:
                reason = decision.reason
            if reason and self._close(s, quotes, reason):
                actions.append(f"closed {s.kind}: {reason}")
                s = None
                self._set_structure(None)

        if s is None and not actions and not halted and decision.target != Regime.FLAT:
            s = self._open(decision, chain, expiry, atm, lower, upper, equity, today)
            if s:
                actions.append(f"opened {s.kind} x{s.units} at {s.entry_price:+.2f}")
                self._set_structure(s)

        if s is not None or self.broker.positions().get(u, 0):
            trade = self._hedge(s, spot, today, iv)
            if trade:
                actions.append(f"hedge {trade:+d} {u}")

        self.state["iv_log"] = (self.state["iv_log"] + [{"date": today.isoformat(), "iv": iv}])[-60:]
        summary = {"date": today.isoformat(), "equity": round(equity, 2), "spot": spot, "iv": round(iv, 4),
                   "forecast": round(forecast, 4), "edge_pts": round(decision.edge_pts, 2),
                   "rv_5d": round(rv5, 4), "target": decision.target.value, "reason": decision.reason,
                   "position": s.kind if s else "flat", "halted": halted, "actions": "; ".join(actions) or "none"}
        self._append_csv("daily_log.csv", summary)
        self.save()
        return summary

    def run_hedge_only(self, today: date | None = None) -> dict:
        """Intraday re-hedge using live deltas (run every hour or so while a structure is open)."""
        today = today or datetime.now(NY).date()
        s = self.reconcile(self.structure)
        spot = self.broker.spot(self.cfg.underlying)
        iv = float(self.state["iv_log"][-1]["iv"]) if self.state["iv_log"] else 0.2
        trade = self._hedge(s, spot, today, iv) if (s or self.broker.positions().get(self.cfg.underlying, 0)) else 0
        self.save()
        return {"date": today.isoformat(), "spot": spot, "hedge_trade": trade, "position": s.kind if s else "flat"}
