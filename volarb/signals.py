"""Turn implied vs forecast volatility into a target position: short vol, long vol or flat."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path

from .config import StrategyConfig


class Regime(str, Enum):
    FLAT = "flat"
    SHORT_VOL = "short_vol"   # implied rich -> sell an iron butterfly
    LONG_VOL = "long_vol"     # implied cheap -> buy a straddle


@dataclass
class SignalInputs:
    iv: float                       # implied vol for ~30 days (decimal)
    rv_forecast: float              # forecast realised vol over the same horizon
    rv_5d: float                    # realised vol over the last 5 days
    iv_5d_ago: float | None = None
    current: Regime = Regime.FLAT
    event_blackout: bool = False


@dataclass
class Decision:
    target: Regime
    edge_pts: float                 # (implied - forecast) in vol points
    reason: str


def decide(x: SignalInputs, cfg: StrategyConfig) -> Decision:
    edge = (x.iv - x.rv_forecast) * 100
    ratio = x.iv / x.rv_forecast if x.rv_forecast > 0 else float("inf")
    stress_reasons = []
    if cfg.stress_rv5_over_iv and x.rv_5d > x.iv:
        stress_reasons.append("5-day realised above implied")
    if x.iv_5d_ago is not None and (x.iv - x.iv_5d_ago) * 100 > cfg.stress_iv_jump:
        stress_reasons.append(f"implied up {(x.iv - x.iv_5d_ago) * 100:.1f} pts in 5 days")

    if x.current == Regime.SHORT_VOL:
        if edge < cfg.exit_edge:
            return Decision(Regime.FLAT, edge, f"edge {edge:.1f} below exit {cfg.exit_edge}")
        return Decision(Regime.SHORT_VOL, edge, "hold short vol")
    if x.current == Regime.LONG_VOL:
        if -edge < cfg.exit_edge:
            return Decision(Regime.FLAT, edge, f"edge {edge:.1f} no longer negative enough")
        return Decision(Regime.LONG_VOL, edge, "hold long vol")

    if edge >= cfg.short_entry_edge and ratio >= cfg.short_entry_ratio:
        if stress_reasons:
            return Decision(Regime.FLAT, edge, "short-vol signal blocked: " + "; ".join(stress_reasons))
        if x.event_blackout:
            return Decision(Regime.FLAT, edge, "short-vol signal blocked: event blackout")
        return Decision(Regime.SHORT_VOL, edge, f"implied {x.iv:.1%} vs forecast {x.rv_forecast:.1%}")
    if cfg.enable_long_vol and -edge >= cfg.long_entry_edge:
        return Decision(Regime.LONG_VOL, edge, f"implied {x.iv:.1%} below forecast {x.rv_forecast:.1%}")
    return Decision(Regime.FLAT, edge, "no edge")


def load_events(path: str | Path | None) -> list[date]:
    """events.csv: one row per scheduled market event (FOMC, CPI ...), columns date,label."""
    if not path or not Path(path).exists():
        return []
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(line for line in f if line.strip() and not line.lstrip().startswith("#")):
            try:
                out.append(date.fromisoformat((row.get("date") or "").strip()))
            except ValueError:
                raise ValueError(f"{path}: bad date {row.get('date')!r}; use YYYY-MM-DD") from None
    return sorted(out)


def in_blackout(today: date, events: list[date], days: int) -> bool:
    """True if an event falls between today and the next `days` calendar days (weekends skipped)."""
    if days <= 0 or not events:
        return False
    horizon, d, counted = today, today, 0
    while counted < days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            counted += 1
            horizon = d
    return any(today <= e <= horizon for e in events)
