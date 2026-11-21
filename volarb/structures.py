"""Option structures: OCC symbols, expiry/strike selection, the short iron butterfly and long straddle.

Prices are per share. A structure's net price follows Alpaca's multi-leg convention:
positive = debit paid, negative = credit received.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date

from .config import StrategyConfig
from .pricing import bs_greeks, bs_price

OCC_RE = re.compile(r"^([A-Z.]{1,6})(\d{6})([CP])(\d{8})$")


def occ_symbol(root: str, expiry: date, right: str, strike: float) -> str:
    return f"{root}{expiry:%y%m%d}{right}{int(round(strike * 1000)):08d}"


def parse_occ(symbol: str) -> tuple[str, date, str, float]:
    m = OCC_RE.match(symbol)
    if not m:
        raise ValueError(f"not an OCC option symbol: {symbol}")
    root, ymd, right, k = m.groups()
    return root, date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:])), right, int(k) / 1000


@dataclass(frozen=True)
class Leg:
    symbol: str
    right: str        # 'C' or 'P'
    strike: float
    expiry: date
    qty: int          # +1 long / -1 short, per unit of the structure


@dataclass
class Structure:
    kind: str                     # 'short_iron_fly' | 'long_straddle'
    legs: list[Leg]
    units: int
    entry_date: date
    entry_price: float            # net per share: + debit, - credit
    meta: dict = field(default_factory=dict)

    @property
    def expiry(self) -> date:
        return min(l.expiry for l in self.legs)

    def dte(self, today: date) -> int:
        return (self.expiry - today).days

    def max_loss_per_unit(self) -> float:
        return max_loss_per_unit(self.kind, self.legs, self.entry_price)

    def pnl(self, current_value: float) -> float:
        """Dollar P&L since entry, given the current net value per share."""
        return (current_value - self.entry_price) * 100 * self.units

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_date"] = self.entry_date.isoformat()
        for leg in d["legs"]:
            leg["expiry"] = leg["expiry"].isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Structure":
        legs = [Leg(l["symbol"], l["right"], float(l["strike"]), date.fromisoformat(l["expiry"]), int(l["qty"])) for l in d["legs"]]
        return cls(d["kind"], legs, int(d["units"]), date.fromisoformat(d["entry_date"]), float(d["entry_price"]), d.get("meta", {}))


def short_iron_fly(root: str, expiry: date, atm: float, lower: float, upper: float) -> list[Leg]:
    """Sell the ATM call and put, buy the wings. Every short leg is covered, as Alpaca requires."""
    return [
        Leg(occ_symbol(root, expiry, "P", lower), "P", lower, expiry, +1),
        Leg(occ_symbol(root, expiry, "P", atm), "P", atm, expiry, -1),
        Leg(occ_symbol(root, expiry, "C", atm), "C", atm, expiry, -1),
        Leg(occ_symbol(root, expiry, "C", upper), "C", upper, expiry, +1),
    ]


def long_straddle(root: str, expiry: date, atm: float) -> list[Leg]:
    return [
        Leg(occ_symbol(root, expiry, "P", atm), "P", atm, expiry, +1),
        Leg(occ_symbol(root, expiry, "C", atm), "C", atm, expiry, +1),
    ]


def max_loss_per_unit(kind: str, legs: list[Leg], entry_price: float) -> float:
    """Worst-case dollar loss per unit, held to expiry."""
    if kind == "long_straddle":
        return entry_price * 100
    if kind == "short_iron_fly":
        shorts = [l.strike for l in legs if l.qty < 0]
        atm = shorts[0]
        width = max(abs(l.strike - atm) for l in legs if l.qty > 0)
        return (width + entry_price) * 100      # entry_price is negative (credit)
    raise ValueError(kind)


def pick_expiry(expiries: list[date], today: date, cfg: StrategyConfig) -> date | None:
    ok = [e for e in expiries if cfg.min_dte <= (e - today).days <= cfg.max_dte]
    return min(ok, key=lambda e: abs((e - today).days - cfg.target_dte)) if ok else None


def nearest(values: list[float], x: float) -> float:
    return min(values, key=lambda v: (abs(v - x), v))


def choose_strikes(strikes: list[float], spot: float, cfg: StrategyConfig) -> tuple[float, float, float]:
    """ATM strike and the wings closest to +/- wing_pct of spot."""
    atm = nearest(strikes, spot)
    lower = nearest([k for k in strikes if k < atm] or [atm], spot * (1 - cfg.wing_pct))
    upper = nearest([k for k in strikes if k > atm] or [atm], spot * (1 + cfg.wing_pct))
    if lower >= atm or upper <= atm:
        raise ValueError("not enough strikes either side of the money for wings")
    return atm, lower, upper


def theoretical_value(legs: list[Leg], spot: float, today: date, iv: float, cfg: StrategyConfig) -> float:
    """Net value per share using one flat implied vol (backtest pricing)."""
    return sum(l.qty * bs_price(spot, l.strike, max((l.expiry - today).days, 0) / 365, cfg.rate, cfg.div_yield, iv, l.right)
               for l in legs)


def theoretical_delta(legs: list[Leg], spot: float, today: date, iv: float, cfg: StrategyConfig) -> float:
    """Net delta per unit, in shares-per-share terms (multiply by 100 x units for share count)."""
    return sum(l.qty * bs_greeks(spot, l.strike, max((l.expiry - today).days, 0) / 365, cfg.rate, cfg.div_yield, iv, l.right)["delta"]
               for l in legs)
