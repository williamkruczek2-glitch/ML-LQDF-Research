"""A fake Alpaca account: options priced by Black-Scholes at a chosen implied vol, instant fills."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from volarb.broker import OptionQuote, OrderLeg, OrderResult
from volarb.config import StrategyConfig
from volarb.pricing import bs_greeks, bs_price
from volarb.structures import occ_symbol, parse_occ


class FakeBroker:
    def __init__(self, closes: pd.Series, spot: float, iv: float, today: date, equity: float = 100_000.0,
                 half_spread: float = 0.02, dry_run: bool = False):
        self.closes, self._spot, self.iv, self.today, self.equity = closes, spot, iv, today, equity
        self.half_spread, self.dry_run = half_spread, dry_run
        self.pos: dict[str, int] = {}
        self.orders: list[dict] = []
        self.stock_orders: list[int] = []
        self.cfg = StrategyConfig()

    def account_equity(self) -> float:
        return self.equity

    def positions(self) -> dict[str, int]:
        return dict(self.pos)

    def daily_closes(self, symbol: str, days: int) -> pd.Series:
        return self.closes

    def spot(self, symbol: str) -> float:
        return self._spot

    def _quote(self, symbol: str) -> OptionQuote:
        _, expiry, right, strike = parse_occ(symbol)
        T = max((expiry - self.today).days, 0) / 365
        px = bs_price(self._spot, strike, T, self.cfg.rate, self.cfg.div_yield, self.iv, right)
        delta = bs_greeks(self._spot, strike, T, self.cfg.rate, self.cfg.div_yield, self.iv, right)["delta"]
        return OptionQuote(symbol, right, strike, expiry, max(round(px - self.half_spread, 2), 0.01),
                           round(px + self.half_spread, 2), self.iv, delta)

    def option_chain(self, symbol, exp_gte: date, exp_lte: date, strike_gte: float, strike_lte: float):
        out, d = [], exp_gte
        while d <= exp_lte:
            if d.weekday() == 4:                      # Friday expiries
                for k in range(int(strike_gte), int(strike_lte) + 1):
                    for right in "CP":
                        out.append(self._quote(occ_symbol(symbol, d, right, float(k))))
            d += timedelta(days=1)
        return out

    def option_quotes(self, symbols):
        return {s: self._quote(s) for s in symbols}

    def place_structure_order(self, legs: list[OrderLeg], units: int, mid: float, natural: float,
                              attempts: int = 4, wait_seconds: int = 20) -> OrderResult:
        self.orders.append({"legs": legs, "units": units, "price": mid})
        if self.dry_run:
            return OrderResult("dry_run", mid)
        for leg in legs:
            self.pos[leg.symbol] = self.pos.get(leg.symbol, 0) + (units if leg.side == "buy" else -units)
            if self.pos[leg.symbol] == 0:
                del self.pos[leg.symbol]
        return OrderResult("filled", mid, f"fake-{len(self.orders)}")

    def submit_stock(self, symbol: str, qty: int) -> OrderResult:
        self.stock_orders.append(qty)
        if not self.dry_run:
            self.pos[symbol] = self.pos.get(symbol, 0) + qty
            if self.pos[symbol] == 0:
                del self.pos[symbol]
        return OrderResult("filled", None)
