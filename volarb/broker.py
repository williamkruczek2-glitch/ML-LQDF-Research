"""Alpaca connection: SPY data, option chains, multi-leg option orders and stock hedges.

The strategy only talks to the small interface below, so tests (and other brokers) can
swap in a fake. Multi-leg limit prices follow Alpaca's rule: positive = debit, negative = credit.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from .structures import parse_occ

log = logging.getLogger("volarb.broker")


@dataclass
class OptionQuote:
    symbol: str
    right: str
    strike: float
    expiry: date
    bid: float | None
    ask: float | None
    iv: float | None = None
    delta: float | None = None

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None or self.ask <= 0 or self.ask < self.bid:
            return None
        return (self.bid + self.ask) / 2


@dataclass
class OrderLeg:
    symbol: str
    side: str          # 'buy' | 'sell'
    intent: str        # 'buy_to_open' | 'sell_to_open' | 'buy_to_close' | 'sell_to_close'


@dataclass
class OrderResult:
    status: str        # 'filled' | 'unfilled' | 'dry_run'
    price: float | None
    order_id: str | None = None


def leg_prices(legs: list[OrderLeg], quotes: dict[str, OptionQuote]) -> tuple[float, float]:
    """(mid, natural) net price per unit for an order: buys at ask, sells at bid for natural."""
    mid = natural = 0.0
    for leg in legs:
        q = quotes[leg.symbol]
        if q.mid is None:
            raise ValueError(f"no usable quote for {leg.symbol}")
        sign = 1 if leg.side == "buy" else -1
        mid += sign * q.mid
        natural += sign * (q.ask if leg.side == "buy" else q.bid)
    return round(mid, 2), round(natural, 2)


class AlpacaBroker:
    def __init__(self, paper: bool = True, dry_run: bool = False, api_key: str | None = None,
                 secret_key: str | None = None):
        from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        key = api_key or os.environ["ALPACA_API_KEY"]
        secret = secret_key or os.environ["ALPACA_SECRET_KEY"]
        self.paper, self.dry_run = paper, dry_run
        self.trading = TradingClient(key, secret, paper=paper)
        self.stocks = StockHistoricalDataClient(key, secret)
        self.options = OptionHistoricalDataClient(key, secret)

    # account ------------------------------------------------------------------------------
    def account_equity(self) -> float:
        return float(self.trading.get_account().equity)

    def options_level(self) -> int | None:
        lvl = getattr(self.trading.get_account(), "options_trading_level", None)
        return int(lvl) if lvl is not None else None

    def market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    def positions(self) -> dict[str, int]:
        """Signed quantities: options in contracts, stock in shares (short = negative)."""
        out = {}
        for p in self.trading.get_all_positions():
            q = float(p.qty)
            if str(getattr(p.side, "value", p.side)) == "short" and q > 0:
                q = -q
            out[p.symbol] = int(round(q))
        return out

    # market data --------------------------------------------------------------------------
    def daily_closes(self, symbol: str, days: int) -> pd.Series:
        from .data import fetch_daily_closes_alpaca
        start = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        return fetch_daily_closes_alpaca(symbol, start)

    def spot(self, symbol: str) -> float:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestTradeRequest
        trade = self.stocks.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX))
        return float(trade[symbol].price)

    def option_chain(self, symbol: str, exp_gte: date, exp_lte: date, strike_gte: float, strike_lte: float) -> list[OptionQuote]:
        from alpaca.data.requests import OptionChainRequest
        snaps = self.options.get_option_chain(OptionChainRequest(
            underlying_symbol=symbol, expiration_date_gte=exp_gte, expiration_date_lte=exp_lte,
            strike_price_gte=strike_gte, strike_price_lte=strike_lte))
        return [self._to_quote(sym, snap) for sym, snap in snaps.items()]

    def option_quotes(self, symbols: list[str]) -> dict[str, OptionQuote]:
        from alpaca.data.requests import OptionSnapshotRequest
        snaps = self.options.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=symbols))
        return {sym: self._to_quote(sym, snap) for sym, snap in snaps.items()}

    @staticmethod
    def _to_quote(symbol: str, snap) -> OptionQuote:
        _, expiry, right, strike = parse_occ(symbol)
        q = snap.latest_quote
        g = snap.greeks
        return OptionQuote(symbol, right, strike, expiry,
                           float(q.bid_price) if q and q.bid_price else None,
                           float(q.ask_price) if q and q.ask_price else None,
                           float(snap.implied_volatility) if snap.implied_volatility else None,
                           float(g.delta) if g and g.delta is not None else None)

    # orders -------------------------------------------------------------------------------
    def place_structure_order(self, legs: list[OrderLeg], units: int, mid: float, natural: float,
                              attempts: int = 4, wait_seconds: int = 20) -> OrderResult:
        """Limit order at mid, then walk towards the natural price until filled or out of attempts."""
        from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest

        req_legs = [OptionLegRequest(symbol=l.symbol, ratio_qty=1, side=OrderSide(l.side),
                                     position_intent=PositionIntent(l.intent)) for l in legs]
        for k in range(attempts):
            price = round(mid + (natural - mid) * (k / max(attempts - 1, 1)), 2)
            if self.dry_run:
                log.info("[dry run] %s x%d at %.2f: %s", "debit" if price > 0 else "credit", units, price,
                         ", ".join(f"{l.side} {l.symbol}" for l in legs))
                return OrderResult("dry_run", price)
            order = self.trading.submit_order(LimitOrderRequest(
                qty=units, order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY, limit_price=price, legs=req_legs))
            filled = self._wait_for_fill(str(order.id), wait_seconds)
            if filled is not None:
                # keep our sign convention (+ debit / - credit) whatever sign the fill is reported with
                fill_price = abs(filled) * (1 if price >= 0 else -1) if filled else price
                return OrderResult("filled", round(fill_price, 2), str(order.id))
            self.trading.cancel_order_by_id(order.id)
            log.info("not filled at %.2f; walking the price", price)
        return OrderResult("unfilled", None)

    def _wait_for_fill(self, order_id: str, wait_seconds: int) -> float | None:
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            o = self.trading.get_order_by_id(order_id)
            status = str(getattr(o.status, "value", o.status))
            if status == "filled":
                return float(o.filled_avg_price) if o.filled_avg_price is not None else 0.0
            if status in ("canceled", "expired", "rejected"):
                raise RuntimeError(f"order {order_id} {status}")
            time.sleep(2)
        return None

    def submit_stock(self, symbol: str, qty: int) -> OrderResult:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        if qty == 0:
            return OrderResult("filled", None)
        if self.dry_run:
            log.info("[dry run] %s %d %s", "buy" if qty > 0 else "sell", abs(qty), symbol)
            return OrderResult("dry_run", None)
        o = self.trading.submit_order(MarketOrderRequest(symbol=symbol, qty=abs(qty),
                                                         side=OrderSide.BUY if qty > 0 else OrderSide.SELL,
                                                         time_in_force=TimeInForce.DAY))
        return OrderResult("submitted", None, str(o.id))
