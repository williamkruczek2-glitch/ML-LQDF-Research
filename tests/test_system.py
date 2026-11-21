"""Backtest behaviour, Alpaca order construction (mocked), and the live loop against a fake account."""
import json
import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fakes import FakeBroker
from volarb.backtest import run_backtest
from volarb.broker import AlpacaBroker, OrderLeg
from volarb.config import StrategyConfig
from volarb.strategy import PositionMismatch, VolArbStrategy
from volarb.synthetic import make_market


class Backtest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = StrategyConfig(har_min_train=300)
        cls.rich = run_backtest(*make_market(n_days=1500, premium_pts=4.0, seed=3), cls.cfg)
        cls.cheap = run_backtest(*make_market(n_days=1500, premium_pts=-4.0, seed=3), cls.cfg)

    def test_makes_money_only_when_implied_is_rich(self):
        self.assertGreater(self.rich.metrics["total_return"], 0)
        self.assertGreater(self.rich.metrics["short_vol_trades"], self.rich.metrics["long_vol_trades"])
        self.assertGreater(self.cheap.metrics["long_vol_trades"], self.cheap.metrics["short_vol_trades"])

    def test_costs_and_accounting_add_up(self):
        d = self.rich.daily
        self.assertGreater(d["costs"].sum(), 0)
        expected = self.cfg.initial_equity + d["option_pnl"].sum() + d["hedge_pnl"].sum() - d["costs"].sum()
        self.assertAlmostEqual(d["equity"].iloc[-1], expected, places=4)

    def test_drawdown_halt_stops_new_trades(self):
        cfg = StrategyConfig(har_min_train=300, max_drawdown_halt=0.002)
        res = run_backtest(*make_market(n_days=1500, premium_pts=4.0, seed=3), cfg)
        halt = res.trades[res.trades["reason"] == "drawdown halt"]
        self.assertEqual(len(halt), 1)
        after = res.daily.loc[pd.Timestamp(halt["exit_date"].iloc[0]):]
        self.assertTrue((after["units"] == 0).all())

    def test_signal_lag_changes_entries(self):
        lagged = run_backtest(*make_market(n_days=1500, premium_pts=4.0, seed=3), self.cfg, signal_lag=1)
        first = lambda r: r.daily[r.daily["note"].str.startswith("open")].index[0]
        self.assertGreater(first(lagged), first(self.rich))


class AlpacaOrders(unittest.TestCase):
    def make(self, dry_run=False):
        b = AlpacaBroker(paper=True, dry_run=dry_run, api_key="k", secret_key="s")
        b.trading = mock.MagicMock()
        b.trading.submit_order.return_value = SimpleNamespace(id="o1")
        b.trading.get_order_by_id.return_value = SimpleNamespace(status="filled", filled_avg_price="1.30")
        return b

    legs = [OrderLeg("SPY261218P00620000", "buy", "buy_to_open"), OrderLeg("SPY261218P00650000", "sell", "sell_to_open"),
            OrderLeg("SPY261218C00650000", "sell", "sell_to_open"), OrderLeg("SPY261218C00680000", "buy", "buy_to_open")]

    def test_iron_fly_is_one_mleg_credit_order(self):
        b = self.make()
        res = b.place_structure_order(self.legs, 2, mid=-18.40, natural=-17.90, attempts=1, wait_seconds=1)
        req = b.trading.submit_order.call_args.args[0]
        fields = req.to_request_fields()
        self.assertEqual(fields["order_class"].value, "mleg")
        self.assertEqual(fields["qty"], 2)
        self.assertEqual(fields["limit_price"], -18.40)                     # negative = credit
        self.assertEqual([(l["side"].value, l["position_intent"].value) for l in fields["legs"]],
                         [("buy", "buy_to_open"), ("sell", "sell_to_open"), ("sell", "sell_to_open"), ("buy", "buy_to_open")])
        self.assertEqual((res.status, res.price), ("filled", -1.30))       # fill keeps the credit sign

    def test_price_walks_towards_natural_when_unfilled(self):
        b = self.make()
        b.trading.get_order_by_id.return_value = SimpleNamespace(status="new", filled_avg_price=None)
        with mock.patch("volarb.broker.time.sleep"), mock.patch("volarb.broker.time.time", side_effect=[0, 0, 99] * 4):
            res = b.place_structure_order(self.legs, 1, mid=-18.40, natural=-17.80, attempts=4, wait_seconds=1)
        prices = [c.args[0].limit_price for c in b.trading.submit_order.call_args_list]
        self.assertEqual(prices, [-18.4, -18.2, -18.0, -17.8])
        self.assertEqual(res.status, "unfilled")
        self.assertEqual(b.trading.cancel_order_by_id.call_count, 4)

    def test_dry_run_places_nothing(self):
        b = self.make(dry_run=True)
        self.assertEqual(b.place_structure_order(self.legs, 1, -18.4, -17.9).status, "dry_run")
        self.assertEqual(b.submit_stock("SPY", 25).status, "dry_run")
        b.trading.submit_order.assert_not_called()


class LiveLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        close, _ = make_market(n_days=800, premium_pts=0, seed=11, stay=0.999, calm=0.14, stressed=0.14)
        self.today = close.index[-1].date() + timedelta(days=3)
        while self.today.weekday() != 0:
            self.today += timedelta(days=1)
        self.close = close
        # tight hedge band so a 3% move triggers a hedge (the wings cap the fly's delta at ~-0.07/unit)
        self.cfg = StrategyConfig(events_file=None, hedge_band=0.02)

    def tearDown(self):
        self.tmp.cleanup()

    def strat(self, broker):
        return VolArbStrategy(broker, self.cfg, self.tmp.name)

    def test_open_hedge_close_then_long_vol(self):
        spot = float(self.close.iloc[-1])
        b = FakeBroker(self.close, spot, iv=0.24, today=self.today)
        out = self.strat(b).run_daily(self.today)
        self.assertEqual(out["position"], "short_iron_fly", out)
        legs = {s: q for s, q in b.pos.items() if s != "SPY"}
        self.assertEqual(len(legs), 4)
        self.assertEqual(sum(legs.values()), 0)                             # 2 long wings, 2 short ATM
        with open(os.path.join(self.tmp.name, "state.json")) as f:
            state = json.load(f)
        self.assertEqual(state["structure"]["kind"], "short_iron_fly")

        # price jumps 3%: the short fly is now short delta, so the hedge buys shares
        b.today += timedelta(days=1)
        b._spot = spot * 1.03
        self.strat(b).run_daily(b.today)
        self.assertGreater(b.pos.get("SPY", 0), 0)

        # implied collapses below the forecast: close everything, no same-day re-entry
        b.today += timedelta(days=1)
        b.iv = 0.08
        out = self.strat(b).run_daily(b.today)
        self.assertIn("closed short_iron_fly", out["actions"])
        self.assertEqual(out["position"], "flat")
        self.assertEqual(b.pos, {})                                         # options closed, hedge flattened

        # next day: still cheap -> buy a straddle
        b.today += timedelta(days=1)
        out = self.strat(b).run_daily(b.today)
        self.assertEqual(out["position"], "long_straddle", out)
        self.assertTrue(all(q > 0 for s, q in b.pos.items() if s != "SPY"))

    def test_refuses_unknown_positions(self):
        b = FakeBroker(self.close, float(self.close.iloc[-1]), 0.24, self.today)
        b.pos["SPY261218C00999000"] = -1
        with self.assertRaises(PositionMismatch):
            self.strat(b).run_daily(self.today)

    def test_dry_run_changes_nothing(self):
        b = FakeBroker(self.close, float(self.close.iloc[-1]), 0.24, self.today, dry_run=True)
        out = self.strat(b).run_daily(self.today)
        self.assertIn("opened short_iron_fly", out["actions"])
        self.assertEqual(b.pos, {})
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "state.json")))


if __name__ == "__main__":
    unittest.main()
