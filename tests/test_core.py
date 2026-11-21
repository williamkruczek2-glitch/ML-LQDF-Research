"""Pricing, volatility forecasting (incl. no look-ahead), signals, structures and risk rules."""
import math
import os
import sys
import unittest
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volarb.config import StrategyConfig
from volarb.leakage import truncation_test
from volarb.pricing import bs_greeks, bs_price, implied_vol
from volarb.risk import DrawdownGuard, exit_reason, hedge_trade, size_units
from volarb.signals import Regime, SignalInputs, decide, in_blackout
from volarb.structures import (Structure, choose_strikes, long_straddle, max_loss_per_unit, occ_symbol,
                               parse_occ, pick_expiry, short_iron_fly, theoretical_value)
from volarb.synthetic import make_market
from volarb.volatility import (HARForecaster, forward_realised_vol, har_features, log_returns,
                               walk_forward_forecast)

CFG = StrategyConfig()


class Pricing(unittest.TestCase):
    S, K, T, r, q, v = 500.0, 510.0, 30 / 365, 0.04, 0.012, 0.18

    def test_put_call_parity(self):
        c = bs_price(self.S, self.K, self.T, self.r, self.q, self.v, "C")
        p = bs_price(self.S, self.K, self.T, self.r, self.q, self.v, "P")
        self.assertAlmostEqual(c - p, self.S * math.exp(-self.q * self.T) - self.K * math.exp(-self.r * self.T), places=8)

    def test_greeks_match_finite_differences(self):
        for right in "CP":
            g = bs_greeks(self.S, self.K, self.T, self.r, self.q, self.v, right)
            h = 0.01
            fd_delta = (bs_price(self.S + h, self.K, self.T, self.r, self.q, self.v, right)
                        - bs_price(self.S - h, self.K, self.T, self.r, self.q, self.v, right)) / (2 * h)
            fd_vega = (bs_price(self.S, self.K, self.T, self.r, self.q, self.v + 1e-4, right)
                       - bs_price(self.S, self.K, self.T, self.r, self.q, self.v - 1e-4, right)) / 2e-4
            self.assertAlmostEqual(g["delta"], fd_delta, places=5)
            self.assertAlmostEqual(g["vega"], fd_vega, places=3)

    def test_implied_vol_round_trip(self):
        for right in "CP":
            px = bs_price(self.S, self.K, self.T, self.r, self.q, self.v, right)
            self.assertAlmostEqual(implied_vol(px, self.S, self.K, self.T, self.r, self.q, right), self.v, places=5)
        self.assertIsNone(implied_vol(0.0, self.S, self.K, self.T, self.r, self.q, "C"))
        self.assertIsNone(implied_vol(10_000, self.S, self.K, self.T, self.r, self.q, "C"))

    def test_expiry_is_intrinsic(self):
        self.assertEqual(bs_price(500, 490, 0, 0.04, 0, 0.2, "C"), 10)
        self.assertEqual(bs_price(500, 490, 0, 0.04, 0, 0.2, "P"), 0)


class Volatility(unittest.TestCase):
    def test_forward_realised_vol_uses_next_h_days(self):
        r = pd.Series([0.0, 0.01, -0.02, 0.03, 0.0, 0.01])
        fwd = forward_realised_vol(r, 2)
        self.assertAlmostEqual(fwd.iloc[0], math.sqrt((0.01 ** 2 + 0.02 ** 2) / 2 * 252))
        self.assertTrue(np.isnan(fwd.iloc[-1]) and np.isnan(fwd.iloc[-2]))

    def test_har_recovers_a_known_relation(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"var_d": rng.uniform(0, 0.1, 500), "var_w": rng.uniform(0, 0.1, 500), "var_m": rng.uniform(0, 0.1, 500)})
        y = 0.01 + 0.2 * X["var_d"] + 0.3 * X["var_w"] + 0.4 * X["var_m"]
        m = HARForecaster().fit(X, y)
        np.testing.assert_allclose(m.coef_, [0.01, 0.2, 0.3, 0.4], atol=1e-10)

    def test_training_never_uses_unfinished_labels(self):
        close, _ = make_market(n_days=900)
        h = CFG.horizon_days
        seen = []

        class Spy(HARForecaster):
            def fit(self, X, y):
                seen.append(X.index.max())
                return super().fit(X, y)

        fc = walk_forward_forecast(close, h, 300, 21, Spy)
        idx = close.index
        first_forecast_days = [d for d in fc.dropna().index]
        self.assertTrue(seen)
        # each fit happens on a forecast day; its last training sample must be >= h days earlier
        fit_days = [first_forecast_days[0]] + [d for d in first_forecast_days][21::21]
        for last_train, fit_day in zip(seen, fit_days):
            self.assertLessEqual(idx.get_loc(last_train) + h, idx.get_loc(fit_day))

    def test_truncation_no_look_ahead(self):
        close, _ = make_market(n_days=900)
        cfg = StrategyConfig(har_min_train=300)
        self.assertTrue(truncation_test(close, cfg, n_cuts=4)["passed"])


class Signals(unittest.TestCase):
    def test_short_vol_entry_and_exit(self):
        d = decide(SignalInputs(iv=0.20, rv_forecast=0.15, rv_5d=0.12), CFG)
        self.assertEqual(d.target, Regime.SHORT_VOL)
        self.assertAlmostEqual(d.edge_pts, 5.0)
        hold = decide(SignalInputs(0.17, 0.15, 0.12, current=Regime.SHORT_VOL), CFG)
        self.assertEqual(hold.target, Regime.SHORT_VOL)            # hysteresis: 2 pts < entry but > exit
        out = decide(SignalInputs(0.152, 0.15, 0.12, current=Regime.SHORT_VOL), CFG)
        self.assertEqual(out.target, Regime.FLAT)

    def test_stress_and_blackout_block_new_short_vol(self):
        self.assertEqual(decide(SignalInputs(0.20, 0.15, 0.25), CFG).target, Regime.FLAT)            # rv5 > iv
        self.assertEqual(decide(SignalInputs(0.20, 0.15, 0.12, iv_5d_ago=0.13), CFG).target, Regime.FLAT)
        self.assertEqual(decide(SignalInputs(0.20, 0.15, 0.12, event_blackout=True), CFG).target, Regime.FLAT)

    def test_long_vol_when_implied_cheap(self):
        self.assertEqual(decide(SignalInputs(0.12, 0.16, 0.15), CFG).target, Regime.LONG_VOL)
        self.assertEqual(decide(SignalInputs(0.12, 0.16, 0.15), StrategyConfig(enable_long_vol=False)).target, Regime.FLAT)

    def test_blackout_counts_trading_days(self):
        fri = date(2026, 11, 13)
        self.assertTrue(in_blackout(fri, [date(2026, 11, 16)], 1))     # Monday is the next trading day
        self.assertFalse(in_blackout(fri, [date(2026, 11, 17)], 1))


class StructuresAndRisk(unittest.TestCase):
    exp = date(2026, 12, 18)

    def test_occ_round_trip(self):
        sym = occ_symbol("SPY", self.exp, "C", 650.5)
        self.assertEqual(sym, "SPY261218C00650500")
        self.assertEqual(parse_occ(sym), ("SPY", self.exp, "C", 650.5))

    def test_iron_fly_is_fully_covered(self):
        legs = short_iron_fly("SPY", self.exp, 650, 620, 680)
        for right in "CP":
            self.assertEqual(sum(l.qty for l in legs if l.right == right), 0)   # each short has a long wing
        self.assertEqual(len(legs), 4)

    def test_max_loss_and_sizing(self):
        legs = short_iron_fly("SPY", self.exp, 650, 620, 680)
        self.assertAlmostEqual(max_loss_per_unit("short_iron_fly", legs, -19.0), 1100.0)
        self.assertEqual(max_loss_per_unit("long_straddle", long_straddle("SPY", self.exp, 650), 25.0), 2500.0)
        self.assertEqual(size_units(1100.0, 100_000, CFG), 2)          # 3% of 100k = 3000 / 1100
        self.assertEqual(size_units(10.0, 100_000, CFG), CFG.max_contracts)

    def test_hedge_band(self):
        self.assertEqual(hedge_trade(-15, 0, 2, CFG), 0)               # band = 10% of 200 shares = 20
        self.assertEqual(hedge_trade(-45, 0, 2, CFG), 45)
        self.assertEqual(hedge_trade(30, 10, 2, CFG), -40)

    def test_exit_rules(self):
        legs = short_iron_fly("SPY", self.exp, 650, 620, 680)
        s = Structure("short_iron_fly", legs, 2, date(2026, 11, 18), -20.0)
        self.assertIsNone(exit_reason(s, -15.0, date(2026, 11, 25), CFG))
        self.assertIn("profit", exit_reason(s, -9.0, date(2026, 11, 25), CFG))
        self.assertIn("stop", exit_reason(s, -41.0, date(2026, 11, 25), CFG))
        self.assertIn("expiry", exit_reason(s, -15.0, date(2026, 12, 12), CFG))

    def test_drawdown_guard(self):
        g = DrawdownGuard(100_000)
        self.assertFalse(g.update(110_000, CFG))
        self.assertTrue(g.update(93_000, CFG))                          # 15.5% below the 110k peak
        self.assertTrue(g.update(120_000, CFG))                         # stays halted

    def test_expiry_and_strike_choice(self):
        today = date(2026, 11, 16)
        exps = [date(2026, 11, 20), date(2026, 12, 11), date(2026, 12, 18), date(2027, 1, 15)]
        self.assertEqual(pick_expiry(exps, today, CFG), date(2026, 12, 18))     # 32 days, closest to 30
        self.assertEqual(choose_strikes([float(k) for k in range(600, 701)], 651.3, CFG), (651.0, 619.0, 684.0))

    def test_iron_fly_value_is_a_credit(self):
        legs = short_iron_fly("SPY", self.exp, 650, 620, 680)
        v = theoretical_value(legs, 650, date(2026, 11, 18), 0.18, CFG)
        self.assertLess(v, 0)
        self.assertGreater(v, -30)                                      # credit < wing width


class Forecasters(unittest.TestCase):
    def test_gbm_plugs_into_walk_forward(self):
        try:
            from volarb.forecasters import GradientBoostingForecaster
        except ImportError:
            self.skipTest("scikit-learn not installed")
        close, _ = make_market(n_days=700)
        fc = walk_forward_forecast(close, 21, 300, 63, lambda: GradientBoostingForecaster(n_estimators=30))
        self.assertGreater(fc.notna().sum(), 300)
        self.assertTrue((fc.dropna() > 0).all())


if __name__ == "__main__":
    unittest.main()
