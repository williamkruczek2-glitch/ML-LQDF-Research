"""Backtest the volatility arbitrage strategy and run the leakage checks.

    python run_backtest.py --vix VIX_History.csv --spy spy_daily.csv
    python run_backtest.py --vix VIX_History.csv --alpaca              # SPY closes from Alpaca
    python run_backtest.py --synthetic                                 # no data needed
    python run_backtest.py --vix ... --spy ... --start 2020-01-01 --end 2020-12-31   # stress window

Outputs in --out (default results/): daily.csv, trades.csv, report.json, equity.png
"""
from __future__ import annotations


import argparse
import json
import sys
from pathlib import Path

from volarb.backtest import run_backtest
from volarb.config import load_config
from volarb.data import align, fetch_daily_closes_alpaca, load_close_csv, load_vix_csv
from volarb.leakage import delay_test, shuffled_label_test, truncation_test
from volarb.signals import load_events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vix", help="Cboe VIX_History.csv")
    ap.add_argument("--spy", help="CSV of SPY daily closes (date, close)")
    ap.add_argument("--alpaca", action="store_true", help="download SPY closes from Alpaca instead of --spy")
    ap.add_argument("--synthetic", action="store_true", help="run on a synthetic market with a known premium")
    ap.add_argument("--premium", type=float, default=3.0, help="synthetic variance premium, vol points")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--config", help="config.toml with parameter overrides")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", choices=["har", "gbm"], default="har", help="volatility forecaster")
    ap.add_argument("--skip-leakage", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        from volarb.synthetic import make_market
        close, iv = make_market(premium_pts=args.premium)
        data_report = {"source": f"synthetic, premium {args.premium} pts"}
    else:
        if not args.vix or not (args.spy or args.alpaca):
            ap.error("need --vix and one of --spy / --alpaca (or --synthetic)")
        iv = load_vix_csv(args.vix)
        close = fetch_daily_closes_alpaca(cfg.underlying, "2010-01-01") if args.alpaca else load_close_csv(args.spy)
        close, iv, data_report = align(close, iv)
        data_report["source"] = "VIX (Cboe) + SPY " + ("(Alpaca)" if args.alpaca else f"({args.spy})")

    from volarb.forecasters import model_factory
    factory = model_factory(args.model)
    events = load_events(cfg.events_file)
    res = run_backtest(close, iv, cfg, events=events, start=args.start, end=args.end, model_factory=factory)
    report = {"data": data_report, "model": args.model, "metrics": res.metrics, "config": cfg.to_dict()}
    if not args.skip_leakage:
        print("running leakage checks...", file=sys.stderr)
        report["leakage"] = {
            "truncation": truncation_test(close, cfg, model_factory=factory),
            "shuffled_labels": shuffled_label_test(close, cfg, model_factory=factory),
            "delay": delay_test(close, iv, cfg, run_backtest, events=events, start=args.start, end=args.end,
                                model_factory=factory),
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.daily.to_csv(out / "daily.csv")
    res.trades.to_csv(out / "trades.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        a1.plot(res.daily.index, res.daily["equity"], lw=1.2)
        a1.set_ylabel("Equity ($)")
        a1.set_title("Volatility arbitrage backtest")
        a2.plot(res.daily.index, res.daily["iv"] * 100, lw=0.8, label="Implied")
        a2.plot(res.daily.index, res.daily["rv_forecast"] * 100, lw=0.8, label="Forecast realised")
        a2.set_ylabel("Vol (%)")
        a2.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        fig.savefig(out / "equity.png", dpi=120)
    except ImportError:
        pass

    m = res.metrics
    print(f"\n{data_report['source']}   {m['start']} to {m['end']}")
    print(f"  total return {m['total_return']:+.1%}   CAGR {m['cagr']:+.1%}   Sharpe {m['sharpe']:.2f}   "
          f"t-stat {m['t_stat']:.2f}   max drawdown {m['max_drawdown']:.1%}")
    print(f"  trades {m['trades']} (short vol {m['short_vol_trades']}, long vol {m['long_vol_trades']})   "
          f"win rate {m['win_rate']:.0%}   costs ${m['total_costs']:,.0f}")
    vp = m["variance_premium"]
    print(f"  implied above the realised that followed on {vp['pct_days_implied_above']:.0%} of days, "
          f"mean gap {vp['mean_gap_vol_pts']:+.1f} vol pts")
    if "leakage" in report:
        lk = report["leakage"]
        print(f"  leakage: truncation {'PASS' if lk['truncation']['passed'] else 'FAIL'}   "
              f"out-of-sample R2 real {lk['shuffled_labels']['real_r2']:.2f} vs shuffled labels {lk['shuffled_labels']['shuffled_r2']:.2f}   "
              f"Sharpe lag0 {lk['delay']['sharpe_lag0']:.2f} vs lag1 {lk['delay']['sharpe_lag1']:.2f}")
    print(f"  files -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
