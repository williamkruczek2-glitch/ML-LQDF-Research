"""Run the volatility arbitrage strategy against Alpaca. Paper trading unless --live.

    export ALPACA_API_KEY=...  ALPACA_SECRET_KEY=...
    python run_live.py --dry-run        # show what it would do; places nothing, changes no state
    python run_live.py                  # paper account: daily run (schedule ~15:45 New York time)
    python run_live.py --hedge-only     # intraday re-hedge (schedule hourly while a position is open)
    python run_live.py --live           # real money: also requires LQDF_ALLOW_LIVE=yes
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from volarb.broker import AlpacaBroker
from volarb.config import load_config
from volarb.strategy import PositionMismatch, VolArbStrategy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="trade the live account (default: paper)")
    ap.add_argument("--dry-run", action="store_true", help="log orders instead of placing them")
    ap.add_argument("--hedge-only", action="store_true", help="only re-hedge delta")
    ap.add_argument("--force", action="store_true", help="run even if the market is closed")
    ap.add_argument("--config", help="config.toml with parameter overrides")
    ap.add_argument("--state-dir", default="state")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if args.live and os.environ.get("LQDF_ALLOW_LIVE") != "yes":
        print("refusing to trade live: set LQDF_ALLOW_LIVE=yes as well as --live", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    broker = AlpacaBroker(paper=not args.live, dry_run=args.dry_run)
    level = broker.options_level()
    if level is not None and level < 3:
        logging.warning("account options level is %s; the short iron butterfly needs level 3 (multi-leg)", level)
    if not args.force and not broker.market_open():
        print("market is closed (use --force to run anyway)", file=sys.stderr)
        return 0

    strategy = VolArbStrategy(broker, cfg, args.state_dir)
    try:
        summary = strategy.run_hedge_only() if args.hedge_only else strategy.run_daily()
    except PositionMismatch as e:
        logging.error("%s", e)
        return 3
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
