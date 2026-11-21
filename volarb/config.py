"""All strategy parameters in one place. Override any of them in config.toml."""
from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class StrategyConfig:
    underlying: str = "SPY"
    strike_step: float = 1.0           # SPY strikes are $1 apart near the money

    # Volatility forecast
    horizon_days: int = 21             # trading days ~ 30 calendar days, matching VIX and 30-DTE options
    har_min_train: int = 504           # ~2 years of labelled samples before the first forecast
    har_refit_every: int = 21          # refit monthly in the backtest
    history_days: int = 1200           # calendar days of SPY history the live run loads

    # Option selection
    target_dte: int = 30
    min_dte: int = 21
    max_dte: int = 45
    wing_pct: float = 0.05             # iron-fly wings at +/-5% of spot

    # Signal, in vol points (2.0 = implied 2 percentage points above the forecast)
    short_entry_edge: float = 2.0
    short_entry_ratio: float = 1.10    # and implied at least 10% above the forecast
    long_entry_edge: float = 2.0
    exit_edge: float = 0.5
    enable_long_vol: bool = True
    stress_rv5_over_iv: bool = True    # no new short vol while last-5-day realised > implied
    stress_iv_jump: float = 5.0        # ...or implied rose more than 5 points in 5 days
    event_blackout_days: int = 1       # no new short vol if an events.csv date is this close
    events_file: str | None = "events.csv"

    # Exits
    exit_dte: int = 7                  # never hold into the last week (gamma risk)
    short_profit_take: float = 0.50    # close the short fly at 50% of the credit
    short_stop_loss: float = 1.00      # ...or when the loss equals the credit
    long_profit_take: float = 0.50     # long straddle: +50% of the debit
    long_stop_loss: float = 0.50       # ...or -50%

    # Risk
    risk_per_trade: float = 0.03       # max loss per structure, as a fraction of equity
    max_contracts: int = 10
    max_drawdown_halt: float = 0.15    # flatten and stop trading 15% below the equity peak
    hedge_band: float = 0.10           # re-hedge when |net delta| > 10% of the position's share-equivalent

    # Pricing and cost assumptions
    rate: float = 0.04
    div_yield: float = 0.012
    option_half_spread: float = 0.03   # $ per share per leg paid on entry and exit
    option_fee_per_contract: float = 0.05
    stock_cost_bps: float = 1.0
    initial_equity: float = 100_000.0

    # Live execution
    order_attempts: int = 4            # limit price walks from mid towards the natural price
    order_wait_seconds: int = 20

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> StrategyConfig:
    cfg = StrategyConfig()
    if path is None:
        path = Path("config.toml")
        if not path.exists():
            return cfg
    with open(path, "rb") as f:
        overrides = tomllib.load(f)
    known = {f.name for f in fields(StrategyConfig)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
