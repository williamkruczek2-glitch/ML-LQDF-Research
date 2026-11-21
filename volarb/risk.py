"""Position sizing, the delta-hedge rule, exit rules and the drawdown kill switch."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from .config import StrategyConfig
from .structures import Structure


def size_units(max_loss_per_unit: float, equity: float, cfg: StrategyConfig) -> int:
    """Units so that the worst case loses at most risk_per_trade of equity."""
    if max_loss_per_unit <= 0:
        return 0
    return max(0, min(cfg.max_contracts, math.floor(cfg.risk_per_trade * equity / max_loss_per_unit)))


def hedge_trade(option_delta_shares: float, stock_shares: int, units: int, cfg: StrategyConfig) -> int:
    """Shares to buy (+) or sell (-) to flatten delta, or 0 while inside the band."""
    net = option_delta_shares + stock_shares
    band = max(1.0, cfg.hedge_band * units * 100)
    if abs(net) <= band:
        return 0
    return int(round(-option_delta_shares)) - stock_shares


def exit_reason(s: Structure, current_value: float, today: date, cfg: StrategyConfig) -> str | None:
    """Rule-based exits, checked before the signal."""
    if s.dte(today) <= cfg.exit_dte:
        return f"{s.dte(today)} days to expiry"
    pnl = s.pnl(current_value)
    basis = abs(s.entry_price) * 100 * s.units
    if s.kind == "short_iron_fly":
        if pnl >= cfg.short_profit_take * basis:
            return f"profit target ({pnl:,.0f})"
        if pnl <= -cfg.short_stop_loss * basis:
            return f"stop loss ({pnl:,.0f})"
    elif s.kind == "long_straddle":
        if pnl >= cfg.long_profit_take * basis:
            return f"profit target ({pnl:,.0f})"
        if pnl <= -cfg.long_stop_loss * basis:
            return f"stop loss ({pnl:,.0f})"
    return None


@dataclass
class DrawdownGuard:
    peak: float
    halted: bool = False

    def update(self, equity: float, cfg: StrategyConfig) -> bool:
        """Returns True if trading must stop."""
        self.peak = max(self.peak, equity)
        if equity < self.peak * (1 - cfg.max_drawdown_halt):
            self.halted = True
        return self.halted
