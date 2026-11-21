"""Performance statistics from a daily equity curve and a trade list."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def summarise(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    rets = equity.pct_change().dropna()
    n = len(rets)
    sd = rets.std(ddof=1) if n > 1 else float("nan")
    years = n / 252 if n else float("nan")
    out = {
        "start": str(equity.index[0].date()) if len(equity) else None,
        "end": str(equity.index[-1].date()) if len(equity) else None,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else float("nan"),
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if n > 20 and equity.iloc[-1] > 0 else float("nan"),
        "ann_vol": float(sd * math.sqrt(252)) if n > 1 else float("nan"),
        "sharpe": float(rets.mean() / sd * math.sqrt(252)) if n > 1 and sd > 0 else float("nan"),
        "t_stat": float(rets.mean() / (sd / math.sqrt(n))) if n > 1 and sd > 0 else float("nan"),
        "max_drawdown": float((equity / equity.cummax() - 1).min()) if len(equity) else float("nan"),
        "days": int(n),
    }
    if trades is not None and len(trades):
        out.update({
            "trades": int(len(trades)),
            "win_rate": float((trades["pnl"] > 0).mean()),
            "avg_trade_pnl": float(trades["pnl"].mean()),
            "short_vol_trades": int((trades["kind"] == "short_iron_fly").sum()),
            "long_vol_trades": int((trades["kind"] == "long_straddle").sum()),
        })
    else:
        out.update({"trades": 0, "win_rate": float("nan"), "avg_trade_pnl": float("nan"),
                    "short_vol_trades": 0, "long_vol_trades": 0})
    return out


def variance_premium(iv: pd.Series, rv_fwd: pd.Series) -> dict:
    """How often, and by how much, implied exceeded the volatility that followed (the evidence chart)."""
    both = pd.concat([iv, rv_fwd], axis=1).dropna()
    gap = (both.iloc[:, 0] - both.iloc[:, 1]) * 100
    return {
        "days": int(len(both)),
        "pct_days_implied_above": float((gap > 0).mean()) if len(both) else float("nan"),
        "mean_gap_vol_pts": float(gap.mean()) if len(both) else float("nan"),
        "median_gap_vol_pts": float(gap.median()) if len(both) else float("nan"),
    }
