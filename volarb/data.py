"""Load the backtest inputs: SPY daily closes (Alpaca or CSV) and VIX history (Cboe CSV)."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def load_vix_csv(path: str | Path) -> pd.Series:
    """Cboe's VIX_History.csv (DATE, OPEN, HIGH, LOW, CLOSE) -> implied vol as a decimal."""
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    s = pd.Series(df["CLOSE"].astype(float).to_numpy() / 100, index=pd.to_datetime(df["DATE"]), name="iv")
    return s.sort_index()[~s.index.duplicated()]


def load_close_csv(path: str | Path) -> pd.Series:
    """Any CSV with a date/timestamp column and a close column (adjusted closes recommended)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date") or cols.get("timestamp")
    ccol = cols.get("adj_close") or cols.get("adj close") or cols.get("close")
    if not dcol or not ccol:
        raise ValueError(f"{path}: need a date/timestamp column and a close column")
    idx = pd.to_datetime(df[dcol], utc=True).dt.tz_convert(None).dt.normalize()
    s = pd.Series(df[ccol].astype(float).to_numpy(), index=idx, name="close")
    return s.sort_index()[~s.index.duplicated()]


def fetch_daily_closes_alpaca(symbol: str, start: str, end: str | None = None) -> pd.Series:
    """Split- and dividend-adjusted daily closes from Alpaca (keys from the environment)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    # free plans may query SIP data older than 15 minutes
    latest = datetime.now(timezone.utc) - timedelta(minutes=16)
    end_dt = min(pd.Timestamp(end, tz="UTC").to_pydatetime(), latest) if end else latest
    bars = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                                                  start=pd.Timestamp(start, tz="UTC").to_pydatetime(), end=end_dt,
                                                  adjustment=Adjustment.ALL, feed=DataFeed.SIP)).df
    bars = bars.xs(symbol, level="symbol") if isinstance(bars.index, pd.MultiIndex) else bars
    idx = pd.to_datetime(bars.index, utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    return pd.Series(bars["close"].to_numpy(), index=idx, name="close").sort_index()


def align(close: pd.Series, iv: pd.Series) -> tuple[pd.Series, pd.Series, dict]:
    both = pd.concat([close.rename("close"), iv.rename("iv")], axis=1)
    report = {"close_only_days": int(both["iv"].isna().sum()), "iv_only_days": int(both["close"].isna().sum())}
    both = both.dropna()
    report["aligned_days"] = int(len(both))
    return both["close"], both["iv"], report
