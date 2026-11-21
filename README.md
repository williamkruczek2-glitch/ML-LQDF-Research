# LQDF volatility arbitrage (SPY options on Alpaca)

A complete, tested volatility arbitrage system: forecast, signal, risk rules, backtest with
leakage checks, and a live loop that trades SPY options on Alpaca. **Paper trading by default.**
This is a learning project for the ML division, not a production system.

## The strategy in five lines

1. Every day near the close, forecast SPY's volatility over the next 21 trading days (HAR model, trained only on past data).
2. Read the ~30-day at-the-money implied volatility from the SPY option chain.
3. **Implied well above forecast** (≥ 2 vol points and ≥ 10% higher): sell a **short iron butterfly**: sell the ATM call and put, buy wings ±5% away.
4. **Implied well below forecast** (≥ 2 points): buy an ATM **straddle**.
5. Delta-hedge with SPY shares, exit at 7 days to expiry, at +50% profit, at the stop loss, or when the edge disappears.

**Why an iron butterfly and not a short straddle?** Alpaca only accepts multi-leg orders where
every short leg is covered in the same order, so naked short options are out. The wings also cap
the worst case, which matters most for short-volatility trades (see February 2018).

## Quick start

```bash
pip install -r requirements.txt
python -m unittest discover tests          # 31 tests, no internet needed
python run_backtest.py --synthetic         # whole pipeline on a synthetic market
```

### Backtest on real data
1. Download `VIX_History.csv` from Cboe's VIX historical data page.
2. Set your Alpaca keys, then run:

```bash
export ALPACA_API_KEY=...  ALPACA_SECRET_KEY=...
python run_backtest.py --vix VIX_History.csv --alpaca
python run_backtest.py --vix VIX_History.csv --alpaca --start 2020-01-01 --end 2020-12-31   # stress window
python run_backtest.py --vix VIX_History.csv --alpaca --model gbm                           # ML forecaster
```

Results go to `results/`: `report.json` (metrics + leakage checks), `daily.csv`, `trades.csv`, `equity.png`.

### Paper trading
The Alpaca paper account needs **options level 3** (multi-leg) for the iron butterfly.

```bash
python run_live.py --dry-run --force    # shows what it would do; places nothing, saves nothing
python run_live.py                      # paper account: schedule daily at ~15:45 New York time
python run_live.py --hedge-only         # optional: hourly re-hedge while a position is open
```

Live money needs both `--live` and `LQDF_ALLOW_LIVE=yes`. Don't, until the fund has an eligible
account and the Committee has approved it.

## How the code fits together

| File | What it does |
|---|---|
| `volarb/config.py` | Every parameter, with defaults. Override in `config.toml`. |
| `volarb/volatility.py` | Realised vol, HAR features, purged walk-forward forecast |
| `volarb/forecasters.py` | Gradient-boosting forecaster that plugs into the same interface |
| `volarb/signals.py` | Implied vs forecast → short vol / long vol / flat; stress filter; event blackout |
| `volarb/structures.py` | OCC symbols, expiry and strike choice, iron butterfly / straddle, max loss |
| `volarb/risk.py` | Sizing (3% of equity max loss per trade), hedge band, exits, drawdown kill switch |
| `volarb/pricing.py` | Black–Scholes prices, Greeks, implied vol solver |
| `volarb/backtest.py` | Daily delta-hedged backtest with costs |
| `volarb/leakage.py` | Truncation, shuffled-label and delay tests |
| `volarb/broker.py` | Alpaca: data, option chain, multi-leg orders (price walks from mid to natural), hedges |
| `volarb/strategy.py` | The live daily run, position reconciliation, state in `state/` |

## Leakage safeguards
- **Purged training:** on day *i* the model only trains on samples whose 21-day label ended by day *i*.
- **Truncation test:** a forecast made with data up to day *c* must equal the one made with the full history.
- **Shuffled labels:** the real model must beat every model trained on shuffled labels.
- **Delay test:** acting one day late should weaken a real edge, not destroy it.

## Known limitations (put these on the Limitations slide)
- **Backtest pricing:** one flat implied vol for every strike, with VIX standing in for ATM implied vol. VIX usually sits above ATM IV because of skew, so the premium and the profits are probably **overstated**. Fix: real option prices (task for the data team, below).
- **Execution assumptions:** fills at the close, an expiry exactly 30 days out, simple cost assumptions.
- **Early assignment:** SPY options are American-style. A short ATM call can be assigned early near an ex-dividend date. The live run detects any mismatch with the state file and stops trading until a human fixes it.
- **Sample size:** about 2 years of training are needed before the first forecast, and there are few independent trades (one structure at a time).

## How the team can contribute

**Milan: risk and market context (little code needed)**
- **Event calendar:** fill `events.csv` with FOMC and CPI dates (the format is in the file). The strategy already avoids opening short vol the day before them.
- **Stress tests:** run the backtest over 2018, 2020 and 2022 with `--start/--end`, and write up what happened and why.
- **Risk limits:** review `risk_per_trade`, `short_stop_loss`, `wing_pct` and `max_drawdown_halt` in `config.toml`, and argue for the values in the presentation.
- **Costs:** research realistic SPY option spreads and fees, and update `option_half_spread` and `option_fee_per_contract`.
- **Limitations slide:** turn the list above into slides.

**Daniel and Zhanar: data**
- **Real option prices:** load real ATM implied vol (ThetaData free tier, or Alpaca option history from 2024) and compare it with VIX. Then pass it to `run_backtest` in place of VIX.
- **Data checks:** add quality checks on the inputs (gaps, duplicates, stale prices) in `volarb/data.py`.

**Next model step (whoever is keenest)**
- Run `--model gbm` and check whether it beats HAR out of sample.
- Add features through `extra_features` (e.g. VIX itself, day of week).
