# TradingAgent

Free futures bar data for strategy backtesting.

A GitHub Actions workflow runs every weekday evening (6:15pm ET), downloads
NQ, ES and gold (GC) futures bars from Yahoo Finance, and appends them to
`data/`:

| File | Bars | Yahoo keeps |
|---|---|---|
| `data/NQ_1m.csv`, `ES_1m.csv`, `GC_1m.csv` | 1 minute | ~7 days |
| `data/NQ_5m.csv`, `ES_5m.csv`, `GC_5m.csv` | 5 minute | ~60 days |
| `data/NQ_15m.csv`, `ES_15m.csv`, `GC_15m.csv` | 15 minute | ~60 days |

Each run merges with what is already saved, so history keeps growing.
MNQ and MGC trade at the same prices as NQ and GC, so these files cover them.

Columns: `ts` (bar open, UTC epoch seconds), `open`, `high`, `low`, `close`, `volume`.

Run it by hand: Actions tab → "Fetch futures bars" → Run workflow.

Yahoo data is unofficial and can have gaps or bad ticks. Good for research,
not exchange-grade.
