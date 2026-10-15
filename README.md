# Perpetual Futures: Funding, Carry and Hedging

This study uses Binance perpetual futures on BTC, ETH and SOL to answer three questions:

1. **Carry.** Does the funding paid by leveraged longs make a cash-and-carry trade (long spot, short perp) worth its risk compared with T-bills?
2. **Hedging.** Does shorting perps make a spot BTC/ETH portfolio safer, and what does it cost?
3. **Signals.** Do funding rates or open interest warn of crashes or predict returns?

Rules and signal cut-offs are tuned on data to Dec 2022 and judged once on Jan 2023 to Oct 2026. The full report is [`reports/report.html`](reports/report.html), a single self-contained file. Tables are saved as CSV in [`reports/tables/`](reports/tables/).

## Findings

- **The funding premium has mostly gone.** Shorting BTC perps paid 31% a year in 2021, 5.1% in 2025 and 2.9% so far in 2026, which is now below the T-bill yield.
- **Out of sample, carry barely beat T-bills:** 5.1% a year for BTC+ETH vs 4.6% for T-bills. Staying in the trade permanently at 1x returned 3.5%.
- **Leverage on the short leg is the real risk, and it is very uneven across assets.** A 3x short is liquidated by a 33% rise in the mark price. BTC's largest 7-day rise was 43%, ETH's 77% and SOL's 120%. In the FTX week, SOL shorts *paid* funding at 1,490% a year.
- **Measure liquidations on the mark price.** On 13 Mar 2020 the ETH perp printed a last-trade high 203% above the previous close, while its mark price rose 31%.
- **Trend-based perp hedges cut drawdowns, at a cost in return.** Hedging 100% when price was below its 200-day average fell 32% at worst (vs 60% unhedged) and returned 28% a year (vs 38%). The short leg earns funding while the hedge is on, which partly offsets the cost.
- **Funding signalled turbulence, not direction.** After the highest-funding days, 10%+ falls within a week were more common. Average returns were not lower, though, and return forecasts from funding, OI and basis had a negative out-of-sample R² for every asset.

## How it works

```
data/raw/               Binance funding, spot, perp and mark-price candles; Bybit OI and
                        funding; FRED 3-month T-bill (pinned to END_DATE)
sql/schema.sql          raw tables
sql/features.sql        daily funding, spot/perp/mark panel, basis, trailing signals,
                        forward outcomes (used only as prediction targets)
sql/queries/            funding by year, Binance vs Bybit check, market tracker, wick table
src/perp_risk/
  fetch.py              downloads everything (public endpoints, no keys)
  db.py                 loads CSVs into SQLite and runs the SQL
  carry.py              carry engine: margin account, funding, mark-price liquidation, resets
  hedge.py              perp hedge overlay on a 50/50 BTC/ETH portfolio
  evaluate.py           train selection, out-of-sample test, cost sensitivity, signal
                        research (quintiles, Newey–West regressions), crash episodes
  report.py             figures, CSV tables, HTML report
tests/                  carry/hedge engine checks, Newey–West, SQL-vs-pandas signals
```

**Design choices**

- **No look-ahead.** Signals at close *d* act from close *d+1*. Forward returns live in a separate table and are only used as prediction targets.
- **Liquidations use mark-price highs, not last-trade highs.** Each test is a daily check against 0.5% maintenance margin, with the margin topped up to target leverage daily or weekly.
- **The cash benchmark is real.** Idle capital earns the 3-month T-bill rate, and results are reported as Sharpe over T-bills.
- **Costs** are 7.5 bps per leg per unit traded, with sensitivity tests from 0 to 30 bps.
- **Not modelled:** exchange counterparty risk, USDT de-pegging, tiered margin for large positions, transfer frictions, and venue or regulatory access.

## Reproduce

Requires Python 3.9+.

```bash
make setup   # virtualenv + pinned dependencies
make all     # build SQLite db, run tests, regenerate reports/
```

*Historical research, not investment advice. Leveraged derivatives can lose more than the capital at stake.*
