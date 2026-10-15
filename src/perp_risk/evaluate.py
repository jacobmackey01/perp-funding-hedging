"""The study. Every parameter is chosen on TRAIN (to 2022-12-31) and the chosen
rules are run once on TEST (2023-01-01 → end). Signal research uses cut-offs
fixed on TRAIN as well.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from . import carry, config, db, hedge, metrics

CARRY_LEVERAGE = (1, 2, 3, 5)
CARRY_RESET = ("D", "W")
CARRY_GATES = ((-np.inf, -np.inf), (0.0, 0.0), (0.05, 0.0), (0.10, 0.02), (0.20, 0.05))
CARRY_ASSETS = ("BTC", "ETH", "SOL")
HEDGE_COLUMNS = ("spot_close", "spot_ret", "perp_ret", "funding", "sma_50", "sma_100",
                 "sma_200", "vol_30d", "funding_7d_ann")


@dataclass
class Study:
    panel: pd.DataFrame
    tbill: pd.Series

    @classmethod
    def load(cls) -> "Study":
        return cls(db.load_panel(), db.load_tbill())

    @property
    def rf(self) -> pd.Series:
        return metrics.tbill_daily(self.tbill)

    def asset(self, a: str) -> pd.DataFrame:
        return self.panel[self.panel.asset == a]

    def carry_start(self, a: str) -> str:
        first = self.asset(a).dropna(subset=["funding_7d_ann"]).date.iloc[0]
        return str((first + pd.Timedelta(days=1)).date())

    def wide(self) -> dict[str, pd.DataFrame]:
        return {c: db.wide(self.panel, c) for c in HEDGE_COLUMNS}

    def hedge_start(self) -> str:
        w = self.wide()
        ok = pd.concat([w[c][list(hedge.PAIR)].notna().all(axis=1) for c in HEDGE_COLUMNS], axis=1).all(axis=1)
        return str((ok[ok].index[0] + pd.Timedelta(days=1)).date())


# ── 1. cash-and-carry ────────────────────────────────────────────────────────

def carry_rules() -> list[carry.CarryRule]:
    return [carry.CarryRule(leverage=L, rebalance=r, enter_above=g[0], exit_below=g[1])
            for L, r, g in product(CARRY_LEVERAGE, CARRY_RESET, CARRY_GATES)]


def carry_summary(study: Study, res: carry.CarryResult) -> dict:
    s = metrics.summarize(res.equity, study.rf)
    years = len(res.equity) / config.DAYS_PER_YEAR
    start_eq = res.equity.iloc[0]
    return {
        "asset": res.asset, "rule": res.rule.label(), "leverage": res.rule.leverage,
        "reset": res.rule.rebalance, "enter_above": res.rule.enter_above, **s,
        "liquidations": len(res.liquidations),
        "time_in_trade": float(res.in_trade.mean()),
        "funding_per_year": res.funding_pnl.sum() / start_eq / years,
        "basis_per_year": res.basis_pnl.sum() / start_eq / years,
        "cash_yield_per_year": res.cash_pnl.sum() / start_eq / years,
        "costs_per_year": res.costs.sum() / start_eq / years,
        "min_margin_ratio": float(res.margin_ratio.min()),
    }


def carry_grid(study: Study) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = [], []
    for a in CARRY_ASSETS:
        ap = study.asset(a)
        for rule in carry_rules():
            train.append(carry_summary(study, carry.run(ap, rule, study.carry_start(a), config.TRAIN_END, tbill=study.tbill)))
            test.append(carry_summary(study, carry.run(ap, rule, config.TEST_START, config.END_DATE, tbill=study.tbill)))
    return pd.DataFrame(train), pd.DataFrame(test)


def carry_select(train: pd.DataFrame) -> dict[str, carry.CarryRule]:
    """Best training Sharpe over T-bills among rules never liquidated in training."""
    rules = {r.label(): r for r in carry_rules()}
    picks = {}
    for a, g in train.groupby("asset"):
        safe = g[g.liquidations == 0]
        pool = safe if len(safe) else g[g.liquidations == g.liquidations.min()]
        picks[a] = rules[pool.loc[pool.sharpe_vs_tbill.idxmax(), "rule"]]
    return picks


def carry_test(study: Study, picks: dict[str, carry.CarryRule], cost_bps: float = config.COST_BPS
               ) -> tuple[pd.DataFrame, dict[str, carry.CarryResult], pd.Series]:
    res = {a: carry.run(study.asset(a), r, config.TEST_START, config.END_DATE, cost_bps, study.tbill)
           for a, r in picks.items()}
    rows = [carry_summary(study, r) for r in res.values()]
    combo = sum(res[a].equity / res[a].equity.iloc[0] for a in ("BTC", "ETH")) / 2
    rows.append({"asset": "BTC+ETH 50/50", "rule": "picked rules", **metrics.summarize(combo, study.rf)})
    tb = metrics.tbill_equity(study.tbill, config.TEST_START, config.END_DATE)
    rows.append({"asset": "T-bills", "rule": "3-month US T-bill", **metrics.summarize(tb, study.rf)})
    return pd.DataFrame(rows), res, combo


def carry_cost_sensitivity(study: Study, picks: dict[str, carry.CarryRule]) -> pd.DataFrame:
    rows = []
    for bps in config.COST_GRID_BPS:
        t, _, _ = carry_test(study, picks, bps)
        for _, r in t[t.asset.isin(picks.keys())].iterrows():
            rows.append({"asset": r.asset, "cost_bps": bps, "cagr": r.cagr, "sharpe_vs_tbill": r.sharpe_vs_tbill})
    return pd.DataFrame(rows)


def carry_by_year(res: dict[str, carry.CarryResult]) -> pd.DataFrame:
    rows = []
    for a, r in res.items():
        eq = r.equity
        for y, g in eq.groupby(eq.index.year):
            start = eq[eq.index < g.index[0]].iloc[-1] if (eq.index < g.index[0]).any() else g.iloc[0]
            rows.append({"asset": a, "year": y, "return": g.iloc[-1] / start - 1,
                         "funding": r.funding_pnl[g.index].sum() / start,
                         "time_in_trade": r.in_trade[g.index].mean()})
    return pd.DataFrame(rows)


# ── 2. hedging the spot portfolio ────────────────────────────────────────────

def hedge_summary(study: Study, res: hedge.HedgeResult) -> dict:
    s = metrics.summarize(res.equity, study.rf)
    prev = res.equity.shift(1)
    per_year = lambda x: float((x / prev).mean() * config.DAYS_PER_YEAR)   # as % of equity
    return {"strategy": res.rule.label(), "family": res.rule.family, **s,
            "avg_hedge": float(res.hedge.mean().mean()),
            "funding_per_year": per_year(res.funding_pnl),
            "hedge_pnl_per_year": per_year(res.hedge_pnl),
            "costs_per_year": per_year(res.costs)}


def hedge_grid(study: Study) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = study.wide()
    start = study.hedge_start()
    rules = ([hedge.HedgeRule("none")] + [r for g in hedge.grid().values() for r in g]
             + [hedge.HedgeRule("static", 1.0)])
    train = pd.DataFrame([hedge_summary(study, hedge.run(r, w, start, config.TRAIN_END)) for r in rules])
    test = pd.DataFrame([hedge_summary(study, hedge.run(r, w, config.TEST_START, config.END_DATE)) for r in rules])
    return train.set_index("strategy"), test.set_index("strategy")


def hedge_select(train: pd.DataFrame) -> list[hedge.HedgeRule]:
    picks = []
    for family, rules in hedge.grid().items():
        scores = train.loc[[r.label() for r in rules], "calmar"]
        picks.append(next(r for r in rules if r.label() == scores.idxmax()))
    return picks


def hedge_test(study: Study, picks: list[hedge.HedgeRule]) -> tuple[pd.DataFrame, list[hedge.HedgeResult]]:
    w = study.wide()
    res = [hedge.run(r, w, config.TEST_START, config.END_DATE) for r in [hedge.HedgeRule("none")] + picks]
    return pd.DataFrame([hedge_summary(study, r) for r in res]).set_index("strategy"), res


# ── 3. do funding and open interest predict anything? ────────────────────────

def _buckets(x: pd.Series, edges: np.ndarray) -> pd.Series:
    return pd.Series(np.searchsorted(edges, x.to_numpy(), side="right") + 1, index=x.index).where(x.notna())


def funding_quintiles(study: Study) -> pd.DataFrame:
    """Forward 7-day returns and crash odds by funding quintile. Quintile
    edges come from each asset's TRAIN data and are reused unchanged on TEST."""
    rows = []
    for a in CARRY_ASSETS:
        d = study.asset(a).dropna(subset=["funding_7d_ann", "fwd_ret_7d"])
        tr = d[d.date <= config.TRAIN_END]
        edges = tr.funding_7d_ann.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        d = d.assign(q=_buckets(d.funding_7d_ann, edges),
                     period=np.where(d.date <= config.TRAIN_END, "train", "test"))
        for (period, q), g in d.groupby(["period", "q"]):
            rows.append({"asset": a, "period": period, "quintile": int(q), "days": len(g),
                         "funding_ann_mean": g.funding_7d_ann.mean(),
                         "fwd_ret_7d_mean": g.fwd_ret_7d.mean(),
                         "p_drop_10pct_7d": (g.fwd_worst_7d < -0.10).mean()})
    return pd.DataFrame(rows)


def leverage_buildup(study: Study) -> pd.DataFrame:
    """'Leverage build-up' days: open interest up into the top decile over 7
    days while price also rose. Are sharp drops more likely afterwards?"""
    rows = []
    for a in CARRY_ASSETS:
        d = study.asset(a).dropna(subset=["oi_change_7d", "spot_ret_7d", "fwd_worst_7d"])
        cut = d[d.date <= config.TRAIN_END].oi_change_7d.quantile(0.9)
        d = d.assign(event=(d.oi_change_7d > cut) & (d.spot_ret_7d > 0),
                     period=np.where(d.date <= config.TRAIN_END, "train", "test"))
        for period, g in d.groupby("period"):
            ev = g[g.event]
            rows.append({"asset": a, "period": period, "oi_cutoff": cut, "event_days": len(ev),
                         "p_drop_10pct_after_event": (ev.fwd_worst_7d < -0.10).mean() if len(ev) else np.nan,
                         "p_drop_10pct_all_days": (g.fwd_worst_7d < -0.10).mean(),
                         "fwd_ret_7d_after_event": ev.fwd_ret_7d.mean() if len(ev) else np.nan,
                         "fwd_ret_7d_all_days": g.fwd_ret_7d.mean()})
    return pd.DataFrame(rows)


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey–West (Bartlett) standard errors."""
    with np.errstate(all="ignore"):   # macOS Accelerate emits spurious matmul warnings
        return _newey_west(y, X, lags)


def _newey_west(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for k in range(1, lags + 1):
        g = Xu[k:].T @ Xu[:-k]
        S += (1 - k / (lags + 1)) * (g + g.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.diag(cov))


PREDICTORS = ("funding_7d_ann", "oi_change_7d", "spot_ret_7d", "basis_7d")


def predictive_regressions(study: Study) -> pd.DataFrame:
    """fwd 7-day return on standardised predictors, fitted on TRAIN with
    Newey–West t-stats (overlapping 7-day windows → 6 lags), scored on TEST
    by out-of-sample R² against the training-mean forecast."""
    rows = []
    for a in CARRY_ASSETS:
        d = study.asset(a).dropna(subset=list(PREDICTORS) + ["fwd_ret_7d"])
        tr, te = d[d.date <= config.TRAIN_END], d[d.date >= config.TEST_START]
        mu, sd = tr[list(PREDICTORS)].mean(), tr[list(PREDICTORS)].std()
        Xtr = np.column_stack([np.ones(len(tr)), ((tr[list(PREDICTORS)] - mu) / sd).to_numpy()])
        Xte = np.column_stack([np.ones(len(te)), ((te[list(PREDICTORS)] - mu) / sd).to_numpy()])
        beta, se = newey_west_ols(tr.fwd_ret_7d.to_numpy(), Xtr, lags=6)
        with np.errstate(all="ignore"):
            pred = Xte @ beta
        y = te.fwd_ret_7d.to_numpy()
        r2_oos = 1 - ((y - pred) ** 2).sum() / ((y - tr.fwd_ret_7d.mean()) ** 2).sum()
        row = {"asset": a, "train_days": len(tr), "test_days": len(te), "oos_r2": r2_oos}
        for name, b, s in zip(("const",) + PREDICTORS, beta, se):
            row[f"{name}_coef"] = b
            row[f"{name}_t"] = b / s
        rows.append(row)
    return pd.DataFrame(rows)


# ── 4. stress tests ──────────────────────────────────────────────────────────

def crash_episodes(study: Study, n: int = 6, gap_days: int = 45) -> pd.DataFrame:
    """The n worst 7-day BTC falls, at least `gap_days` apart."""
    btc = study.asset("BTC").set_index("date")
    r7 = (btc.spot_close / btc.spot_close.shift(7) - 1).dropna().sort_values()
    picked = []
    for d, v in r7.items():
        if all(abs((d - p).days) > gap_days for p, _ in picked):
            picked.append((d, v))
        if len(picked) == n:
            break
    rows = []
    for end, _ in sorted(picked):
        start = end - pd.Timedelta(days=7)
        row = {"start": start.date(), "end": end.date()}
        for a in CARRY_ASSETS:
            ap = study.asset(a).set_index("date")
            if start in ap.index and end in ap.index:
                row[f"{a}_ret"] = ap.spot_close[end] / ap.spot_close[start] - 1
                row[f"{a}_funding_before"] = ap.funding_7d_ann.get(start, np.nan)
                window = ap.loc[start + pd.Timedelta(days=1):end]
                row[f"{a}_funding_during"] = window.funding.mean() * 365
                if pd.notna(ap.oi_usd.get(start)) and pd.notna(ap.oi_usd.get(end)):
                    row[f"{a}_oi_change"] = ap.oi_usd[end] / ap.oi_usd[start] - 1
        rows.append(row)
    return pd.DataFrame(rows)


def episode_strategy_returns(episodes: pd.DataFrame, curves: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for _, e in episodes.iterrows():
        row = {"episode": f"{e.start} → {e.end}"}
        for name, eq in curves.items():
            s, t = pd.Timestamp(e.start), pd.Timestamp(e.end)
            row[name] = eq[t] / eq[s] - 1 if (s in eq.index and t in eq.index) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("episode")


def leverage_stress(study: Study) -> pd.DataFrame:
    """Price rise that liquidates a short at each leverage, against the
    largest mark-price rises actually seen within 1 and 7 days."""
    rows = []
    for a in CARRY_ASSETS:
        ap = study.asset(a).set_index("date")
        up1 = ap.mark_high / ap.perp_close.shift(1) - 1
        up7 = ap.mark_high.rolling(7).max() / ap.perp_close.shift(7) - 1
        for L in CARRY_LEVERAGE + (10,):
            x = carry.liquidation_move(L)
            rows.append({"asset": a, "leverage": L, "liquidating_rise": x,
                         "max_1d_rise": up1.max(), "max_7d_rise": up7.max(),
                         "days_1d_rise_exceeds": int((up1 > x).sum()),
                         "weeks_7d_rise_exceeds": int((up7 > x).sum())})
    return pd.DataFrame(rows)
