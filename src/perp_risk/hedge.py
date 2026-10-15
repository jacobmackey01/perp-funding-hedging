"""Hedging a spot BTC/ETH portfolio by shorting perpetuals.

The portfolio holds BTC and ETH 50/50 (rebalanced on the first of each month,
as in the spot study). A hedge rule sets, for each asset, the fraction h of
that asset's position to short in its perp. The short pays or earns funding:
in bull markets longs pay shorts, so a hedge can *earn* while it is on.

Daily P&L per asset a, with spot value X_a and hedge notional N_a = h_a * X_a
(both set at the previous close):
    X_a * r_spot  -  N_a * r_perp  +  N_a * funding
The hedge is covered by the coins held, so it is not liquidated on its own;
in practice that needs portfolio margin or a separate collateral buffer.

Hedge targets use signals known at the previous close, so a rule that turns
the hedge on at close d acts on day d+1's returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

PAIR = ("BTC", "ETH")


@dataclass(frozen=True)
class HedgeRule:
    kind: str                  # none | static | trend | vol | crowding
    hmax: float = 0.0          # hedge fraction when the rule is on
    param: float = 0.0         # SMA days / vol threshold / funding threshold

    def label(self) -> str:
        if self.kind == "none":
            return "Unhedged 50/50"
        if self.kind == "static":
            return f"Static hedge {self.hmax:.0%}"
        what = {"trend": f"price < {self.param:g}-day average",
                "vol": f"30-day vol > {self.param:.0%}",
                "crowding": f"7-day funding > {self.param:.0%} a year"}[self.kind]
        return f"Hedge {self.hmax:.0%} when {what}"

    @property
    def family(self) -> str:
        return {"none": "Unhedged", "static": "Static hedge", "trend": "Trend hedge",
                "vol": "Volatility hedge", "crowding": "Crowded-funding hedge"}[self.kind]


def grid() -> dict[str, list[HedgeRule]]:
    return {
        # A 100% static hedge is the cash-and-carry trade, not a hedge; it is
        # reported as a reference point, not selected.
        "Static hedge": [HedgeRule("static", h) for h in (0.25, 0.5)],
        "Trend hedge": [HedgeRule("trend", h, n) for h in (0.5, 1.0) for n in (50, 100, 200)],
        "Volatility hedge": [HedgeRule("vol", h, v) for h in (0.5, 1.0) for v in (0.6, 0.8, 1.0)],
        "Crowded-funding hedge": [HedgeRule("crowding", h, f) for h in (0.5, 1.0) for f in (0.15, 0.3, 0.5)],
    }


def targets(rule: HedgeRule, w: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Hedge fraction per asset per day, decided at that day's close."""
    close = w["spot_close"][list(PAIR)]
    if rule.kind == "none":
        h = close * 0.0
    elif rule.kind == "static":
        h = close * 0.0 + rule.hmax
    elif rule.kind == "trend":
        sma = w[f"sma_{int(rule.param)}"][list(PAIR)]
        h = (close < sma).astype(float) * rule.hmax
        h[sma.isna()] = np.nan
    elif rule.kind == "vol":
        v = w["vol_30d"][list(PAIR)]
        h = (v > rule.param).astype(float) * rule.hmax
        h[v.isna()] = np.nan
    elif rule.kind == "crowding":
        f = w["funding_7d_ann"][list(PAIR)]
        h = (f > rule.param).astype(float) * rule.hmax
        h[f.isna()] = np.nan
    else:
        raise ValueError(rule.kind)
    return h


@dataclass
class HedgeResult:
    rule: HedgeRule
    equity: pd.Series
    hedge: pd.DataFrame          # hedge fraction in force each day
    funding_pnl: pd.Series
    hedge_pnl: pd.Series         # perp mark-to-market on the short
    costs: pd.Series


def run(rule: HedgeRule, w: dict[str, pd.DataFrame], start: str, end: str,
        cost_bps: float = config.COST_BPS) -> HedgeResult:
    h_target = targets(rule, w).shift(1)            # decided at the previous close
    idx = w["spot_close"].loc[start:end].index
    rs = w["spot_ret"].loc[idx, list(PAIR)].to_numpy()
    rp = w["perp_ret"].loc[idx, list(PAIR)].to_numpy()
    fu = w["funding"].loc[idx, list(PAIR)].fillna(0).to_numpy()
    H = h_target.loc[idx].to_numpy()
    if np.isnan(H[0]).any():
        raise ValueError(f"{rule.label()}: no signal at {idx[0].date()}")
    month_start = np.asarray(idx.day == 1)
    c = cost_bps / 10_000

    n = len(idx)
    eq = np.empty(n); f_pnl = np.zeros(n); h_pnl = np.zeros(n); cost = np.zeros(n)
    held = np.zeros((n, 2))
    X = np.array([0.5, 0.5])          # spot value per asset (start equity = 1)
    N = H[0] * X                      # hedge notional
    for i in range(n):
        if i > 0:
            spot = X * rs[i]
            perp = -N * rp[i]
            fund = N * fu[i]
            X = X * (1 + rs[i])
            f_pnl[i] = fund.sum(); h_pnl[i] = perp.sum()
            # P&L from the perp leg settles in cash, which sits with the portfolio
            cash_delta = perp.sum() + fund.sum()
            X = X + cash_delta * X / X.sum()
        V = X.sum()
        if month_start[i]:
            new_X = np.array([V / 2, V / 2])
            cost[i] += c * np.abs(new_X - X).sum()
            X = new_X
        target = np.nan_to_num(H[i]) * X
        traded = np.abs(target - N)
        # only trade when the hedge decision changes or positions drift > 10%
        if (np.abs(np.nan_to_num(H[i]) - (N / np.maximum(X, 1e-12))) > 0.1).any() or month_start[i]:
            cost[i] += c * traded.sum()
            N = target
        X = X - cost[i] * X / X.sum()
        eq[i] = X.sum()
        held[i] = N / X
    return HedgeResult(rule, pd.Series(eq, idx), pd.DataFrame(held, idx, columns=list(PAIR)),
                       pd.Series(f_pnl, idx), pd.Series(h_pnl, idx), pd.Series(cost, idx))
