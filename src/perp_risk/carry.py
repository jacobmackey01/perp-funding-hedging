"""Cash-and-carry: long spot, short the same number of coins in the perpetual,
collect funding. Delta-neutral, but not risk-free.

Accounting (one asset, daily, USDT):
  * spot leg     q coins, worth q * S
  * perp margin  M, the collateral posted for the short. Each day it gains
                 the funding and loses the short's mark-to-market:
                     M += funding_d * q * P_{d-1}  -  q * (P_d - P_{d-1})
  * idle cash    C, when out of the trade; earns the T-bill rate if enabled.
  * equity       q * S + M + C

Leverage L means the short's notional is L times its margin, so a position
needs 1 + 1/L dollars of capital per dollar of carry. A rise in the perp price
drains the margin; if the intraday MARK-price high (what Binance liquidates on;
last-trade highs include flash wicks) would have pushed margin below the
maintenance requirement before the next scheduled top-up, the short is
liquidated: the margin is lost, the spot leg is sold at the close and the
strategy waits for its next entry signal.

Signals are lagged one day: a decision at close d is executed at close d+1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

MAINTENANCE = 0.005     # maintenance margin, fraction of notional (Binance tier 1 is 0.4%)


@dataclass(frozen=True)
class CarryRule:
    leverage: float = 2.0
    rebalance: str = "W"          # D, W or M: how often margin is reset to target
    enter_above: float = -np.inf  # trailing 7-day annualised funding to (re)enter
    exit_below: float = -np.inf   # ... and to exit
    cash_yield: bool = True       # idle cash earns the T-bill rate

    def label(self) -> str:
        if np.isinf(self.enter_above):
            gate = "always on"
        else:
            gate = f"enter >{self.enter_above:.0%}, exit <{self.exit_below:.0%}"
        return f"{self.leverage:g}x, {self.rebalance} reset, {gate}"


@dataclass
class CarryResult:
    asset: str
    rule: CarryRule
    equity: pd.Series
    in_trade: pd.Series
    funding_pnl: pd.Series
    basis_pnl: pd.Series
    costs: pd.Series
    cash_pnl: pd.Series
    margin_ratio: pd.Series          # margin / required maintenance at the day's perp high
    liquidations: list = field(default_factory=list)


def _reset_day(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    if freq == "D":
        return np.ones(len(dates), bool)
    if freq == "W":
        return np.asarray(dates.dayofweek == 0)
    if freq == "M":
        return np.asarray(dates.day == 1)
    raise ValueError(freq)


def run(asset_panel: pd.DataFrame, rule: CarryRule, start: str, end: str,
        cost_bps: float = config.COST_BPS, tbill: pd.Series | None = None) -> CarryResult:
    """`asset_panel`: one asset's rows from db.load_panel(), any date range."""
    p = asset_panel.set_index("date").sort_index()
    signal = p["funding_7d_ann"].shift(1)                 # known at the previous close
    p = p.loc[start:end]
    signal = signal.loc[start:end]
    dates = p.index
    S, P = p.spot_close.to_numpy(), p.perp_close.to_numpy()
    H, F = p.mark_high.to_numpy(), p.funding.to_numpy()
    sig = signal.to_numpy()
    reset = _reset_day(dates, rule.rebalance)
    c = cost_bps / 10_000
    L = rule.leverage
    rf = (tbill.reindex(dates).ffill().fillna(0).to_numpy() / config.DAYS_PER_YEAR
          if (tbill is not None and rule.cash_yield) else np.zeros(len(dates)))

    n = len(dates)
    eq = np.empty(n); live = np.zeros(n, bool)
    f_pnl = np.zeros(n); b_pnl = np.zeros(n); cost = np.zeros(n); cash_pnl = np.zeros(n)
    mratio = np.full(n, np.nan)
    liquidations = []

    q, M, C = 0.0, 0.0, 1.0
    want_in = False

    def open_position(i: int) -> None:
        nonlocal q, M, C
        equity = C
        # q*S + q*P/L + 2*c*q*(S+P)/2 ≈ equity  →  size so costs come out of equity
        q = equity / (S[i] + P[i] / L + c * (S[i] + P[i]))
        M = q * P[i] / L
        cost[i] += c * q * (S[i] + P[i])
        C = equity - q * S[i] - M - c * q * (S[i] + P[i])

    def close_position(i: int) -> None:
        nonlocal q, M, C
        cost[i] += c * q * (S[i] + P[i])
        C += q * S[i] + M - c * q * (S[i] + P[i])
        q, M = 0.0, 0.0

    for i in range(n):
        if i > 0:
            cash_pnl[i] = C * rf[i]
            C += cash_pnl[i]
            if q > 0:
                # liquidation check at the day's worst price for the short
                adverse = q * (H[i] - P[i - 1])
                if M - adverse < MAINTENANCE * q * H[i]:
                    liquidations.append({"date": dates[i], "asset": p.asset.iloc[0],
                                         "move": H[i] / P[i - 1] - 1, "leverage": L})
                    b_pnl[i] += -M                       # margin gone
                    M = 0.0
                    cost[i] += c * q * S[i]
                    C += q * S[i] * (1 - c)              # sell the now-unhedged spot
                    b_pnl[i] += q * (S[i] - S[i - 1])
                    q = 0.0
                    want_in = False
                else:
                    fund = F[i] * q * P[i - 1]
                    spot_move = q * (S[i] - S[i - 1])
                    perp_move = -q * (P[i] - P[i - 1])
                    M += fund + perp_move
                    f_pnl[i] = fund
                    b_pnl[i] = spot_move + perp_move
                    mratio[i] = (M - q * max(H[i] - P[i], 0)) / (MAINTENANCE * q * H[i])

        # entry / exit decisions use yesterday's signal
        s = sig[i]
        if not np.isnan(s):
            if q == 0 and s > rule.enter_above:
                want_in = True
            elif q > 0 and s < rule.exit_below:
                want_in = False
        elif np.isinf(rule.enter_above) and rule.enter_above < 0:
            want_in = True

        if q == 0 and want_in:
            open_position(i)
        elif q > 0 and not want_in:
            close_position(i)
        elif q > 0 and reset[i]:
            # restore target leverage: resize both legs to the current equity
            equity = q * S[i] + M + C
            q_new = (equity - C) / (S[i] + P[i] / L)
            traded = abs(q_new - q)
            cost[i] += c * traded * (S[i] + P[i])
            q = q_new
            M = q * P[i] / L - c * traded * (S[i] + P[i])

        eq[i] = q * S[i] + M + C
        live[i] = q > 0

    idx = dates
    return CarryResult(
        asset=p.asset.iloc[0], rule=rule, equity=pd.Series(eq, idx), in_trade=pd.Series(live, idx),
        funding_pnl=pd.Series(f_pnl, idx), basis_pnl=pd.Series(b_pnl, idx), costs=pd.Series(cost, idx),
        cash_pnl=pd.Series(cash_pnl, idx), margin_ratio=pd.Series(mratio, idx), liquidations=liquidations,
    )


def liquidation_move(leverage: float, mm: float = MAINTENANCE) -> float:
    """Price rise that liquidates a fresh short at `leverage` with no top-up:
    margin 1/L must fall to mm * (1 + x)  →  x = (1/L - mm) / (1 + mm)."""
    return (1 / leverage - mm) / (1 + mm)
