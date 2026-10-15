"""Return and risk metrics on a daily equity curve."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DAYS_PER_YEAR


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def summarize(equity: pd.Series, rf_daily: pd.Series | None = None) -> dict:
    """`equity` starts at the value held at the start of the window."""
    r = equity.pct_change().dropna()
    years = len(r) / DAYS_PER_YEAR
    ex = r - (rf_daily.reindex(r.index).fillna(0) if rf_daily is not None else 0)
    mdd = float(drawdown(equity).min())
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
    return {
        "cagr": cagr,
        "ann_vol": float(r.std() * np.sqrt(DAYS_PER_YEAR)),
        "sharpe_vs_tbill": float(ex.mean() / r.std() * np.sqrt(DAYS_PER_YEAR)) if r.std() > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "worst_day": float(r.min()),
        "worst_month": float((1 + r).resample("MS").prod().sub(1).min()),
    }


def tbill_daily(tbill: pd.Series) -> pd.Series:
    return tbill / DAYS_PER_YEAR


def tbill_equity(tbill: pd.Series, start: str, end: str) -> pd.Series:
    d = tbill_daily(tbill).loc[start:end]
    return (1 + d.shift(1).fillna(0)).cumprod()
