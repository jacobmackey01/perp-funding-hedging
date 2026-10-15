"""Engine checks on synthetic data plus data checks on the built database."""
import numpy as np
import pandas as pd
import pytest

from perp_risk import carry, config, evaluate, hedge

DAYS = 60


def synthetic(spot, perp=None, funding=0.0, mark_high=None, start="2021-01-01"):
    spot = np.asarray(spot, float)
    perp = spot.copy() if perp is None else np.asarray(perp, float)
    idx = pd.date_range(start, periods=len(spot), freq="D")
    df = pd.DataFrame({"asset": "BTC", "date": idx, "spot_close": spot, "perp_close": perp,
                       "mark_high": perp if mark_high is None else mark_high,
                       "funding": funding, "funding_7d_ann": 0.2})
    return df


def test_flat_market_earns_exactly_the_funding():
    df = synthetic(np.full(DAYS, 100.0), funding=0.001)
    # one month, so no margin reset compounds the funding back into the position
    r = carry.run(df, carry.CarryRule(leverage=1, rebalance="M", cash_yield=False),
                  "2021-01-01", "2021-01-31", cost_bps=0)
    # 1x: half the capital is spot, half margin, so carry = funding × notional
    notional = 1 / (1 + 1)
    assert r.funding_pnl.sum() == pytest.approx(0.001 * notional * (len(r.equity) - 1))
    assert r.equity.iloc[-1] == pytest.approx(1 + r.funding_pnl.sum())


def test_delta_neutral_when_spot_and_perp_move_together():
    path = 100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.03, DAYS))
    df = synthetic(path)
    r = carry.run(df, carry.CarryRule(leverage=1, rebalance="W", cash_yield=False),
                  "2021-01-01", "2021-03-01", cost_bps=0)
    assert r.equity.iloc[-1] == pytest.approx(1.0, abs=1e-12)
    assert not r.liquidations


def test_liquidation_when_mark_high_exceeds_the_threshold():
    spot = np.full(DAYS, 100.0)
    lev = 5
    x = carry.liquidation_move(lev)
    high = spot.copy()
    high[10] = 100 * (1 + x + 0.01)
    df = synthetic(spot, mark_high=high)
    r = carry.run(df, carry.CarryRule(leverage=lev, rebalance="D", cash_yield=False),
                  "2021-01-01", "2021-03-01", cost_bps=0)
    assert len(r.liquidations) >= 1 and r.liquidations[0]["date"] == df.date[10]
    # just below the threshold: no liquidation
    high[10] = 100 * (1 + x - 0.01)
    r2 = carry.run(synthetic(spot, mark_high=high), carry.CarryRule(leverage=lev, rebalance="D", cash_yield=False),
                   "2021-01-01", "2021-03-01", cost_bps=0)
    assert not r2.liquidations


def test_entry_gate_uses_yesterdays_signal():
    df = synthetic(np.full(DAYS, 100.0), funding=0.001)
    df["funding_7d_ann"] = 0.0
    df.loc[20:, "funding_7d_ann"] = 0.5
    r = carry.run(df, carry.CarryRule(enter_above=0.1, exit_below=0.0, cash_yield=False),
                  "2021-01-01", "2021-03-01", cost_bps=0)
    first_in = r.in_trade[r.in_trade].index[0]
    assert first_in == df.date[21]


def test_liquidation_move_formula():
    for lev in (1, 2, 3, 5, 10):
        x = carry.liquidation_move(lev)
        margin_left = 1 / lev - x               # per unit of starting notional
        assert margin_left == pytest.approx(carry.MAINTENANCE * (1 + x))


def test_newey_west_matches_ols_when_lags_zero():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(500), rng.normal(size=500)])
    y = X @ np.array([0.5, 2.0]) + rng.normal(size=500)
    beta, se = evaluate.newey_west_ols(y, X, lags=0)
    assert beta == pytest.approx([0.5, 2.0], abs=0.15)
    u = y - X @ beta
    white = np.sqrt(np.diag(np.linalg.inv(X.T @ X) @ (X.T * u**2) @ X @ np.linalg.inv(X.T @ X)))
    assert se == pytest.approx(white)


def test_full_hedge_removes_price_risk():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2021-01-01", periods=DAYS)
    r = pd.DataFrame(rng.normal(0, 0.04, (DAYS, 2)), idx, list(hedge.PAIR))
    w = {"spot_ret": r, "perp_ret": r, "funding": r * 0, "spot_close": (1 + r).cumprod()}
    full = hedge.run(hedge.HedgeRule("static", 1.0), w, "2021-01-02", "2021-03-01", cost_bps=0)
    none = hedge.run(hedge.HedgeRule("none"), w, "2021-01-02", "2021-03-01", cost_bps=0)
    # the hedge is re-sized only when it drifts >10% from target, so a little
    # price risk remains between re-sizes
    assert full.equity.pct_change().std() < 0.1 * none.equity.pct_change().std()


@pytest.mark.skipif(not config.DB_PATH.exists(), reason="run `make db` first")
def test_panel_has_no_lookahead_in_signals():
    from perp_risk import db
    p = db.load_panel()
    btc = p[p.asset == "BTC"].set_index("date")
    f7 = btc.funding.rolling(7).mean() * 365
    assert np.allclose(f7.dropna(), btc.funding_7d_ann.dropna(), atol=1e-12)
    sma = btc.spot_close.rolling(200).mean()
    assert np.allclose(sma.dropna(), btc.sma_200.dropna())


@pytest.mark.skipif(not config.DB_PATH.exists(), reason="run `make db` first")
def test_mark_high_is_used_and_sane():
    from perp_risk import db
    p = db.load_panel()
    assert (p.mark_high >= p.spot_close * 0.8).all()
    assert p.mark_imputed.sum() < 150          # only the months before Dec 2019
