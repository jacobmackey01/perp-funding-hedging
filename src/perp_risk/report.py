"""Run the whole study and write tables, figures and the HTML report.

Run:  python -m perp_risk.report        (after `python -m perp_risk.db`)
"""
from __future__ import annotations

import base64
import html
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import carry, config, db, hedge, metrics  # noqa: E402
from . import evaluate as ev  # noqa: E402

TABLES = config.REPORT_DIR / "tables"
FIGURES = config.REPORT_DIR / "figures"

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
FAINT = "#d6d5ce"
ASSET_COLOR = dict(zip(config.ASSETS, SLOT))
STROKE = matplotlib.patheffects.withStroke(linewidth=3, foreground=SURFACE)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.3, "legend.frameon": False, "axes.titlelocation": "left",
    "axes.titlesize": 11, "axes.titleweight": "bold",
})


# ── run everything ───────────────────────────────────────────────────────────

def run_all() -> dict:
    s = ev.Study.load()
    c_train, c_test_grid = ev.carry_grid(s)
    c_picks = ev.carry_select(c_train)
    c_test, c_res, c_combo = ev.carry_test(s, c_picks)
    h_train, h_test_grid = ev.hedge_grid(s)
    h_picks = ev.hedge_select(h_train)
    h_test, h_res = ev.hedge_test(s, h_picks)

    # full-history curves for the crash episodes (train and test together)
    w = s.wide()
    curves = {}
    for rule in [hedge.HedgeRule("none"), hedge.HedgeRule("static", 0.5)] + [p for p in h_picks if p.kind == "trend"]:
        curves[rule.label()] = hedge.run(rule, w, s.hedge_start(), config.END_DATE).equity
    for a in ("BTC", "ETH"):
        curves[f"{a} carry, picked rule"] = carry.run(s.asset(a), c_picks[a], s.carry_start(a), config.END_DATE, tbill=s.tbill).equity
        curves[f"{a} carry, 3x always on"] = carry.run(s.asset(a), carry.CarryRule(leverage=3, rebalance="W"),
                                                     s.carry_start(a), config.END_DATE, tbill=s.tbill).equity
    episodes = ev.crash_episodes(s)

    return {
        "study": s, "carry_train": c_train, "carry_test_grid": c_test_grid, "carry_picks": c_picks,
        "carry_test": c_test, "carry_res": c_res, "carry_combo": c_combo,
        "carry_by_year": ev.carry_by_year(c_res), "carry_costs": ev.carry_cost_sensitivity(s, c_picks),
        "hedge_train": h_train, "hedge_test_grid": h_test_grid, "hedge_picks": h_picks,
        "hedge_test": h_test, "hedge_res": h_res,
        "quintiles": ev.funding_quintiles(s), "buildup": ev.leverage_buildup(s),
        "regressions": ev.predictive_regressions(s),
        "episodes": episodes, "episode_returns": ev.episode_strategy_returns(episodes, curves),
        "leverage": ev.leverage_stress(s),
        "funding_by_year": db.query_file("funding_by_year.sql"),
        "source_check": db.query_file("funding_source_check.sql"),
        "snapshot": db.query_file("market_snapshot.sql"),
        "wicks": db.query_file("wicks.sql"),
    }


def save_tables(r: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for key in ("carry_train", "carry_test_grid", "carry_test", "carry_by_year", "carry_costs",
                "quintiles", "buildup", "regressions", "episodes", "leverage",
                "funding_by_year", "source_check", "snapshot", "wicks"):
        r[key].to_csv(TABLES / f"{key}.csv", index=False)
    for key in ("hedge_train", "hedge_test_grid", "hedge_test", "episode_returns"):
        r[key].to_csv(TABLES / f"{key}.csv")


# ── figures ──────────────────────────────────────────────────────────────────

def _pct(ax, axis="y", decimals=0):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _save(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=160)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode((FIGURES / name).read_bytes()).decode()


def fig_funding(r: dict) -> str:
    s = r["study"]
    tb = s.tbill
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.4), sharex=True)
    for ax, a in zip(axes, config.ASSETS):
        d = s.asset(a).set_index("date")
        ax.plot(d.index, d.funding_30d_ann, color=ASSET_COLOR[a], label=f"{a} perp funding, 30-day average")
        ax.plot(tb.index, tb, color=INK_2, lw=1.1, label="3-month T-bill")
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(pd.Timestamp(config.TEST_START), color=AXIS, lw=1)
        ax.set_ylim(-0.6, 1.0)
        _pct(ax)
        ax.set_title(f"{a}: annualised funding paid to shorts", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, ncols=2)
    axes[0].annotate("  out-of-sample period", (pd.Timestamp(config.TEST_START), 1), xycoords=("data", "axes fraction"),
                     va="top", fontsize=8.5, color=INK_2)
    axes[-1].annotate("SOL fell below −60% in Nov 2022 (FTX); axis clipped", (0.01, 0.04), xycoords="axes fraction",
                      fontsize=8, color=INK_2)
    fig.tight_layout()
    return _save(fig, "funding.png")


def fig_carry(r: dict) -> str:
    s = r["study"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.set_title("Cash-and-carry vs T-bills, out-of-sample, after costs")
    tb = metrics.tbill_equity(s.tbill, config.TEST_START, config.END_DATE)
    ax.plot(tb.index, tb, color=INK_2, label="T-bills")
    for a, res in r["carry_res"].items():
        eq = res.equity / res.equity.iloc[0]
        ax.plot(eq.index, eq, color=ASSET_COLOR[a], label=f"{a}: {res.rule.label()}")
    ref = carry.run(s.asset("BTC"), carry.CarryRule(leverage=1, rebalance="W"), config.TEST_START, config.END_DATE, tbill=s.tbill)
    ax.plot(ref.equity.index, ref.equity, color=ASSET_COLOR["BTC"], lw=1, ls=(0, (1, 1.5)),
            label="BTC: 1x, always on (reference)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return _save(fig, "carry.png")


def fig_hedge(r: dict) -> str:
    res = {x.rule.label(): x for x in r["hedge_res"]}
    trend = next(p for p in r["hedge_picks"] if p.kind == "trend").label()
    static = next(p for p in r["hedge_picks"] if p.kind == "static").label()
    feat = [("Unhedged 50/50", INK_2), (static, SLOT[0]), (trend, SLOT[1])]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    a1.set_title("BTC/ETH 50/50 with perp hedges, out-of-sample, after costs")
    a2.set_title("Drawdown from previous peak")
    for lab, color in feat:
        eq = res[lab].equity
        a1.plot(eq.index, eq, color=color, label=lab)
        a2.plot(eq.index, metrics.drawdown(eq), color=color, lw=1.1)
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    a1.legend(loc="upper left", fontsize=8.5)
    _pct(a2)
    fig.tight_layout()
    return _save(fig, "hedge.png")


def fig_quintiles(r: dict) -> str:
    q = r["quintiles"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    width = 0.38
    for ax, a in zip(axes, config.ASSETS):
        d = q[q.asset == a]
        for k, (period, color) in enumerate((("train", FAINT), ("test", SLOT[0]))):
            g = d[d.period == period].set_index("quintile")
            ax.bar(g.index + (k - 0.5) * width, g.p_drop_10pct_7d, width=width - 0.04, color=color,
                   label="training" if period == "train" else "out-of-sample")
        ax.set_title(a, fontsize=10)
        ax.set_xticks(range(1, 6), ["lowest", "2", "3", "4", "highest"])
        ax.set_xlabel("7-day funding quintile")
        _pct(ax)
    axes[0].set_ylabel("Chance of a 10%+ fall\nwithin the next 7 days")
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Does crowded positioning (high funding) precede sharp falls?", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, "quintiles.png")


# ── HTML ─────────────────────────────────────────────────────────────────────

PCT_COLS = {"cagr", "ann_vol", "max_drawdown", "worst_day", "worst_month", "time_in_trade", "funding_per_year",
            "basis_per_year", "cash_yield_per_year", "costs_per_year", "avg_hedge", "hedge_pnl_per_year",
            "funding_total", "funding_ann", "share_days_positive", "best_day", "worst_day", "avg_basis",
            "binance_ann", "bybit_ann", "same_sign", "return", "funding", "liquidating_rise", "max_1d_rise",
            "max_7d_rise", "funding_ann_mean", "fwd_ret_7d_mean", "p_drop_10pct_7d", "oi_cutoff",
            "p_drop_10pct_after_event", "p_drop_10pct_all_days", "fwd_ret_7d_after_event", "fwd_ret_7d_all_days",
            "basis", "funding_7d_ann", "funding_30d_ann", "oi_change_7d", "vol_30d", "vs_sma_200",
            "funding_7d_pctile", "oi_change_pctile", "vol_pctile", "last_trade_up", "mark_up", "spot_up",
            "cagr_vs_tbill"}
NAMES = {
    "cagr": "CAGR", "ann_vol": "Volatility", "sharpe_vs_tbill": "Sharpe vs T-bills", "max_drawdown": "Max drawdown",
    "calmar": "Calmar", "worst_day": "Worst day", "worst_month": "Worst month", "liquidations": "Liquidations",
    "time_in_trade": "Time in trade", "funding_per_year": "Funding / yr", "basis_per_year": "Basis & price / yr",
    "cash_yield_per_year": "T-bill yield / yr", "costs_per_year": "Costs / yr", "min_margin_ratio": "Min margin ÷ maintenance",
    "avg_hedge": "Avg hedge", "hedge_pnl_per_year": "Perp P&L / yr", "funding_ann": "Funding, annualised",
    "share_days_positive": "Days positive", "best_day": "Best day", "worst_day ": "Worst day", "avg_basis": "Avg basis",
    "avg_oi_usd_bn": "Avg OI ($bn)", "funding_total": "Funding, total", "rule": "Rule (picked on training data)",
    "liquidating_rise": "Rise that liquidates", "max_1d_rise": "Largest 1-day rise", "max_7d_rise": "Largest 7-day rise",
    "days_1d_rise_exceeds": "Days it was exceeded", "weeks_7d_rise_exceeds": "7-day windows it was exceeded",
    "funding_ann_mean": "Avg funding (ann.)", "fwd_ret_7d_mean": "Next 7-day return", "p_drop_10pct_7d": "P(10%+ fall in 7d)",
    "p_drop_10pct_after_event": "P(10%+ fall), after build-up", "p_drop_10pct_all_days": "P(10%+ fall), all days",
    "fwd_ret_7d_after_event": "Next 7d return, after build-up", "fwd_ret_7d_all_days": "Next 7d return, all days",
    "oi_cutoff": "OI 7-day rise cut-off", "event_days": "Build-up days", "oos_r2": "Out-of-sample R²",
    "spot_close": "Spot", "basis": "Basis", "funding_7d_ann": "Funding 7d (ann.)", "funding_30d_ann": "Funding 30d (ann.)",
    "oi_change_7d": "OI change 7d", "oi_usd": "OI ($bn)", "vol_30d": "Vol 30d", "vs_sma_200": "vs 200-day avg",
    "funding_7d_pctile": "Funding percentile", "oi_change_pctile": "OI-change percentile", "vol_pctile": "Vol percentile",
    "last_trade_up": "Last-trade high", "mark_up": "Mark-price high", "spot_up": "Spot high",
    "binance_ann": "Binance (ann.)", "bybit_ann": "Bybit (ann.)", "same_sign": "Same sign", "correlation": "Correlation",
    "overlapping_days": "Days", "first_day": "From",
}


def _fmt(col, v) -> str:
    if isinstance(v, str):
        return html.escape(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if col in PCT_COLS or str(col).endswith("_ret") or str(col).endswith("_funding_before") \
            or str(col).endswith("_funding_during") or str(col).endswith("_oi_change") or col.startswith("ret_"):
        return f"{v:.1%}"
    if col == "oi_usd":
        return f"{v / 1e9:.1f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,d}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"
    return html.escape(str(v))


def _table(df: pd.DataFrame, cols=None, index=True) -> str:
    df = df if cols is None else df[cols]
    head = "".join(f"<th>{html.escape(NAMES.get(c, str(c).replace('_', ' ')))}</th>" for c in df.columns)
    rows = []
    for i, row in df.iterrows():
        cells = "".join(f"<td>{_fmt(c, row[c])}</td>" for c in df.columns)
        rows.append(f"<tr><th scope='row'>{html.escape(str(i))}</th>{cells}</tr>" if index else f"<tr>{cells}</tr>")
    first = "<th></th>" if index else ""
    return f"<div class='scroll'><table><thead><tr>{first}{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def write_html(r: dict, figs: dict) -> None:
    s = r["study"]
    ct = r["carry_test"].set_index("asset")
    fy = r["funding_by_year"].set_index(["asset", "year"])
    ht = r["hedge_test"]
    trend = next(p for p in r["hedge_picks"] if p.kind == "trend").label()
    static = next(p for p in r["hedge_picks"] if p.kind == "static").label()
    crowd = next(p for p in r["hedge_picks"] if p.kind == "crowding").label()
    un = ht.loc["Unhedged 50/50"]
    regs = r["regressions"].set_index("asset")
    lev = r["leverage"].set_index(["asset", "leverage"])
    tg = r["carry_test_grid"]
    always = tg[(tg.rule == "1x, W reset, always on")].set_index("asset")
    liq_test = tg.groupby(["asset", "leverage"]).liquidations.sum().unstack()
    liq_train = r["carry_train"].groupby(["asset", "leverage"]).liquidations.sum().unstack()
    q = r["quintiles"]
    sol_ftx = r["episodes"].dropna(subset=["SOL_funding_during"]).sort_values("SOL_funding_during").iloc[0]
    by_year = r["carry_by_year"]
    last_year = by_year.year.max()
    ep_ret = r["episode_returns"]

    def qtop(a):
        g = q[(q.asset == a) & (q.period == "test")].set_index("quintile")
        return f"{g.loc[5, 'p_drop_10pct_7d']:.0%} vs {g.loc[1:4, 'p_drop_10pct_7d'].mean():.0%}"

    def qdays(a):
        return int(q[(q.asset == a) & (q.period == "test") & (q.quintile == 5)].days.iloc[0])

    epi = r["episodes"]
    n_lower = int((epi.BTC_funding_during < epi.BTC_funding_before).sum())
    n_high = int((epi.BTC_funding_before > 0.30).sum())
    high_when = ", ".join(pd.to_datetime(epi[epi.BTC_funding_before > 0.30].end).dt.strftime("%b %Y"))
    l3 = pd.concat({"train": r["carry_train"], "test": tg}).reset_index(level=0).rename(columns={"level_0": "period"})
    l3 = l3[l3.rule == "3x, W reset, always on"].set_index(["period", "asset"]).liquidations

    def fy_ann(a, y):
        return fy.loc[(a, str(y)), "funding_ann"]

    findings = f"""
<ol class="findings">
<li><b>The funding premium has mostly gone.</b> Shorting BTC perps paid {fy_ann('BTC', 2021):.0%} a year in 2021 and
{fy_ann('ETH', 2021):.0%} for ETH. Since then it has fallen to {fy_ann('BTC', 2025):.1%} and {fy_ann('ETH', 2025):.1%} in 2025,
and {fy_ann('BTC', last_year):.1%} and {fy_ann('ETH', last_year):.1%} annualised so far in {last_year}. That is at or below the
3-month T-bill yield ({s.tbill.loc[str(last_year)].mean():.1%} in {last_year}).</li>
<li><b>Out of sample, cash-and-carry barely beat T-bills.</b> With rules picked on 2019–2022 data, BTC+ETH carry returned
{ct.loc['BTC+ETH 50/50', 'cagr']:.1%} a year from 2023, against {ct.loc['T-bills', 'cagr']:.1%} for T-bills. The picked rules only
entered when funding was high, so they spent most of the time in T-bills ({ct.loc['BTC', 'time_in_trade']:.0%} of days in the trade for BTC,
{ct.loc['ETH', 'time_in_trade']:.0%} for ETH). Staying in the trade all the time at 1x would have returned
{always.loc['BTC', 'cagr']:.1%} (BTC) and {always.loc['ETH', 'cagr']:.1%} (ETH), below T-bills.</li>
<li><b>The real risk in carry is leverage on the short leg, and it is very uneven across assets.</b> With no top-up, a 3x short is
liquidated by a {carry.liquidation_move(3):.0%} rise in the mark price. BTC's largest 7-day rise was {lev.loc[('BTC', 3), 'max_7d_rise']:.0%};
ETH's was {lev.loc[('ETH', 3), 'max_7d_rise']:.0%} and SOL's {lev.loc[('SOL', 3), 'max_7d_rise']:.0%}. Even with the margin topped up weekly, an always-on 3x carry was liquidated
{int(l3.loc[('train', 'ETH')])} times on ETH in training and {int(l3.loc[('test', 'SOL')])} times on SOL out of sample;
on BTC it was liquidated {int(l3.loc[('train', 'BTC')] + l3.loc[('test', 'BTC')])} times in total. Funding can also flip hard:
in the FTX week, SOL shorts paid funding at {abs(sol_ftx.SOL_funding_during):.0%} a year.</li>
<li><b>Trend-based perp hedges cut drawdowns, at a cost in return.</b> Out of sample, the trend rule picked on training data
({html.escape(trend.lower())}) fell {abs(ht.loc[trend, 'max_drawdown']):.0%} at its worst, against {abs(un.max_drawdown):.0%} unhedged,
and returned {ht.loc[trend, 'cagr']:.0%} a year against {un.cagr:.0%}. A permanent {static.split()[-1]} hedge fell {abs(ht.loc[static, 'max_drawdown']):.0%}
and returned {ht.loc[static, 'cagr']:.0%}. Because longs usually pay shorts, a hedge earns funding while it is on; the funding received partly offsets what the hedge gives up.</li>
<li><b>Funding signalled turbulence, not direction.</b> For BTC and ETH, the highest-funding fifth of days was followed by a
10%+ fall within a week more often than other days. Out of sample that was {qtop('BTC')} for BTC and {qtop('ETH')} for ETH
(the ETH figure rests on only {qdays('ETH')} days). Average returns after those days were not lower, though, because crowded
markets also kept rallying. A regression of next-week returns on funding, OI change, past return and basis had a negative
out-of-sample R² for all three assets ({regs.oos_r2.min():.3f} to {regs.oos_r2.max():.3f}), so it forecast worse than the historical
average. Hedging only when funding was crowded ({html.escape(crowd.lower())}) did no better than staying unhedged.</li>
<li><b>Crashes show up in the funding data during the fall, not reliably before it.</b> In the {len(r['episodes'])} worst BTC
weeks since 2019, BTC funding was lower during the fall than in the week before in {n_lower} of {len(r['episodes'])} cases.
Funding was above 30% a year going into {n_high} of them ({high_when}) and at ordinary levels before the rest.</li>
</ol>"""

    snap = r["snapshot"].set_index("asset")
    snap_cols = ["date", "spot_close", "basis", "funding_7d_ann", "funding_30d_ann", "oi_usd", "oi_change_7d",
                 "vol_30d", "vs_sma_200", "funding_7d_pctile", "oi_change_pctile", "vol_pctile"]

    carry_cols = ["rule", "cagr", "ann_vol", "sharpe_vs_tbill", "max_drawdown", "liquidations", "time_in_trade",
                  "funding_per_year", "cash_yield_per_year", "costs_per_year"]
    ref = always.reset_index()[["asset"] + carry_cols].assign(rule=lambda d: d.rule + " (reference)")
    carry_view = pd.concat([r["carry_test"][["asset"] + carry_cols], ref]).set_index("asset")

    lev_view = r["leverage"].pivot(index="leverage", columns="asset", values="liquidating_rise")
    lev_view.columns = [f"{c}" for c in lev_view.columns]
    liq_view = pd.concat({"training": liq_train, "out-of-sample": liq_test}, axis=1)
    liq_view.columns = [f"{a} {b}x" for a, b in liq_view.columns]
    liq_tbl = pd.concat({"training": liq_train.T, "out-of-sample": liq_test.T}, axis=1)
    liq_tbl.index = [f"{i}x" for i in liq_tbl.index]
    liq_tbl.columns = [f"{b} ({a})" for a, b in liq_tbl.columns]
    lev_tbl = r["leverage"][r["leverage"].asset != "x"].copy()
    lev_tbl.index = lev_tbl.asset + " " + lev_tbl.leverage.astype(str) + "x"
    lev_tbl = lev_tbl.drop(columns=["asset", "leverage"])

    cc = r["carry_costs"].pivot(index="asset", columns="cost_bps", values="cagr")
    cc.columns = [f"{c:g} bps" for c in cc.columns]
    PCT_COLS.update(cc.columns)

    by_year_view = by_year.pivot(index="year", columns="asset", values="return")
    by_year_view.columns = [f"{c} return" for c in by_year_view.columns]
    PCT_COLS.update(by_year_view.columns)

    fyv = r["funding_by_year"].pivot(index="year", columns="asset", values="funding_ann")
    fyv.columns = [f"{c} funding (ann.)" for c in fyv.columns]
    fyv["T-bill"] = s.tbill.groupby(s.tbill.index.year).mean().reindex(fyv.index.astype(int)).to_numpy()
    PCT_COLS.update(fyv.columns)

    hedge_cols = ["cagr", "ann_vol", "sharpe_vs_tbill", "max_drawdown", "calmar", "avg_hedge",
                  "funding_per_year", "hedge_pnl_per_year", "costs_per_year"]

    qv = q.pivot_table(index=["asset", "quintile"], columns="period", values=["funding_ann_mean", "p_drop_10pct_7d", "fwd_ret_7d_mean"])
    qv.columns = [f"{a} ({b})" for a, b in qv.columns]
    PCT_COLS.update(qv.columns)
    qv.index = [f"{a} Q{k}" for a, k in qv.index]

    reg_cols = ["oos_r2"] + [f"{p}_t" for p in ev.PREDICTORS]
    regv = regs[reg_cols].rename(columns={f"{p}_t": f"t: {p.replace('_', ' ')}" for p in ev.PREDICTORS})

    bu = r["buildup"].set_index(["asset", "period"])
    bu.index = [f"{a} {p}" for a, p in bu.index]

    ep = r["episodes"].copy()
    ep.index = ep.start.astype(str) + " → " + ep.end.astype(str)
    ep = ep.drop(columns=["start", "end"])
    epr = ep_ret.copy()
    epr.columns = [f"ret_{c}" for c in epr.columns]
    NAMES.update({c: c[4:] for c in epr.columns})

    body = f"""
<header>
<p class="eyebrow">Derivatives research · Python, SQL · data to {config.END_DATE}</p>
<h1>Perpetual futures: funding, carry and hedging</h1>
<p class="lede">This study uses Binance perpetual futures on BTC, ETH and SOL to test three things. Does the funding paid by
leveraged longs give a carry trade that is worth its risk? Does shorting perps make a spot portfolio safer, and at what cost?
Does funding or open interest warn of crashes? Rules were tuned on data to {config.TRAIN_END[:7]}
and then judged once on {config.TEST_START[:7]} to {config.END_DATE[:7]}.</p>
</header>

<section class="tiles">
<div class="tile"><span>BTC+ETH carry, out-of-sample</span><b>{ct.loc['BTC+ETH 50/50', 'cagr']:.1%}</b><small>a year, vs {ct.loc['T-bills', 'cagr']:.1%} T-bills</small></div>
<div class="tile"><span>BTC funding, 2021 → {last_year}</span><b>{fy_ann('BTC', 2021):.0%} → {fy_ann('BTC', last_year):.0%}</b><small>annualised</small></div>
<div class="tile"><span>Trend hedge, max drawdown</span><b>{ht.loc[trend, 'max_drawdown']:.0%}</b><small>vs {un.max_drawdown:.0%} unhedged</small></div>
<div class="tile"><span>Funding/OI return forecasts, out-of-sample R²</span><b>{regs.oos_r2.max():.3f}</b><small>best of 3 assets; below 0 = no skill</small></div>
</section>

<h2>Findings</h2>
{findings}

<h2>Market tracker, {config.END_DATE}</h2>
<p>The latest readings, with each one's percentile against that asset's own history since its perp launched. OI is Bybit
open interest in USD. Volume ratios and on-chain flows are not included.</p>
{_table(snap, snap_cols)}

<h2>1. Funding and basis</h2>
<figure><img src="{figs['funding']}" alt="Funding history"></figure>
<h3>Funding by year (from <code>sql/queries/funding_by_year.sql</code>)</h3>
{_table(fyv)}
<h3>Is Binance representative? Daily funding vs Bybit (<code>funding_source_check.sql</code>)</h3>
{_table(r['source_check'].set_index('asset'))}

<h2>2. Cash-and-carry: long spot, short perp</h2>
<p>The trade holds spot and shorts the same number of coins in the perp. It earns funding whenever longs pay shorts, and
idle cash earns the T-bill rate. Leverage applies to the short leg only, so at <i>L</i>x the trade needs 1 + 1/<i>L</i> dollars of capital per
dollar of position. Each asset's rule was picked on training data: the best Sharpe ratio over T-bills among rules never liquidated in training.</p>
{_table(carry_view, carry_cols)}
<figure><img src="{figs['carry']}" alt="Carry equity curves"></figure>
<h3>Picked rules by calendar year (out-of-sample)</h3>
{_table(by_year_view)}
<h3>Leverage: the rise in the mark price that liquidates a short, against the largest rises seen</h3>
<p>The rise that liquidates a short with no top-up is (1/<i>L</i> − 0.5%) / 1.005. The largest rises use mark-price highs
(the price Binance liquidates on), measured from the previous close.</p>
{_table(lev_tbl, ["liquidating_rise", "max_1d_rise", "max_7d_rise", "days_1d_rise_exceeds", "weeks_7d_rise_exceeds"])}
<h3>Liquidations in the backtest, all rules at each leverage (margin topped up daily or weekly)</h3>
{_table(liq_tbl)}
<h3>Why the mark price matters: biggest last-trade wicks against a short</h3>
<p>Last-trade highs on the perp include flash wicks that the mark price never reached. On 13 March 2020 the ETH perp
printed a last-trade high 203% above the previous close, while its mark price rose 31%. A backtest using last-trade highs
would show liquidations that the exchange's own rules would not have triggered.</p>
{_table(r['wicks'].set_index('asset'), ["date", "last_trade_up", "mark_up", "spot_up"])}
<h3>Cost sensitivity: out-of-sample CAGR of the picked rules</h3>
{_table(cc)}

<h2>3. Hedging a spot BTC/ETH portfolio with perps</h2>
<p>The base portfolio holds BTC and ETH 50/50, rebalanced monthly. Each rule shorts perps against part of each position.
In each family, the parameters with the best training Calmar ratio were picked. The 100% static hedge is the carry trade itself, so it
appears in the full grid only.</p>
{_table(ht, hedge_cols)}
<figure><img src="{figs['hedge']}" alt="Hedged portfolio"></figure>
<details><summary>Every hedge rule, out-of-sample (diagnostic only, not used for selection)</summary>
{_table(r['hedge_test_grid'], hedge_cols)}</details>

<h2>4. Do funding and open interest predict anything?</h2>
<figure><img src="{figs['quintiles']}" alt="Funding quintiles"></figure>
<h3>Predictive regression: next 7-day return (training fit, Newey–West t-stats, out-of-sample R²)</h3>
<p>All predictors are standardised on training data. The t-statistics are from the training fit. An out-of-sample R² below zero
means the model forecast worse than simply using the training-period average return.</p>
{_table(regv)}
<h3>Leverage build-up days: 7-day OI rise in the top training decile, with price also up</h3>
{_table(bu, ["event_days", "p_drop_10pct_after_event", "p_drop_10pct_all_days", "fwd_ret_7d_after_event", "fwd_ret_7d_all_days"])}
<details><summary>Funding quintile detail</summary>{_table(qv)}</details>

<h2>5. Stress test: the worst BTC weeks</h2>
<p>The six worst 7-day BTC falls since September 2019, at least 45 days apart, with funding before and during each fall and
the change in open interest.</p>
{_table(ep)}
<h3>How the strategies did in those weeks</h3>
<p>"–" means the strategy had not started yet: the hedges need 200 days of history.</p>
{_table(epr)}

<h2>Method</h2>
<ul>
<li><b>Data.</b> Binance USDT-margined perpetuals and spot markets: daily candles, perp mark-price candles, and every funding
payment (BTC from September 2019, ETH from November 2019, SOL from September 2020). Daily open interest and a second funding
series come from Bybit. The cash benchmark is the 3-month US T-bill yield (FRED DTB3). Prices are in USDT, treated as USD.</li>
<li><b>SQL layer.</b> SQLite does the daily funding aggregation, the spot/perp/mark panel, the trailing signals (7- and 30-day funding,
OI change, volatility, moving averages), the forward outcomes, the yearly regimes, the venue cross-check and the market tracker.
A test checks the SQL signals against pandas.</li>
<li><b>Carry engine.</b> Each day it marks the short to market against posted margin and credits funding on the previous close's
notional. It checks for liquidation at the day's mark-price high, with 0.5% maintenance margin, and resizes both legs to target leverage
on the reset schedule. Entry and exit use the previous day's 7-day funding, so decisions are lagged by one day.</li>
<li><b>Hedge engine.</b> Hedge targets are decided at the previous close. The short earns or pays funding and its mark-to-market,
and it is re-sized when it drifts more than 10% from target. The hedge is assumed fully collateralised by the coins held.</li>
<li><b>Out-of-sample.</b> Training runs to {config.TRAIN_END} and testing from {config.TEST_START}. Rules and signal cut-offs were fixed on training data.
Costs are {config.COST_BPS} bps per leg per unit traded, tested from 0 to 30 bps.</li>
<li><b>Not modelled.</b> Exchange counterparty risk (FTX-style failures), USDT de-pegging, withdrawal limits, borrow and
transfer frictions, tiered maintenance margin for large positions, and regulatory limits on venue access (for example, which
venues a UAE-regulated firm may use). Results are for one venue and one decade.</li>
</ul>
<p class="caveat">This is historical research, not investment advice. Leverage, perpetual futures and exchange custody can
lose more than the capital at stake.</p>
<footer>Generated {date.today().isoformat()} by <code>make report</code>. Tables as CSV in <code>reports/tables/</code>.</footer>
"""
    css = """
:root { color-scheme: light; --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1000px; margin: 0 auto; padding: 40px 16px 64px; background: var(--surface); }
h1 { font-size: 30px; line-height: 1.2; margin: 4px 0 12px; text-wrap: balance; }
h2 { font-size: 20px; margin: 44px 0 10px; padding-top: 16px; border-top: 1px solid var(--grid); }
h3 { font-size: 15px; margin: 28px 0 8px; }
.eyebrow { color: var(--muted); font-size: 13px; margin: 0; }
.lede { color: var(--ink2); font-size: 16px; max-width: 72ch; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 12px; margin: 24px 0; }
.tile { border: 1px solid var(--grid); border-radius: 8px; padding: 14px 16px; }
.tile span, .tile small { display:block; color: var(--ink2); font-size: 13px; }
.tile b { font-size: 26px; font-weight: 600; }
.findings li { margin-bottom: 10px; max-width: 80ch; }
.scroll { overflow-x: auto; margin: 8px 0 16px; }
table { border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; width: 100%; }
th, td { padding: 6px 8px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
thead th { color: var(--ink2); font-weight: 600; vertical-align: bottom; white-space: normal; min-width: 60px; }
tbody th { text-align: left; font-weight: 500; }
details { margin: 12px 0; } summary { cursor: pointer; color: var(--ink2); }
figure { margin: 16px 0 24px; } img { max-width: 100%; height: auto; display: block; }
code { font-size: 12.5px; background: #f0efec; padding: 1px 4px; border-radius: 3px; }
.caveat { color: var(--ink2); font-size: 13.5px; border-left: 3px solid var(--grid); padding-left: 12px; }
footer { margin-top: 40px; color: var(--muted); font-size: 12.5px; }
"""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Perp Funding and Hedging</title><style>{css}</style></head>
<body><main>{body}</main></body></html>"""
    (config.REPORT_DIR / "report.html").write_text(page)


def main() -> None:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    r = run_all()
    save_tables(r)
    figs = {"funding": fig_funding(r), "carry": fig_carry(r), "hedge": fig_hedge(r), "quintiles": fig_quintiles(r)}
    write_html(r, figs)
    pd.set_option("display.width", 220)
    print(r["carry_test"][["asset", "rule", "cagr", "max_drawdown", "time_in_trade"]].round(3).to_string())
    print(r["hedge_test"][["cagr", "max_drawdown", "calmar"]].round(3).to_string())
    print(f"\nwrote {config.REPORT_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
