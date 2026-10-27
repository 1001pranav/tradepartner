"""Out-of-sample evaluation and the promotion gate (spec 10.5).

The gate is evaluated automatically; **approval is never automatic** (check 7 is always pending
until Pranav clicks Approve). Strategy returns here come from a light vectorised simulation of the
walk-forward predictions (top-K per day, one position per symbol at a time, forward return from the
next open, round-trip costs and slippage deducted). The full event-driven backtester remains the
final word and uses the same risk rules.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

TRADING_DAYS = 252
GATE_SHARPE_MARGIN = 0.3
GATE_DD_MULTIPLE = 1.2
GATE_MIN_TRADES = 100
GATE_MIN_POSITIVE_WINDOWS = 0.60
GATE_MAX_FEATURE_SHARE = 0.40


@dataclass
class Sim:
    trades: pd.DataFrame  # date, symbol, net, window
    daily: pd.Series  # portfolio daily return indexed by date
    # Returns per NON-OVERLAPPING block of ``horizon`` days, used for Sharpe/drawdown. Spreading a
    # multi-week trade evenly over its days (``daily``) smooths away volatility and inflates Sharpe.
    period: pd.Series | None = None
    periods_per_year: float = TRADING_DAYS


def simulate(
    oos: pd.DataFrame,
    score: pd.Series,
    horizon: int,
    top_k: int,
    cost_rt: float,
    floor: float | None = None,
) -> Sim:
    """Hold at most ``top_k`` positions at once. Each day, free slots go to the best-scoring
    symbols (above ``floor``) that are not already held; a position lasts ``horizon`` days."""
    df = oos.assign(score=score.reindex(oos.index)).dropna(subset=["fwd_ret", "score"])
    dates = sorted(df.index.get_level_values("date").unique())
    pos = {d: i for i, d in enumerate(dates)}
    daily = np.zeros(len(dates))
    held_until: dict[str, int] = {}  # symbol -> first day index it is free again
    rows: list[dict[str, Any]] = []
    for d in dates:
        i = pos[d]
        open_now = sum(1 for free in held_until.values() if free > i)
        slots = top_k - open_now
        if slots <= 0:
            continue
        day = cast(pd.DataFrame, df.xs(d, level="date"))
        day = day[day["score"] >= floor] if floor is not None else day
        for sym, r in day.sort_values(by="score", ascending=False).iterrows():
            if slots <= 0:
                break
            if held_until.get(str(sym), -1) > i:
                continue
            net = float(r["fwd_ret"]) - cost_rt
            held_until[str(sym)] = i + 1 + horizon
            slots -= 1
            rows.append({"date": d, "symbol": sym, "net": net, "window": int(r["window"])})
            for k in range(i + 1, min(i + 1 + horizon, len(dates))):
                daily[k] += net / horizon / top_k
    trades = pd.DataFrame(rows, columns=["date", "symbol", "net", "window"])
    block = {d: pos[d] // horizon for d in dates}
    n_blocks = (len(dates) + horizon - 1) // horizon
    per = pd.Series(0.0, index=range(n_blocks))
    for trade in rows:
        per[block[trade["date"]]] += trade["net"] / top_k
    return Sim(trades, pd.Series(daily, index=dates), per, TRADING_DAYS / horizon)


def sharpe(daily: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    sd = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    return float(daily.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")


def max_drawdown(daily: pd.Series) -> float:
    if daily.empty:
        return 0.0
    eq = (1 + daily).cumprod()
    return float((eq / eq.cummax() - 1).min())


def positive_window_fraction(sim: Sim, windows: list[int]) -> float:
    if not windows:
        return float("nan")
    wins = 0
    for w in windows:
        wins += float(sim.trades.loc[sim.trades["window"] == w, "net"].sum()) > 0
    return wins / len(windows)


@dataclass
class GateCheck:
    name: str
    passed: bool
    value: float | None
    threshold: str
    detail: str = ""


@dataclass
class GateReport:
    checks: list[GateCheck]
    ready_for_review: bool  # checks 1-6 pass; check 7 is Pranav
    approved: bool = False  # always False here: the system never self-approves
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed and c.name != "human_approval"]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "failed": self.failed}


def _series(sim: Sim) -> pd.Series:
    return sim.period if sim.period is not None else sim.daily


def assemble_gate(
    model: Sim, base: Sim, harsh: Sim, windows: list[int], importance: dict[str, float]
) -> GateReport:
    """The seven checks of spec 10.5 from three simulations (model, baseline, doubled slippage)."""
    sm = sharpe(_series(model), model.periods_per_year)
    sb = sharpe(_series(base), base.periods_per_year)
    dm, db = max_drawdown(_series(model)), max_drawdown(_series(base))
    pos_frac = positive_window_fraction(model, windows)
    harsh_total = float(harsh.trades["net"].sum()) if len(harsh.trades) else float("-inf")
    top_feature = max(importance.items(), key=lambda kv: kv[1]) if importance else ("-", 0.0)
    checks = [
        GateCheck(
            "sharpe_vs_baseline",
            bool(sm == sm and sb == sb and sm >= sb + GATE_SHARPE_MARGIN),
            None if sm != sm else sm,
            f">= baseline {sb:.2f} + {GATE_SHARPE_MARGIN}",
        ),
        GateCheck(
            "max_drawdown",
            abs(dm) <= GATE_DD_MULTIPLE * abs(db),
            abs(dm),
            f"<= {GATE_DD_MULTIPLE} x baseline {abs(db):.3f}",
        ),
        GateCheck(
            "min_trades",
            len(model.trades) >= GATE_MIN_TRADES,
            float(len(model.trades)),
            f">= {GATE_MIN_TRADES}",
        ),
        GateCheck(
            "positive_windows",
            bool(pos_frac == pos_frac and pos_frac >= GATE_MIN_POSITIVE_WINDOWS),
            None if pos_frac != pos_frac else pos_frac,
            f">= {GATE_MIN_POSITIVE_WINDOWS:.0%} of windows",
        ),
        GateCheck(
            "double_slippage_positive",
            harsh_total > 0,
            None if harsh_total == float("-inf") else harsh_total,
            "> 0 with slippage doubled",
        ),
        GateCheck(
            "feature_concentration",
            top_feature[1] <= GATE_MAX_FEATURE_SHARE,
            top_feature[1],
            f"<= {GATE_MAX_FEATURE_SHARE:.0%} gain share",
            f"top feature: {top_feature[0]}",
        ),
    ]
    ready = all(c.passed for c in checks)
    checks.append(
        GateCheck("human_approval", False, None, "Pranav clicks Approve", "never automatic")
    )
    return GateReport(
        checks,
        ready,
        False,
        {
            "sharpe": sm,
            "baseline_sharpe": sb,
            "max_drawdown": dm,
            "baseline_max_drawdown": db,
            "trades": len(model.trades),
            "windows": len(windows),
            "positive_window_fraction": pos_frac,
            "net_return_total": float(model.trades["net"].sum()) if len(model.trades) else 0.0,
            "double_slippage_total": harsh_total,
        },
    )


def evaluate_gate(
    oos: pd.DataFrame,
    baseline_score: pd.Series,
    importance: dict[str, float],
    *,
    horizon: int,
    cost_rt: float,
    slippage_bps: float,
    top_k: int = 5,
    floor: float | None = None,
) -> GateReport:
    """Run the seven checks on daily-horizon predictions. ``oos`` has ``pred``, ``fwd_ret``,
    ``window``; scores for the baseline are aligned to the same rows."""
    model = simulate(oos, oos["pred"], horizon, top_k, cost_rt, floor)
    base = simulate(oos, baseline_score, horizon, top_k, cost_rt, None)
    harsh = simulate(oos, oos["pred"], horizon, top_k, cost_rt + 2 * slippage_bps / 10_000, floor)
    windows = sorted(int(w) for w in oos["window"].unique())
    return assemble_gate(model, base, harsh, windows, importance)


def simulate_intraday(
    oos: pd.DataFrame,
    score: pd.Series,
    hold_bars: int,
    top_k: int,
    cost_rt: float,
    floor: float | None = None,
) -> Sim:
    """Intraday version of ``simulate``: bar-by-bar, at most ``top_k`` positions, each held
    ``hold_bars`` bars within its day. The daily series sums that day's trades over ``top_k``."""
    df = oos.assign(score=score.reindex(oos.index)).dropna(subset=["fwd_ret", "score"])
    df = df.reset_index()
    stamps = sorted(df["ts"].unique())
    pos = {t: i for i, t in enumerate(stamps)}
    held_until: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for t, grp in df.groupby("ts", sort=True):
        i = pos[t]
        slots = top_k - sum(1 for free in held_until.values() if free > i)
        if slots <= 0:
            continue
        g = grp[grp["score"] >= floor] if floor is not None else grp
        for _, r in g.sort_values(by="score", ascending=False).iterrows():
            if slots <= 0:
                break
            sym = str(r["symbol"])
            if held_until.get(sym, -1) > i:
                continue
            held_until[sym] = i + 1 + hold_bars
            slots -= 1
            rows.append(
                {
                    "date": r["date"],
                    "symbol": sym,
                    "net": float(r["fwd_ret"]) - cost_rt,
                    "window": int(r["window"]),
                }
            )
    trades = pd.DataFrame(rows, columns=["date", "symbol", "net", "window"])
    days = sorted(df["date"].unique())
    daily = (
        trades.groupby("date")["net"].sum().div(top_k).reindex(days, fill_value=0.0)
        if len(trades)
        else pd.Series(0.0, index=days)
    )
    return Sim(trades, daily)


def evaluate_intraday_gate(
    oos: pd.DataFrame,
    baseline_score: pd.Series,
    importance: dict[str, float],
    *,
    hold_bars: int,
    cost_rt: float,
    slippage_bps: float,
    top_k: int = 3,
    floor: float | None = None,
) -> GateReport:
    model = simulate_intraday(oos, oos["pred"], hold_bars, top_k, cost_rt, floor)
    base = simulate_intraday(oos, baseline_score, hold_bars, top_k, cost_rt, None)
    harsh = simulate_intraday(
        oos, oos["pred"], hold_bars, top_k, cost_rt + 2 * slippage_bps / 10_000, floor
    )
    windows = sorted(int(w) for w in oos["window"].unique())
    return assemble_gate(model, base, harsh, windows, importance)


def prediction_quality(oos: pd.DataFrame, binary: bool) -> dict[str, float]:
    """AUC and Brier for binary targets; rank correlation for regression targets."""
    d = oos.dropna(subset=["y", "pred"])
    if d.empty:
        return {}
    if binary and d["y"].nunique() == 2:
        from sklearn.metrics import brier_score_loss, roc_auc_score

        return {
            "auc": float(roc_auc_score(d["y"], d["pred"])),
            "brier": float(brier_score_loss(d["y"], d["pred"])),
            "base_rate": float(d["y"].mean()),
        }
    return {"spearman": float(d["y"].corr(d["pred"], method="spearman"))}
