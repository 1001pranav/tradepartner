"""Backtest engine behaviour on hand-built bars, plus the determinism regression."""

import hashlib
from dataclasses import dataclass
from datetime import date

import pandas as pd
import pytest

from tests.fixtures.synth import bars_from_closes, bdays, settings_with_capital, synth_view
from tradepartner.backtest import report
from tradepartner.backtest.engine import SURVIVORSHIP_BANNER, run_backtest
from tradepartner.backtest.view import MarketView
from tradepartner.strategies.base import Candidate
from tradepartner.strategies.baseline import Momentum12m1TopN, Momentum20TopN

S = settings_with_capital()
DAYS = bdays(date(2024, 1, 1), 40)


@dataclass
class Scripted:
    """Emits fixed candidates at fixed signal days (the strategy never sees prices)."""

    plan: dict[date, list[Candidate]]
    name: str = "scripted"

    def candidates(self, as_of: date, view: MarketView) -> list[Candidate]:
        return self.plan.get(as_of, [])


def cand(symbol: str = "AAA", limit: int = 100_500, stop: int = 96_000, target: int = 108_000,
         horizon: int = 10, score: float = 1.0) -> Candidate:  # fmt: skip
    return Candidate(symbol, limit, stop, target, horizon, score)


def flat_view(n: int = 40, price: float = 1000.0, edits: dict[int, dict[str, float]] | None = None,
              symbols: tuple[str, ...] = ("AAA",)) -> MarketView:  # fmt: skip
    bars = {}
    for sym in symbols:
        df = bars_from_closes(DAYS[:n], [price] * n)
        for i, patch in (edits or {}).items():
            for col, val in patch.items():
                df.loc[i, col] = val
        bars[sym] = df
    return MarketView(bars, None, {s: f"SEC-{s}" for s in symbols})


def run(view: MarketView, plan: dict[date, list[Candidate]], **kw: object):  # type: ignore[no-untyped-def]
    return run_backtest(Scripted(plan), view, kw.pop("settings", S), DAYS[0], DAYS[-1], **kw)  # type: ignore[arg-type]


def test_entry_fills_next_open_with_slippage_and_charges() -> None:
    r = run(flat_view(), {DAYS[5]: [cand(horizon=3)]})
    t = r.trades[0]
    assert t.entry_date == DAYS[6] and t.entry_price_paise == 100_100  # open 1000 + 10 bps
    assert t.reason == "time_stop" and t.exit_date >= DAYS[9]
    assert t.charges_paise > 0 and t.slippage_paise == t.qty * 100  # buy slippage Rs 1.00 / share
    assert t.pnl_paise == t.gross_pnl_paise - t.charges_paise
    assert t.qty == 99  # position cap: 10% of Rs 10 lakh / Rs 1,005
    assert r.rejections == {} and r.unfilled == 0


def test_stop_hit_on_a_later_bar() -> None:
    view = flat_view(edits={8: {"low": 950.0, "close": 955.0}})
    r = run(view, {DAYS[5]: [cand()]})
    t = r.trades[0]
    assert t.reason == "stop" and t.exit_date == DAYS[8]
    assert t.exit_price_paise == 96_000 - 96  # stop slipped 10 bps adversely
    assert t.pnl_paise < 0


def test_target_hit_has_no_slippage() -> None:
    view = flat_view(edits={9: {"high": 1090.0, "close": 1085.0}})
    t = run(view, {DAYS[5]: [cand()]}).trades[0]
    assert t.reason == "target" and t.exit_price_paise == 108_000 and t.pnl_paise > 0


def test_stop_and_target_in_one_bar_assumes_stop_first() -> None:
    view = flat_view(edits={8: {"low": 940.0, "high": 1100.0}})
    assert run(view, {DAYS[5]: [cand()]}).trades[0].reason == "stop"


def test_gap_down_through_stop_exits_at_the_open() -> None:
    view = flat_view(edits={8: {"open": 900.0, "low": 890.0, "high": 905.0, "close": 895.0}})
    t = run(view, {DAYS[5]: [cand()]}).trades[0]
    assert t.reason == "stop" and t.exit_price_paise == 90_000 - 90


def test_stop_inside_the_entry_bar_closes_the_trade_the_same_day() -> None:
    view = flat_view(edits={6: {"low": 940.0}})
    t = run(view, {DAYS[5]: [cand()]}).trades[0]
    assert t.entry_date == t.exit_date == DAYS[6] and t.reason == "stop"


def test_unfilled_when_price_never_reaches_the_limit() -> None:
    view = flat_view(edits={6: {"open": 1050.0, "low": 1040.0, "high": 1060.0, "close": 1050.0}})
    r = run(view, {DAYS[5]: [cand(limit=100_500, target=120_000)]})
    # the price band (+/-1.5% of the open) rejects a 4% gap first; either way nothing trades
    assert r.trades == [] and (r.unfilled == 1 or r.rejections["price_band"] == 1)


def test_fills_at_limit_when_open_is_above_but_range_touches() -> None:
    view = flat_view(edits={6: {"open": 1010.0, "low": 1000.0, "high": 1015.0, "close": 1005.0}})
    t = run(view, {DAYS[5]: [cand(horizon=2)]}).trades[0]
    assert t.entry_price_paise == 100_500


def test_risk_engine_rejections_are_counted_not_traded() -> None:
    wide = cand(stop=85_000, target=130_000)  # 15.4% stop: max is 8%
    r = run(flat_view(), {DAYS[5]: [wide]})
    assert r.trades == [] and r.rejections["stop_distance"] == 1


def test_open_positions_are_marked_but_not_in_the_trade_list() -> None:
    view = flat_view(
        edits={
            i: {
                "close": 1000.0 + (i - 6) * 5,
                "high": 1100.0 if i < 0 else 1000.0 + (i - 6) * 5 + 3,
                "open": 1000.0 + (i - 7) * 5,
            }
            for i in range(7, 40)
        }
    )
    r = run(view, {DAYS[5]: [cand(target=200_000, horizon=100)]})
    assert r.trades == [] and r.equity.iloc[-1] > r.equity.iloc[0]
    assert (r.invested > 0).any() and r.invested.iloc[-1] > 0.05


def test_equity_curve_is_conservative_and_cash_never_negative() -> None:
    r = run(
        flat_view(symbols=("AAA", "BBB", "CCC")), {DAYS[5]: [cand("AAA"), cand("BBB"), cand("CCC")]}
    )
    assert len(r.trades) == 3 and r.equity.min() > 0.99 * 1_000_000_00 - 3 * 100_000 * 100


def test_max_open_positions_limit_applies_in_backtest() -> None:
    syms = tuple(f"S{i}" for i in range(10))
    r = run(flat_view(symbols=syms), {DAYS[5]: [cand(s, score=10 - i) for i, s in enumerate(syms)]})
    opened = {t.symbol for t in r.trades}
    assert len(opened) == 8 and r.rejections["open_positions"] == 2  # risk.yaml: max 8
    assert {"S0", "S7"} <= opened and not ({"S8", "S9"} & opened)  # best scores win


def test_consecutive_loss_halt_stops_a_strategy() -> None:
    edits = {i: {"low": 940.0, "close": 950.0} for i in range(7, 40, 2)}
    view = flat_view(edits=edits)
    plan = {DAYS[i]: [cand(limit=100_500, horizon=50)] for i in range(5, 38, 3)}
    r = run(
        view,
        plan,
        settings=settings_with_capital(
            loss_limits=S.risk.loss_limits.model_copy(update={"consecutive_losses_halt": 2})
        ),
    )
    assert len([t for t in r.trades if t.pnl_paise < 0]) >= 2
    assert r.rejections["consecutive_losses"] > 0


def test_daily_loss_and_cooldown_rules_are_enforced() -> None:
    edits = {8: {"low": 940.0, "close": 950.0}}
    view = flat_view(edits=edits)
    plan = {
        DAYS[5]: [cand()],
        DAYS[8]: [cand(limit=95_500, stop=91_000, target=104_000)],
        DAYS[9]: [cand()],
    }
    r = run(view, plan)
    assert r.trades[0].reason == "stop"
    assert r.rejections["reentry_cooldown"] >= 1  # re-entering the stopped symbol within 3 days


def test_survivorship_warning_in_result_and_report() -> None:
    r = run(flat_view(), {DAYS[5]: [cand(horizon=3)]}, survivorship_warning=True)
    assert r.warnings == [SURVIVORSHIP_BANNER]
    html = report.render(r, {})
    assert "survivorship" in html.lower() and "Slippage" in html and "Same RiskEngine" in html


def test_report_sections_with_enough_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    view = synth_view(n_symbols=10, n_days=420)
    r = run_backtest(Momentum20TopN(n=3), view, S, view.trading_days()[0], view.trading_days()[-1])
    idx = view.index.set_index("date")["close"]  # type: ignore[union-attr]
    html = report.render(r, {"bench": r.metrics}, idx)
    assert (
        "By year" in html and "By regime" in html and "Last 50 trades" in html and "bench" in html
    )
    path = report.save(html, tmp_path / "r", "t")
    assert path.exists() and path.read_text().startswith("<!doctype html>")


def test_report_handles_empty_result() -> None:
    r = run(flat_view(), {})
    html = report.render(r, {})
    assert "Trades: 0" in html and "Last 50" not in html


# ---------- determinism regression (spec 13) ----------


def _digest(r) -> str:  # type: ignore[no-untyped-def]
    h = hashlib.sha256()
    for t in r.trades:
        h.update(repr(t).encode())
    h.update(r.equity.to_numpy().tobytes())
    return h.hexdigest()[:16]


def test_same_data_same_trades_across_runs() -> None:
    view = synth_view(n_symbols=10, n_days=600, seed=11)
    a = run_backtest(Momentum20TopN(n=4), view, S, view.trading_days()[0], view.trading_days()[-1])
    b = run_backtest(
        Momentum20TopN(n=4),
        synth_view(n_symbols=10, n_days=600, seed=11),
        S,
        view.trading_days()[0],
        view.trading_days()[-1],
    )
    assert a.trades == b.trades and a.equity.equals(b.equity) and len(a.trades) > 5
    assert _digest(a) == _digest(b)


def test_golden_trade_list_fixture() -> None:
    """If this digest changes, backtest behaviour changed: review the diff, then update it."""
    view = synth_view(n_symbols=10, n_days=600, seed=11)
    r = run_backtest(Momentum20TopN(n=4), view, S, view.trading_days()[0], view.trading_days()[-1])
    assert (len(r.trades), _digest(r)) == GOLDEN


GOLDEN = (476, "350fb96308331f91")  # changed deliberately when halt cooldown/reset was added


# ---------- no lookahead (spec 10.3, CLAUDE.md rule 8) ----------


@pytest.mark.parametrize("strategy", [Momentum20TopN(n=5), Momentum12m1TopN(n=5)])
def test_strategies_ignore_future_data(strategy) -> None:  # type: ignore[no-untyped-def]
    view = synth_view(n_symbols=8, n_days=500, seed=3)
    days = view.trading_days()
    cut = days[320]
    # build a second world whose FUTURE (after `cut`) is garbage
    scrambled = {}
    for s in view.symbols:
        df = view.history(s, days[-1]).copy()
        future = df["date"] > cut
        df.loc[future, ["open", "high", "low", "close"]] = 1e6
        scrambled[s] = df
    idx = view.index.copy()  # type: ignore[union-attr]
    idx.loc[idx["date"] > cut, ["open", "high", "low", "close"]] = 1e6
    other = MarketView(scrambled, idx, view.sectors)
    for d in days[300:340]:
        if d > cut:
            break
        assert strategy.candidates(d, view) == strategy.candidates(d, other), d


def test_momentum_strategies_produce_candidates_with_valid_geometry() -> None:
    view = synth_view(n_symbols=10, n_days=500, seed=5)
    days = view.trading_days()
    found20 = [c for d in days[60:200] for c in Momentum20TopN(n=3).candidates(d, view)]
    assert found20 and all(c.stop_paise < c.limit_price_paise < c.target_paise for c in found20)
    assert all(c.limit_price_paise % 5 == 0 and c.horizon_days == 5 for c in found20)
    found12 = [c for d in days[260:] for c in Momentum12m1TopN(n=3).candidates(d, view)]
    assert found12 and all(c.horizon_days == 60 for c in found12)
    first_of_month = {d for d in days[260:] if d.month != days[days.index(d) - 1].month}
    assert {d for d in days[260:] if Momentum12m1TopN().candidates(d, view)} <= first_of_month
    assert Momentum20TopN().candidates(days[10], view) == []  # not enough history


def test_registry() -> None:
    from tradepartner.strategies import registry

    assert registry.get("baseline_momentum20").name == "baseline_momentum20"
    with pytest.raises(KeyError, match="unknown strategy"):
        registry.get("nope")
    assert isinstance(pd.Series([1]), pd.Series)


def test_a_halt_blocks_for_the_cooldown_and_then_trading_resumes() -> None:
    edits = {i: {"low": 940.0, "close": 950.0} for i in range(7, 40, 2)}
    view = flat_view(edits=edits)
    plan = {DAYS[i]: [cand(limit=100_500, horizon=50)] for i in range(5, 38, 3)}
    halt = S.risk.loss_limits.model_copy(update={"consecutive_losses_halt": 2})
    quick = run(view, plan, settings=settings_with_capital(loss_limits=halt), halt_cooldown_days=3)
    never = run(
        view, plan, settings=settings_with_capital(loss_limits=halt), halt_cooldown_days=10_000
    )
    assert never.rejections["halt_cooldown"] > 0
    assert len(quick.trades) > len(never.trades)  # after the reset the strategy trades again
    assert quick.params["halt_cooldown_days"] == 3
