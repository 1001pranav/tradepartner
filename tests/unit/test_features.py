"""Price/market/calendar features, labels, the matrix builder and their no-lookahead guarantees."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from tests.fixtures.synth import bars_from_closes, bdays, synth_view
from tradepartner.backtest.view import MarketView
from tradepartner.calendar import session_close_utc
from tradepartner.db.models import NewsRaw, NewsScore, ResultDate
from tradepartner.features import build, labels, market
from tradepartner.features import news as nf
from tradepartner.features.price import PRICE_COLUMNS, price_features, rsi
from tradepartner.settings import LLMConfig, load_settings

DAYS = bdays(date(2023, 1, 2), 400)


def test_return_and_sma_features_match_hand_calculation() -> None:
    closes = [100.0 + i for i in range(300)]
    f = price_features(bars_from_closes(DAYS[:300], closes))
    assert list(f.columns) == PRICE_COLUMNS
    assert f["ret_1"].iloc[10] == pytest.approx(110 / 109 - 1)
    assert f["ret_20"].iloc[50] == pytest.approx(150 / 130 - 1)
    sma20 = np.mean(closes[31:51])
    assert f["dist_sma20"].iloc[50] == pytest.approx(150 / sma20 - 1)
    assert f["dist_52w_high"].iloc[299] == 0.0 and f["dist_52w_low"].iloc[299] > 0
    assert f["ret_60"].iloc[:60].isna().all() and f["dist_sma200"].iloc[:199].isna().all()
    assert f["gap_pct"].iloc[5] == pytest.approx(
        0.0, abs=1e-9
    )  # open == previous close in the fixture


def test_rsi_extremes_and_close_location() -> None:
    up = pd.Series(np.arange(1.0, 60.0))
    assert rsi(up).iloc[-1] == 100.0
    down = pd.Series(np.arange(60.0, 1.0, -1.0))
    assert rsi(down).iloc[-1] == pytest.approx(0.0, abs=1e-6)
    df = bars_from_closes(DAYS[:30], [100.0] * 30)
    df["high"] = df["low"] = df["close"]  # zero range -> neutral 0.5
    assert (price_features(df)["close_loc"] == 0.5).all()


def test_price_features_at_t_never_depend_on_later_bars() -> None:
    rng = np.random.default_rng(1)
    closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))
    full = price_features(bars_from_closes(DAYS, closes))
    for t in (120, 250, 399):
        part = price_features(bars_from_closes(DAYS[: t + 1], closes[: t + 1]))
        pd.testing.assert_series_equal(full.iloc[t], part.iloc[t], check_names=False)


# ---------- matrix ----------


def test_matrix_shape_columns_and_cross_sectional_rank() -> None:
    view = synth_view(n_symbols=6, n_days=320)
    m = build.build_matrix(view)
    assert list(m.columns) == build.ALL_FEATURE_COLUMNS and m.index.names == ["date", "symbol"]
    last = m.xs(view.trading_days()[-1], level="date")
    assert set(last.index) == set(view.symbols)
    assert last["turnover_rank"].between(0, 1).all()
    assert m[build.NEWS_FEATURE_COLUMNS].isna().all().all()  # price-only by default
    assert last["breadth_50"].between(0, 1).all() and last["nifty_dist_sma200"].notna().all()
    assert (
        "dow" not in m.columns and "days_to_month_end" not in m.columns
    )  # fs2: no calendar timing
    assert build.build_matrix(MarketView({})).empty


def test_matrix_rows_up_to_t_are_identical_when_the_future_is_removed() -> None:
    view = synth_view(n_symbols=6, n_days=320, seed=9)
    days = view.trading_days()
    cut = days[260]
    trunc = MarketView(
        {s: view.history(s, cut) for s in view.symbols},
        view.index[view.index["date"] <= cut], view.sectors,
    )  # fmt: skip
    a, b = build.build_matrix(view), build.build_matrix(trunc)
    common = b.index
    pd.testing.assert_frame_equal(a.loc[common], b, check_exact=False, rtol=1e-9, atol=1e-12)


def test_market_calendar_and_relative_helpers() -> None:
    dates = pd.Index(DAYS[:5])
    cal = market.calendar_features(dates, {"A": [DAYS[3], DAYS[1]]}, ["A", "B"])
    assert cal["days_to_results"]["A"].tolist()[0] == (DAYS[1] - DAYS[0]).days
    assert (
        np.isnan(cal["days_to_results"]["A"].iloc[4]) and cal["days_to_results"]["B"].isna().all()
    )
    assert (
        cal["dow"]["A"].iloc[0] == DAYS[0].weekday() and cal["days_to_month_end"]["A"].iloc[0] == 29
    )
    mk = market.market_features(
        None, None, pd.DataFrame({"A": [1.0, -1.0], "B": [1.0, 1.0]}, index=[0, 1])
    )
    assert mk["breadth_50"].tolist() == [1.0, 0.5] and mk["nifty_dist_sma200"].isna().all()
    vix = pd.DataFrame({"date": DAYS[:10], "close": np.arange(10.0, 20.0)})
    mk2 = market.market_features(None, vix, pd.DataFrame({"A": range(10)}, index=DAYS[:10]))
    assert mk2["vix_chg_5d"].iloc[5] == pytest.approx(15 / 10 - 1)
    rets = {n: pd.DataFrame({"A": [0.1], "B": [0.3]}, index=[DAYS[0]]) for n in (5, 20, 60)}
    rel = market.relative_features(rets, None, {"A": "X", "B": "X"})
    assert (
        rel["sector_rel_ret_20"]["A"].iloc[0] == pytest.approx(-0.1)
        and rel["rel_ret_5"].isna().all().all()
    )
    assert rel["sector_mom_rank"]["B"].iloc[0] == 1.0


def test_save_matrix(tmp_path) -> None:  # type: ignore[no-untyped-def]
    m = build.build_matrix(synth_view(n_symbols=3, n_days=260))
    path = build.save_matrix(m, tmp_path)
    assert path.name == f"features_{build.FEATURE_SET_VERSION}.parquet"
    assert len(pd.read_parquet(path)) == len(m)


# ---------- labels ----------


def _flat_with_jump() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    n = 30
    s = bars_from_closes(DAYS[:n], [100.0] * 10 + [110.0] * 20)
    s.loc[10, "open"] = 110.0  # the jump happens at the open of bar 10
    idx = bars_from_closes(DAYS[:n], [1000.0] * n)
    return {"A": s}, idx


def test_labels_start_at_the_next_open_not_the_signal_close() -> None:
    stock, idx = _flat_with_jump()
    lab = labels.swing_labels(stock, idx, horizon=5, threshold=0.02)
    row = lab.xs("A", level="symbol")
    # signal at bar 9's close: enters at bar 10's open (110) and exits at bar 14's close (110) -> 0%
    assert row["fwd_ret"].iloc[9] == pytest.approx(0.0)
    # signal at bar 8: enters at bar 9's open (100) -> exits bar 13's close (110): +10%
    assert row["fwd_ret"].iloc[8] == pytest.approx(0.10) and row["y"].iloc[8] == 1.0
    assert row["y"].iloc[9] == 0.0  # the move already happened before the next open
    assert (
        row["fwd_ret"].iloc[-5:].isna().all() and row["y"].iloc[-5:].isna().all()
    )  # unknown future


def test_label_threshold_and_excess_over_the_index() -> None:
    stock, idx = _flat_with_jump()
    idx["close"] = [1000.0 * 1.01**i for i in range(30)]
    idx["open"] = idx["close"].shift(1).fillna(1000.0)
    lab = labels.swing_labels(stock, idx, 5, 0.02).xs("A", level="symbol")
    assert lab["excess"].iloc[8] < lab["fwd_ret"].iloc[8]
    assert lab["y"].iloc[8] == (1.0 if lab["excess"].iloc[8] > 0.02 else 0.0)


def test_positional_labels_are_cross_sectional_ranks() -> None:
    view = synth_view(n_symbols=6, n_days=200)
    stock = {s: view.history(s, view.trading_days()[-1]) for s in view.symbols}
    lab = labels.positional_labels(stock, view.index, 60)
    day = lab.xs(view.trading_days()[50], level="date")
    assert day["y"].between(0, 1).all() and day["y"].max() == 1.0 and day["y"].nunique() == 6
    assert lab.xs(view.trading_days()[-1], level="date")["y"].isna().all()


def test_round_trip_cost_comes_from_config() -> None:
    costs = load_settings().costs
    delivery = labels.round_trip_cost_fraction(costs)
    assert 0.002 < delivery < 0.01  # ~0.2% charges + 0.2% slippage
    assert labels.round_trip_cost_fraction(costs, product="MIS") != delivery
    zero = costs.model_copy(
        update={"slippage": costs.slippage.model_copy(update={"default_bps": 0})}
    )
    assert labels.round_trip_cost_fraction(zero) < delivery


# ---------- news panel equals the per-date function ----------

LLM = LLMConfig(
    model="m", knowledge_cutoff="2020-01-01", cutoff_buffer_days=0, num_ctx=4096, keep_alive="5m"
)


def test_news_panel_matches_news_features_and_respects_the_cutoff(db: Session) -> None:
    base = datetime(2026, 3, 2, 5, 0, tzinfo=UTC)
    for i in range(12):
        n = NewsRaw(source="t", url=f"u{i}", title="t", published_at=base + timedelta(days=i // 2, hours=i),
                    fetched_at=base + timedelta(days=30), content_hash=f"h{i}")  # fmt: skip
        db.add(n)
        db.flush()
        db.add(NewsScore(news_id=n.id, symbol="AAA" if i % 2 else "BBB", llm_model="m", prompt_version="v1",
                         relevance=0.9, sentiment=(-1) ** i * 0.3, event_type="earnings", materiality=0.7,
                         horizon="days", confidence=0.8, summary="s", raw_json={}, scored_at=base))  # fmt: skip
    db.add(ResultDate(symbol="AAA", result_date=date(2026, 3, 9), source="t", created_at=base))
    db.commit()
    days = [date(2026, 3, d) for d in (3, 4, 5, 6, 9)]
    panel = nf.news_panel(db, ["AAA", "BBB"], days, LLM)
    for d in days:
        for sym in ("AAA", "BBB"):
            one = nf.news_features(db, [sym], session_close_utc(d), LLM, mode="backtest").loc[sym]
            pd.testing.assert_series_equal(panel.loc[(d, sym)], one, check_names=False)
    cut = LLMConfig(
        model="m",
        knowledge_cutoff="2026-03-04",
        cutoff_buffer_days=0,
        num_ctx=4096,
        keep_alive="5m",
    )
    blocked = nf.news_panel(db, ["AAA"], days, cut)
    assert blocked.loc[(days[0], "AAA"), "llm_features_valid"] == 0
    assert blocked.loc[(days[0], "AAA")].drop("llm_features_valid").isna().all()
    assert blocked.loc[(days[-1], "AAA"), "llm_features_valid"] == 1
    assert nf.news_panel(db, ["AAA"], [], LLM).empty
