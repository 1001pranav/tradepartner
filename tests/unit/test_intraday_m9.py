"""M9 building blocks: bars from ticks, causal 5-minute features and labels, the ORB scanner."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.intraday import BARS_PER_DAY, breakout_day, session_bar_ends, synth_bars
from tradepartner.calendar import IST
from tradepartner.data import candles
from tradepartner.data.live import Bar, BarBuilder, Tick, bars_to_frame, bucket_end
from tradepartner.features.intraday import (
    INTRADAY_COLUMNS,
    OR_BARS,
    intraday_features,
    intraday_labels,
)
from tradepartner.strategies.intraday import IntradayScanner

DAY = date(2025, 6, 2)


def ist(h: int, m: int, s: int = 0, day: date = DAY) -> datetime:
    return datetime(day.year, day.month, day.day, h, m, s, tzinfo=IST)


# ---------------- ticks -> bars ----------------


def test_bucket_boundaries_and_session_limits() -> None:
    assert bucket_end(ist(9, 15)) == ist(9, 20).astimezone(UTC)
    assert bucket_end(ist(9, 19, 59)) == ist(9, 20).astimezone(UTC)
    assert bucket_end(ist(9, 20)) == ist(9, 25).astimezone(UTC)
    assert bucket_end(ist(15, 29, 59)) == ist(15, 30).astimezone(UTC)
    assert bucket_end(ist(9, 14, 59)) is None and bucket_end(ist(15, 30)) is None
    assert bucket_end(ist(10, 0), interval_min=15) == ist(10, 15).astimezone(UTC)


def test_bar_builder_ohlc_and_volume_deltas() -> None:
    bb = BarBuilder()
    assert bb.on_tick(Tick("X", 10000, 1000, ist(9, 15, 5))) == []
    assert bb.on_tick(Tick("X", 10050, 1400, ist(9, 16))) == []
    assert bb.on_tick(Tick("X", 9990, 1900, ist(9, 18))) == []
    assert bb.on_tick(Tick("X", 10010, 2000, ist(9, 19, 50))) == []
    done = bb.on_tick(Tick("X", 10100, 2300, ist(9, 20, 1)))  # the next bucket closes the first bar
    assert len(done) == 1
    b = done[0]
    assert (b.open, b.high, b.low, b.close) == (10000, 10050, 9990, 10010) and b.end == ist(
        9, 20
    ).astimezone(UTC)
    assert b.volume == 1000 and b.day == DAY  # 2000 - 1000 cumulative at the first tick
    later = bb.flush(ist(9, 25, 1).astimezone(UTC))
    assert (
        len(later) == 1 and later[0].volume == 300 and later[0].open == 10100
    )  # delta continues from the last bar
    assert bb.flush(ist(9, 40).astimezone(UTC)) == []
    assert (
        bb.on_tick(Tick("X", 1, 1, ist(8, 0))) == []
        and bb.on_tick(Tick("X", 1, 1, ist(16, 0))) == []
    )


def test_bar_builder_keeps_symbols_apart_and_never_goes_negative_on_volume_resets() -> None:
    bb = BarBuilder()
    for sym, px in (("A", 100), ("B", 200)):
        bb.on_tick(Tick(sym, px, 500, ist(9, 16)))
    out = bb.flush(ist(9, 21).astimezone(UTC))
    assert [(b.symbol, b.close) for b in out] == [("A", 100), ("B", 200)]
    bb2 = BarBuilder()
    bb2.on_tick(Tick("A", 100, 900, ist(9, 16)))
    bb2.on_tick(Tick("A", 101, 100, ist(9, 17)))  # a bad cumulative figure
    assert bb2.flush(ist(9, 21).astimezone(UTC))[0].volume == 0


def test_bars_roundtrip_into_the_candle_store(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bars = [
        Bar("X", ist(9, 20).astimezone(UTC), DAY, 100, 110, 90, 105, 50),
        Bar("X", ist(9, 25).astimezone(UTC), DAY, 105, 106, 100, 101, 60),
    ]
    frame = bars_to_frame(bars)
    assert (
        list(frame.columns) == ["ts", "date", "open", "high", "low", "close", "volume"]
        and len(frame) == 2
    )
    store = candles.CandleStore(tmp_path)
    store.write("5minute", "X", frame)
    store.write("5minute", "X", frame)  # idempotent
    got = store.read("5minute", "X")
    assert len(got) == 2 and got["close"].tolist() == [105, 101]
    assert (
        len(store.read("5minute", "X", as_of=ist(9, 22))) == 1
    )  # point-in-time: only finished bars


# ---------------- features and labels ----------------


def test_features_are_causal_and_within_day() -> None:
    bars = synth_bars(1, n_days=6)
    full = intraday_features(bars)
    assert list(full.columns) == INTRADAY_COLUMNS
    for cut in (40, 100, 300):
        part = intraday_features(bars.iloc[: cut + 1])
        pd.testing.assert_series_equal(full.iloc[cut], part.iloc[cut], check_names=False)
    first = full.iloc[:BARS_PER_DAY]
    assert first["ret_1"].iloc[0] != first["ret_1"].iloc[0]  # NaN: no previous bar in the session
    day2_first = full.iloc[BARS_PER_DAY]
    assert np.isnan(day2_first["ret_1"]) and np.isnan(
        day2_first["ret_12"]
    )  # a return never spans the overnight gap


def test_opening_range_only_exists_after_it_completes() -> None:
    f = intraday_features(breakout_day())
    assert (
        f["or_high_dist"].iloc[: OR_BARS - 1].isna().all()
        and f["or_high_dist"].iloc[OR_BARS - 1] == f["or_high_dist"].iloc[OR_BARS - 1]
    )
    assert f["or_breakout"].iloc[: OR_BARS + 4].eq(0).all() and f["or_breakout"].iloc[7] == 1.0
    assert (
        f["vol_ratio"].iloc[7] > 3
        and f["vwap_dist"].iloc[7] > 0
        and f["minutes_since_open"].iloc[0] == 5
    )
    assert f["day_ret"].iloc[7] > 0 and 0 <= f["range_pos"].iloc[7] <= 1


def test_vwap_matches_a_hand_calculation() -> None:
    b = breakout_day().iloc[:2]
    expected = float(
        (((b["high"] + b["low"] + b["close"]) / 3) * b["volume"]).sum() / b["volume"].sum()
    )
    f = intraday_features(b)
    assert f["vwap_dist"].iloc[1] == pytest.approx(float(b["close"].iloc[1]) / expected - 1)


def test_labels_start_at_the_next_open_and_never_cross_the_day() -> None:
    bars = synth_bars(2, n_days=3)
    lab = intraday_labels(bars, 6, 0.001)
    i = 10
    expected = bars["close"].iloc[i + 6] / bars["open"].iloc[i + 1] - 1
    assert lab["fwd_ret"].iloc[i] == pytest.approx(float(expected))
    assert (
        lab["fwd_ret"].iloc[BARS_PER_DAY - 6 : BARS_PER_DAY].isna().all()
    )  # last 6 bars of a session: unknown
    assert lab["fwd_ret"].iloc[BARS_PER_DAY - 7] == lab["fwd_ret"].iloc[BARS_PER_DAY - 7]
    assert set(lab["y"].dropna().unique()) <= {0.0, 1.0} and lab.index.equals(pd.Index(bars["ts"]))


# ---------------- scanner ----------------


def scanner(**p: object) -> IntradayScanner:
    return IntradayScanner(None, None, {"min_vol_ratio": 1.5, "top_k": 3, **p})


def now_at(bars: pd.DataFrame, i: int) -> datetime:
    return bars["ts"].iloc[i].to_pydatetime()


def test_scanner_finds_the_breakout_and_builds_valid_geometry() -> None:
    bars = breakout_day()
    found = scanner().scan(now_at(bars, 7), {"ORB": bars})
    assert [c.symbol for c in found] == ["ORB"]
    c = found[0]
    assert c.product == "MIS" and c.side == "BUY" and c.horizon_days == 0
    assert c.stop_paise < c.limit_price_paise < c.target_paise and c.limit_price_paise % 5 == 0
    assert (c.target_paise - c.limit_price_paise) >= 1.5 * (
        c.limit_price_paise - c.stop_paise
    )  # reward:risk
    assert (
        c.limit_price_paise - c.stop_paise
    ) / c.limit_price_paise <= 0.015 + 1e-9  # stop within 1.5%
    assert "ORB breakout" in c.reason and c.prob is None


def test_scanner_filters_window_blocks_stale_data_and_weak_volume() -> None:
    bars = breakout_day()
    t = now_at(bars, 7)
    assert t.astimezone(IST).hour == 9 and t.astimezone(IST).minute == 55
    assert scanner().scan(now_at(bars, 7), {"ORB": bars}, frozenset({"ORB"})) == []  # news-blocked
    assert (
        scanner().scan(now_at(bars, 7) + timedelta(minutes=10), {"ORB": bars}) == []
    )  # last bar is stale
    assert scanner(min_vol_ratio=9.0).scan(t, {"ORB": bars}) == []
    assert scanner().scan(now_at(bars, 5), {"ORB": bars}) == []  # no breakout yet
    early = datetime(2025, 6, 2, 9, 29, tzinfo=IST)
    assert scanner().scan(early, {"ORB": bars}) == []  # before 09:30
    late = breakout_day().assign(ts=lambda d: d["ts"] + pd.Timedelta(hours=5))
    assert scanner().scan(now_at(late, 7), {"ORB": late}) == []  # after 14:30
    other_day = breakout_day(day=date(2025, 6, 3))
    assert scanner().scan(now_at(bars, 7), {"ORB": other_day}) == []
    assert scanner().in_window(t) and not scanner().in_window(
        datetime(2025, 6, 2, 14, 30, tzinfo=IST)
    )


def test_scanner_only_uses_bars_that_have_ended() -> None:
    bars = breakout_day()
    t = now_at(bars, 6)  # the breakout bar (index 7) has not finished yet
    assert scanner().scan(t, {"ORB": bars}) == []
    assert len(scanner().scan(now_at(bars, 7), {"ORB": bars})) == 1
    scrambled = bars.copy()
    scrambled.loc[8:, ["open", "high", "low", "close"]] = 1
    assert scanner().scan(now_at(bars, 7), {"ORB": scrambled}) == scanner().scan(
        now_at(bars, 7), {"ORB": bars}
    )


def test_scanner_ranks_and_caps_top_k() -> None:
    bars = {
        f"S{i}": breakout_day(100_000 + 1000 * i).assign(
            volume=lambda d, k=i: d["volume"] * (1 + k)
        )
        for i in range(5)
    }
    t = now_at(bars["S0"], 7)
    out = scanner(top_k=2).scan(t, bars)
    assert len(out) == 2 and out[0].score >= out[1].score
    assert {c.symbol for c in scanner(top_k=9).scan(t, bars)} <= set(bars)


def test_session_bar_ends_helper() -> None:
    ends = session_bar_ends(DAY)
    assert (
        len(ends) == BARS_PER_DAY
        and ends[0] == ist(9, 20).astimezone(UTC)
        and ends[-1] == ist(15, 30).astimezone(UTC)
    )
