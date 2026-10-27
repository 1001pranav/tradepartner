import threading
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from tests.conftest import MemoryKeyring
from tests.fixtures.market import FakeKite, daily_rows, dt, make_cal
from tradepartner.calendar import utc_now
from tradepartner.data import (
    candles,
    corporate_actions,
    instruments,
    kite_auth,
    quality,
)
from tradepartner.data import universe as uni
from tradepartner.db.migrate import upgrade
from tradepartner.db.session import make_engine, make_session_factory

CAL = make_cal()
NOW = datetime(2026, 3, 10, 16, 0, tzinfo=dt(2026, 1, 1).tzinfo)  # after close


@pytest.fixture
def session(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 't.sqlite'}"
    upgrade(url)
    return make_session_factory(make_engine(url))()


# ---------- kite auth ----------


def test_credentials_and_token_expiry(memory_keyring: MemoryKeyring) -> None:
    assert kite_auth.load_credentials() is None
    kite_auth.save_credentials("k", "s")
    assert kite_auth.load_credentials() == ("k", "s")
    kite_auth.save_access_token("tok", date(2026, 3, 10))
    assert kite_auth.load_access_token(date(2026, 3, 10)) == "tok"
    assert kite_auth.load_access_token(date(2026, 3, 11)) is None  # expired next day
    kite_auth.clear_access_token()
    kite_auth.clear_access_token()  # idempotent
    assert kite_auth.load_access_token(date(2026, 3, 10)) is None


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _hit(port: int, query: str) -> None:
    def go() -> None:
        import time

        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/?{query}", timeout=2).read()
                return
            except urllib.error.HTTPError:
                return
            except OSError:
                time.sleep(0.05)

    threading.Thread(target=go, daemon=True).start()


def test_redirect_server_receives_token() -> None:
    port = _free_port()
    _hit(port, "status=success&request_token=abc123")
    assert kite_auth.wait_for_request_token(port, timeout_sec=10) == "abc123"


def test_redirect_server_failure_and_timeout() -> None:
    port = _free_port()
    _hit(port, "status=error")
    with pytest.raises(kite_auth.LoginError):
        kite_auth.wait_for_request_token(port, timeout_sec=3)
    with pytest.raises(kite_auth.LoginError):
        kite_auth.wait_for_request_token(_free_port(), timeout_sec=0.4)


class _Client:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def login_url(self) -> str:
        return "u"

    def generate_session(self, request_token: str, api_secret: str) -> dict[str, Any]:
        assert (request_token, api_secret) == ("rt", "sec")
        return self.data


def test_complete_login_stores_token() -> None:
    assert kite_auth.complete_login(_Client({"access_token": "AT"}), "sec", "rt") == "AT"
    assert kite_auth.load_access_token() == "AT"
    with pytest.raises(kite_auth.LoginError):
        kite_auth.complete_login(_Client({}), "sec", "rt")


# ---------- instruments & universe ----------

ROWS = [
    {
        "instrument_token": 1,
        "exchange": "NSE",
        "segment": "NSE",
        "instrument_type": "EQ",
        "tradingsymbol": "INFY",
        "isin": "INE009A01021",
        "name": "INFOSYS",
        "tick_size": 0.05,
        "lot_size": 1,
    },
    {
        "instrument_token": 2,
        "exchange": "NFO",
        "segment": "NFO-FUT",
        "instrument_type": "FUT",
        "tradingsymbol": "INFYFUT",
    },
    {
        "instrument_token": 3,
        "exchange": "NSE",
        "segment": "INDICES",
        "instrument_type": "EQ",
        "tradingsymbol": "NIFTY 50",
    },
    {
        "instrument_token": 4,
        "exchange": "NSE",
        "segment": "NSE",
        "instrument_type": "EQ",
        "tradingsymbol": "TCS",
        "tick_size": 0.1,
    },
]


def test_only_nse_equity_kept_and_upsert_idempotent(session: Session) -> None:
    kite = FakeKite(CAL)
    kite.instrument_rows = ROWS
    assert instruments.refresh_instruments(session, kite, utc_now()) == 2
    assert instruments.refresh_instruments(session, kite, utc_now()) == 2
    assert instruments.symbol_tokens(session) == {"INFY": 1, "TCS": 4}
    assert instruments.token_for(session, "INFY") == 1
    assert instruments.token_for(session, "FUT") is None


def test_universe_seed_and_point_in_time(session: Session, tmp_path: Path) -> None:
    (tmp_path / "u.csv").write_text("# comment\nsymbol\nINFY\nTCS\n")
    assert uni.read_seed("u", tmp_path) == ["INFY", "TCS"]
    assert uni.seed_universe(session, "u", tmp_path) == 2
    assert uni.seed_universe(session, "u", tmp_path) == 0
    assert uni.members_as_of(session, "u", date(2020, 1, 1)) == ["INFY", "TCS"]
    assert uni.members_as_of(session, "u", date(1999, 1, 1)) == []
    assert uni.has_survivorship_bias(session, "u")
    assert not uni.has_survivorship_bias(session, "empty")
    kite = FakeKite(CAL)
    kite.instrument_rows = ROWS
    instruments.refresh_instruments(session, kite, utc_now())
    assert uni.tradable_members(session, "u", date(2026, 1, 1)) == {"INFY": 1, "TCS": 4}


def test_repo_nifty100_seed_has_about_100_symbols() -> None:
    symbols = uni.read_seed("nifty100")
    assert 95 <= len(symbols) <= 105 and len(set(symbols)) == len(symbols)


def test_closed_membership_excluded(session: Session) -> None:
    from tradepartner.db.models import UniverseMember

    session.add(
        UniverseMember(
            universe_name="x", symbol="OLD", valid_from=date(2010, 1, 1), valid_to=date(2015, 1, 1)
        )
    )
    session.commit()
    assert uni.members_as_of(session, "x", date(2012, 1, 1)) == ["OLD"]
    assert uni.members_as_of(session, "x", date(2016, 1, 1)) == []
    assert not uni.has_survivorship_bias(session, "x")


# ---------- candles ----------


def test_rate_limiter_spaces_calls() -> None:
    t = [0.0]
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        t[0] += s

    lim = candles.RateLimiter(2.0, clock=lambda: t[0], sleep=sleep)
    for _ in range(4):
        lim.acquire()
    assert sleeps == pytest.approx([0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        candles.RateLimiter(0)


def test_normalize_money_is_int_paise_and_ts_is_close() -> None:
    df = candles.normalize_rows(
        daily_rows(date(2026, 3, 2), date(2026, 3, 2), CAL, base=1234.56), "day"
    )
    assert df.loc[0, "open"] == 123456 and df["open"].dtype == "int64"
    assert df.loc[0, "ts"].hour == 10 and df.loc[0, "date"] == date(2026, 3, 2)
    assert candles.normalize_rows([], "day").empty


def test_normalize_intraday_rows() -> None:
    row = {"date": dt(2026, 3, 2, 9, 15), "open": 1, "high": 2, "low": 1, "close": 2, "volume": 5}
    df = candles.normalize_rows([row], "5minute")
    assert df.loc[0, "date"] == date(2026, 3, 2) and df.loc[0, "ts"].hour == 3
    naive = dict(row, date=datetime(2026, 3, 2, 9, 20))
    assert candles.normalize_rows([naive], "5minute").loc[0, "ts"].minute == 50


def test_store_roundtrip_dedupe_asof_and_query(tmp_path: Path) -> None:
    store = candles.CandleStore(tmp_path)
    assert store.read("day", "X").empty and store.last_date("day", "X") is None
    df = candles.normalize_rows(daily_rows(date(2026, 3, 2), date(2026, 3, 6), CAL), "day")
    store.write("day", "M&M", df)
    assert store.write("day", "M&M", df) == 5  # idempotent
    assert store.symbols("day") == ["M_and_M"] and store.symbols("none") == []
    assert store.last_date("day", "M&M") == date(2026, 3, 6)
    assert len(store.read("day", "M&M", start=date(2026, 3, 4), end=date(2026, 3, 5))) == 2
    # point-in-time: a bar is only visible after the session close
    just_before = datetime(2026, 3, 4, 10, 0, tzinfo=dt(2026, 1, 1).tzinfo).replace(
        hour=15, minute=29
    )
    assert store.read("day", "M&M", as_of=just_before)["date"].max() == date(2026, 3, 3)
    out = store.query("SELECT count(*) AS n FROM read_parquet('{root}/day/*.parquet')")
    assert out.loc[0, "n"] == 5


def _run(
    kite: FakeKite,
    store: candles.CandleStore,
    targets: dict[str, int],
    years: int = 1,
    now: datetime = NOW,
) -> candles.BackfillResult:
    return candles.backfill(
        kite,
        store,
        targets,
        years=years,
        now=now,
        calendar=CAL,
        limiter=candles.RateLimiter(1000),
        sleep=lambda s: None,
    )


def test_backfill_chunks_resumes_and_isolates_failures(tmp_path: Path) -> None:
    store = candles.CandleStore(tmp_path)
    kite = FakeKite(CAL, fail_tokens={9})
    res = _run(kite, store, {"A": 1, "BAD": 9}, years=6)
    assert res.done == ["A"] and "BAD" in res.failed and res.rows > 1000
    assert len([c for c in kite.calls if c[0] == 1]) == 2  # >2000 days -> two chunks
    assert len([c for c in kite.calls if c[0] == 9]) == 3  # retried 3 times
    # resume: nothing missing -> no new calls for A
    kite.calls.clear()
    res2 = _run(kite, store, {"A": 1})
    assert res2.skipped == ["A"] and kite.calls == []
    # a later day only fetches the increment
    later = datetime(2026, 3, 12, 16, 0, tzinfo=NOW.tzinfo)
    res3 = _run(kite, store, {"A": 1}, now=later)
    assert res3.done == ["A"] and kite.calls[0][1] == date(2026, 3, 11)


def test_backfill_interrupted_then_resumed(tmp_path: Path) -> None:
    store = candles.CandleStore(tmp_path)

    class Dying(FakeKite):
        def historical_data(self, *a: Any, **k: Any) -> list[dict[str, Any]]:
            if len(self.calls) >= 1:
                self.calls.append((0, date.min, date.min, ""))
                raise KeyboardInterrupt
            return super().historical_data(*a, **k)

    with pytest.raises(KeyboardInterrupt):
        _run(Dying(CAL), store, {"A": 1}, years=6)
    assert store.last_date("day", "A") is not None  # first chunk persisted
    kite = FakeKite(CAL)
    assert _run(kite, store, {"A": 1}, years=6).done == ["A"]
    assert kite.calls[0][1] > date(2020, 1, 1)  # resumed, did not restart from scratch
    assert candles.detect_gaps(store.read("day", "A"), CAL) == []


def test_last_completed_session_before_and_after_close() -> None:
    tz = NOW.tzinfo
    assert candles.last_completed_session(datetime(2026, 3, 10, 16, 0, tzinfo=tz), CAL) == date(
        2026, 3, 10
    )
    assert candles.last_completed_session(datetime(2026, 3, 10, 10, 0, tzinfo=tz), CAL) == date(
        2026, 3, 9
    )
    assert candles.last_completed_session(datetime(2026, 3, 8, 16, 0, tzinfo=tz), CAL) == date(
        2026, 3, 6
    )


def test_detect_gaps() -> None:
    df = candles.normalize_rows(daily_rows(date(2026, 3, 2), date(2026, 3, 6), CAL), "day")
    holey = df[df["date"] != date(2026, 3, 4)]
    assert candles.detect_gaps(holey, CAL) == [date(2026, 3, 4)]
    assert candles.detect_gaps(candles.empty_frame(), CAL) == []


# ---------- corporate actions ----------


def _split_frame() -> pd.DataFrame:
    """Raw prices: ~1000 before a 5:1 split on 2026-03-05, ~200 after."""
    rows = []
    for d in CAL.trading_days(date(2026, 3, 2), date(2026, 3, 9)):
        px = 100000 if d < date(2026, 3, 5) else 20000
        rows.append({"date": d, "open": px, "high": px, "low": px, "close": px, "volume": 1000})
    return pd.DataFrame(rows)


def test_split_adjustment_is_continuous() -> None:
    adj = corporate_actions.adjust_candles(
        _split_frame(), [corporate_actions.Action("X", "split", date(2026, 3, 5), 5, 1)]
    )
    assert adj["close"].nunique() == 1 and adj.loc[0, "close"] == pytest.approx(200.0)
    assert adj.loc[0, "volume"] == pytest.approx(5000)  # volume scaled inversely


def test_bonus_and_unadjusted_actions() -> None:
    bonus = corporate_actions.Action("X", "bonus", date(2026, 3, 5), 1, 1)
    assert corporate_actions.share_multiplier(bonus) == (2, 1)
    adj = corporate_actions.adjust_candles(_split_frame(), [bonus])
    assert adj.loc[0, "close"] == pytest.approx(500.0)
    div = corporate_actions.Action("X", "dividend", date(2026, 3, 5))
    assert corporate_actions.adjust_candles(_split_frame(), [div]).loc[0, "close"] == pytest.approx(
        1000.0
    )
    with pytest.raises(ValueError):
        corporate_actions.share_multiplier(div)
    with pytest.raises(ValueError):
        corporate_actions.share_multiplier(corporate_actions.Action("X", "split", date(2026, 3, 5)))
    assert corporate_actions.share_multiplier(
        corporate_actions.Action("X", "split", date(2026, 3, 5), 5, 1)
    ) == (5, 1)


def test_actions_csv_import_idempotent_and_strict(session: Session, tmp_path: Path) -> None:
    f = tmp_path / "ca.csv"
    f.write_text(
        "symbol,isin,action_type,ex_date,ratio_num,ratio_den,amount,source\n"
        "X,INE1,split,2026-03-05,5,1,,nse\nX,,dividend,2026-04-01,,,12.50,nse\n"
    )
    assert corporate_actions.import_actions_csv(session, f) == 2
    assert corporate_actions.import_actions_csv(session, f) == 0
    acts = corporate_actions.load_actions(session, "X")
    assert [a.action_type for a in acts] == ["split", "dividend"]
    assert len(corporate_actions.load_actions(session)) == 2
    f.write_text("symbol,action_type,ex_date,mystery\nX,split,2026-03-05,1\n")
    with pytest.raises(ValueError, match="unknown columns"):
        corporate_actions.import_actions_csv(session, f)
    f.write_text("symbol,action_type,ex_date\nX,spinoff,2026-03-05\n")
    with pytest.raises(ValueError, match="action_type"):
        corporate_actions.import_actions_csv(session, f)


# ---------- quality ----------


def _frame(**over: Any) -> pd.DataFrame:
    df = candles.normalize_rows(daily_rows(date(2026, 3, 2), date(2026, 3, 6), CAL), "day")
    for k, v in over.items():
        df.loc[2, k] = v
    return df


def _kinds(df: pd.DataFrame, actions: list[corporate_actions.Action] | None = None) -> set[str]:
    return {i.kind for i in quality.check_symbol("X", df, CAL, actions)}


def test_quality_clean_and_each_defect() -> None:
    assert _kinds(_frame()) == set()
    assert _kinds(candles.empty_frame()) == {"empty"}
    assert "ohlc_inconsistent" in _kinds(_frame(high=1))
    assert "non_positive_price" in _kinds(_frame(low=0))
    assert "negative_volume" in _kinds(_frame(volume=-1))
    assert "zero_volume" in _kinds(_frame(volume=0))
    dup = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)
    assert {"duplicate_ts", "not_sorted"} <= _kinds(dup)
    assert "gaps" in _kinds(_frame().drop(index=2))


def test_unexplained_jump_flagged_unless_corporate_action() -> None:
    df = _split_frame()
    df["ts"] = [candles.session_close_utc(d) for d in df["date"]]
    assert "unexplained_jump" in _kinds(df)
    act = corporate_actions.Action("X", "split", date(2026, 3, 5), 5, 1)
    assert "unexplained_jump" not in _kinds(df, [act])


def test_report_and_stale(tmp_path: Path) -> None:
    store = candles.CandleStore(tmp_path)
    store.write(
        "day",
        "A",
        candles.normalize_rows(daily_rows(date(2026, 3, 2), date(2026, 3, 6), CAL), "day"),
    )
    assert quality.report(store, ["A"], CAL) == []
    assert quality.report(store, ["MISSING"], CAL)[0].kind == "empty"
    assert quality.stale_symbols(store, ["A", "B"], date(2026, 3, 9)) == ["A", "B"]
    assert quality.stale_symbols(store, ["A"], date(2026, 3, 6)) == []
    assert timedelta(0) == timedelta(days=0)


# ---------- point-in-time membership (removes survivorship bias) ----------


def test_membership_import_replaces_the_static_seed_and_is_point_in_time(
    session: Session, tmp_path: Path
) -> None:
    (tmp_path / "u.csv").write_text("symbol\nINFY\nOLDCO\n")
    uni.seed_universe(session, "u", tmp_path)
    assert uni.has_survivorship_bias(session, "u")
    f = tmp_path / "m.csv"
    f.write_text(
        "# real history\nsymbol,valid_from,valid_to\nINFY,2010-01-01,\nOLDCO,2010-01-01,2020-06-30\nNEWCO,2021-04-01,\n"
    )
    assert uni.import_membership_csv(session, "u", f) == (3, 2)
    assert uni.import_membership_csv(session, "u", f) == (0, 0)  # idempotent
    assert not uni.has_survivorship_bias(session, "u")
    members = uni.membership_fn(session, "u")
    assert members(date(2019, 1, 1)) == ["INFY", "OLDCO"]
    assert members(date(2020, 6, 30)) == ["INFY"]  # valid_to is exclusive
    assert members(date(2022, 1, 1)) == ["INFY", "NEWCO"]
    assert uni.all_symbols(session, "u") == ["INFY", "NEWCO", "OLDCO"]


def test_membership_import_rejects_bad_files(session: Session, tmp_path: Path) -> None:
    bad = tmp_path / "b.csv"
    bad.write_text("symbol,from\nA,2020-01-01\n")
    with pytest.raises(ValueError, match="exactly columns"):
        uni.import_membership_csv(session, "u", bad)
    bad.write_text("symbol,valid_from,valid_to\nA,2020-01-01,2019-01-01\n")
    with pytest.raises(ValueError, match="after valid_from"):
        uni.import_membership_csv(session, "u", bad)


def test_view_universe_follows_membership_dates() -> None:
    from tests.fixtures.synth import synth_view

    base = synth_view(n_symbols=3, n_days=200)
    days = base.trading_days()
    from tradepartner.backtest.view import MarketView

    view = MarketView({s: base.history(s, days[-1]) for s in base.symbols}, base.index, base.sectors,
                      members=lambda d: ["SYM00"] if d < days[100] else ["SYM00", "SYM01"])  # fmt: skip
    assert view.symbols_as_of(days[50]) == ["SYM00"] and view.symbols_as_of(days[150]) == [
        "SYM00",
        "SYM01",
    ]


def test_gap_check_starts_at_first_real_bar_not_prelisting_junk() -> None:
    rows = candles.normalize_rows(daily_rows(date(2026, 3, 2), date(2026, 3, 6), CAL), "day")
    junk = rows.iloc[[0]].copy()
    junk[["open", "high", "low", "close"]] = 0
    junk["ts"] = junk["ts"] - pd.Timedelta(days=700)
    junk["date"] = junk["date"].map(lambda d: d - timedelta(days=700))
    df = pd.concat([junk, rows], ignore_index=True)
    kinds = _kinds(df)
    assert "gaps" not in kinds and "non_positive_price" in kinds


def _store_with_hole(
    tmp_path: Path, hole: date, syms: tuple[str, ...] = ("A",)
) -> candles.CandleStore:
    store = candles.CandleStore(tmp_path)
    rows = [
        r for r in daily_rows(date(2026, 3, 2), date(2026, 3, 13), CAL) if r["date"].date() != hole
    ]
    for s in syms:
        store.write("day", s, candles.normalize_rows(rows, "day"))
    return store


def test_refill_gaps_fills_hole_inside_history_and_reports_unavailable(tmp_path: Path) -> None:
    hole = date(2026, 3, 5)
    store = _store_with_hole(tmp_path, hole)
    assert candles.real_gaps(store.read("day", "A"), CAL) == [hole]
    kite = FakeKite(CAL)
    filled, still, failed = candles.refill_gaps(
        kite, store, {"A": 1}, calendar=CAL, limiter=candles.RateLimiter(1000)
    )
    assert filled == {"A": [hole]} and still == {"A": []} and not failed
    assert candles.real_gaps(store.read("day", "A"), CAL) == []

    class Empty(FakeKite):
        def historical_data(self, *a: Any, **k: Any) -> list[dict[str, Any]]:
            return []

    store2 = _store_with_hole(tmp_path / "b", hole)
    _, still2, _ = candles.refill_gaps(
        Empty(CAL), store2, {"A": 1}, calendar=CAL, limiter=candles.RateLimiter(1000)
    )
    assert still2 == {"A": [hole]}

    class Boom(FakeKite):
        def historical_data(self, *a: Any, **k: Any) -> list[dict[str, Any]]:
            raise RuntimeError("down")

    store3 = _store_with_hole(tmp_path / "c", hole)
    _, _, bad = candles.refill_gaps(
        Boom(CAL), store3, {"A": 1}, calendar=CAL, limiter=candles.RateLimiter(1000),
        retries=1, sleep=lambda s: None,
    )  # fmt: skip
    assert "A" in bad


def test_refill_skips_market_wide_holes_and_honours_only(tmp_path: Path) -> None:
    hole = date(2026, 3, 5)
    syms = ("A", "B", "C", "D")
    store = _store_with_hole(tmp_path, hole, syms)
    kite = FakeKite(CAL)
    tokens = {s: i + 1 for i, s in enumerate(syms)}
    f, s, _ = candles.refill_gaps(
        kite, store, tokens, calendar=CAL, limiter=candles.RateLimiter(1000)
    )
    assert f == {} and s == {} and kite.calls == []  # looks like a market holiday: left alone
    f, _, _ = candles.refill_gaps(
        kite, store, tokens, calendar=CAL, limiter=candles.RateLimiter(1000),
        only={"A": (date(2026, 3, 1), date(2026, 3, 31))},
    )  # fmt: skip
    assert f == {"A": [hole]}
