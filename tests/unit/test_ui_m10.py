"""Data / Train and Test / Config pages and their safety rules."""

import json
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from tests.fixtures.synth import synth_view
from tests.unit.test_cli_m4 import store_view
from tradepartner import cli
from tradepartner.calendar import TradingCalendar
from tradepartner.data.candles import CandleStore
from tradepartner.db.models import AuditLog, UniverseMember
from tradepartner.paths import config_dir, reports_dir
from tradepartner.ui import configedit, data, jobs

PAGES = Path(__file__).parents[2] / "src" / "tradepartner" / "ui" / "pages"


def page(name: str) -> AppTest:
    return AppTest.from_file(str(PAGES / name), default_timeout=60).run()


# ---------------- job whitelist ----------------


def test_only_whitelisted_research_commands_with_validated_arguments() -> None:
    assert jobs.build_args("train", {"strategy": "swing_v1", "trials": 5}) == [
        "train",
        "--strategy",
        "swing_v1",
        "--no-news",
        "--trials",
        "5",
    ]
    assert jobs.build_args("replay", {"as_of": "2026-09-01", "days": 5, "top": 3})[:3] == [
        "replay",
        "--as-of",
        "2026-09-01",
    ]
    assert "--end" in jobs.build_args(
        "paper-sim", {"strategy": "positional_v1", "start": "2024-01-01", "end": "2025-01-01"}
    )
    assert jobs.build_args("data-quality", {}) == ["data", "quality"]
    for bad_action in (
        "kill",
        "live-test",
        "unlock-auto",
        "models approve",
        "mode",
        "sync",
        "rm -rf",
        "",
    ):
        with pytest.raises(jobs.JobError, match="unknown action"):
            jobs.build_args(bad_action, {})
    bad = [
        ("train", {"strategy": "intraday_v1"}),
        ("train", {"strategy": "swing_v1; rm -rf /"}),
        ("train", {"strategy": "swing_v1", "trials": 99}),
        ("replay", {"as_of": "tomorrow"}),
        ("replay", {"as_of": "2026-01-01", "days": 500}),
        ("paper-sim", {"strategy": "swing_v1", "start": "2024-01-01", "capital_inr": 5}),
        ("backtest", {"strategy": "swing_v1", "capital_inr": 100000}),
        ("data-backfill", {"years": 50}),
    ]
    for action, params in bad:
        with pytest.raises(jobs.JobError):
            jobs.build_args(action, params)


def test_launch_runs_one_shell_wrapper_logs_and_reports_status(tmp_path: Path) -> None:
    started: list[list[str]] = []
    job = jobs.launch(
        "data-quality",
        {},
        popen=lambda cmd, **k: (started.append(cmd), SimpleNamespace(pid=4242))[1],
    )
    assert (
        started[0][:2] == ["/bin/sh", "-c"]
        and "tradepartner.cli" in started[0][2]
        and job.pid == 4242
    )
    assert job.status == "running" and jobs.list_jobs()[0].id == job.id
    job.log_path.write_text("Context impl SQLiteImpl.\nhello\n[exit 0]\n")
    assert job.exit_code() == 0 and job.status == "done" and jobs.tail(job) == "hello\n[exit 0]"
    job.log_path.write_text("boom\n[exit 2]\n")
    assert job.status == "failed (2)"
    assert (
        jobs.list_jobs(limit=0) == []
        and jobs.tail(jobs.UiJob("x", "a", [], 1, 0.0, str(tmp_path / "nope.log"))) == ""
    )
    with pytest.raises(jobs.JobError):
        jobs.launch("train", {"strategy": "nope"}, popen=lambda *a, **k: None)
    assert not list(jobs.jobs_dir().glob("*nope*"))  # a refused job leaves nothing behind


def test_jobs_dir_empty_listing() -> None:
    assert jobs.list_jobs() == []


# ---------------- config editor ----------------


def test_risk_and_app_config_are_never_editable_and_paths_cannot_escape() -> None:
    names = set(configedit.editable())
    assert "risk.yaml" not in names and "app.yaml" not in names
    assert set(configedit.READ_ONLY) == {"risk.yaml", "app.yaml"}
    for bad in (
        "risk.yaml",
        "app.yaml",
        "../pyproject.toml",
        "/etc/hosts",
        "strategies/../risk.yaml",
    ):
        with pytest.raises(configedit.ConfigEditError):
            configedit.save(bad, "x: 1\n")
        with pytest.raises(configedit.ConfigEditError):
            configedit.read(bad)


def test_strategy_params_are_validated_before_saving() -> None:
    text = configedit.read("strategies/swing_v1.yaml")
    assert configedit.validate("strategies/swing_v1.yaml", text) == []
    assert configedit.validate("strategies/swing_v1.yaml", text.replace("top_k: 5", "top_k: 99"))
    assert any(
        "n_trials" in e
        for e in configedit.validate(
            "strategies/swing_v1.yaml", text.replace("n_trials: 25", "n_trials: 500")
        )
    )
    assert any(
        "min_prob" in e
        for e in configedit.validate(
            "strategies/swing_v1.yaml", text.replace("min_prob: 0.45", "min_prob: 7")
        )
    )
    assert configedit.validate(
        "strategies/swing_v1.yaml", text.replace("name: swing_v1", "name: other")
    )
    assert configedit.validate("strategies/swing_v1.yaml", "name: [unclosed")
    assert configedit.validate("strategies/swing_v1.yaml", "- a list\n")


def test_cost_tax_holiday_and_csv_validation() -> None:
    costs = configedit.read("costs.yaml")
    assert configedit.validate("costs.yaml", costs) == []
    assert configedit.validate("costs.yaml", costs.replace("gst_pct: 18", "gst_pct: lots"))
    assert configedit.validate(
        "tax.yaml", configedit.read("tax.yaml").replace("capital_gains", "gambling")
    )
    assert configedit.validate("holidays.yaml", configedit.read("holidays.yaml")) == []
    assert any(
        "weekend" in e
        for e in configedit.validate(
            "holidays.yaml", "holidays:\n  - 2026-10-31\nextra_trading_days: []\n"
        )
    )
    assert any(
        "not YYYY" in e
        for e in configedit.validate(
            "holidays.yaml", "holidays:\n  - soon\nextra_trading_days: []\n"
        )
    )
    assert configedit.validate("holidays.yaml", "holidays: []\n")
    assert configedit.validate("universe/excluded.csv", "symbol,reason\nVEDL,demerger\n") == []
    assert configedit.validate("universe/excluded.csv", "symbol\nVEDL\n")
    assert configedit.validate("universe/excluded.csv", "symbol,reason\n,oops\n")
    assert configedit.validate("universe/nifty100.csv", "# only comments\n")


def test_save_backs_up_audits_and_is_a_noop_when_unchanged() -> None:
    cli._session().close()
    new = configedit.read("universe/excluded.csv") + "WIPRO,test exclusion\n"
    with cli._session() as s:
        path = configedit.save("universe/excluded.csv", new, s)
        assert "WIPRO" in path.read_text()
        assert (
            s.scalars(select(AuditLog).where(AuditLog.action == "config_edit"))
            .one()
            .payload_json["file"]
            == "universe/excluded.csv"
        )
    backups = list((reports_dir().parent / "backups" / "config-edits").glob("*excluded.csv"))
    assert len(backups) == 1 and "WIPRO" not in backups[0].read_text()
    assert (
        configedit.save("universe/excluded.csv", new) == path
        and len(list(backups[0].parent.glob("*"))) == 1
    )  # no second backup
    with pytest.raises(configedit.ConfigEditError, match="top_k"):
        configedit.save(
            "strategies/swing_v1.yaml",
            configedit.read("strategies/swing_v1.yaml").replace("top_k: 5", "top_k: 0"),
        )
    assert configedit.diff("a\n", "b\n").startswith("--- current")


# ---------------- data helpers ----------------


def seeded() -> CandleStore:
    v = synth_view(n_symbols=4, n_days=300)
    store = CandleStore()
    store_view(store, v)
    with cli._session() as s:
        for sym in v.symbols:
            s.add(UniverseMember(universe_name="nifty100", symbol=sym, valid_from=date(2000, 1, 1)))
        s.add(UniverseMember(universe_name="nifty100", symbol="GONE", valid_from=date(2000, 1, 1)))
        s.commit()
    return store


def test_data_helpers() -> None:
    store = seeded()
    cov = data.coverage(store)
    assert len(cov) == 5 and set(cov["bars"]) == {300}
    frame = data.symbol_frame(store, "SYM00")
    assert len(frame) == 300 and 100 < frame["close"].iloc[0] < 1000  # rupees, not paise
    assert data.symbol_frame(store, "NOPE").empty
    q = data.quality_frame(
        store,
        TradingCalendar(
            extra_trading_days=frozenset(synth_view(n_symbols=1, n_days=300).trading_days())
        ),
    )
    assert list(q.columns) == ["symbol", "kind", "severity", "detail"]
    with cli._session() as s:
        u = data.universe_summary(s, store)
    assert (
        u.members == 5
        and u.without_data == ["GONE"]
        and u.static_bias
        and isinstance(u.excluded, dict)
    )


# ---------------- pages ----------------


def test_data_page_renders_issues_symbols_and_universe() -> None:
    seeded()
    at = page("data.py")
    assert not at.exception and at.metric[0].value == "5"
    assert len(at.tabs) == 5 and at.selectbox
    at.selectbox[0].select("SYM01").run()
    assert not at.exception
    at.button[
        0
    ].click().run()  # backfill: starts a background job (fake popen not needed: refused without Kite)
    assert not at.exception


def test_data_page_with_no_data() -> None:
    at = page("data.py")
    assert not at.exception and at.info and "No candles yet" in at.info[0].value


def test_train_test_page_validates_and_lists_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        jobs,
        "launch",
        lambda action, params, **k: (
            started.append((action, params)),
            jobs.UiJob("j", action, [], 1, 0.0, "x"),
        )[1],
    )
    reports_dir().mkdir(parents=True, exist_ok=True)
    (reports_dir() / "train-swing_v1-20260101-000000.json").write_text(json.dumps(
        {"strategy": "swing_v1", "chosen_variant": "price_only", "chosen_reason": "x", "quality": {"auc": 0.53},
         "importance": {"a": 0.6, "b": 0.4}, "warnings": ["w"],
         "gate": {"checks": [{"name": "min_trades", "passed": True, "value": 120.0, "threshold": ">= 100"}]}}))  # fmt: skip
    at = page("train_test.py")
    assert not at.exception and at.warning and at.dataframe
    at.selectbox[0].select("replay").run()
    at.button[0].click().run()
    assert started and started[0][0] == "replay" and not at.exception


def test_config_page_edit_validate_and_save() -> None:
    at = page("config.py")
    assert not at.exception and "risk.yaml" in at.expander[0].label
    at.selectbox[0].select("universe/excluded.csv").run()
    at.text_area[0].set_value("symbol\nVEDL\n").run()
    assert at.error  # wrong columns: refused
    assert at.button[0].disabled
    at.text_area[0].set_value(configedit.read("universe/excluded.csv") + "WIPRO,test\n").run()
    assert not at.error and not at.button[0].disabled
    at.button[0].click().run()
    assert (
        "WIPRO" in configedit.read("universe/excluded.csv") and shutil.which("sh") and config_dir()
    )


def test_html_reports_are_embedded_with_st_iframe_and_downloadable() -> None:
    reports_dir().mkdir(parents=True, exist_ok=True)
    (reports_dir() / "paper-sim-swing_v1-20260101-000000.html").write_text("<h1>Hello report</h1>")
    at = page("train_test.py")
    assert not at.exception
    kinds = {e.type for e in at.main}
    assert "download_button" in kinds  # the report can be downloaded


def test_text_reports_land_in_the_reports_folder(tmp_path: Path) -> None:
    from tradepartner.reports.text import save_text_report

    p = save_text_report("ew-sim", "A <title>", ["line 1", "<b>x</b>"], tmp_path)
    html = p.read_text()
    assert (
        p.parent == tmp_path and "&lt;b&gt;x&lt;/b&gt;" in html and "A &lt;title&gt;" in html
    )  # escaped


def test_news_job_args_are_validated() -> None:
    assert jobs.build_args("news-fetch", {"hours": 12}) == ["news", "fetch", "--hours", "12"]
    assert jobs.build_args("news-score", {"limit": 5, "max_minutes": 2}) == [
        "news", "score", "--limit", "5", "--max-minutes", "2",
    ]  # fmt: skip
    assert "--news" in jobs.build_args("train", {"strategy": "swing_v1", "news": True})
    assert "--no-news" in jobs.build_args("train", {"strategy": "swing_v1"})
    for action, params in (("news-fetch", {"hours": 0}), ("news-score", {"limit": "x"})):
        with pytest.raises(jobs.JobError):
            jobs.build_args(action, params)


def test_news_summary_and_recent(db) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from tradepartner.db.models import NewsEntity, NewsRaw, NewsScore

    assert data.news_summary(db).articles == 0 and data.recent_news(db).empty
    now = datetime(2026, 5, 1, tzinfo=UTC)
    db.add(
        NewsRaw(
            source="rss", url="u", title="T", published_at=now, fetched_at=now, content_hash="h"
        )
    )
    db.flush()
    nid = db.query(NewsRaw).one().id
    db.add(NewsEntity(news_id=nid, symbol="TCS", match_method="alias", confidence=0.9))
    db.add(
        NewsScore(
            news_id=nid, symbol="TCS", llm_model="m", prompt_version="v", relevance=0.5,
            sentiment=0.4, event_type="results", materiality=0.5, horizon="days",
            confidence=0.7, summary="s", raw_json={}, scored_at=now,
        )
    )  # fmt: skip
    db.commit()
    s = data.news_summary(db)
    assert (s.articles, s.scored, s.valid_rate, s.per_hour) == (1, 1, 1.0, None)
    row = data.recent_news(db).iloc[0]
    assert row["symbols"] == "TCS" and row["sentiment"] == 0.4


def test_add_holiday_keeps_order_and_is_idempotent(db) -> None:  # type: ignore[no-untyped-def]
    from datetime import date

    from tradepartner.paths import config_dir

    configedit.add_holiday(date(2017, 10, 18), db)
    configedit.add_holiday(date(2017, 10, 18), db)
    text = (config_dir() / "holidays.yaml").read_text()
    assert text.count("2017-10-18") == 1
    items = [ln for ln in text.splitlines() if ln.startswith("  - 2017-10")]
    assert items == sorted(items)
    with pytest.raises(configedit.ConfigEditError):
        configedit.add_holiday(date(2017, 10, 21), db)  # Saturday


def test_exclude_symbol_and_missing_days(db, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from datetime import date

    import pandas as pd

    from tradepartner.calendar import TradingCalendar

    configedit.exclude_symbol("abc", "demerger, price jump", db)
    configedit.exclude_symbol("ABC", "again", db)
    from tradepartner.data.universe import read_excluded

    assert read_excluded()["ABC"] == "demerger; price jump"
    with pytest.raises(configedit.ConfigEditError):
        configedit.exclude_symbol("X", " ")

    class Store:
        def symbols(self, tf: str) -> list[str]:
            return ["A", "B"]

        def read(self, tf: str, sym: str) -> pd.DataFrame:
            days = [date(2026, 5, 4), date(2026, 5, 6)] + ([date(2026, 5, 5)] if sym == "A" else [])
            return pd.DataFrame({"date": sorted(days), "open": 1, "high": 1, "low": 1, "close": 1})

    out = data.missing_days(Store(), TradingCalendar())  # type: ignore[arg-type]
    assert out.iloc[0]["symbols_missing"] == 1 and out.iloc[0]["likely"] == "data gap"
    assert len(out) == 1


def test_explain_jumps_job_args() -> None:
    assert jobs.build_args("explain-jumps", {"limit": 5}) == [
        "data", "explain-jumps", "--limit", "5",
    ]  # fmt: skip
    assert jobs.build_args("explain-jumps", {"limit": 5, "symbol": "M&M"})[-2:] == [
        "--symbol",
        "M&M",
    ]
    with pytest.raises(jobs.JobError):
        jobs.build_args("explain-jumps", {"limit": 5, "symbol": "x; rm -rf"})
