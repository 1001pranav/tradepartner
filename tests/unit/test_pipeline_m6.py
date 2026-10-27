"""End-to-end M6: features -> labels -> walk-forward -> gate -> candidate. Nothing is approved."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from tests.fixtures.synth import synth_view
from tests.unit.test_cli_m4 import store_view
from tradepartner import cli
from tradepartner.data.candles import CandleStore
from tradepartner.db.models import Model, UniverseMember
from tradepartner.features import build
from tradepartner.models import pipeline, registry
from tradepartner.settings import StrategyConfig, load_settings

NOW = datetime(2026, 6, 1, 5, 0, tzinfo=UTC)
runner = CliRunner()


def fast_settings(strategy: str = "swing_v1", **over: object):  # type: ignore[no-untyped-def]
    s = load_settings()
    cfg = {**s.strategies[strategy].model_dump(), "n_trials": 1, "min_train_days": 300, **over}
    return s.model_copy(update={"strategies": {**s.strategies, strategy: StrategyConfig(**cfg)}})


@pytest.fixture(scope="module")
def view():  # type: ignore[no-untyped-def]
    return synth_view(n_symbols=8, n_days=760, seed=21, momentum=0.25)


def test_price_only_pipeline_registers_a_candidate_only(db: Session, view, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    out = pipeline.train_strategy(db, "swing_v1", view, fast_settings(), NOW, models_root=tmp_path / "m",
                                  reports_root=tmp_path / "r", survivorship_warning=True)  # fmt: skip
    assert out.chosen == "price_only" and set(out.variants) == {"price_only"}
    assert any("price-only" in w for w in out.warnings) and any(
        "survivorship" in w.lower() for w in out.warnings
    )
    m = db.scalars(select(Model)).one()
    assert (
        m.status == "candidate" and m.approved_by is None and m.approved_at is None
    )  # never auto-approved
    mj = m.metrics_json
    assert mj["gate"]["checks"][-1]["name"] == "human_approval" and mj["gate"]["approved"] is False
    assert mj["feature_columns"] == build.PRICE_ONLY_COLUMNS and mj["paper_gate_passed"] is False
    assert mj["feature_bins"] and mj["objective"] == "binary" and mj["horizon_days"] == 5
    assert registry.load_booster(m).num_feature() == len(build.PRICE_ONLY_COLUMNS)
    report = json.loads(next((tmp_path / "r").glob("train-swing_v1-*.json")).read_text())
    assert report["chosen_variant"] == "price_only" and "gate" in report
    assert (
        out.report["comparison"]["price_only"]["quality"]["auc"] > 0.5
    )  # the planted momentum is learnt


def test_with_news_variant_is_compared_and_the_llm_period_reported(
    db: Session, view, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    matrix_index = build.build_matrix(view).index
    panel = pd.DataFrame(rng.normal(size=(len(matrix_index), len(build.NEWS_FEATURE_COLUMNS))),
                         index=matrix_index, columns=build.NEWS_FEATURE_COLUMNS)  # fmt: skip
    days = matrix_index.get_level_values("date")
    cutoff = sorted(set(days))[400]
    panel["llm_features_valid"] = (days >= cutoff).astype(float)
    panel.loc[days < cutoff, build.NEWS_FEATURE_COLUMNS[:-1]] = np.nan
    out = pipeline.train_strategy(
        db, "swing_v1", view, fast_settings(), NOW, news_panel=panel, models_root=tmp_path
    )
    assert set(out.variants) == {"price_only", "with_news"} and out.chosen in {
        "price_only",
        "with_news",
    }
    news = out.variants["with_news"]
    assert news.llm_period is not None and date.fromisoformat(news.llm_period[0]) >= cutoff
    cmp = out.report["comparison"]
    assert (
        tuple(cmp["with_news"]["llm_period"]) == news.llm_period
        and cmp["price_only"]["llm_period"] is None
    )
    assert db.scalars(select(Model)).one().status == "candidate"


def test_choose_variant_rules() -> None:
    class G:
        def __init__(self, ready: bool, sharpe: float) -> None:
            self.ready_for_review, self.metrics = ready, {"sharpe": sharpe}

    class V:
        def __init__(self, ready: bool, sharpe: float) -> None:
            self.gate = G(ready, sharpe)

    p = V(True, 1.0)
    assert pipeline.choose_variant(p, None)[0] == "price_only"  # type: ignore[arg-type]
    assert pipeline.choose_variant(p, V(False, 9.0))[0] == "price_only"  # type: ignore[arg-type]
    assert pipeline.choose_variant(p, V(True, 0.5))[0] == "price_only"  # type: ignore[arg-type]
    assert pipeline.choose_variant(p, V(True, 1.5))[0] == "with_news"  # type: ignore[arg-type]
    assert pipeline.choose_variant(V(True, float("nan")), V(True, 0.1))[0] == "with_news"  # type: ignore[arg-type]
    assert pipeline.finite(1.0) and not pipeline.finite(None) and not pipeline.finite(float("nan"))


def test_positional_pipeline_uses_rank_regression(db: Session, tmp_path: Path) -> None:
    v = synth_view(n_symbols=8, n_days=1100, seed=3, momentum=0.2)
    out = pipeline.train_strategy(
        db, "positional_v1", v, fast_settings("positional_v1"), NOW, models_root=tmp_path
    )
    assert out.report["objective"] == "regression" and out.report["horizon_days"] == 60
    assert "spearman" in out.report["comparison"]["price_only"]["quality"]


def test_pipeline_input_errors(db: Session, view, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(registry.ModelError, match="no ML pipeline"):
        pipeline.strategy_params(load_settings(), "intraday_v1")
    from tradepartner.backtest.view import MarketView

    no_index = MarketView({s: view.history(s, view.trading_days()[-1]) for s in view.symbols})
    with pytest.raises(registry.ModelError, match="index"):
        pipeline.train_strategy(db, "swing_v1", no_index, fast_settings(), NOW, register=False)


# ---------- CLI ----------


def _seed_store() -> None:
    v = synth_view(n_symbols=8, n_days=760, seed=21, momentum=0.25)
    with cli._session() as s:
        for sym in v.symbols:
            s.add(UniverseMember(universe_name="nifty100", symbol=sym, valid_from=date(2000, 1, 1)))
        s.commit()
    store_view(CandleStore(), v)


def test_train_and_models_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_store()
    real = load_settings
    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: fast_settings())
    r = runner.invoke(cli.app, ["train", "--trials", "1", "--no-news"])
    assert r.exit_code == 0, r.output
    assert "price_only: Sharpe" in r.output and "Nothing was approved." in r.output
    listing = runner.invoke(cli.app, ["models", "list"]).output
    assert "swing_v1" in listing and "candidate" in listing
    assert runner.invoke(cli.app, ["models", "approve", "1"], input="wrong\n").exit_code == 1
    blocked = runner.invoke(cli.app, ["models", "approve", "1"], input="APPROVE swing_v1\n")
    assert (
        blocked.exit_code == 1 and "shadow first" in blocked.output
    )  # the gate process cannot be skipped
    assert runner.invoke(cli.app, ["models", "shadow", "1"]).exit_code == 0
    assert runner.invoke(cli.app, ["models", "shadow", "1"]).exit_code == 1  # already in shadow
    assert runner.invoke(cli.app, ["models", "rollback", "swing_v1"]).exit_code == 1
    assert real is not None


def test_train_errors() -> None:
    assert runner.invoke(cli.app, ["train", "--no-news"]).exit_code == 1  # no stored candles
    assert "No models yet" in runner.invoke(cli.app, ["models", "list"]).output
    assert runner.invoke(
        cli.app, ["train", "--strategy", "intraday_v1", "--no-news"]
    ).exit_code in (1, 2)


def test_strategies_page_renders_and_blocks_unapproved(db: Session, tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    with cli._session() as s:
        m = registry.register_candidate(
            s, "swing_v1", _booster(),
            algo="lightgbm", feature_set_version="fs1", train_start=date(2020, 1, 1), train_end=date(2025, 1, 1),
            params={}, metrics={"gate": {"ready_for_review": True, "checks": [{"name": "min_trades", "passed": True, "value": 120, "threshold": ">= 100"},
                                                                              {"name": "human_approval", "passed": False, "value": None, "threshold": "Pranav"}],
                                         "failed": []}, "importance": {"a": 0.6, "b": 0.4}, "warnings": ["w1"]},
            now=NOW, root=tmp_path,
        )  # fmt: skip
        mid = m.id
    page = Path(__file__).parents[2] / "src/tradepartner/ui/pages/strategies.py"
    at = AppTest.from_file(str(page), default_timeout=30).run()
    assert not at.exception and at.warning
    at.text_input[0].set_value("APPROVE swing_v1")
    at.button[1].click().run()
    assert at.error and "shadow first" in at.error[0].value  # the UI cannot skip shadow either
    at.button[0].click().run()  # start shadow
    with cli._session() as s:
        assert registry.get(s, mid).status == "shadow"
    assert timedelta(0) == timedelta()


def _booster():  # type: ignore[no-untyped-def]
    from tests.unit.test_models_core import _booster as b

    return b()
