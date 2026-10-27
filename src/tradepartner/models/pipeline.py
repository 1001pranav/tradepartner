"""Train -> evaluate -> gate -> register (spec 10.5, 10.6). Nothing here approves a model."""

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from tradepartner.backtest.view import MarketView
from tradepartner.features import build, labels
from tradepartner.models import drift, evaluate, registry, train
from tradepartner.models.registry import ModelError
from tradepartner.settings import Settings

DEFAULTS: dict[str, dict[str, Any]] = {
    "swing_v1": {
        "objective": "binary",
        "horizon_days": 5,
        "top_k": 5,
        "n_trials": 25,
        "test_months": 3,
        "min_train_days": 504,
        "baseline": "ret_20",
    },
    "positional_v1": {
        "objective": "regression",
        "horizon_days": 60,
        "top_k": 10,
        "n_trials": 25,
        "test_months": 3,
        "min_train_days": 504,
        "baseline": "mom_12_1",
    },
}
TOP_FEATURES_FOR_DRIFT = 10


@dataclass
class Variant:
    name: str  # price_only | with_news
    wf: train.WalkForward
    gate: evaluate.GateReport
    quality: dict[str, float]
    llm_period: tuple[str, str] | None = None


@dataclass
class FinalFit:
    booster: lgb.Booster
    start: date
    end: date


@dataclass
class TrainOutcome:
    strategy: str
    variants: dict[str, Variant]
    chosen: str
    model_id: int | None
    report: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def strategy_params(settings: Settings, strategy: str) -> dict[str, Any]:
    if strategy not in DEFAULTS:
        raise ModelError(f"no ML pipeline for {strategy!r}; known: {sorted(DEFAULTS)}")
    cfg = settings.strategies.get(strategy)
    extra = cfg.model_dump() if cfg else {}
    return {**DEFAULTS[strategy], **{k: v for k, v in extra.items() if k in DEFAULTS[strategy]}}


def _labels(view: MarketView, p: dict[str, Any], settings: Settings) -> pd.DataFrame:
    stock = {s: view.history(s, view.trading_days()[-1]) for s in view.symbols}
    if view.index is None:
        raise ModelError("the Nifty 50 index series is required for labels")
    if p["objective"] == "binary":
        cost = labels.round_trip_cost_fraction(settings.costs)
        return labels.swing_labels(stock, view.index, p["horizon_days"], cost + labels.EXTRA_MARGIN)
    return labels.positional_labels(stock, view.index, p["horizon_days"])


def _run_variant(
    name: str,
    matrix: pd.DataFrame,
    lab: pd.DataFrame,
    columns: list[str],
    p: dict[str, Any],
    settings: Settings,
    seed: int,
    only_valid_llm: bool,
) -> tuple[Variant, FinalFit]:
    wf = train.walk_forward(
        matrix,
        lab,
        columns,
        objective=p["objective"],
        horizon=p["horizon_days"],
        n_trials=p["n_trials"],
        seed=seed,
        test_months=p["test_months"],
        min_train_days=p["min_train_days"],
    )
    booster, t0, t1 = train.fit_final(matrix, lab, columns, wf, p["objective"], seed)
    importance = train.gain_share(booster)
    oos = wf.oos
    llm_period = None
    if only_valid_llm and len(oos):
        valid = matrix["llm_features_valid"].reindex(oos.index).fillna(0) == 1
        oos = oos[valid.to_numpy()]
        if len(oos):
            d = oos.index.get_level_values("date")
            llm_period = (str(min(d)), str(max(d)))
    baseline = matrix[p["baseline"]].reindex(oos.index) if len(oos) else pd.Series(dtype=float)
    cost = labels.round_trip_cost_fraction(settings.costs)
    gate = evaluate.evaluate_gate(
        oos,
        baseline,
        importance,
        horizon=p["horizon_days"],
        cost_rt=cost,
        slippage_bps=settings.costs.slippage.default_bps,
        top_k=p["top_k"],
        floor=p.get("min_prob") if p["objective"] == "binary" else None,
    )
    quality = evaluate.prediction_quality(oos, p["objective"] == "binary")
    wf.importance = importance
    return Variant(name, wf, gate, quality, llm_period), FinalFit(booster, t0, t1)


def choose_variant(price: Variant, news: Variant | None) -> tuple[str, str]:
    """With-news wins only if it is ready for review AND beats price-only on the same rows."""
    if news is None:
        return "price_only", "no news variant (no valid LLM-era history yet)"
    if not news.gate.ready_for_review:
        return "price_only", "news variant did not pass the gate"
    sn, sp = news.gate.metrics["sharpe"], price.gate.metrics["sharpe"]
    if sn == sn and (sp != sp or sn > sp):
        return "with_news", f"news Sharpe {sn:.2f} > price-only {sp:.2f} on the LLM-valid period"
    return "price_only", "news did not improve Sharpe on the LLM-valid period"


def train_strategy(
    session: Session,
    strategy: str,
    view: MarketView,
    settings: Settings,
    now: datetime,
    *,
    news_panel: pd.DataFrame | None = None,
    vix: pd.DataFrame | None = None,
    results: dict[str, list[date]] | None = None,
    seed: int = 42,
    survivorship_warning: bool = False,
    register: bool = True,
    models_root: Path | None = None,
    reports_root: Path | None = None,
) -> TrainOutcome:
    p = strategy_params(settings, strategy)
    warnings: list[str] = []
    if survivorship_warning:
        warnings.append(
            "Universe membership is static (survivorship bias): results are optimistic."
        )
    matrix = build.build_matrix(view, vix=vix, results=results, news_panel=news_panel)
    lab = _labels(view, p, settings)

    price, fit_p = _run_variant(
        "price_only", matrix, lab, build.PRICE_ONLY_COLUMNS, p, settings, seed, False
    )
    news: Variant | None = None
    fit_n: FinalFit | None = None
    has_news = news_panel is not None and bool((news_panel["llm_features_valid"] == 1).any())
    if has_news:
        news, fit_n = _run_variant(
            "with_news", matrix, lab, build.ALL_FEATURE_COLUMNS, p, settings, seed, True
        )
    else:
        warnings.append(
            "No LLM-era news features available: only the price-only model was trained."
        )
    chosen, why = choose_variant(price, news)
    use_news = chosen == "with_news" and news is not None and fit_n is not None
    variant = news if use_news and news is not None else price
    final = fit_n if use_news and fit_n is not None else fit_p
    booster, t0, t1 = final.booster, final.start, final.end

    columns = variant.wf.feature_names
    top = sorted(variant.wf.importance, key=lambda k: -variant.wf.importance[k])[
        :TOP_FEATURES_FOR_DRIFT
    ]
    fitted = matrix.loc[matrix.index.get_level_values("date") <= t1, columns]
    metrics: dict[str, Any] = {
        "gate": variant.gate.to_dict(),
        "quality": variant.quality,
        "chosen_variant": chosen,
        "chosen_reason": why,
        "feature_columns": columns,
        "importance": variant.wf.importance,
        "llm_period": variant.llm_period,
        "warnings": warnings,
        "paper_gate_passed": False,
        "feature_bins": {f: drift.feature_bins(fitted[f]) for f in top if fitted[f].notna().any()},
        "train_brier": variant.quality.get("brier"),
        "comparison": {
            v.name: {"gate": v.gate.to_dict(), "quality": v.quality, "llm_period": v.llm_period}
            for v in [price, *([news] if news else [])]
        },
        "horizon_days": p["horizon_days"],
        "objective": p["objective"],
        "top_k": p["top_k"],
    }
    report = {"strategy": strategy, "generated": now.isoformat(), **metrics}
    model_id: int | None = None
    if register:
        row = registry.register_candidate(
            session,
            strategy,
            booster,
            algo="lightgbm",
            feature_set_version=build.FEATURE_SET_VERSION,
            train_start=t0,
            train_end=t1,
            params={**variant.wf.params, "rounds": variant.wf.rounds},
            metrics=json.loads(json.dumps(metrics, default=str)),
            now=now,
            root=models_root,
            notes=why,
        )
        model_id = row.id
    if reports_root is not None:
        reports_root.mkdir(parents=True, exist_ok=True)
        (reports_root / f"train-{strategy}-{now:%Y%m%d-%H%M%S}.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
    return TrainOutcome(
        strategy,
        {price.name: price, **({news.name: news} if news else {})},
        chosen,
        model_id,
        report,
        warnings,
    )


def finite(x: float | None) -> bool:
    return x is not None and bool(np.isfinite(x))
