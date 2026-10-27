"""Intraday model training and gate (spec 10.5, M9): 5-minute bars, sign of the next-30-minute
return after costs, LightGBM binary, evaluated walk-forward and gated like every other model.
Nothing here approves anything."""

import json
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from tradepartner.features import labels
from tradepartner.features.intraday import INTRADAY_COLUMNS, intraday_features, intraday_labels
from tradepartner.models import evaluate, registry, train
from tradepartner.models.registry import ModelError
from tradepartner.settings import Settings
from tradepartner.strategies.intraday import NAME

HORIZON_BARS = 6  # 30 minutes
INDEX = ["date", "symbol", "ts"]


def intraday_cost_fraction(settings: Settings) -> float:
    """Round-trip MIS charges + 2 x intraday slippage, as a fraction of notional (from config)."""
    return labels.round_trip_cost_fraction(
        settings.costs, product="MIS", slippage_bps=settings.costs.slippage.intraday_bps
    )


def build_matrix(
    bars: dict[str, pd.DataFrame], cost_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features, labels) indexed by (date, symbol, ts)."""
    feats, labs = [], []
    for sym, df in sorted(bars.items()):
        if len(df) < 50:
            continue
        f = intraday_features(df)
        lab = intraday_labels(df, HORIZON_BARS, cost_fraction)
        dates = df.set_index("ts")["date"]
        for part in (f, lab):
            part["date"] = dates.reindex(part.index).to_numpy()
            part["symbol"] = sym
            part["ts"] = part.index
        feats.append(f)
        labs.append(lab)
    if not feats:
        raise ModelError("no usable 5-minute bars")
    fm = pd.concat(feats).reset_index(drop=True).set_index(INDEX).sort_index()
    lm = pd.concat(labs).reset_index(drop=True).set_index(INDEX).sort_index()
    return fm, lm


def train_intraday(
    session: Session,
    bars: dict[str, pd.DataFrame],
    settings: Settings,
    now: datetime,
    *,
    n_trials: int = 10,
    seed: int = 42,
    test_months: int = 1,
    min_train_days: int = 60,
    top_k: int = 3,
    register: bool = True,
    models_root: object | None = None,
) -> dict[str, object]:
    cost = intraday_cost_fraction(settings)
    fm, lm = build_matrix(bars, cost)
    wf = train.walk_forward(
        fm,
        lm,
        INTRADAY_COLUMNS,
        objective="binary",
        horizon=1,
        n_trials=n_trials,
        seed=seed,
        test_months=test_months,
        min_train_days=min_train_days,
    )
    booster, t0, t1 = train.fit_final(fm, lm, INTRADAY_COLUMNS, wf, "binary", seed)
    importance = train.gain_share(booster)
    oos = wf.oos
    baseline = (fm["or_breakout"] * fm["vol_ratio"].fillna(0)).reindex(oos.index)
    gate = evaluate.evaluate_intraday_gate(
        oos,
        baseline,
        importance,
        hold_bars=HORIZON_BARS,
        cost_rt=cost,
        slippage_bps=settings.costs.slippage.intraday_bps,
        top_k=top_k,
    )
    quality = evaluate.prediction_quality(oos, True)
    report: dict[str, object] = {
        "gate": gate.to_dict(),
        "quality": quality,
        "feature_columns": INTRADAY_COLUMNS,
        "importance": importance,
        "objective": "binary",
        "horizon_days": 0,
        "top_k": top_k,
        "horizon_bars": HORIZON_BARS,
        "paper_gate_passed": False,
        "cost_fraction": cost,
        "warnings": ["Intraday is paper-only until 3 months of paper trading pass the gate."],
    }
    if register:
        row = registry.register_candidate(
            session,
            NAME,
            booster,
            algo="lightgbm",
            feature_set_version="intraday-fs1",
            train_start=t0,
            train_end=t1,
            params={**wf.params, "rounds": wf.rounds},
            metrics=json.loads(json.dumps(report, default=str)),
            now=now,
            notes="intraday: paper only until the paper gate passes",
        )
        report["model_id"] = row.id
    return report
