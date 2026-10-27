"""Model registry (spec 6.3, 10.6): versions, statuses and artifacts. Read/write of candidate
and shadow state. Approval and rollback live in ``models.approval`` and are importable only from
``cli`` and ``ui`` (import-linter), so strategy code can never approve a model."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db import audit
from tradepartner.db.models import Model
from tradepartner.paths import models_dir

STATUSES = ("candidate", "shadow", "approved", "retired")


class ModelError(Exception):
    pass


def next_version(session: Session, strategy: str, today: date) -> str:
    n = len(list(session.scalars(select(Model.id).where(Model.strategy == strategy))))
    return f"{today:%Y%m%d}-{n + 1}"


def register_candidate(
    session: Session,
    strategy: str,
    booster: lgb.Booster,
    *,
    algo: str,
    feature_set_version: str,
    train_start: date,
    train_end: date,
    params: dict[str, Any],
    metrics: dict[str, Any],
    now: datetime,
    root: Path | None = None,
    notes: str | None = None,
) -> Model:
    """Save the artifact and add a ``candidate`` row. Never anything but ``candidate``."""
    version = next_version(session, strategy, now.date())
    path = (root or models_dir()) / strategy / f"{version}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    row = Model(
        strategy=strategy, version=version, algo=algo, feature_set_version=feature_set_version,
        train_start=train_start, train_end=train_end, params_json=params, metrics_json=metrics,
        artifact_path=str(path), status="candidate", notes=notes,
    )  # fmt: skip
    session.add(row)
    audit.log(session, "system", "model_candidate", {"strategy": strategy, "version": version}, now)
    session.commit()
    return row


def get(session: Session, model_id: int) -> Model:
    row = session.get(Model, model_id)
    if row is None:
        raise ModelError(f"model {model_id} not found")
    return row


def by_status(session: Session, strategy: str, status: str) -> list[Model]:
    return list(
        session.scalars(
            select(Model)
            .where(Model.strategy == strategy, Model.status == status)
            .order_by(Model.id)
        )
    )


def current_approved(session: Session, strategy: str) -> Model | None:
    rows = by_status(session, strategy, "approved")
    return rows[-1] if rows else None


def load_booster(model: Model) -> lgb.Booster:
    return lgb.Booster(model_file=model.artifact_path)


def start_shadow(session: Session, model_id: int, now: datetime) -> Model:
    """candidate -> shadow: signals are generated and logged but never shown (spec 10.6)."""
    row = get(session, model_id)
    if row.status != "candidate":
        raise ModelError(f"only a candidate can enter shadow (is {row.status})")
    row.status = "shadow"
    row.metrics_json = {**row.metrics_json, "shadow_start": now.isoformat()}
    audit.log(session, "system", "model_shadow", {"model_id": model_id}, now)
    session.commit()
    return row


def record_shadow_comparison(
    session: Session, model_id: int, comparison: dict[str, Any], now: datetime
) -> Model:
    """Store how the shadow model did against the approved one (written by the shadow job)."""
    row = get(session, model_id)
    row.metrics_json = {**row.metrics_json, "shadow_comparison": comparison}
    audit.log(
        session, "system", "model_shadow_comparison", {"model_id": model_id, **comparison}, now
    )
    session.commit()
    return row
