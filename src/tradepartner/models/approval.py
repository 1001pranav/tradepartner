"""Approving and rolling back models. ONLY ``cli`` and ``ui`` may import this module.

The system never self-approves a model (spec 10.5.7, CLAUDE.md rule 3).
"""

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from tradepartner.db import audit
from tradepartner.db.models import Model
from tradepartner.models.registry import ModelError, by_status, current_approved, get

MIN_SHADOW_DAYS = 14


def approval_blockers(model: Model, now: datetime) -> list[str]:
    out: list[str] = []
    gate = model.metrics_json.get("gate", {})
    if not gate.get("ready_for_review"):
        out.append(
            "promotion gate not passed: " + ", ".join(gate.get("failed", ["no gate report"]))
        )
    if model.status != "shadow":
        out.append(f"must run in shadow first (is {model.status})")
    else:
        started = model.metrics_json.get("shadow_start")
        if not started or now < datetime.fromisoformat(started) + timedelta(days=MIN_SHADOW_DAYS):
            out.append(f"needs at least {MIN_SHADOW_DAYS} days in shadow")
        if not model.metrics_json.get("shadow_comparison", {}).get("passed"):
            out.append("shadow comparison against the approved model has not passed")
    return out


def approve(session: Session, model_id: int, actor: str, now: datetime) -> Model:
    if actor != "user":
        raise ModelError("only the user can approve a model")
    model = get(session, model_id)
    blockers = approval_blockers(model, now)
    if blockers:
        raise ModelError("cannot approve: " + "; ".join(blockers))
    previous = current_approved(session, model.strategy)
    if previous is not None:  # kept for instant rollback
        previous.status = "retired"
        previous.metrics_json = {**previous.metrics_json, "replaced_by": model.id}
    model.status = "approved"
    model.approved_by = "user"
    model.approved_at = now
    audit.log(
        session, "user", "model_approved", {"model_id": model.id, "strategy": model.strategy}, now
    )
    session.commit()
    return model


def rollback(session: Session, strategy: str, actor: str, now: datetime) -> Model:
    """Retire the approved model and reinstate the one it replaced."""
    if actor != "user":
        raise ModelError("only the user can roll back a model")
    current = current_approved(session, strategy)
    if current is None:
        raise ModelError("no approved model to roll back")
    replaced = [
        m
        for m in by_status(session, strategy, "retired")
        if m.metrics_json.get("replaced_by") == current.id
    ]
    if not replaced:
        raise ModelError("no previous model to roll back to")
    prev = replaced[-1]
    current.status = "retired"
    prev.status = "approved"
    prev.metrics_json = {k: v for k, v in prev.metrics_json.items() if k != "replaced_by"}
    audit.log(session, "user", "model_rollback", {"from": current.id, "to": prev.id}, now)
    session.commit()
    return prev


def retire(session: Session, model_id: int, actor: str, now: datetime) -> Model:
    if actor != "user":
        raise ModelError("only the user can retire a model")
    model = get(session, model_id)
    model.status = "retired"
    audit.log(session, "user", "model_retired", {"model_id": model_id}, now)
    session.commit()
    return model
