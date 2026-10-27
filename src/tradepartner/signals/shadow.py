"""Shadow comparison (spec 10.6): a candidate's logged-but-hidden signals vs the approved model."""

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from tradepartner.db.models import Model
from tradepartner.models import registry
from tradepartner.signals.outcomes import outcome_frame

MIN_OUTCOMES = 20


def _stats(df: Any) -> dict[str, float]:
    return {
        "n": float(len(df)),
        "mean_excess": float(df["excess"].mean()) if len(df) else float("nan"),
        "hit_rate": float((df["excess"] > 0).mean()) if len(df) else float("nan"),
    }


def shadow_comparison(
    session: Session,
    candidate: Model,
    approved: Model | None,
    now: datetime,
    min_outcomes: int = MIN_OUTCOMES,
) -> dict[str, Any]:
    """Compare revealed excess returns of the shadow signals with the approved model's signals
    created over the same period. Passes only with enough outcomes and no worse performance."""
    start = candidate.metrics_json.get("shadow_start")
    cand = outcome_frame(session, candidate.strategy, candidate.id)
    if start:
        cand = cand[cand["created_at"] >= datetime.fromisoformat(start)]
    cstats = _stats(cand)
    result: dict[str, Any] = {"candidate": cstats, "min_outcomes": min_outcomes}
    if approved is None:
        result["approved"] = None
        result["passed"] = bool(cstats["n"] >= min_outcomes and cstats["mean_excess"] > 0)
    else:
        base = outcome_frame(session, approved.strategy, approved.id)
        if start:
            base = base[base["created_at"] >= datetime.fromisoformat(start)]
        astats = _stats(base)
        result["approved"] = astats
        result["passed"] = bool(
            cstats["n"] >= min_outcomes
            and cstats["mean_excess"] == cstats["mean_excess"]
            and (astats["n"] == 0 or cstats["mean_excess"] >= astats["mean_excess"])
        )
    registry.record_shadow_comparison(session, candidate.id, result, now)
    return result
