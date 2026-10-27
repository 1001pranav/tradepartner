"""Audit log helper: append-only rows with a content hash (spec 6.5)."""

import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from tradepartner.db.models import AuditLog


def payload_hash(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def log(
    session: Session,
    actor: str,
    action: str,
    payload: dict[str, object],
    at: datetime | None = None,
) -> AuditLog:
    row = AuditLog(
        at=at or datetime.now(UTC),
        actor=actor,
        action=action,
        payload_json=payload,
        payload_hash=payload_hash(payload),
    )
    session.add(row)
    return row
