'''【041】单元测试：验证审计事件哈希确定性与链位置校验。'''

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from financial_risk_agent.services.audit import (
    AuditEvent,
    AuditEventDraft,
    audit_event_hash,
    build_audit_event,
)


def make_draft(**overrides: object) -> AuditEventDraft:
    values: dict[str, object] = {
        "event_type": "STATE_CHANGED",
        "occurred_at": datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        "actor_id": "coordinator",
        "resource_ref": "task/task-audit-001",
        "metadata": {"status": "VALIDATING_INPUT", "content_hash": "a" * 64},
    }
    values.update(overrides)
    return AuditEventDraft.model_validate(values)


def test_audit_event_hash_is_deterministic_and_binds_the_payload() -> None:
    first = build_audit_event(
        task_id="task-audit-001",
        sequence_number=1,
        draft=make_draft(),
        previous_event_hash=None,
    )
    repeated = build_audit_event(
        task_id="task-audit-001",
        sequence_number=1,
        draft=make_draft(),
        previous_event_hash=None,
    )
    changed = build_audit_event(
        task_id="task-audit-001",
        sequence_number=1,
        draft=make_draft(metadata={"status": "FAILED"}),
        previous_event_hash=None,
    )

    assert first.event_hash == repeated.event_hash
    assert first.event_hash != changed.event_hash
    assert first.event_hash == audit_event_hash(first.hash_payload())


def test_audit_event_rejects_an_invalid_hash_or_chain_position() -> None:
    first = build_audit_event(
        task_id="task-audit-001",
        sequence_number=1,
        draft=make_draft(),
        previous_event_hash=None,
    )

    with pytest.raises(ValidationError, match="event_hash"):
        AuditEvent.model_validate({**first.model_dump(), "event_hash": "b" * 64})

    with pytest.raises(ValidationError, match="previous event hash"):
        AuditEvent.model_validate({**first.model_dump(), "previous_event_hash": "a" * 64})
