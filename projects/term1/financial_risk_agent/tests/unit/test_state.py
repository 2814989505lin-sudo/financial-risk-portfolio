'''【046】单元测试：验证工作流状态校验、枚举完整性和 JSON 往返。'''

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from financial_risk_agent.core.enums import WorkflowStatus
from financial_risk_agent.core.schemas import InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState


def make_workflow_state(**overrides: object) -> WorkflowState:
    now = datetime(2026, 7, 16, 9, 0, tzinfo=UTC)
    values: dict[str, object] = {
        "task_id": "task-001",
        "session_id": "session-001",
        "correlation_id": "correlation-001",
        "input": InputSnapshot(
            source_type="news",
            article_id="article-001",
            title="Example risk event",
            content_ref="data/raw/article-001.json",
            content_hash=content_hash("Example risk event content"),
        ),
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return WorkflowState.model_validate(values)


def test_workflow_status_contains_every_architecture_state() -> None:
    assert {status.value for status in WorkflowStatus} == {
        "CREATED",
        "VALIDATING_INPUT",
        "EXTRACTING_GRAPH",
        "PREDICTING_RISK",
        "ASSESSING_COMPLIANCE",
        "REVIEWING",
        "AWAITING_APPROVAL",
        "EXECUTING",
        "VERIFYING",
        "GENERATING_REPORT",
        "COMPLETED",
        "REJECTED",
        "FAILED",
        "NEEDS_MANUAL_REVIEW",
    }


def test_workflow_state_round_trips_through_json() -> None:
    state = make_workflow_state(
        status=WorkflowStatus.VALIDATING_INPUT,
        current_phase=WorkflowStatus.VALIDATING_INPUT,
        audit_event_refs=["audit-001"],
    )

    restored = WorkflowState.model_validate_json(state.model_dump_json())

    assert restored == state


def test_workflow_state_rejects_blank_task_id_and_unknown_status() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        make_workflow_state(task_id="")

    with pytest.raises(ValidationError, match="Input should be"):
        make_workflow_state(status="UNKNOWN")


def test_workflow_state_rejects_reverse_timestamps() -> None:
    with pytest.raises(ValidationError, match="updated_at"):
        make_workflow_state(
            created_at=datetime(2026, 7, 16, 10, 0, tzinfo=UTC),
            updated_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        )
