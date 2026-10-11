'''【036】集成测试：验证合法任务状态流转可持久化并能在重启后恢复。'''

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from financial_risk_agent.core.enums import WorkflowStatus
from financial_risk_agent.core.schemas import InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.orchestration.state_store import InvalidStateTransition, StateStore
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.task_repository import TaskRepository


def make_state() -> WorkflowState:
    now = datetime(2026, 7, 16, 9, 0, tzinfo=UTC)
    return WorkflowState(
        task_id="task-store-001",
        session_id="session-001",
        correlation_id="correlation-001",
        created_at=now,
        updated_at=now,
        input=InputSnapshot(
            source_type="news",
            article_id="article-001",
            title="State store test event",
            content_ref="data/samples/article-001.json",
            content_hash=content_hash("State store test event"),
        ),
    )


def test_transition_is_persisted_and_survives_store_recreation(tmp_path) -> None:
    database_path = tmp_path / "tasks.sqlite3"
    first_store = StateStore(TaskRepository(SQLiteDatabase(database_path)))
    first_store.create(make_state())

    transitioned = first_store.transition("task-store-001", WorkflowStatus.VALIDATING_INPUT)
    restarted_store = StateStore(TaskRepository(SQLiteDatabase(database_path)))

    restored = restarted_store.load("task-store-001")
    assert restored == transitioned
    assert restored.state.status is WorkflowStatus.VALIDATING_INPUT
    assert restored.state.current_phase is WorkflowStatus.VALIDATING_INPUT


def test_illegal_transition_never_changes_the_persisted_state(tmp_path) -> None:
    store = StateStore(TaskRepository(SQLiteDatabase(tmp_path / "tasks.sqlite3")))
    created = store.create(make_state())

    with pytest.raises(InvalidStateTransition, match="Cannot transition"):
        store.transition(created.state.task_id, WorkflowStatus.COMPLETED)

    assert store.load(created.state.task_id) == created
