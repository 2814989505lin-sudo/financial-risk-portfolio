'''【037】集成测试：验证任务状态的创建、读取、事务更新和乐观锁。'''

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from financial_risk_agent.core.enums import WorkflowStatus
from financial_risk_agent.core.schemas import InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.task_repository import (
    ConcurrentTaskUpdate,
    TaskAlreadyExists,
    TaskRepository,
)


def make_state() -> WorkflowState:
    now = datetime(2026, 7, 16, 9, 0, tzinfo=UTC)
    return WorkflowState(
        task_id="task-repository-001",
        session_id="session-001",
        correlation_id="correlation-001",
        created_at=now,
        updated_at=now,
        input=InputSnapshot(
            source_type="news",
            article_id="article-001",
            title="Repository test event",
            content_ref="data/samples/article-001.json",
            content_hash=content_hash("Repository test event"),
        ),
    )


def test_creates_and_reads_workflow_state_by_task_id(tmp_path) -> None:
    repository = TaskRepository(SQLiteDatabase(tmp_path / "tasks.sqlite3"))
    created = repository.create(make_state())

    restored = repository.get(created.state.task_id)

    assert restored == created


def test_update_is_transactional_and_uses_optimistic_locking(tmp_path) -> None:
    repository = TaskRepository(SQLiteDatabase(tmp_path / "tasks.sqlite3"))
    created = repository.create(make_state())
    updated_state = created.state.model_copy(
        update={
            "status": WorkflowStatus.VALIDATING_INPUT,
            "current_phase": WorkflowStatus.VALIDATING_INPUT,
            "updated_at": created.state.updated_at + timedelta(seconds=1),
        }
    )

    updated = repository.update(updated_state, expected_version=created.version)

    assert updated.version == 2
    assert repository.get(updated.state.task_id) == updated
    with pytest.raises(ConcurrentTaskUpdate):
        repository.update(updated_state, expected_version=created.version)
    assert repository.get(updated.state.task_id) == updated


def test_rejects_duplicate_task_creation(tmp_path) -> None:
    repository = TaskRepository(SQLiteDatabase(tmp_path / "tasks.sqlite3"))
    repository.create(make_state())

    with pytest.raises(TaskAlreadyExists):
        repository.create(make_state())
