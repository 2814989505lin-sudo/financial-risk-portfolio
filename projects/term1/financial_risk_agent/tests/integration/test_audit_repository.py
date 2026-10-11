'''【033】集成测试：验证审计事件追加、哈希链连续性和篡改检测。'''

from __future__ import annotations

from datetime import UTC, datetime

from financial_risk_agent.core.schemas import InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.services.audit import AuditEventDraft, AuditService
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.audit_repository import AuditRepository
from financial_risk_agent.storage.repositories.task_repository import TaskRepository


def create_task(database: SQLiteDatabase) -> str:
    task = WorkflowState(
        task_id="task-audit-001",
        session_id="session-001",
        correlation_id="correlation-001",
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        input=InputSnapshot(
            source_type="news",
            article_id="article-001",
            title="Audit test event",
            content_ref="data/samples/article-001.json",
            content_hash=content_hash("Audit test event"),
        ),
    )
    TaskRepository(database).create(task)
    return task.task_id


def make_draft(event_type: str, **metadata: object) -> AuditEventDraft:
    return AuditEventDraft(
        event_type=event_type,
        occurred_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        actor_id="coordinator",
        metadata=metadata,
    )


def test_appends_a_continuous_hash_chain_for_a_persisted_task(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "tasks.sqlite3")
    task_id = create_task(database)
    repository = AuditRepository(database)
    service = AuditService(repository)

    first = service.record(task_id, make_draft("TASK_READ", status="CREATED"))
    second = service.record(
        task_id,
        make_draft("STATE_CHANGED", status="VALIDATING_INPUT", phase="VALIDATING_INPUT"),
    )

    assert (first.sequence_number, second.sequence_number) == (1, 2)
    assert second.previous_event_hash == first.event_hash
    assert repository.list_for_task(task_id) == [first, second]
    assert service.verify_task_chain(task_id)
    assert not hasattr(repository, "update")
    assert not hasattr(repository, "delete")


def test_integrity_verification_fails_after_historical_event_tampering(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "tasks.sqlite3")
    task_id = create_task(database)
    service = AuditService(AuditRepository(database))
    service.record(task_id, make_draft("TASK_READ", status="CREATED"))
    service.record(task_id, make_draft("STATE_CHANGED", status="VALIDATING_INPUT"))

    with database.transaction() as connection:
        connection.execute(
            "UPDATE audit_events SET event_type = ? WHERE task_id = ? AND sequence_number = 1",
            ("TAMPERED", task_id),
        )

    assert not service.verify_task_chain(task_id)
