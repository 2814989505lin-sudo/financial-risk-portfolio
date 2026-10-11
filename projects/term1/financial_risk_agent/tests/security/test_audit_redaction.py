'''【038】安全测试：验证审计事件拒绝密钥与原文内容，仅保留安全元数据。'''

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from financial_risk_agent.core.schemas import InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.services.audit import AuditEventDraft, AuditService
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.audit_repository import AuditRepository
from financial_risk_agent.storage.repositories.task_repository import TaskRepository


def create_audit_service(database: SQLiteDatabase) -> tuple[AuditService, AuditRepository]:
    task = WorkflowState(
        task_id="task-audit-security-001",
        session_id="session-001",
        correlation_id="correlation-001",
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        input=InputSnapshot(
            source_type="news",
            article_id="article-001",
            title="Audit security test",
            content_ref="data/samples/article-001.json",
            content_hash=content_hash("Audit security test"),
        ),
    )
    TaskRepository(database).create(task)
    repository = AuditRepository(database)
    return AuditService(repository), repository


def test_audit_event_rejects_credential_and_raw_content_fields(tmp_path) -> None:
    service, repository = create_audit_service(SQLiteDatabase(tmp_path / "tasks.sqlite3"))

    with pytest.raises(ValueError, match="credentials or secrets"):
        AuditEventDraft(event_type="TASK_READ", metadata={"api_key": "test-secret-value"})
    with pytest.raises(ValueError, match="unsupported or raw-content"):
        AuditEventDraft(event_type="TASK_READ", metadata={"article_body": "full source text"})

    assert repository.list_for_task("task-audit-security-001") == []
    assert not hasattr(service, "record_raw_payload")


def test_stored_audit_event_contains_only_safe_metadata(tmp_path) -> None:
    service, repository = create_audit_service(SQLiteDatabase(tmp_path / "tasks.sqlite3"))
    event = service.record(
        "task-audit-security-001",
        AuditEventDraft(
            event_type="TASK_READ",
            occurred_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
            metadata={"content_hash": "a" * 64, "record_count": 1},
        ),
    )

    serialized = event.model_dump_json()
    assert "api_key" not in serialized
    assert "secret" not in serialized
    assert repository.list_for_task(event.task_id) == [event]
