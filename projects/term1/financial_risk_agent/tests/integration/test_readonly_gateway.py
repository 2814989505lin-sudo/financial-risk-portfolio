'''【069】集成测试：验证只读网关统一校验、执行、超时和安全审计链。'''

from __future__ import annotations

import time
from datetime import UTC, datetime

from pydantic import Field

from financial_risk_agent.core.schemas import InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.services.audit import AuditService
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.audit_repository import AuditRepository
from financial_risk_agent.storage.repositories.task_repository import TaskRepository
from financial_risk_agent.tools.gateway import SafeToolGateway, ToolCallStatus
from financial_risk_agent.tools.registry import ToolCallParameters, ToolRegistry


class EchoParameters(ToolCallParameters):
    message: str = Field(min_length=1, max_length=100)


def _create_gateway(database: SQLiteDatabase) -> tuple[SafeToolGateway, AuditRepository]:
    task = WorkflowState(
        task_id="task-gateway-001",
        session_id="session-001",
        correlation_id="correlation-001",
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        input=InputSnapshot(
            source_type="sample",
            article_id="bootstrap-001",
            title="Gateway test input",
            content_ref="sample://bootstrap-001",
            content_hash=content_hash("Gateway test input"),
        ),
    )
    TaskRepository(database).create(task)
    registry = ToolRegistry()
    registry.register(
        name="echo",
        parameters_model=EchoParameters,
        handler=lambda *, message: {"message": message},
    )
    registry.register(
        name="slow_echo",
        parameters_model=EchoParameters,
        handler=lambda *, message: (time.sleep(0.05), {"message": message})[1],
        timeout_seconds=0.01,
    )
    repository = AuditRepository(database)
    return SafeToolGateway(registry, AuditService(repository)), repository


def test_gateway_validates_executes_times_out_and_audits_every_call(tmp_path) -> None:
    gateway, repository = _create_gateway(SQLiteDatabase(tmp_path / "tasks.sqlite3"))

    success = gateway.invoke(
        task_id="task-gateway-001", tool_name="echo", parameters={"message": "safe"}
    )
    invalid = gateway.invoke(
        task_id="task-gateway-001", tool_name="echo", parameters={"unexpected": "field"}
    )
    timeout = gateway.invoke(
        task_id="task-gateway-001", tool_name="slow_echo", parameters={"message": "slow"}
    )

    assert success.status is ToolCallStatus.SUCCEEDED
    assert success.result == {"message": "safe"}
    assert invalid.status is ToolCallStatus.REJECTED
    assert invalid.error_code == "INVALID_PARAMETERS"
    assert timeout.status is ToolCallStatus.TIMED_OUT
    assert timeout.error_code == "TOOL_TIMEOUT"
    events = repository.list_for_task("task-gateway-001")
    assert [event.event_type for event in events] == [
        "TOOL_CALL_SUCCEEDED",
        "TOOL_CALL_FAILED",
        "TOOL_CALL_FAILED",
    ]
    assert all(event.metadata["input_hash"] for event in events)
    assert all(event.metadata["parameter_summary"] == "top_level_fields=1" for event in events)
    assert all("duration_ms" in event.metadata for event in events)
