'''【070】安全测试：验证网关永久拒绝未知、修改、代码执行、Shell、SQL 和用户脚本请求。'''

from __future__ import annotations

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


class SearchParameters(ToolCallParameters):
    query: str = Field(min_length=1, max_length=500)


def _gateway(database: SQLiteDatabase) -> tuple[SafeToolGateway, AuditRepository]:
    task = WorkflowState(
        task_id="task-gateway-security-001",
        session_id="session-001",
        correlation_id="correlation-001",
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        input=InputSnapshot(
            source_type="sample",
            article_id="bootstrap-001",
            title="Gateway security input",
            content_ref="sample://bootstrap-001",
            content_hash=content_hash("Gateway security input"),
        ),
    )
    TaskRepository(database).create(task)
    registry = ToolRegistry()
    registry.register(
        name="search_records",
        parameters_model=SearchParameters,
        handler=lambda *, query: {"query": query},
    )
    repository = AuditRepository(database)
    return SafeToolGateway(registry, AuditService(repository)), repository


def test_gateway_permanently_denies_unsafe_and_unregistered_requests(tmp_path) -> None:
    gateway, repository = _gateway(SQLiteDatabase(tmp_path / "tasks.sqlite3"))
    requests = [
        ("unknown_tool", {"query": "safe"}, "UNKNOWN_TOOL"),
        ("update_case_status", {"case_id": "case-001"}, "MUTATION_TOOL_DENIED"),
        ("eval", {"expression": "1 + 1"}, "PROHIBITED_REQUEST"),
        ("search_records", {"query": "SELECT * FROM cases"}, "PROHIBITED_REQUEST"),
        ("search_records", {"query": "bash -c 'whoami'"}, "PROHIBITED_REQUEST"),
        ("search_records", {"query": "user_script: print('x')"}, "PROHIBITED_REQUEST"),
    ]

    for tool_name, parameters, expected_code in requests:
        result = gateway.invoke(
            task_id="task-gateway-security-001",
            tool_name=tool_name,
            parameters=parameters,
        )
        assert result.status is ToolCallStatus.REJECTED
        assert result.error_code == expected_code
        assert result.result is None

    events = repository.list_for_task("task-gateway-security-001")
    assert len(events) == len(requests)
    assert all(event.event_type == "TOOL_CALL_FAILED" for event in events)
    assert all(event.metadata["status"] == "REJECTED" for event in events)
    assert all("query" not in event.model_dump_json() for event in events)
