'''【091】工作流测试：验证控制器只走预定义状态、持久化检查点并记录每次转换审计。'''

from __future__ import annotations

from datetime import UTC, datetime

from financial_risk_agent.core.enums import WorkflowStatus
from financial_risk_agent.core.schemas import InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.orchestration.state_store import StateStore
from financial_risk_agent.orchestration.team import AnalysisTeam, AnalysisTeamAgents
from financial_risk_agent.orchestration.workflow import WorkflowController
from financial_risk_agent.services.audit import AuditService
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.audit_repository import AuditRepository
from financial_risk_agent.storage.repositories.task_repository import TaskRepository
from tests.workflow.test_analysis_team import RecordingAgents, _input


def _create_controller(tmp_path):
    database = SQLiteDatabase(tmp_path / "tasks.sqlite3")
    repository = TaskRepository(database)
    state = WorkflowState(
        task_id="task-controller-001",
        session_id="session-001",
        correlation_id="correlation-001",
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        input=InputSnapshot(
            source_type="fixture",
            article_id="article-001",
            title="Fixture article",
            content_ref="fixture://article-001",
            content_hash=content_hash("fixture content"),
        ),
    )
    repository.create(state)
    calls: list[str] = []
    agents = RecordingAgents(calls)
    controller = WorkflowController(
        state_store=StateStore(repository),
        audit_service=AuditService(AuditRepository(database)),
        analysis_team=AnalysisTeam(
            AnalysisTeamAgents(
                intake=agents.intake,
                graph=agents.graph,
                rgcn=agents.rgcn,
                compliance=agents.compliance,
                critic=agents.critic,
                report=agents.report,
            )
        ),
    )
    return controller, repository, AuditRepository(database), calls


def test_controller_persists_each_predefined_transition_and_checkpoint(tmp_path) -> None:
    controller, repository, audit_repository, calls = _create_controller(tmp_path)

    result = controller.run(
        "task-controller-001", content="fixture content", graph_contract=_input().graph_contract
    )

    assert calls == ["intake", "graph", "rgcn", "compliance", "critic", "report"]
    assert result.stored_state.state.status is WorkflowStatus.COMPLETED
    restored = repository.get("task-controller-001")
    assert restored == result.stored_state
    assert restored.state.risk_prediction is not None
    assert restored.state.compliance_assessment is not None
    assert restored.state.analysis_checkpoint["report"]
    events = audit_repository.list_for_task("task-controller-001")
    assert [event.event_type for event in events] == ["WORKFLOW_TRANSITIONED"] * 7
    assert AuditService(audit_repository).verify_task_chain("task-controller-001")
