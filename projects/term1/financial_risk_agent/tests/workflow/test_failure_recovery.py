'''【093】工作流测试：验证失败仅有限重试，超过上限后持久化为 FAILED 并留下安全错误记录。'''

from __future__ import annotations

from financial_risk_agent.orchestration.state_store import StateStore
from financial_risk_agent.orchestration.team import AnalysisTeam, AnalysisTeamAgents
from financial_risk_agent.orchestration.workflow import WorkflowController
from financial_risk_agent.services.audit import AuditService
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.audit_repository import AuditRepository
from financial_risk_agent.storage.repositories.task_repository import TaskRepository
from tests.workflow.test_analysis_team import RecordingAgents, _input
from tests.workflow.test_state_transitions import _create_controller


class AlwaysFailingIntake:
    allowed_tool_names = ()

    def __init__(self) -> None:
        self.calls = 0

    def run(self, _):
        self.calls += 1
        raise RuntimeError("do not persist this internal detail")


def test_failure_retry_then_retry_exhaustion_transitions_to_failed(tmp_path) -> None:
    _, _, _, _ = _create_controller(tmp_path)
    database = SQLiteDatabase(tmp_path / "tasks.sqlite3")
    failing = AlwaysFailingIntake()
    agents = RecordingAgents([])
    controller = WorkflowController(
        state_store=StateStore(TaskRepository(database)),
        audit_service=AuditService(AuditRepository(database)),
        analysis_team=AnalysisTeam(
            AnalysisTeamAgents(
                intake=failing,
                graph=agents.graph,
                rgcn=agents.rgcn,
                compliance=agents.compliance,
                critic=agents.critic,
                report=agents.report,
            )
        ),
        max_retries_per_phase=1,
    )

    first = controller.run(
        "task-controller-001", content="fixture content", graph_contract=_input().graph_contract
    )
    second = controller.run(
        "task-controller-001", content="fixture content", graph_contract=_input().graph_contract
    )

    assert failing.calls == 2
    assert first.retry_scheduled
    assert first.stored_state.state.status.value == "VALIDATING_INPUT"
    assert second.stored_state.state.status.value == "FAILED"
    assert second.stored_state.state.retry_counters["VALIDATING_INPUT"] == 2
    assert "internal detail" not in second.stored_state.state.error_records[-1].message
