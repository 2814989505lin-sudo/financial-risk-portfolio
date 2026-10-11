'''【092】工作流测试：验证暂停后的持久化检查点可在新控制器中恢复且不重复已完成阶段。'''

from __future__ import annotations

from financial_risk_agent.orchestration.selector import AnalysisStep
from financial_risk_agent.orchestration.state_store import StateStore
from financial_risk_agent.orchestration.team import AnalysisTeam, AnalysisTeamAgents
from financial_risk_agent.orchestration.workflow import WorkflowController
from financial_risk_agent.services.audit import AuditService
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.audit_repository import AuditRepository
from financial_risk_agent.storage.repositories.task_repository import TaskRepository
from tests.workflow.test_analysis_team import RecordingAgents, _input
from tests.workflow.test_state_transitions import _create_controller


def test_pause_resume_uses_persisted_checkpoint_without_repeating_completed_agents(
    tmp_path,
) -> None:
    controller, repository, _, initial_calls = _create_controller(tmp_path)

    paused = controller.run(
        "task-controller-001",
        content="fixture content",
        graph_contract=_input().graph_contract,
        pause_after_step=AnalysisStep.GRAPH,
    )

    assert paused.paused
    assert initial_calls == ["intake", "graph"]
    assert paused.stored_state.state.analysis_checkpoint["graph"]

    database = SQLiteDatabase(tmp_path / "tasks.sqlite3")
    resumed_calls: list[str] = []
    agents = RecordingAgents(resumed_calls)
    resumed = WorkflowController(
        state_store=StateStore(TaskRepository(database)),
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
    ).run("task-controller-001", content="fixture content", graph_contract=_input().graph_contract)

    assert resumed_calls == ["rgcn", "compliance", "critic", "report"]
    assert resumed.stored_state.state.status.value == "COMPLETED"
