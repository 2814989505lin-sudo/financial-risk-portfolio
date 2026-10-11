'''【102】端到端测试：以模拟只读服务验证案例选择、任务创建、分析启动和刷新后的结果展示。'''

from __future__ import annotations

from datetime import UTC, datetime

from app.pages.case_analysis import (
    CaseAnalysisTask,
    SampleCase,
    render_case_analysis,
)
from financial_risk_agent.agents.graph import GraphAgentResult
from financial_risk_agent.agents.report import ReportDraft
from financial_risk_agent.agents.rgcn import RGCNAgentResult
from financial_risk_agent.core.schemas import InputSnapshot, RiskPrediction
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.domain.knowledge_graph.models import (
    GraphLineage,
    KnowledgeGraphArtifact,
    KnowledgeGraphEntity,
)
from financial_risk_agent.orchestration.team import AnalysisTeamResult


class FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, str] = {}
        self.clicked = {"创建只读分析任务", "启动或继续只读分析"}
        self.writes: list[str] = []
        self.button_labels: list[str] = []

    def title(self, value: str) -> None:
        self.writes.append(value)

    def subheader(self, value: str) -> None:
        self.writes.append(value)

    def selectbox(self, _, *, options, format_func):
        self.writes.append(format_func(options[0]))
        return options[0]

    def button(self, label: str) -> bool:
        self.button_labels.append(label)
        if label not in self.clicked:
            return False
        self.clicked.remove(label)
        return True

    def write(self, value) -> None:
        self.writes.append(str(value))

    def warning(self, value: str) -> None:
        self.writes.append(value)

    def info(self, value: str) -> None:
        self.writes.append(value)


class FakeCaseAnalysisService:
    def __init__(self) -> None:
        self.created_case_ids: list[str] = []
        self.started_task_ids: list[str] = []
        self.tasks: dict[str, CaseAnalysisTask] = {}

    def list_sample_cases(self) -> list[SampleCase]:
        return [
            SampleCase(
                case_id="sample-001",
                title="Sample fraud case",
                source_ref="fixture://sample-001",
            )
        ]

    def create_task(self, case_id: str) -> CaseAnalysisTask:
        self.created_case_ids.append(case_id)
        task = CaseAnalysisTask(state=_state("CREATED"))
        self.tasks[task.state.task_id] = task
        return task

    def get_task(self, task_id: str) -> CaseAnalysisTask:
        return self.tasks[task_id]

    def start_or_resume_readonly_analysis(self, task_id: str) -> CaseAnalysisTask:
        self.started_task_ids.append(task_id)
        task = CaseAnalysisTask(state=_state("COMPLETED"), analysis_result=_analysis_result())
        self.tasks[task_id] = task
        return task


def _state(status: str) -> WorkflowState:
    return WorkflowState(
        task_id="task-ui-e2e-001",
        session_id="session-001",
        correlation_id="correlation-001",
        status=status,
        current_phase=status,
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 16, 9, 1, tzinfo=UTC),
        input=InputSnapshot(
            source_type="fixture",
            article_id="sample-001",
            title="Sample fraud case",
            content_ref="fixture://sample-001",
            content_hash=content_hash("fixture content"),
        ),
    )


def _analysis_result() -> AnalysisTeamResult:
    artifact = KnowledgeGraphArtifact(
        artifact_id="graph-e2e-001",
        entities=[
            KnowledgeGraphEntity(
                entity_id="entity-001",
                entity_type="ORG",
                name="Acme",
                source_document_refs=["sample-001"],
            )
        ],
        lineage=GraphLineage(
            source_asset_id="fixture-asset",
            source_asset_hash="a" * 64,
            source_schema_version="knowledge_graph.v1",
            source_document_id="sample-001",
        ),
    )
    prediction = RiskPrediction(
        model_version="fixture-rgcn-v1",
        graph_artifact_hash=artifact.artifact_hash,
        credit_probability=0.7,
        market_probability=0.2,
        operational_probability=0.1,
        fraud_probability=0.4,
    )
    return AnalysisTeamResult(
        graph=GraphAgentResult(artifact=artifact),
        rgcn=RGCNAgentResult(prediction=prediction),
        report=ReportDraft(
            report_id="report:task-ui-e2e-001",
            article_ref="fixture://sample-001",
            executive_summary="Readonly sample report.",
            critic_accepted=True,
        ),
    )


def test_case_analysis_page_creates_starts_and_refreshes_a_readonly_task() -> None:
    st = FakeStreamlit()
    service = FakeCaseAnalysisService()

    task = render_case_analysis(st, service)
    refreshed = render_case_analysis(st, service)

    assert service.created_case_ids == ["sample-001"]
    assert service.started_task_ids == ["task-ui-e2e-001"]
    assert task and task.analysis_result is not None
    assert refreshed and refreshed.state.status.value == "COMPLETED"
    assert "credit：70.00%" in st.writes
    assert "Readonly sample report." in st.writes
    assert set(st.button_labels) == {"创建只读分析任务", "启动或继续只读分析"}
