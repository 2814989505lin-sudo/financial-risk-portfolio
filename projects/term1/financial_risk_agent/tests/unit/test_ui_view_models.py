'''【101】单元测试：验证只读界面的任务状态、图谱、风险概率、证据和报告视图模型。'''

from __future__ import annotations

from datetime import UTC, datetime

from app.components.report_view import ReportViewModel
from app.components.task_status import TaskStatusViewModel
from financial_risk_agent.agents.graph import GraphAgentResult
from financial_risk_agent.agents.report import ReportDraft
from financial_risk_agent.agents.rgcn import RGCNAgentResult
from financial_risk_agent.core.schemas import ErrorRecord, InputSnapshot, RiskPrediction
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.domain.knowledge_graph.models import (
    GraphLineage,
    KnowledgeGraphArtifact,
    KnowledgeGraphEntity,
)
from financial_risk_agent.orchestration.team import AnalysisTeamResult


def _state() -> WorkflowState:
    return WorkflowState(
        task_id="task-ui-001",
        session_id="session-001",
        correlation_id="correlation-001",
        status="PREDICTING_RISK",
        current_phase="PREDICTING_RISK",
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 16, 9, 1, tzinfo=UTC),
        input=InputSnapshot(
            source_type="fixture",
            article_id="article-001",
            title="Fixture case",
            content_ref="fixture://article-001",
            content_hash=content_hash("fixture content"),
        ),
        retry_counters={"PREDICTING_RISK": 1},
        error_records=[
            ErrorRecord(
                code="AGENT_TIMEOUT",
                message="internal detail is not exposed by the view model",
                occurred_at=datetime(2026, 7, 16, 9, 1, tzinfo=UTC),
            )
        ],
    )


def _analysis_result() -> AnalysisTeamResult:
    artifact = KnowledgeGraphArtifact(
        artifact_id="graph-001",
        entities=[
            KnowledgeGraphEntity(
                entity_id="entity-001",
                entity_type="ORG",
                name="Acme",
                source_document_refs=["article-001"],
            )
        ],
        lineage=GraphLineage(
            source_asset_id="fixture-asset",
            source_asset_hash="a" * 64,
            source_schema_version="knowledge_graph.v1",
            source_document_id="article-001",
        ),
    )
    prediction = RiskPrediction(
        model_version="fixture-rgcn-v1",
        graph_artifact_hash=artifact.artifact_hash,
        credit_probability=0.8,
        market_probability=0.2,
        operational_probability=0.1,
        fraud_probability=0.3,
        thresholds={"credit": 0.5, "market": 0.5, "operational": 0.5, "fraud": 0.5},
    )
    return AnalysisTeamResult(
        graph=GraphAgentResult(artifact=artifact),
        rgcn=RGCNAgentResult(prediction=prediction),
        report=ReportDraft(
            report_id="report:task-ui-001",
            article_ref="fixture://article-001",
            executive_summary="Risk summary.",
            conclusions=["Credit risk is elevated."],
            recommended_next_steps=["Request human review."],
            critic_accepted=True,
        ),
    )


def test_task_status_view_model_excludes_internal_error_message() -> None:
    view_model = TaskStatusViewModel.from_state(_state())

    assert view_model.status == "PREDICTING_RISK"
    assert view_model.retry_count == 1
    assert view_model.latest_error_code == "AGENT_TIMEOUT"
    assert "internal detail" not in view_model.model_dump_json()


def test_report_view_model_displays_graph_probabilities_and_report_draft() -> None:
    view_model = ReportViewModel.from_analysis_result(_analysis_result())

    assert view_model.graph_entity_count == 1
    assert view_model.graph_relation_count == 0
    assert [risk.category for risk in view_model.risk_probabilities] == [
        "credit",
        "market",
        "operational",
        "fraud",
    ]
    assert view_model.risk_probabilities[0].probability == 0.8
    assert view_model.report_summary == "Risk summary."
    assert view_model.critic_accepted
