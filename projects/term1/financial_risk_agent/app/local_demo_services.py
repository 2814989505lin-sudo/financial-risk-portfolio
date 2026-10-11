'''【135】本地样例服务：为未注入部署服务时的 Streamlit 提供可直接运行的确定性演示分析。'''

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.pages.approvals import ApprovalUIError, PendingApproval
from app.pages.case_analysis import CaseAnalysisTask, SampleCase
from financial_risk_agent.agents.graph import GraphAgentResult
from financial_risk_agent.agents.report import ReportDraft
from financial_risk_agent.agents.rgcn import RGCNAgentResult
from financial_risk_agent.core.enums import ApprovalDecision, WorkflowStatus
from financial_risk_agent.core.schemas import ApprovalRecord, InputSnapshot, RiskPrediction
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.domain.knowledge_graph.models import (
    GraphLineage,
    KnowledgeGraphArtifact,
    KnowledgeGraphEntity,
)
from financial_risk_agent.orchestration.team import AnalysisTeamResult
from financial_risk_agent.services.audit import AuditEvent


class LocalDemoServices:
    """Supply safe, session-local sample analysis data when no deployment host is configured."""

    def __init__(self) -> None:
        self._articles = _load_sample_articles()
        self._tasks: dict[str, CaseAnalysisTask] = {}

    def list_sample_cases(self) -> list[SampleCase]:
        return [
            SampleCase(
                case_id=item["article_id"], title=item["title"], source_ref=item["source_url"]
            )
            for item in self._articles
        ]

    def create_task(self, case_id: str) -> CaseAnalysisTask:
        article = self._article(case_id)
        task_id = f"local-demo-{uuid4().hex}"
        now = datetime.now(UTC)
        task = CaseAnalysisTask(
            state=WorkflowState(
                task_id=task_id,
                session_id="local-demo-session",
                correlation_id=f"local-demo-{uuid4().hex}",
                status=WorkflowStatus.CREATED,
                current_phase=WorkflowStatus.CREATED,
                created_at=now,
                updated_at=now,
                input=InputSnapshot(
                    source_type="local-demo-sample",
                    article_id=article["article_id"],
                    title=article["title"],
                    content_ref=article["source_url"],
                    content_hash=content_hash(article["content"]),
                ),
            )
        )
        self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> CaseAnalysisTask:
        try:
            return self._tasks[task_id]
        except KeyError as error:
            raise ValueError("Unknown local demo task.") from error

    def start_or_resume_readonly_analysis(self, task_id: str) -> CaseAnalysisTask:
        task = self.get_task(task_id)
        article = self._article(task.state.input.article_id)
        completed = CaseAnalysisTask(
            state=task.state.model_copy(
                update={
                    "status": WorkflowStatus.COMPLETED,
                    "current_phase": WorkflowStatus.COMPLETED,
                    "updated_at": datetime.now(UTC),
                }
            ),
            analysis_result=_demo_analysis(task.state.task_id, task.state.input, article),
        )
        self._tasks[task_id] = completed
        return completed

    def list_task_history(self) -> list[WorkflowState]:
        return [task.state for task in reversed(self._tasks.values())]

    def list_audit_task_ids(self) -> list[str]:
        return []

    def list_audit_events(self, task_id: str, *, event_type: str | None = None) -> list[AuditEvent]:
        return []

    def list_pending_approvals(self) -> list[PendingApproval]:
        return []

    def get_pending_approval(self, task_id: str, action_id: str) -> PendingApproval | None:
        return None

    def decide(
        self,
        *,
        task_id: str,
        action_id: str,
        proposal_hash: str,
        approver_id: str,
        decision: ApprovalDecision,
        comment: str | None = None,
    ) -> ApprovalRecord:
        raise ApprovalUIError("Local demo mode does not allow approval decisions.")

    def _article(self, case_id: str) -> dict[str, str]:
        for article in self._articles:
            if article["article_id"] == case_id:
                return article
        raise ValueError("Unknown local demo sample.")


def _load_sample_articles() -> list[dict[str, str]]:
    path = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_articles.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        {key: value for key, value in item.items() if isinstance(value, str)}
        for item in payload
        if isinstance(item, dict)
    ]


def _demo_analysis(
    task_id: str,
    snapshot: InputSnapshot,
    article: dict[str, str],
) -> AnalysisTeamResult:
    artifact = KnowledgeGraphArtifact(
        artifact_id=f"local-demo-graph:{task_id}",
        entities=[
            KnowledgeGraphEntity(
                entity_id="local-demo-entity",
                entity_type="ORG",
                name="本地样例主体",
                source_document_refs=[snapshot.article_id],
            )
        ],
        lineage=GraphLineage(
            source_asset_id="local-demo-samples",
            source_asset_hash=content_hash(article["content"]),
            source_schema_version="local-demo.v1",
            source_document_id=snapshot.article_id,
        ),
    )
    prediction = RiskPrediction(
        model_version="local-demo-deterministic-v1",
        graph_artifact_hash=artifact.artifact_hash or "",
        credit_probability=0.35,
        market_probability=0.15,
        operational_probability=0.25,
        fraud_probability=0.45,
    )
    return AnalysisTeamResult(
        graph=GraphAgentResult(artifact=artifact),
        rgcn=RGCNAgentResult(prediction=prediction),
        report=ReportDraft(
            report_id=f"local-demo-report:{task_id}",
            article_ref=snapshot.content_ref,
            executive_summary="这是本地确定性样例报告，不代表生产模型或真实风险结论。",
            conclusions=["已加载内置新闻样例。", "风险概率仅用于演示界面流程。"],
            recommended_next_steps=["接入生产案例服务后运行真实分析。"],
            critic_accepted=True,
        ),
    )
