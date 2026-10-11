'''【050】集成测试：验证模拟 Provider 的抽取契约、审计写入和原文不入日志。'''

from __future__ import annotations

from datetime import UTC, datetime

from financial_risk_agent.core.schemas import InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.integrations.graph_adapter import (
    ExtractedEntity,
    ExtractedRelation,
    GraphExtractionAdapter,
    GraphExtractionResponse,
)
from financial_risk_agent.services.audit import AuditService
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.audit_repository import AuditRepository
from financial_risk_agent.storage.repositories.task_repository import TaskRepository
from financial_risk_agent.tools.readonly.extract_graph import ExtractGraphTool


class SimulatedProvider:
    """A deterministic local Provider used to prove the contract runs without an LLM."""

    def extract(self, request) -> GraphExtractionResponse:
        content = request.content
        return GraphExtractionResponse(
            entities=[
                ExtractedEntity(entity_id="bank", entity_type="ORG", name="Acme Bank"),
                ExtractedEntity(entity_id="vendor", entity_type="ORG", name="Vendor X"),
            ],
            relations=[
                ExtractedRelation(
                    relation_id="bank-issued-invoice",
                    source_entity_id="bank",
                    target_entity_id="vendor",
                    predicate="issued_invoice_to",
                    evidence_start=0,
                    evidence_end=len(content),
                )
            ],
        )


def make_snapshot(content: str) -> InputSnapshot:
    return InputSnapshot(
        source_type="sample",
        article_id="article-contract-001",
        title="Bank and vendor article",
        content_ref="sample://article-contract-001",
        content_hash=content_hash(content),
        source_metadata={
            "asset_id": "sample-articles-v1",
            "asset_sha256": "a" * 64,
            "schema_version": "financial_risk_article_sample.v1",
        },
    )


def create_task(database: SQLiteDatabase, snapshot: InputSnapshot) -> str:
    task = WorkflowState(
        task_id="task-graph-contract-001",
        session_id="session-001",
        correlation_id="correlation-001",
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        input=snapshot,
    )
    TaskRepository(database).create(task)
    return task.task_id


def test_offline_extraction_contract_builds_a_graph_and_audits_only_metadata(tmp_path) -> None:
    content = "Acme Bank issued an invoice to Vendor X."
    snapshot = make_snapshot(content)
    database = SQLiteDatabase(tmp_path / "tasks.sqlite3")
    task_id = create_task(database, snapshot)
    audit_service = AuditService(AuditRepository(database))
    tool = ExtractGraphTool(GraphExtractionAdapter(SimulatedProvider()), audit_service)

    artifact = tool.extract(task_id=task_id, input_snapshot=snapshot, content=content)

    assert artifact.relations[0].evidence_refs[0].excerpt_hash == content_hash(content)
    assert artifact.relations[0].evidence_refs[0].locator == f"chars:0:{len(content)}"
    events = AuditRepository(database).list_for_task(task_id)
    assert [event.event_type for event in events] == ["GRAPH_EXTRACTED"]
    assert content not in events[0].model_dump_json()
    assert audit_service.verify_task_chain(task_id)
