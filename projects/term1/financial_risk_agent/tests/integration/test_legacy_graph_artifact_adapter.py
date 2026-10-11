'''【035】集成测试：验证两类历史图谱的确定性转换、低证据标记和只读行为。'''

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from financial_risk_agent.core.schemas import InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.domain.knowledge_graph.models import EvidenceStrength
from financial_risk_agent.integrations.legacy.graph_artifact_adapter import (
    GraphAsset,
    LegacyGraphArtifactAdapter,
    load_graph_asset,
)
from financial_risk_agent.services.audit import AuditService
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.audit_repository import AuditRepository
from financial_risk_agent.storage.repositories.task_repository import TaskRepository
from financial_risk_agent.tools.readonly.load_graph_artifact import LoadGraphArtifactTool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/legacy_assets.json"
FIXTURE_ROOT = PROJECT_ROOT / "tests/fixtures/graph_cases"


def fixture_asset(name: str, schema_version: str) -> GraphAsset:
    path = FIXTURE_ROOT / name
    return GraphAsset(
        asset_id=f"fixture-{path.stem}",
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        schema_version=schema_version,
    )


def create_task(database: SQLiteDatabase) -> str:
    task = WorkflowState(
        task_id="task-graph-tools-001",
        session_id="session-001",
        correlation_id="correlation-001",
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        input=InputSnapshot(
            source_type="sample",
            article_id="bootstrap-graph-001",
            title="Bootstrap graph task input",
            content_ref="sample://bootstrap-graph-001",
            content_hash=content_hash("Bootstrap graph task input"),
        ),
    )
    TaskRepository(database).create(task)
    return task.task_id


def test_converts_document_graph_with_low_evidence_and_deterministic_hashes() -> None:
    adapter = LegacyGraphArtifactAdapter(
        fixture_asset("document_graph.json", "document_graph_multilabel.v1")
    )

    first = adapter.convert("fixture-document-001")
    repeated = adapter.convert("fixture-document-001")

    assert len(first.entities) == 3
    assert len(first.relations) == 2
    assert all(relation.evidence_strength is EvidenceStrength.LOW for relation in first.relations)
    assert all(not relation.evidence_refs for relation in first.relations)
    assert first.artifact_hash == repeated.artifact_hash
    assert first.lineage.lineage_hash == repeated.lineage.lineage_hash


def test_converts_nodes_edges_docs_and_audits_read_only_tool_access(tmp_path) -> None:
    adapter = LegacyGraphArtifactAdapter(
        fixture_asset("nodes_edges_docs.json", "legacy_nodes_edges_docs.v1")
    )
    database = SQLiteDatabase(tmp_path / "tasks.sqlite3")
    task_id = create_task(database)
    audit_service = AuditService(AuditRepository(database))

    artifact = LoadGraphArtifactTool(adapter, audit_service).load(
        task_id=task_id,
        document_id="fixture-nodes-001",
    )

    relation = artifact.relations[0]
    assert {entity.entity_id for entity in artifact.entities} == {"entity:bank", "entity:vendor"}
    assert relation.predicate == "co_occurs_in_sentence"
    assert relation.evidence_strength is EvidenceStrength.LOW
    assert relation.evidence_refs == []
    events = AuditRepository(database).list_for_task(task_id)
    assert [event.event_type for event in events] == ["GRAPH_ARTIFACT_LOADED"]
    assert audit_service.verify_task_chain(task_id)


def test_manifest_registered_document_graph_is_converted_without_source_mutation() -> None:
    asset = load_graph_asset(MANIFEST_PATH, "document-graph-train-v1")
    before_hash = hashlib.sha256(asset.path.read_bytes()).hexdigest()

    artifact = LegacyGraphArtifactAdapter(asset).convert(
        "business/2026/apr/12/gsk-promising-early-results-ovarian-womb-cancer-drug-trial"
    )

    assert artifact.entities
    assert artifact.relations
    assert artifact.lineage.source_asset_hash == asset.sha256
    assert hashlib.sha256(asset.path.read_bytes()).hexdigest() == before_hash


def test_manifest_registered_nodes_edges_docs_graph_marks_missing_evidence_as_low() -> None:
    asset = load_graph_asset(MANIFEST_PATH, "legacy-kg-train-clean-v1")
    before_hash = hashlib.sha256(asset.path.read_bytes()).hexdigest()

    artifact = LegacyGraphArtifactAdapter(asset).convert(
        "society/2025/oct/31/woman-flight-italy-did-not-board-child-benefits-stopped"
    )

    assert artifact.entities
    assert artifact.relations
    assert all(
        relation.evidence_strength is EvidenceStrength.LOW for relation in artifact.relations
    )
    assert all(not relation.evidence_refs for relation in artifact.relations)
    assert hashlib.sha256(asset.path.read_bytes()).hexdigest() == before_hash
