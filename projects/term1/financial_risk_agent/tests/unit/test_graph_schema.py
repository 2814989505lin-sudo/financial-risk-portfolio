'''【042】单元测试：验证图谱 Schema、低证据约束与谱系哈希确定性。'''

from __future__ import annotations

import pytest
from pydantic import ValidationError

from financial_risk_agent.domain.knowledge_graph.models import (
    EvidenceStrength,
    GraphLineage,
    KnowledgeGraphArtifact,
    KnowledgeGraphEntity,
    KnowledgeGraphRelation,
)


def make_lineage() -> GraphLineage:
    return GraphLineage(
        source_asset_id="fixture-graph-v1",
        source_asset_hash="a" * 64,
        source_schema_version="fixture_graph.v1",
        source_document_id="document-001",
    )


def make_artifact() -> KnowledgeGraphArtifact:
    entity = KnowledgeGraphEntity(
        entity_id="entity-001",
        entity_type="ORG",
        name="Acme Bank",
        source_document_refs=["document-001"],
    )
    relation = KnowledgeGraphRelation(
        relation_id="relation-001",
        source_entity_id="entity-001",
        target_entity_id="entity-001",
        predicate="related_to",
        evidence_strength=EvidenceStrength.LOW,
        source_document_refs=["document-001"],
    )
    return KnowledgeGraphArtifact(
        artifact_id="fixture-graph-v1:document-001",
        entities=[entity],
        relations=[relation],
        lineage=make_lineage(),
    )


def test_graph_artifact_hash_and_lineage_hash_are_deterministic() -> None:
    first = make_artifact()
    repeated = make_artifact()

    assert first.artifact_hash == repeated.artifact_hash
    assert first.lineage.lineage_hash == repeated.lineage.lineage_hash
    assert first.relations[0].evidence_strength is EvidenceStrength.LOW
    assert first.relations[0].evidence_refs == []


def test_graph_schema_rejects_invalid_relation_references_and_unbacked_high_evidence() -> None:
    with pytest.raises(ValidationError, match="HIGH evidence"):
        KnowledgeGraphRelation(
            relation_id="relation-001",
            source_entity_id="entity-001",
            target_entity_id="entity-002",
            predicate="related_to",
            evidence_strength=EvidenceStrength.HIGH,
            source_document_refs=["document-001"],
        )

    artifact = make_artifact()
    with pytest.raises(ValidationError, match="source_entity_id"):
        KnowledgeGraphArtifact.model_validate(
            {
                **artifact.model_dump(),
                "relations": [
                    {
                        **artifact.relations[0].model_dump(),
                        "source_entity_id": "missing-entity",
                    }
                ],
                "artifact_hash": None,
            }
        )
