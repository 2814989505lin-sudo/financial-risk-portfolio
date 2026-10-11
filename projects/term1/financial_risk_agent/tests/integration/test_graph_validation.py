'''【054】集成测试：验证图谱清理记录、降级删除和原图谱不被修改。'''

from __future__ import annotations

from financial_risk_agent.core.schemas import EvidenceReference
from financial_risk_agent.domain.knowledge_graph.models import (
    EvidenceStrength,
    GraphLineage,
    KnowledgeGraphArtifact,
    KnowledgeGraphEntity,
    KnowledgeGraphRelation,
)
from financial_risk_agent.services.graph_validation import (
    GraphCleanupAction,
    GraphValidationService,
)


def make_artifact() -> KnowledgeGraphArtifact:
    lineage = GraphLineage(
        source_asset_id="fixture-graph-v1",
        source_asset_hash="a" * 64,
        source_schema_version="fixture_graph.v1",
        source_document_id="document-001",
    )
    entities = [
        KnowledgeGraphEntity(
            entity_id="company",
            entity_type="ORGANIZATION",
            name="Acme Bank",
            source_document_refs=["document-001"],
        ),
        KnowledgeGraphEntity(
            entity_id="person",
            entity_type="PERSON",
            name="Alice",
            source_document_refs=["document-001"],
        ),
        KnowledgeGraphEntity(
            entity_id="unknown",
            entity_type="ALIEN",
            name="Unknown",
            source_document_refs=["document-001"],
        ),
    ]
    evidence = EvidenceReference(
        source_ref="fixture://document-001",
        locator="chars:0:10",
        excerpt_hash="b" * 64,
    )
    relations = [
        KnowledgeGraphRelation(
            relation_id="downgrade",
            source_entity_id="company",
            target_entity_id="person",
            predicate="related_to",
            evidence_strength=EvidenceStrength.HIGH,
            evidence_refs=[evidence],
            source_document_refs=["document-001"],
            legacy_confidence=0.2,
        ),
        KnowledgeGraphRelation(
            relation_id="bad-type-pair",
            source_entity_id="person",
            target_entity_id="company",
            predicate="issued_invoice_to",
            evidence_strength=EvidenceStrength.LOW,
            source_document_refs=["document-001"],
        ),
        KnowledgeGraphRelation(
            relation_id="missing-entity",
            source_entity_id="company",
            target_entity_id="unknown",
            predicate="related_to",
            evidence_strength=EvidenceStrength.LOW,
            source_document_refs=["document-001"],
        ),
        KnowledgeGraphRelation(
            relation_id="bad-predicate",
            source_entity_id="company",
            target_entity_id="person",
            predicate="invented_predicate",
            evidence_strength=EvidenceStrength.LOW,
            source_document_refs=["document-001"],
        ),
    ]
    return KnowledgeGraphArtifact(
        artifact_id="fixture-graph-v1:document-001",
        entities=entities,
        relations=relations,
        lineage=lineage,
    )


def test_validation_returns_cleaned_copy_and_complete_cleanup_records() -> None:
    original = make_artifact()
    original_hash = original.artifact_hash

    result = GraphValidationService().validate(original)

    assert {entity.entity_id for entity in result.validated_artifact.entities} == {
        "company",
        "person",
    }
    assert result.validated_artifact.entities[0].entity_type == "ORG"
    assert [relation.relation_id for relation in result.validated_artifact.relations] == [
        "downgrade"
    ]
    assert result.validated_artifact.relations[0].evidence_strength is EvidenceStrength.LOW
    assert original.artifact_hash == original_hash
    assert original.entities[0].entity_type == "ORGANIZATION"
    assert original.relations[0].evidence_strength is EvidenceStrength.HIGH
    assert [record.action for record in result.cleanup_records] == [
        GraphCleanupAction.CORRECT_ENTITY_TYPE,
        GraphCleanupAction.DELETE_ENTITY,
        GraphCleanupAction.DOWNGRADE_RELATION,
        GraphCleanupAction.DELETE_RELATION,
        GraphCleanupAction.DELETE_RELATION,
        GraphCleanupAction.DELETE_RELATION,
    ]


def test_validation_result_is_deterministic_for_a_fixed_input() -> None:
    first = GraphValidationService().validate(make_artifact())
    repeated = GraphValidationService().validate(make_artifact())

    assert first.validated_artifact.artifact_hash == repeated.validated_artifact.artifact_hash
    assert first.cleanup_records == repeated.cleanup_records
    assert first.validation_hash == repeated.validation_hash
