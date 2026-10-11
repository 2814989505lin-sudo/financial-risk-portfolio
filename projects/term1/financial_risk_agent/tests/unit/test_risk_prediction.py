'''【063】单元测试：验证 RGCN 风险概率映射、模型追溯信息与只读元数据输出。'''

from __future__ import annotations

from pathlib import Path

from financial_risk_agent.core.schemas import EvidenceReference
from financial_risk_agent.domain.knowledge_graph.models import (
    EvidenceStrength,
    GraphLineage,
    KnowledgeGraphArtifact,
    KnowledgeGraphEntity,
    KnowledgeGraphRelation,
)
from financial_risk_agent.domain.risk.models import GraphFeatureContract
from financial_risk_agent.integrations.rgcn_adapter import RGCNInferenceAdapter
from financial_risk_agent.tools.readonly.get_model_metadata import GetModelMetadataTool
from financial_risk_agent.tools.readonly.predict_risk_rgcn import PredictRiskRGCNTool

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures/rgcn"


def _graph_contract() -> GraphFeatureContract:
    return GraphFeatureContract.model_validate_json(
        (FIXTURE_ROOT / "compatible_graph_contract.json").read_text(encoding="utf-8")
    )


def _artifact() -> KnowledgeGraphArtifact:
    lineage = GraphLineage(
        source_asset_id="fixture-graph-v1",
        source_asset_hash="a" * 64,
        source_schema_version="knowledge_graph.v1",
        source_document_id="document-001",
    )
    evidence = EvidenceReference(
        source_ref="fixture://document-001",
        locator="chars:0:10",
        excerpt_hash="b" * 64,
    )
    return KnowledgeGraphArtifact(
        artifact_id="fixture-graph-v1:document-001",
        entities=[
            KnowledgeGraphEntity(
                entity_id="org",
                entity_type="ORG",
                name="Acme",
                source_document_refs=["document-001"],
            ),
            KnowledgeGraphEntity(
                entity_id="person",
                entity_type="PERSON",
                name="Alex",
                source_document_refs=["document-001"],
            ),
        ],
        relations=[
            KnowledgeGraphRelation(
                relation_id="relation-001",
                source_entity_id="org",
                target_entity_id="person",
                predicate="related_to",
                evidence_strength=EvidenceStrength.HIGH,
                evidence_refs=[evidence],
                source_document_refs=["document-001"],
            )
        ],
        lineage=lineage,
    )


def test_prediction_has_fixed_probability_mapping_and_deterministic_provenance() -> None:
    adapter = RGCNInferenceAdapter.from_bundle_directory(FIXTURE_ROOT / "trusted_bundle")
    tool = PredictRiskRGCNTool(adapter)

    first = tool.predict(artifact=_artifact(), graph_contract=_graph_contract())
    repeated = tool.predict(artifact=_artifact(), graph_contract=_graph_contract())

    assert first == repeated
    assert all(
        0 <= probability <= 1
        for probability in (
            first.credit_probability,
            first.market_probability,
            first.operational_probability,
            first.fraud_probability,
        )
    )
    assert list(first.thresholds) == ["credit", "market", "operational", "fraud"]
    assert first.model_version == "fixture-rgcn-v1"
    assert first.model_bundle_ref == "model:fixture-rgcn-v1"
    assert first.model_bundle_manifest_hash
    assert first.graph_artifact_hash == _artifact().artifact_hash


def test_model_metadata_exposes_versions_and_thresholds_but_not_checkpoint_contents() -> None:
    adapter = RGCNInferenceAdapter.from_bundle_directory(FIXTURE_ROOT / "trusted_bundle")

    metadata = GetModelMetadataTool(adapter).get()

    assert [category.value for category in metadata.risk_category_order] == [
        "credit",
        "market",
        "operational",
        "fraud",
    ]
    assert metadata.thresholds == {
        "credit": 0.5,
        "market": 0.5,
        "operational": 0.5,
        "fraud": 0.5,
    }
    assert "checkpoint" not in metadata.model_dump_json()
