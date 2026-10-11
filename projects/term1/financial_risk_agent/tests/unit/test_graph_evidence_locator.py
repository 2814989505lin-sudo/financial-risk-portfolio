'''【049】单元测试：验证图谱关系的原文字符定位、哈希绑定与资源上限。'''

from __future__ import annotations

import pytest

from financial_risk_agent.core.schemas import InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.integrations.graph_adapter import (
    ExtractedEntity,
    ExtractedRelation,
    GraphEvidenceLocationError,
    GraphExtractionAdapter,
    GraphExtractionLimitError,
    GraphExtractionLimits,
    GraphExtractionResponse,
)


class StaticProvider:
    """Return configured offline responses and expose calls for retry assertions."""

    def __init__(self, responses: list[GraphExtractionResponse]) -> None:
        self._responses = iter(responses)
        self.calls = 0

    def extract(self, request) -> GraphExtractionResponse:
        self.calls += 1
        return next(self._responses)


def make_snapshot(content: str) -> InputSnapshot:
    return InputSnapshot(
        source_type="sample",
        article_id="article-graph-001",
        title="Bank and vendor article",
        content_ref="sample://article-graph-001",
        content_hash=content_hash(content),
        source_metadata={
            "asset_id": "sample-articles-v1",
            "asset_sha256": "a" * 64,
            "schema_version": "financial_risk_article_sample.v1",
        },
    )


def make_response(*, evidence_end: int) -> GraphExtractionResponse:
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
                evidence_end=evidence_end,
            )
        ],
    )


def test_locates_relation_evidence_and_hashes_the_exact_source_span() -> None:
    content = "Acme Bank issued an invoice to Vendor X."
    adapter = GraphExtractionAdapter(StaticProvider([make_response(evidence_end=len(content))]))

    artifact = adapter.extract(make_snapshot(content), content)

    relation = artifact.relations[0]
    assert relation.evidence_refs[0].locator == f"chars:0:{len(content)}"
    assert relation.evidence_refs[0].excerpt_hash == content_hash(content)
    assert relation.evidence_strength.value == "HIGH"


def test_rejects_unlocatable_evidence_after_the_retry_budget_is_exhausted() -> None:
    content = "Acme Bank issued an invoice to Vendor X."
    adapter = GraphExtractionAdapter(
        StaticProvider([make_response(evidence_end=len(content) + 1)]),
        limits=GraphExtractionLimits(max_attempts=1),
    )

    with pytest.raises(GraphEvidenceLocationError, match="outside"):
        adapter.extract(make_snapshot(content), content)


def test_enforces_input_and_response_limits_and_retries_invalid_response() -> None:
    content = "Acme Bank issued an invoice to Vendor X."
    invalid = make_response(evidence_end=len(content))
    valid = GraphExtractionResponse(
        entities=[ExtractedEntity(entity_id="bank", entity_type="ORG", name="Acme Bank")],
        relations=[],
    )
    provider = StaticProvider([invalid, valid])
    adapter = GraphExtractionAdapter(
        provider,
        limits=GraphExtractionLimits(max_entities=1, max_attempts=2),
    )

    artifact = adapter.extract(make_snapshot(content), content)
    assert provider.calls == 2
    assert [entity.entity_id for entity in artifact.entities] == ["bank"]

    with pytest.raises(GraphExtractionLimitError, match="Input content"):
        GraphExtractionAdapter(
            StaticProvider([valid]), limits=GraphExtractionLimits(max_input_characters=5)
        ).extract(make_snapshot(content), content)


def test_rejects_responses_that_exceed_the_relation_limit() -> None:
    content = "Acme Bank issued an invoice to Vendor X."
    response = GraphExtractionResponse(
        entities=[ExtractedEntity(entity_id="bank", entity_type="ORG", name="Acme Bank")],
        relations=[
            ExtractedRelation(
                relation_id="relation-001",
                source_entity_id="bank",
                target_entity_id="bank",
                predicate="related_to",
                evidence_start=0,
                evidence_end=len(content),
            ),
            ExtractedRelation(
                relation_id="relation-002",
                source_entity_id="bank",
                target_entity_id="bank",
                predicate="related_to",
                evidence_start=0,
                evidence_end=len(content),
            ),
        ],
    )

    with pytest.raises(GraphExtractionLimitError, match="relations"):
        GraphExtractionAdapter(
            StaticProvider([response]),
            limits=GraphExtractionLimits(max_relations=1, max_attempts=1),
        ).extract(make_snapshot(content), content)
