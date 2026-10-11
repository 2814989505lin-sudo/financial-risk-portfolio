'''【053】单元测试：验证图谱实体、谓词、类型组合和强关系置信度规则。'''

from __future__ import annotations

from financial_risk_agent.domain.knowledge_graph.models import EvidenceStrength
from financial_risk_agent.domain.knowledge_graph.rules import (
    canonical_entity_type,
    has_valid_entity_type_pair,
    is_allowed_entity_type,
    is_allowed_predicate,
    requires_confidence_downgrade,
)


def test_canonical_entity_types_are_deterministic_and_unknown_types_are_rejected() -> None:
    assert canonical_entity_type("organization") == "ORG"
    assert canonical_entity_type("geo political entity") == "GPE"
    assert is_allowed_entity_type("COMPANY")
    assert not is_allowed_entity_type("ALIEN")


def test_predicate_and_entity_type_pair_rules_are_explicit() -> None:
    assert is_allowed_predicate("issued_invoice_to")
    assert not is_allowed_predicate("invented_predicate")
    assert has_valid_entity_type_pair("issued_invoice_to", "ORG", "ORG")
    assert not has_valid_entity_type_pair("issued_invoice_to", "PERSON", "ORG")
    assert has_valid_entity_type_pair("related_to", "PERSON", "PRODUCT")


def test_low_confidence_strong_relationships_require_downgrade() -> None:
    assert requires_confidence_downgrade(EvidenceStrength.HIGH, 0.69)
    assert requires_confidence_downgrade(EvidenceStrength.MEDIUM, 0.0)
    assert not requires_confidence_downgrade(EvidenceStrength.LOW, 0.1)
    assert not requires_confidence_downgrade(EvidenceStrength.HIGH, None)
