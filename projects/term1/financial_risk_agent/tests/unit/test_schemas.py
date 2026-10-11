'''【043】单元测试：验证业务 Pydantic 模型、工具约束和提案哈希。'''

from __future__ import annotations

import pytest
from pydantic import ValidationError

from financial_risk_agent.core.enums import MutationToolName, OperationType, RiskLevel
from financial_risk_agent.core.schemas import ActionProposal, InputSnapshot, RiskPrediction
from financial_risk_agent.core.security import content_hash


def make_input_snapshot(**overrides: object) -> InputSnapshot:
    values: dict[str, object] = {
        "source_type": "news",
        "article_id": "article-001",
        "title": "Example risk event",
        "content_ref": "data/raw/article-001.json",
        "content_hash": content_hash("Example risk event content"),
        "source_metadata": {"publisher": "example"},
    }
    values.update(overrides)
    return InputSnapshot.model_validate(values)


def make_action_proposal(**overrides: object) -> ActionProposal:
    values: dict[str, object] = {
        "action_id": "action-001",
        "tool_name": MutationToolName.SAVE_ANALYSIS_DRAFT,
        "operation_type": OperationType.SAVE,
        "risk_level": RiskLevel.LOW,
        "target_resource": "artifacts/reports/draft-001.md",
        "validated_arguments": {"report_id": "draft-001", "format": "markdown"},
        "reason": "Preserve the reviewed analysis draft.",
        "expected_effect": "A new local draft is available for review.",
        "requires_approval": False,
        "idempotency_key": "save-draft-001",
        "proposed_by": "report-draft-agent",
    }
    values.update(overrides)
    return ActionProposal.model_validate(values)


def test_input_snapshot_rejects_sensitive_metadata() -> None:
    with pytest.raises(ValidationError, match="credentials or secrets"):
        make_input_snapshot(source_metadata={"api_key": "must-not-be-stored"})


def test_risk_prediction_rejects_probability_outside_unit_interval() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        RiskPrediction(
            model_version="rgcn-v1",
            graph_artifact_hash=content_hash("graph-001"),
            credit_probability=1.1,
            market_probability=0.2,
            operational_probability=0.3,
            fraud_probability=0.4,
        )


def test_action_proposal_rejects_unknown_tool_name() -> None:
    with pytest.raises(ValidationError, match="Input should be"):
        make_action_proposal(tool_name="delete_everything")


def test_action_proposal_hash_is_deterministic_and_content_bound() -> None:
    first = make_action_proposal()
    second = make_action_proposal()
    changed = make_action_proposal(target_resource="artifacts/reports/draft-002.md")

    assert first.proposal_hash == second.proposal_hash
    assert first.proposal_hash != changed.proposal_hash


def test_action_proposal_enforces_registered_risk_and_approval_policy() -> None:
    with pytest.raises(ValidationError, match="risk_level"):
        make_action_proposal(risk_level=RiskLevel.HIGH, requires_approval=True)

    with pytest.raises(ValidationError, match="require approval"):
        make_action_proposal(
            tool_name=MutationToolName.ADD_TO_WATCHLIST,
            operation_type=OperationType.ADD,
            risk_level=RiskLevel.HIGH,
            requires_approval=False,
        )


def test_model_serialization_contains_no_credential_fields() -> None:
    serialized = make_action_proposal().model_dump_json()

    assert "api_key" not in serialized
    assert "password" not in serialized
    assert "secret" not in serialized
