'''【106】单元测试：验证审批绑定的行动提案哈希覆盖目标、参数和预期影响。'''

from __future__ import annotations

from financial_risk_agent.core.enums import MutationToolName, OperationType, RiskLevel
from financial_risk_agent.core.schemas import ActionProposal


def _proposal(**overrides: object) -> ActionProposal:
    values: dict[str, object] = {
        "action_id": "action-watchlist-001",
        "tool_name": MutationToolName.ADD_TO_WATCHLIST,
        "operation_type": OperationType.ADD,
        "risk_level": RiskLevel.HIGH,
        "target_resource": "case:case-001",
        "validated_arguments": {"case_id": "case-001", "watchlist": "heightened"},
        "reason": "The analysis found a sustained elevated fraud signal.",
        "expected_effect": "The case is available to the enhanced review queue.",
        "requires_approval": True,
        "idempotency_key": "watchlist-case-001",
        "proposed_by": "compliance-agent",
    }
    values.update(overrides)
    return ActionProposal.model_validate(values)


def test_proposal_hash_changes_for_every_approval_relevant_field() -> None:
    proposal = _proposal()
    changed_target = _proposal(target_resource="case:case-002")
    changed_arguments = _proposal(validated_arguments={"case_id": "case-002"})
    changed_effect = _proposal(expected_effect="A different review queue is used.")

    assert proposal.proposal_hash != changed_target.proposal_hash
    assert proposal.proposal_hash != changed_arguments.proposal_hash
    assert proposal.proposal_hash != changed_effect.proposal_hash
