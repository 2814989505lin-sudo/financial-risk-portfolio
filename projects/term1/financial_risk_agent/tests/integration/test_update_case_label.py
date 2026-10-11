'''【131】集成测试：验证 update_case_label 的固定参数、审批约束、幂等执行与权威重读校验。'''

from __future__ import annotations

from datetime import timedelta

from financial_risk_agent.core.enums import (
    ApprovalDecision,
    MutationToolName,
    OperationType,
    RiskLevel,
    VerificationStatus,
)
from financial_risk_agent.core.schemas import ActionProposal
from tests.integration.test_update_case_status import _NOW, _runtime


def _label_proposal(**overrides: object) -> ActionProposal:
    values: dict[str, object] = {
        "action_id": "action-case-label-001",
        "tool_name": MutationToolName.UPDATE_CASE_LABEL,
        "operation_type": OperationType.UPDATE,
        "risk_level": RiskLevel.MEDIUM,
        "target_resource": "case:case-001",
        "validated_arguments": {
            "case_id": "case-001",
            "new_label": "ELEVATED_RISK",
            "expected_version": 1,
        },
        "reason": "The approved review requires a persistent case classification label.",
        "expected_effect": "The case retains the elevated risk classification.",
        "requires_approval": True,
        "idempotency_key": "case-label-case-001-v1",
        "proposed_by": "compliance-agent",
    }
    values.update(overrides)
    return ActionProposal.model_validate(values)


def test_update_case_label_requires_approval_is_idempotent_and_re_reads_case(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    approval_service, execution_service, case_repository, execution_repository, _, _ = runtime
    proposal = _label_proposal()
    approval_service.submit_proposal("task-execution-001", proposal)
    approval = approval_service.record_decision(
        "task-execution-001",
        action_id=proposal.action_id,
        proposal_hash=proposal.proposal_hash,
        approver_id="reviewer-label-001",
        decision=ApprovalDecision.APPROVED,
        expires_at=_NOW + timedelta(hours=1),
        approval_id="approval-case-label-001",
    )

    first = execution_service.execute_update_case_label(
        "task-execution-001",
        proposal=proposal,
        approval_id=approval.approval_id,
    )
    replay = execution_service.execute_update_case_label(
        "task-execution-001",
        proposal=proposal,
        approval_id=approval.approval_id,
    )

    case = case_repository.get("case-001")
    assert case.label == "ELEVATED_RISK"
    assert case.version == 2
    assert first.receipt.tool_name is MutationToolName.UPDATE_CASE_LABEL
    assert first.verification.status is VerificationStatus.VERIFIED
    assert replay.idempotent_replay is True
    assert execution_repository.get_by_idempotency_key(proposal.idempotency_key) is not None
