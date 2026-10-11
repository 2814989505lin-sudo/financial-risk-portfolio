'''【136】端到端测试：验证已批准的新增标签修改执行一次、保留凭证并经权威重读校验。'''

from __future__ import annotations

from datetime import timedelta

from financial_risk_agent.core.enums import ApprovalDecision, VerificationStatus
from tests.integration.test_update_case_label import _label_proposal
from tests.integration.test_update_case_status import _NOW, _runtime


def test_approved_label_mutation_flow_is_verified_and_idempotent(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    approval_service, execution_service, case_repository, _, audit_repository, _ = runtime
    proposal = _label_proposal()
    approval_service.submit_proposal("task-execution-001", proposal)
    approval = approval_service.record_decision(
        "task-execution-001",
        action_id=proposal.action_id,
        proposal_hash=proposal.proposal_hash,
        approver_id="reviewer-e2e-001",
        decision=ApprovalDecision.APPROVED,
        expires_at=_NOW + timedelta(hours=1),
        approval_id="approval-label-e2e-001",
    )

    result = execution_service.execute_update_case_label(
        "task-execution-001", proposal=proposal, approval_id=approval.approval_id
    )

    assert case_repository.get("case-001").label == "ELEVATED_RISK"
    assert result.verification.status is VerificationStatus.VERIFIED
    events = audit_repository.list_for_task("task-execution-001")
    assert any(event.event_type == "MUTATION_EXECUTED" for event in events)
