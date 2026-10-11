'''【137】端到端测试：验证拒绝后的提案没有执行入口，案例权威数据保持不变。'''

from __future__ import annotations

from datetime import timedelta

import pytest

from financial_risk_agent.core.enums import ApprovalDecision
from financial_risk_agent.services.execution import MutationExecutionDenied
from tests.integration.test_update_case_label import _label_proposal
from tests.integration.test_update_case_status import _NOW, _runtime


def test_rejected_label_mutation_does_not_execute(tmp_path) -> None:
    approval_service, execution_service, case_repository, _, _, _ = _runtime(tmp_path)
    proposal = _label_proposal()
    approval_service.submit_proposal("task-execution-001", proposal)
    approval_service.record_decision(
        "task-execution-001",
        action_id=proposal.action_id,
        proposal_hash=proposal.proposal_hash,
        approver_id="reviewer-e2e-reject",
        decision=ApprovalDecision.REJECTED,
        expires_at=_NOW + timedelta(hours=1),
        approval_id="approval-label-e2e-reject",
    )

    with pytest.raises(MutationExecutionDenied, match="VALID_APPROVAL_REQUIRED"):
        execution_service.execute_update_case_label(
            "task-execution-001", proposal=proposal, approval_id="approval-label-e2e-reject"
        )

    assert case_repository.get("case-001").label == "UNLABELED"
    assert case_repository.get("case-001").version == 1
