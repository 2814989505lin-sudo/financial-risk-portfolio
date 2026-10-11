'''【115】安全测试：验证案例状态修改拒绝无审批、过期审批及与批准内容不一致的请求。'''

from __future__ import annotations

import pytest

from financial_risk_agent.services.execution import MutationExecutionDenied
from tests.integration.test_update_case_status import _approve, _proposal, _runtime


def test_execution_without_valid_approval_cannot_modify_case(tmp_path) -> None:
    approval_service, execution_service, case_repository, _, _, _ = _runtime(tmp_path)
    proposal = _proposal()
    approval_service.submit_proposal("task-execution-001", proposal)

    with pytest.raises(MutationExecutionDenied, match="VALID_APPROVAL_REQUIRED"):
        execution_service.execute_update_case_status(
            "task-execution-001", proposal=proposal, approval_id="approval-missing"
        )

    assert case_repository.get("case-001").status == "OPEN"


def test_execution_rejects_parameters_that_differ_from_approved_proposal(tmp_path) -> None:
    approval_service, execution_service, case_repository, _, _, _ = _runtime(tmp_path)
    proposal = _proposal()
    approval_id = _approve(approval_service, proposal)
    changed = _proposal(
        validated_arguments={
            "case_id": "case-001",
            "new_status": "CLOSED",
            "expected_version": 1,
        }
    )

    with pytest.raises(MutationExecutionDenied, match="PROPOSAL_CONTENT_MISMATCH"):
        execution_service.execute_update_case_status(
            "task-execution-001", proposal=changed, approval_id=approval_id
        )

    assert case_repository.get("case-001").status == "OPEN"
