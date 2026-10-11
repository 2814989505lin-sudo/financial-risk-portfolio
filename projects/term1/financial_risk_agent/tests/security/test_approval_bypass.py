'''【133】安全测试：验证高风险提案和未审批的新增修改工具均无法绕过受控执行服务。'''

from __future__ import annotations

import pytest

from financial_risk_agent.core.enums import MutationToolName, OperationType, RiskLevel
from financial_risk_agent.core.schemas import ActionProposal
from financial_risk_agent.services.execution import MutationExecutionDenied
from tests.integration.test_update_case_label import _label_proposal
from tests.integration.test_update_case_status import _runtime


def test_label_tool_and_high_risk_actions_cannot_bypass_approval(tmp_path) -> None:
    approval_service, execution_service, case_repository, _, _, _ = _runtime(tmp_path)
    label_proposal = _label_proposal()
    approval_service.submit_proposal("task-execution-001", label_proposal)

    with pytest.raises(MutationExecutionDenied, match="VALID_APPROVAL_REQUIRED"):
        execution_service.execute_update_case_label(
            "task-execution-001",
            proposal=label_proposal,
            approval_id="missing-approval",
        )

    high_risk_proposal = ActionProposal(
        action_id="action-high-risk-001",
        tool_name=MutationToolName.ADD_TO_WATCHLIST,
        operation_type=OperationType.ADD,
        risk_level=RiskLevel.HIGH,
        target_resource="case:case-001",
        validated_arguments={"case_id": "case-001"},
        reason="A high-risk action requires explicit approval and a dedicated implementation.",
        expected_effect="The case would be added to a watchlist.",
        requires_approval=True,
        idempotency_key="high-risk-case-001",
        proposed_by="compliance-agent",
    )
    with pytest.raises(MutationExecutionDenied, match="UNSUPPORTED_OR_UNAPPROVED_MUTATION"):
        execution_service.execute_update_case_label(
            "task-execution-001",
            proposal=high_risk_proposal,
            approval_id="any-approval",
        )

    case = case_repository.get("case-001")
    assert case.label == "UNLABELED"
    assert case.version == 1
