'''【105】单元测试：验证行动提案策略的确定性审批路由与永久拒绝规则。'''

from __future__ import annotations

import pytest

from financial_risk_agent.core.enums import RiskLevel
from financial_risk_agent.services.policy import ActionPolicyDenied, ActionProposalPolicy


def test_high_risk_action_requires_human_approval() -> None:
    decision = ActionProposalPolicy().decision_for_risk_level(RiskLevel.HIGH)

    assert decision.risk_level is RiskLevel.HIGH
    assert decision.requires_approval is True


def test_prohibited_action_is_rejected_without_exception_path() -> None:
    with pytest.raises(ActionPolicyDenied, match="PROHIBITED_ACTION_DENIED"):
        ActionProposalPolicy().decision_for_risk_level(RiskLevel.PROHIBITED)
