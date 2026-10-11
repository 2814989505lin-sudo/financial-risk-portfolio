'''【128】单元测试：验证审批摘要完整展示风险信息、证据、可回滚性和有效期。'''

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.components.proposal_summary import ProposalSummaryViewModel
from financial_risk_agent.core.enums import MutationToolName, OperationType, RiskLevel
from financial_risk_agent.core.schemas import ActionProposal, EvidenceReference
from financial_risk_agent.core.security import content_hash


def _proposal() -> ActionProposal:
    return ActionProposal(
        action_id="action-ui-001",
        tool_name=MutationToolName.UPDATE_CASE_STATUS,
        operation_type=OperationType.UPDATE,
        risk_level=RiskLevel.MEDIUM,
        target_resource="case:case-001",
        validated_arguments={
            "case_id": "case-001",
            "new_status": "UNDER_REVIEW",
            "expected_version": 1,
        },
        reason="The case requires enhanced manual review after the risk assessment.",
        evidence_refs=[
            EvidenceReference(
                source_ref="article:article-001",
                locator="paragraph:2",
                excerpt_hash=content_hash("evidence excerpt"),
            )
        ],
        expected_effect="The case enters the enhanced review queue.",
        requires_approval=True,
        idempotency_key="case-status-ui-001",
        proposed_by="compliance-agent",
    )


def test_proposal_summary_includes_all_human_approval_context() -> None:
    expires_at = datetime(2026, 7, 16, 14, 0, tzinfo=UTC)
    proposal = _proposal()

    view_model = ProposalSummaryViewModel.from_proposal(proposal, expires_at=expires_at)

    assert view_model.target_resource == "case:case-001"
    assert "case_id=case-001" in view_model.parameter_summary
    assert view_model.risk_reason == proposal.reason
    assert view_model.evidence[0].locator == "paragraph:2"
    assert view_model.expected_effect == proposal.expected_effect
    assert view_model.reversible is True
    assert view_model.expires_at == expires_at
    assert view_model.proposal_hash == proposal.proposal_hash


def test_proposal_summary_requires_timezone_aware_expiry() -> None:
    proposal = _proposal()
    naive_expiry = datetime(2026, 7, 16, 14, 0) + timedelta(minutes=1)

    try:
        ProposalSummaryViewModel.from_proposal(proposal, expires_at=naive_expiry)
    except ValueError as error:
        assert "timezone" in str(error)
    else:
        raise AssertionError("A naive approval expiry must be rejected.")
