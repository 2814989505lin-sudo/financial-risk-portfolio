'''【122】工作流测试：验证不可完成的补偿不会伪造成功，而是进入人工处理状态。'''

from __future__ import annotations

from datetime import UTC, datetime

from financial_risk_agent.core.enums import VerificationStatus, WorkflowStatus
from financial_risk_agent.core.schemas import VerificationResult
from financial_risk_agent.storage.repositories.case_repository import CaseVersionConflict
from tests.integration.test_update_case_status import _proposal
from tests.workflow.test_mutation_team import _TASK_ID, _approve, _runtime


class ForcedFailureVerifier:
    """Force the first verification failure while leaving compensation verification independent."""

    allowed_tool_names = ("verify_update_case_status",)

    def run(self, *, task_id, proposal, receipt) -> VerificationResult:
        return VerificationResult(
            verification_id="verification-compensation-failure",
            action_id=proposal.action_id,
            status=VerificationStatus.FAILED,
            verified_at=datetime(2026, 7, 16, 12, 10, tzinfo=UTC),
            authoritative_ref="case:case-001:version:2",
            details="Forced failed re-read before compensation failure testing.",
        )


class BlockingRecoveryCaseRepository:
    """Reject the bounded compensation update to model a non-recoverable authoritative conflict."""

    def update_status(self, **kwargs):
        raise CaseVersionConflict("CASE_VERSION_CONFLICT")


def test_failed_compensation_enters_needs_manual_review(tmp_path) -> None:
    blocking_repository = BlockingRecoveryCaseRepository()
    team, approval_service, case_repository, task_repository, _ = _runtime(
        tmp_path,
        verifier=ForcedFailureVerifier(),
        recovery_case_repository=blocking_repository,
    )
    proposal = _proposal()
    approval_id = _approve(approval_service, proposal)

    result = team.run(_TASK_ID, proposal=proposal, approval_id=approval_id)

    state = task_repository.get(_TASK_ID).state
    assert result.recovery is not None
    assert result.recovery.compensated is False
    assert result.terminal_status is WorkflowStatus.NEEDS_MANUAL_REVIEW
    assert state.status is WorkflowStatus.NEEDS_MANUAL_REVIEW
    assert result.final_report.recovery_outcome == "NEEDS_MANUAL_REVIEW"
    assert case_repository.get("case-001").status == "UNDER_REVIEW"
