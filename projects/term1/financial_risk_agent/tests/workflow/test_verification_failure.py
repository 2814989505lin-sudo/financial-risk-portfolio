'''【121】工作流测试：验证初次重读失败时 MutationTeam 不信任执行回执，并转入补偿流程。'''

from __future__ import annotations

from datetime import UTC, datetime

from financial_risk_agent.core.enums import VerificationStatus, WorkflowStatus
from financial_risk_agent.core.schemas import VerificationResult
from tests.integration.test_update_case_status import _proposal
from tests.workflow.test_mutation_team import _TASK_ID, _approve, _runtime


class ForcedFailureVerifier:
    """Return a failed initial verification to ensure recovery is driven by re-read conclusions."""

    allowed_tool_names = ("verify_update_case_status",)

    def run(self, *, task_id, proposal, receipt) -> VerificationResult:
        return VerificationResult(
            verification_id="verification-forced-failure",
            action_id=proposal.action_id,
            status=VerificationStatus.FAILED,
            verified_at=datetime(2026, 7, 16, 12, 5, tzinfo=UTC),
            authoritative_ref="case:case-001:version:2",
            details="Forced failed re-read for controlled recovery testing.",
        )


def test_failed_initial_verification_is_compensated_and_reverified(tmp_path) -> None:
    team, approval_service, case_repository, task_repository, _ = _runtime(
        tmp_path,
        verifier=ForcedFailureVerifier(),
    )
    proposal = _proposal()
    approval_id = _approve(approval_service, proposal)

    result = team.run(_TASK_ID, proposal=proposal, approval_id=approval_id)

    state = task_repository.get(_TASK_ID).state
    assert result.verification.status is VerificationStatus.FAILED
    assert result.recovery is not None
    assert result.recovery.compensated is True
    assert result.recovery.verification is not None
    assert result.recovery.verification.status is VerificationStatus.VERIFIED
    assert result.final_report.recovery_outcome == "COMPENSATED"
    assert result.terminal_status is WorkflowStatus.COMPLETED
    assert state.status is WorkflowStatus.COMPLETED
    assert case_repository.get("case-001").status == "OPEN"
