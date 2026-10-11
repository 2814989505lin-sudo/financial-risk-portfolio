'''【116】集成测试：验证修改后必须重读权威案例数据，且仅匹配时产生 VERIFIED 结论。'''

from __future__ import annotations

from financial_risk_agent.core.enums import VerificationStatus
from financial_risk_agent.services.verification import MutationVerificationService
from tests.integration.test_update_case_status import _approve, _proposal, _runtime


class ReReadSpy:
    """Count authoritative re-reads while delegating all values to the real repository."""

    def __init__(self, repository) -> None:
        self._repository = repository
        self.get_calls = 0

    def get(self, case_id: str):
        self.get_calls += 1
        return self._repository.get(case_id)


class DivergentReRead:
    """Provide a mismatching authoritative re-read for deterministic verification failure."""

    def __init__(self, repository) -> None:
        self._repository = repository

    def get(self, case_id: str):
        return self._repository.get(case_id).model_copy(update={"status": "OPEN"})


def test_verification_re_reads_authoritative_case_and_marks_verified_on_match(tmp_path) -> None:
    approval_service, execution_service, case_repository, _, _, audit_service = _runtime(tmp_path)
    proposal = _proposal()
    approval_id = _approve(approval_service, proposal)
    result = execution_service.execute_update_case_status(
        "task-execution-001", proposal=proposal, approval_id=approval_id
    )

    spy = ReReadSpy(case_repository)
    verifier = MutationVerificationService(spy, audit_service)
    replay_verification = verifier.verify_update_case_status(
        "task-execution-001", proposal=proposal, receipt=result.receipt
    )

    assert spy.get_calls == 1
    assert result.verification.status is VerificationStatus.VERIFIED
    assert replay_verification.status is VerificationStatus.VERIFIED


def test_verification_reports_failed_when_authoritative_re_read_does_not_match(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    (
        approval_service,
        execution_service,
        case_repository,
        _,
        audit_repository,
        audit_service,
    ) = runtime
    proposal = _proposal()
    approval_id = _approve(approval_service, proposal)
    result = execution_service.execute_update_case_status(
        "task-execution-001", proposal=proposal, approval_id=approval_id
    )
    verifier = MutationVerificationService(
        DivergentReRead(case_repository),
        audit_service,
    )

    verification = verifier.verify_update_case_status(
        "task-execution-001", proposal=proposal, receipt=result.receipt
    )

    assert audit_repository.list_for_task("task-execution-001")
    assert verification.status is VerificationStatus.FAILED
