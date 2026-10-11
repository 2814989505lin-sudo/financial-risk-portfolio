'''【138】端到端测试：验证校验与补偿失败时任务保留在 NEEDS_MANUAL_REVIEW 且不伪造完成。'''

from __future__ import annotations

from financial_risk_agent.core.enums import WorkflowStatus
from tests.integration.test_update_case_status import _proposal
from tests.workflow.test_compensation import BlockingRecoveryCaseRepository, ForcedFailureVerifier
from tests.workflow.test_mutation_team import _TASK_ID, _approve, _runtime


def test_manual_review_flow_remains_durable_after_failed_recovery(tmp_path) -> None:
    team, approval_service, case_repository, task_repository, _ = _runtime(
        tmp_path,
        verifier=ForcedFailureVerifier(),
        recovery_case_repository=BlockingRecoveryCaseRepository(),
    )
    proposal = _proposal()
    approval_id = _approve(approval_service, proposal)

    result = team.run(_TASK_ID, proposal=proposal, approval_id=approval_id)

    assert result.terminal_status is WorkflowStatus.NEEDS_MANUAL_REVIEW
    assert task_repository.get(_TASK_ID).state.status is WorkflowStatus.NEEDS_MANUAL_REVIEW
    assert case_repository.get("case-001").status == "UNDER_REVIEW"
