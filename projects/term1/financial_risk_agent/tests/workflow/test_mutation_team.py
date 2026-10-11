'''【120】工作流测试：验证 MutationTeam 仅在有效审批后按执行、重读校验和最终报告顺序运行。'''

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from financial_risk_agent.agents.executor import ExecutorAgent
from financial_risk_agent.agents.verifier import VerifierAgent
from financial_risk_agent.core.enums import ApprovalDecision, WorkflowStatus
from financial_risk_agent.core.schemas import ActionProposal, InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.orchestration.state_store import StateStore
from financial_risk_agent.orchestration.team import MutationTeam, MutationTeamError
from financial_risk_agent.services.approval import ApprovalService
from financial_risk_agent.services.audit import AuditService
from financial_risk_agent.services.execution import MutationExecutionService
from financial_risk_agent.services.recovery import RecoveryService
from financial_risk_agent.services.verification import MutationVerificationService
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.approval_repository import ApprovalRepository
from financial_risk_agent.storage.repositories.audit_repository import AuditRepository
from financial_risk_agent.storage.repositories.case_repository import CaseRepository
from financial_risk_agent.storage.repositories.execution_repository import ExecutionRepository
from financial_risk_agent.storage.repositories.task_repository import TaskRepository
from tests.integration.test_update_case_status import _proposal

_NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
_TASK_ID = "task-mutation-001"


def _runtime(tmp_path, *, verifier=None, recovery_case_repository=None):
    database = SQLiteDatabase(tmp_path / "mutation-team.sqlite3")
    task_repository = TaskRepository(database)
    task_repository.create(
        WorkflowState(
            task_id=_TASK_ID,
            session_id="session-mutation-001",
            correlation_id="correlation-mutation-001",
            status=WorkflowStatus.REVIEWING,
            current_phase=WorkflowStatus.REVIEWING,
            created_at=_NOW,
            updated_at=_NOW,
            input=InputSnapshot(
                source_type="fixture",
                article_id="article-mutation-001",
                title="Mutation team fixture",
                content_ref="fixture://article-mutation-001",
                content_hash=content_hash("mutation team fixture"),
            ),
        )
    )
    state_store = StateStore(task_repository)
    audit_repository = AuditRepository(database)
    audit_service = AuditService(audit_repository)
    approval_service = ApprovalService(
        state_store,
        ApprovalRepository(database),
        audit_service,
        now=lambda: _NOW,
    )
    case_repository = CaseRepository(database)
    case_repository.create(case_id="case-001", status="OPEN")
    execution_repository = ExecutionRepository(database)
    verification_service = MutationVerificationService(
        case_repository,
        audit_service,
        now=lambda: _NOW,
    )
    execution_service = MutationExecutionService(
        state_store,
        approval_service,
        case_repository,
        execution_repository,
        verification_service,
        audit_service,
        now=lambda: _NOW,
    )
    recovery_service = RecoveryService(
        recovery_case_repository or case_repository,
        execution_repository,
        verification_service,
        audit_service,
    )
    team = MutationTeam(
        state_store,
        approval_service,
        ExecutorAgent(execution_service),
        verifier or VerifierAgent(verification_service),
        recovery_service,
        audit_service,
    )
    return team, approval_service, case_repository, task_repository, audit_repository


def _approve(approval_service: ApprovalService, proposal: ActionProposal) -> str:
    approval_service.submit_proposal(_TASK_ID, proposal)
    approval = approval_service.record_decision(
        _TASK_ID,
        action_id=proposal.action_id,
        proposal_hash=proposal.proposal_hash,
        approver_id="reviewer-mutation-001",
        decision=ApprovalDecision.APPROVED,
        expires_at=_NOW + timedelta(hours=1),
        approval_id="approval-mutation-001",
    )
    return approval.approval_id


def test_mutation_team_runs_only_with_valid_approval_and_generates_final_report(tmp_path) -> None:
    team, approval_service, case_repository, task_repository, audit_repository = _runtime(tmp_path)
    proposal = _proposal()
    approval_id = _approve(approval_service, proposal)

    result = team.run(_TASK_ID, proposal=proposal, approval_id=approval_id)

    state = task_repository.get(_TASK_ID).state
    assert result.terminal_status is WorkflowStatus.COMPLETED
    assert result.final_report.approval_id == approval_id
    assert result.final_report.execution_receipt_id == result.receipt.receipt_id
    assert result.final_report.recovery_outcome == "NOT_REQUIRED"
    assert state.execution_receipts == [result.receipt]
    assert state.verification_results == [result.verification]
    assert case_repository.get("case-001").status == "UNDER_REVIEW"
    assert any(
        event.event_type == "MUTATION_FINAL_REPORT_GENERATED"
        for event in audit_repository.list_for_task(_TASK_ID)
    )


def test_mutation_team_refuses_to_start_without_a_current_approval(tmp_path) -> None:
    team, approval_service, case_repository, task_repository, _ = _runtime(tmp_path)
    proposal = _proposal()
    approval_service.submit_proposal(_TASK_ID, proposal)

    with pytest.raises(MutationTeamError, match="VALID_APPROVAL_REQUIRED"):
        team.run(_TASK_ID, proposal=proposal, approval_id="missing-approval")

    assert task_repository.get(_TASK_ID).state.status is WorkflowStatus.AWAITING_APPROVAL
    assert case_repository.get("case-001").status == "OPEN"
