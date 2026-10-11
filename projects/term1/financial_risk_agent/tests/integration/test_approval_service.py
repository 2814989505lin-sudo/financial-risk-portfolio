'''【107】集成测试：验证提案审批持久化、哈希失效与拒绝后的安全终止。'''

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from financial_risk_agent.core.enums import (
    ApprovalDecision,
    MutationToolName,
    OperationType,
    RiskLevel,
    WorkflowStatus,
)
from financial_risk_agent.core.schemas import ActionProposal, InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.orchestration.state_store import StateStore
from financial_risk_agent.services.approval import ApprovalService, ProposalHashMismatch
from financial_risk_agent.services.audit import AuditService
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.approval_repository import (
    ApprovalRepository,
)
from financial_risk_agent.storage.repositories.audit_repository import AuditRepository
from financial_risk_agent.storage.repositories.task_repository import TaskRepository

_NOW = datetime(2026, 7, 16, 10, 0, tzinfo=UTC)


def _proposal(**overrides: object) -> ActionProposal:
    values: dict[str, object] = {
        "action_id": "action-watchlist-001",
        "tool_name": MutationToolName.ADD_TO_WATCHLIST,
        "operation_type": OperationType.ADD,
        "risk_level": RiskLevel.HIGH,
        "target_resource": "case:case-001",
        "validated_arguments": {"case_id": "case-001"},
        "reason": "The reviewed analysis indicates elevated fraud risk.",
        "expected_effect": "The case enters the enhanced review queue.",
        "requires_approval": True,
        "idempotency_key": "watchlist-case-001",
        "proposed_by": "compliance-agent",
    }
    values.update(overrides)
    return ActionProposal.model_validate(values)


def _service(tmp_path) -> tuple[ApprovalService, TaskRepository, ApprovalRepository]:
    database = SQLiteDatabase(tmp_path / "approvals.sqlite3")
    task_repository = TaskRepository(database)
    task_repository.create(
        WorkflowState(
            task_id="task-approval-001",
            session_id="session-approval-001",
            correlation_id="correlation-approval-001",
            status=WorkflowStatus.REVIEWING,
            current_phase=WorkflowStatus.REVIEWING,
            created_at=_NOW,
            updated_at=_NOW,
            input=InputSnapshot(
                source_type="fixture",
                article_id="article-approval-001",
                title="Approval fixture",
                content_ref="fixture://article-approval-001",
                content_hash=content_hash("approval fixture"),
            ),
        )
    )
    approval_repository = ApprovalRepository(database)
    return (
        ApprovalService(
            StateStore(task_repository),
            approval_repository,
            AuditService(AuditRepository(database)),
            now=lambda: _NOW,
        ),
        task_repository,
        approval_repository,
    )


def test_high_proposal_enters_awaiting_approval_and_binds_valid_decision(tmp_path) -> None:
    service, task_repository, approval_repository = _service(tmp_path)
    proposal = _proposal()

    submitted = service.submit_proposal("task-approval-001", proposal)
    approval = service.record_decision(
        "task-approval-001",
        action_id=proposal.action_id,
        proposal_hash=proposal.proposal_hash,
        approver_id="reviewer-001",
        decision=ApprovalDecision.APPROVED,
        expires_at=_NOW + timedelta(hours=1),
        approval_id="approval-001",
    )

    assert submitted.state.status is WorkflowStatus.AWAITING_APPROVAL
    assert approval_repository.get("approval-001") == approval
    assert task_repository.get("task-approval-001").state.approvals == [approval]
    assert service.is_approval_valid(
        "task-approval-001", action_id=proposal.action_id, approval_id=approval.approval_id
    )


def test_changed_proposal_invalidates_old_approval_and_rejects_stale_hash(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    original = _proposal()
    service.submit_proposal("task-approval-001", original)
    approval = service.record_decision(
        "task-approval-001",
        action_id=original.action_id,
        proposal_hash=original.proposal_hash,
        approver_id="reviewer-001",
        decision=ApprovalDecision.APPROVED,
        expires_at=_NOW + timedelta(hours=1),
        approval_id="approval-original",
    )
    changed = _proposal(target_resource="case:case-002")
    service.submit_proposal("task-approval-001", changed)

    assert not service.is_approval_valid(
        "task-approval-001", action_id=changed.action_id, approval_id=approval.approval_id
    )
    with pytest.raises(ProposalHashMismatch, match="PROPOSAL_HASH_MISMATCH"):
        service.record_decision(
            "task-approval-001",
            action_id=changed.action_id,
            proposal_hash=original.proposal_hash,
            approver_id="reviewer-002",
            decision=ApprovalDecision.APPROVED,
            expires_at=_NOW + timedelta(hours=1),
        )


def test_rejection_moves_task_to_rejected_without_execution_receipt(tmp_path) -> None:
    service, task_repository, _ = _service(tmp_path)
    proposal = _proposal()
    service.submit_proposal("task-approval-001", proposal)

    service.record_decision(
        "task-approval-001",
        action_id=proposal.action_id,
        proposal_hash=proposal.proposal_hash,
        approver_id="reviewer-001",
        decision=ApprovalDecision.REJECTED,
        expires_at=_NOW + timedelta(hours=1),
        approval_id="approval-rejected",
    )

    state = task_repository.get("task-approval-001").state
    assert state.status is WorkflowStatus.REJECTED
    assert state.execution_receipts == []
