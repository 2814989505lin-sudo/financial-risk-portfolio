'''【114】集成测试：验证唯一案例状态修改工具的参数约束、版本控制和幂等执行。'''

from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
from financial_risk_agent.services.approval import ApprovalService
from financial_risk_agent.services.audit import AuditService
from financial_risk_agent.services.execution import MutationExecutionService
from financial_risk_agent.services.verification import MutationVerificationService
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.approval_repository import ApprovalRepository
from financial_risk_agent.storage.repositories.audit_repository import AuditRepository
from financial_risk_agent.storage.repositories.case_repository import CaseRepository
from financial_risk_agent.storage.repositories.execution_repository import ExecutionRepository
from financial_risk_agent.storage.repositories.task_repository import TaskRepository

_NOW = datetime(2026, 7, 16, 11, 0, tzinfo=UTC)


def _proposal(**overrides: object) -> ActionProposal:
    values: dict[str, object] = {
        "action_id": "action-case-status-001",
        "tool_name": MutationToolName.UPDATE_CASE_STATUS,
        "operation_type": OperationType.UPDATE,
        "risk_level": RiskLevel.MEDIUM,
        "target_resource": "case:case-001",
        "validated_arguments": {
            "case_id": "case-001",
            "new_status": "UNDER_REVIEW",
            "expected_version": 1,
        },
        "reason": "The reviewed analysis requires a case status escalation.",
        "expected_effect": "The case is marked for enhanced review.",
        "requires_approval": True,
        "idempotency_key": "case-status-case-001-v1",
        "proposed_by": "compliance-agent",
    }
    values.update(overrides)
    return ActionProposal.model_validate(values)


def _runtime(tmp_path):
    database = SQLiteDatabase(tmp_path / "executions.sqlite3")
    task_repository = TaskRepository(database)
    task_repository.create(
        WorkflowState(
            task_id="task-execution-001",
            session_id="session-execution-001",
            correlation_id="correlation-execution-001",
            status=WorkflowStatus.REVIEWING,
            current_phase=WorkflowStatus.REVIEWING,
            created_at=_NOW,
            updated_at=_NOW,
            input=InputSnapshot(
                source_type="fixture",
                article_id="article-execution-001",
                title="Execution fixture",
                content_ref="fixture://article-execution-001",
                content_hash=content_hash("execution fixture"),
            ),
        )
    )
    audit_repository = AuditRepository(database)
    audit_service = AuditService(audit_repository)
    approval_service = ApprovalService(
        StateStore(task_repository),
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
        StateStore(task_repository),
        approval_service,
        case_repository,
        execution_repository,
        verification_service,
        audit_service,
        now=lambda: _NOW,
    )
    return (
        approval_service,
        execution_service,
        case_repository,
        execution_repository,
        audit_repository,
        audit_service,
    )


def _approve(approval_service: ApprovalService, proposal: ActionProposal) -> str:
    approval_service.submit_proposal("task-execution-001", proposal)
    approval = approval_service.record_decision(
        "task-execution-001",
        action_id=proposal.action_id,
        proposal_hash=proposal.proposal_hash,
        approver_id="reviewer-001",
        decision=ApprovalDecision.APPROVED,
        expires_at=_NOW + timedelta(hours=1),
        approval_id="approval-case-status-001",
    )
    return approval.approval_id


def test_update_case_status_is_idempotent_and_audits_digests(tmp_path) -> None:
    (
        approval_service,
        execution_service,
        case_repository,
        execution_repository,
        audit_repository,
        _,
    ) = _runtime(tmp_path)
    proposal = _proposal()
    approval_id = _approve(approval_service, proposal)

    first = execution_service.execute_update_case_status(
        "task-execution-001", proposal=proposal, approval_id=approval_id
    )
    replay = execution_service.execute_update_case_status(
        "task-execution-001", proposal=proposal, approval_id=approval_id
    )

    assert first.idempotent_replay is False
    assert replay.idempotent_replay is True
    assert replay.receipt == first.receipt
    assert case_repository.get("case-001").version == 2
    assert execution_repository.get_by_idempotency_key(proposal.idempotency_key) is not None
    mutation_events = [
        event
        for event in audit_repository.list_for_task("task-execution-001")
        if event.event_type == "MUTATION_EXECUTED"
    ]
    assert len(mutation_events) == 1
    assert mutation_events[0].metadata["before_digest"] == first.receipt.before_digest
    assert mutation_events[0].metadata["after_digest"] == first.receipt.after_digest
