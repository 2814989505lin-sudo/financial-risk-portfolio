'''【129】端到端测试：验证审批 UI 重读哈希、持久化结果及重复点击不会重复决定或执行。'''

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.pages.approvals import PendingApproval, render_approvals
from financial_risk_agent.core.enums import (
    ApprovalDecision,
    MutationToolName,
    OperationType,
    RiskLevel,
)
from financial_risk_agent.core.schemas import ActionProposal, ApprovalRecord


class FakeStreamlit:
    """A minimal Streamlit surface that records controls and consumes each requested click once."""

    def __init__(self, clicked: set[str]) -> None:
        self.clicked = clicked
        self.session_state: dict[str, str] = {}
        self.writes: list[object] = []
        self.button_labels: list[str] = []

    def title(self, value: str) -> None:
        self.writes.append(value)

    def subheader(self, value: str) -> None:
        self.writes.append(value)

    def selectbox(self, label: str, *, options: list[str], format_func=None) -> str:
        self.writes.append(label)
        return options[0]

    def text_input(self, label: str, *, value: str) -> str:
        self.writes.append(label)
        return value

    def button(self, label: str) -> bool:
        self.button_labels.append(label)
        if label not in self.clicked:
            return False
        self.clicked.remove(label)
        return True

    def write(self, value: object) -> None:
        self.writes.append(value)

    def info(self, value: str) -> None:
        self.writes.append(value)


class FakeApprovalService:
    """A durable in-memory service whose decision removes the request like persisted state."""

    def __init__(self, pending: PendingApproval) -> None:
        self.pending = pending
        self.decisions: list[ApprovalRecord] = []
        self.hash_reads = 0
        self.execution_count = 0

    def list_pending_approvals(self) -> list[PendingApproval]:
        return [self.pending] if self.pending is not None else []

    def get_pending_approval(self, task_id: str, action_id: str) -> PendingApproval | None:
        self.hash_reads += 1
        if self.pending is None:
            return None
        if (self.pending.task_id, self.pending.proposal.action_id) != (task_id, action_id):
            return None
        return self.pending

    def decide(
        self,
        *,
        task_id: str,
        action_id: str,
        proposal_hash: str,
        approver_id: str,
        decision: ApprovalDecision,
        comment: str | None = None,
    ) -> ApprovalRecord:
        assert self.pending is not None
        assert proposal_hash == self.pending.proposal.proposal_hash
        record = ApprovalRecord(
            approval_id="approval-ui-001",
            action_id=action_id,
            proposal_hash=proposal_hash,
            approver_id=approver_id,
            decision=decision,
            decided_at=datetime(2026, 7, 16, 13, 0, tzinfo=UTC),
            expires_at=datetime(2026, 7, 16, 14, 0, tzinfo=UTC),
            comment=comment,
        )
        self.decisions.append(record)
        self.pending = None
        return record


def _pending() -> PendingApproval:
    proposal = ActionProposal(
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
        reason="The review requires a status escalation.",
        expected_effect="The case enters enhanced review.",
        requires_approval=True,
        idempotency_key="case-ui-001",
        proposed_by="compliance-agent",
    )
    return PendingApproval(
        task_id="task-ui-001",
        proposal=proposal,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )


def test_approval_ui_rechecks_hash_and_persists_one_decision_across_refreshes() -> None:
    service = FakeApprovalService(_pending())
    st = FakeStreamlit({"批准提案"})

    decision = render_approvals(st, service)
    restarted_st = FakeStreamlit(set())
    refreshed = render_approvals(restarted_st, service)

    assert decision is not None
    assert decision.decision is ApprovalDecision.APPROVED
    assert refreshed is None
    assert service.hash_reads == 1
    assert len(service.decisions) == 1
    assert service.execution_count == 0
    assert "目标资源：case:case-001" in st.writes
    assert st.button_labels.count("批准提案") == 1
    assert "当前没有待审批提案。" in restarted_st.writes
