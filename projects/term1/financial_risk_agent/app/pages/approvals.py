'''【125】审批页面：重新验证当前 proposal_hash 后委托应用服务批准或拒绝待审批提案。'''

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import Field, field_validator

from app.components.proposal_summary import ProposalSummaryViewModel, render_proposal_summary
from financial_risk_agent.core.enums import ApprovalDecision
from financial_risk_agent.core.schemas import ActionProposal, ApprovalRecord, StrictModel

_SELECTED_APPROVAL_KEY = "selected_pending_approval"


class PendingApproval(StrictModel):
    """A current application-level approval request, rather than a database row exposed to UI."""

    task_id: str = Field(min_length=1, max_length=200)
    proposal: ActionProposal
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("expires_at must include a timezone.")
        return value


class ApprovalUIService(Protocol):
    """The UI boundary for durable approval operations and no mutation execution operations."""

    def list_pending_approvals(self) -> list[PendingApproval]:
        """Return current approval requests from durable state."""

    def get_pending_approval(self, task_id: str, action_id: str) -> PendingApproval | None:
        """Re-read the current request immediately before an approval decision."""

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
        """Persist one approval decision after the implementation validates the supplied hash."""


class ApprovalUIError(RuntimeError):
    """A safe UI-level error raised without surfacing storage or workflow implementation details."""


def render_approvals(st, service: ApprovalUIService) -> ApprovalRecord | None:
    """Render durable approvals and re-check proposal content immediately before a decision."""
    st.title("人工审批")
    pending = service.list_pending_approvals()
    if not pending:
        st.info("当前没有待审批提案。")
        return None
    pending_by_key = {_pending_key(item): item for item in pending}
    selected_key = st.selectbox(
        "选择待审批提案",
        options=list(pending_by_key),
        format_func=lambda key: _pending_label(pending_by_key[key]),
    )
    selected = pending_by_key[selected_key]
    st.session_state[_SELECTED_APPROVAL_KEY] = selected_key
    summary = ProposalSummaryViewModel.from_proposal(
        selected.proposal,
        expires_at=selected.expires_at,
    )
    render_proposal_summary(st, summary)
    approver_id = st.text_input("审批人编号", value="reviewer-ui")
    comment = st.text_input("审批备注（可选）", value="")

    if st.button("批准提案"):
        return _decide_current(
            service,
            selected,
            approver_id=approver_id,
            decision=ApprovalDecision.APPROVED,
            comment=comment or None,
        )
    if st.button("拒绝提案"):
        return _decide_current(
            service,
            selected,
            approver_id=approver_id,
            decision=ApprovalDecision.REJECTED,
            comment=comment or None,
        )
    return None


def _decide_current(
    service: ApprovalUIService,
    selected: PendingApproval,
    *,
    approver_id: str,
    decision: ApprovalDecision,
    comment: str | None,
) -> ApprovalRecord:
    if not approver_id.strip():
        raise ApprovalUIError("Approver identifier is required.")
    selected_hash = selected.proposal.proposal_hash
    if selected_hash is None:
        raise ApprovalUIError("The approval request has no valid proposal hash.")
    current = service.get_pending_approval(selected.task_id, selected.proposal.action_id)
    current_hash = current.proposal.proposal_hash if current is not None else None
    if current is None or current_hash is None or current_hash != selected_hash:
        raise ApprovalUIError("The approval request changed before the decision was submitted.")
    try:
        return service.decide(
            task_id=current.task_id,
            action_id=current.proposal.action_id,
            proposal_hash=current_hash,
            approver_id=approver_id,
            decision=decision,
            comment=comment,
        )
    except Exception as error:
        raise ApprovalUIError("Unable to persist the approval decision.") from error


def _pending_key(pending: PendingApproval) -> str:
    return f"{pending.task_id}:{pending.proposal.action_id}"


def _pending_label(pending: PendingApproval) -> str:
    return (
        f"{pending.task_id} · {pending.proposal.tool_name.value} · "
        f"{pending.proposal.risk_level.value}"
    )
