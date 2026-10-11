'''【123】审批提案组件：将不可执行的 ActionProposal 转为审批页所需的安全摘要。'''

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from financial_risk_agent.core.enums import MutationToolName
from financial_risk_agent.core.schemas import ActionProposal, StrictModel


class ProposalEvidenceView(StrictModel):
    """A display-safe evidence locator that omits the original content body."""

    source_ref: str = Field(min_length=1, max_length=500)
    locator: str = Field(min_length=1, max_length=500)
    excerpt_hash: str = Field(min_length=64, max_length=64)


class ProposalSummaryViewModel(StrictModel):
    """The complete read-only proposal summary required before a human makes a decision."""

    action_id: str = Field(min_length=1, max_length=200)
    proposal_hash: str = Field(min_length=64, max_length=64)
    tool_name: str = Field(min_length=1, max_length=100)
    target_resource: str = Field(min_length=1, max_length=1_000)
    parameter_summary: list[str] = Field(default_factory=list)
    risk_level: str = Field(min_length=1, max_length=100)
    risk_reason: str = Field(min_length=1, max_length=10_000)
    evidence: list[ProposalEvidenceView] = Field(default_factory=list)
    expected_effect: str = Field(min_length=1, max_length=10_000)
    reversible: bool
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("expires_at must include a timezone.")
        return value

    @classmethod
    def from_proposal(
        cls,
        proposal: ActionProposal,
        *,
        expires_at: datetime,
    ) -> ProposalSummaryViewModel:
        """Build the approval payload from structured, credential-screened proposal fields only."""
        proposal_hash = proposal.proposal_hash
        if proposal_hash is None:
            raise ValueError("proposal_hash is required for approval display.")
        return cls(
            action_id=proposal.action_id,
            proposal_hash=proposal_hash,
            tool_name=proposal.tool_name.value,
            target_resource=proposal.target_resource,
            parameter_summary=[
                f"{key}={_summarize_value(value)}"
                for key, value in sorted(proposal.validated_arguments.items())
            ],
            risk_level=proposal.risk_level.value,
            risk_reason=proposal.reason,
            evidence=[
                ProposalEvidenceView(
                    source_ref=reference.source_ref,
                    locator=reference.locator,
                    excerpt_hash=reference.excerpt_hash,
                )
                for reference in proposal.evidence_refs
            ],
            expected_effect=proposal.expected_effect,
            reversible=proposal.tool_name is MutationToolName.UPDATE_CASE_STATUS,
            expires_at=expires_at,
        )


def render_proposal_summary(st, view_model: ProposalSummaryViewModel) -> None:
    """Render proposal details without giving the component execution or storage access."""
    st.subheader("待审批提案")
    st.write(f"操作编号：{view_model.action_id}")
    st.write(f"提案哈希：{view_model.proposal_hash}")
    st.write(f"操作工具：{view_model.tool_name}")
    st.write(f"目标资源：{view_model.target_resource}")
    st.write(f"风险等级：{view_model.risk_level}")
    st.write(f"风险原因：{view_model.risk_reason}")
    st.write(f"参数摘要：{view_model.parameter_summary}")
    st.write(f"预期变化：{view_model.expected_effect}")
    st.write(f"可回滚：{'是' if view_model.reversible else '否'}")
    st.write(f"审批有效期至：{view_model.expires_at.isoformat()}")
    st.write("证据引用：")
    st.write([evidence.model_dump() for evidence in view_model.evidence])


def _summarize_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"object[{len(value)}]"
    return type(value).__name__
