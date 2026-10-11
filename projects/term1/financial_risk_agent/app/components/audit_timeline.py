'''【124】审计时间线组件：将追加式 AuditEvent 转为可筛选但不可编辑的安全展示数据。'''

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from pydantic import Field, field_validator

from financial_risk_agent.core.schemas import StrictModel
from financial_risk_agent.services.audit import AuditEvent


class AuditTimelineItemViewModel(StrictModel):
    """One append-only audit item with only already screened metadata for the UI."""

    sequence_number: int = Field(ge=1)
    event_type: str = Field(min_length=1, max_length=100)
    occurred_at: datetime
    actor_id: str | None = Field(default=None, max_length=200)
    resource_ref: str | None = Field(default=None, max_length=2_000)
    status: str | None = Field(default=None, max_length=100)
    operation: str | None = Field(default=None, max_length=100)

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("occurred_at must include a timezone.")
        return value

    @classmethod
    def from_event(cls, event: AuditEvent) -> AuditTimelineItemViewModel:
        """Discard hashes and metadata unrelated to an operator-visible audit timeline."""
        return cls(
            sequence_number=event.sequence_number,
            event_type=event.event_type,
            occurred_at=event.occurred_at,
            actor_id=event.actor_id,
            resource_ref=event.resource_ref,
            status=_metadata_str(event.metadata, "status"),
            operation=_metadata_str(event.metadata, "operation"),
        )


def render_audit_timeline(st, items: list[AuditTimelineItemViewModel]) -> None:
    """Render read-only chronological audit data and deliberately expose no edit controls."""
    st.subheader("审计时间线")
    if not items:
        st.write("没有符合筛选条件的审计事件。")
        return
    for item in items:
        st.write(
            {
                "序号": item.sequence_number,
                "事件": item.event_type,
                "时间": item.occurred_at.isoformat(),
                "操作者": item.actor_id,
                "资源": item.resource_ref,
                "状态": item.status,
                "操作": item.operation,
            }
        )


def _metadata_str(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None
