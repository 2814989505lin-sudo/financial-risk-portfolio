'''【127】审计页面：按任务和事件类型读取追加式审计记录，页面不提供编辑入口。'''

from __future__ import annotations

from typing import Protocol

from app.components.audit_timeline import AuditTimelineItemViewModel, render_audit_timeline
from financial_risk_agent.services.audit import AuditEvent


class AuditTrailService(Protocol):
    """The read-only application boundary for audited event retrieval and filtering."""

    def list_audit_task_ids(self) -> list[str]:
        """Return task identifiers that have audit events."""

    def list_audit_events(self, task_id: str, *, event_type: str | None = None) -> list[AuditEvent]:
        """Return append-only audit events filtered by task and optionally event type."""


def render_audit_trail(st, service: AuditTrailService) -> list[AuditTimelineItemViewModel]:
    """Render a task and event-type-filtered audit timeline without any modification controls."""
    st.title("审计记录")
    task_ids = service.list_audit_task_ids()
    if not task_ids:
        st.info("当前没有审计记录。")
        return []
    task_id = st.selectbox("选择审计任务", options=task_ids)
    event_type = st.text_input("事件类型筛选（可选）", value="").strip() or None
    events = service.list_audit_events(task_id, event_type=event_type)
    items = [AuditTimelineItemViewModel.from_event(event) for event in events]
    render_audit_timeline(st, items)
    return items
