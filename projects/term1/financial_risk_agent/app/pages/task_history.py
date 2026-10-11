'''【126】任务历史页面：通过应用服务展示持久化任务状态，不修改任务或工作流数据。'''

from __future__ import annotations

from typing import Protocol

from app.components.task_status import TaskStatusViewModel, render_task_status
from financial_risk_agent.core.state import WorkflowState


class TaskHistoryService(Protocol):
    """The UI boundary for durable task-history reads only."""

    def list_task_history(self) -> list[WorkflowState]:
        """Return persisted task snapshots ordered by the implementation's history policy."""


def render_task_history(st, service: TaskHistoryService) -> WorkflowState | None:
    """Render a selectable task history without direct Repository or database access."""
    st.title("任务历史")
    states = service.list_task_history()
    if not states:
        st.info("当前没有历史任务。")
        return None
    states_by_id = {state.task_id: state for state in states}
    selected_task_id = st.selectbox(
        "选择历史任务",
        options=list(states_by_id),
        format_func=lambda task_id: states_by_id[task_id].input.title,
    )
    selected = states_by_id[selected_task_id]
    render_task_status(st, TaskStatusViewModel.from_state(selected))
    return selected
