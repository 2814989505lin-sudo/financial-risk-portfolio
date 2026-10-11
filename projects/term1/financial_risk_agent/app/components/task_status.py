'''【097】任务状态组件：将持久化工作流状态转换为不含敏感信息的界面展示数据。'''

from __future__ import annotations

from pydantic import Field

from financial_risk_agent.core.schemas import StrictModel
from financial_risk_agent.core.state import WorkflowState


class TaskStatusViewModel(StrictModel):
    """A secret-free summary of one durable workflow task for the Streamlit status panel."""

    task_id: str = Field(min_length=1, max_length=200)
    article_title: str = Field(min_length=1, max_length=1_000)
    status: str = Field(min_length=1, max_length=100)
    current_phase: str = Field(min_length=1, max_length=100)
    retry_count: int = Field(ge=0)
    latest_error_code: str | None = Field(default=None, max_length=100)

    @classmethod
    def from_state(cls, state: WorkflowState) -> TaskStatusViewModel:
        """Build a display-only status summary without exposing input content or error messages."""
        return cls(
            task_id=state.task_id,
            article_title=state.input.title,
            status=state.status.value,
            current_phase=state.current_phase.value,
            retry_count=sum(state.retry_counters.values()),
            latest_error_code=state.error_records[-1].code if state.error_records else None,
        )


def render_task_status(st, view_model: TaskStatusViewModel) -> None:
    """Render task metadata using only the supplied view model and no storage dependencies."""
    st.subheader("任务状态")
    st.write(f"任务编号：{view_model.task_id}")
    st.write(f"案例：{view_model.article_title}")
    st.write(f"状态：{view_model.status}")
    st.write(f"当前阶段：{view_model.current_phase}")
    st.write(f"累计重试：{view_model.retry_count}")
    if view_model.latest_error_code is not None:
        st.warning(f"最近错误代码：{view_model.latest_error_code}")
