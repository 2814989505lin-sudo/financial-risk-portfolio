'''【099】案例分析页面：通过只读应用服务创建、启动、恢复并展示金融风险分析任务。'''

from __future__ import annotations

from typing import Protocol

from pydantic import Field

from app.components.report_view import ReportViewModel, render_report_view
from app.components.task_status import TaskStatusViewModel, render_task_status
from financial_risk_agent.core.schemas import StrictModel
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.orchestration.team import AnalysisTeamResult

_SELECTED_TASK_KEY = "selected_readonly_case_task_id"


class SampleCase(StrictModel):
    """Safe case-selection metadata that intentionally excludes raw article body text."""

    case_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=1_000)
    source_ref: str = Field(min_length=1, max_length=2_000)


class CaseAnalysisTask(StrictModel):
    """One task snapshot supplied by the application service to the Streamlit view layer."""

    state: WorkflowState
    analysis_result: AnalysisTeamResult | None = None


class CaseAnalysisService(Protocol):
    """Application boundary used by the UI; implementations own storage, models, and workflows."""

    def list_sample_cases(self) -> list[SampleCase]:
        """Return selectable sample case metadata without article bodies."""

    def create_task(self, case_id: str) -> CaseAnalysisTask:
        """Create and persist one read-only analysis task for the selected case."""

    def get_task(self, task_id: str) -> CaseAnalysisTask:
        """Load one persisted task for refresh-safe status and result display."""

    def start_or_resume_readonly_analysis(self, task_id: str) -> CaseAnalysisTask:
        """Run or resume only the predefined read-only analysis workflow."""


class CaseAnalysisUIError(RuntimeError):
    """The UI service failed; its internal error detail is never displayed."""


def render_case_analysis(st, service: CaseAnalysisService) -> CaseAnalysisTask | None:
    """Render a refresh-safe workflow without database or model access."""
    st.title("只读案例分析")
    samples = service.list_sample_cases()
    if not samples:
        st.info("当前没有可用的样例案例。")
        return None
    samples_by_id = {sample.case_id: sample for sample in samples}
    selected_case_id = st.selectbox(
        "选择样例案例",
        options=list(samples_by_id),
        format_func=lambda case_id: samples_by_id[case_id].title,
    )
    task: CaseAnalysisTask | None = None
    if st.button("创建只读分析任务"):
        try:
            task = service.create_task(selected_case_id)
        except Exception as error:
            raise CaseAnalysisUIError("Unable to create read-only analysis task.") from error
        st.session_state[_SELECTED_TASK_KEY] = task.state.task_id
    task_id = st.session_state.get(_SELECTED_TASK_KEY)
    if task is None and task_id:
        try:
            task = service.get_task(task_id)
        except Exception as error:
            raise CaseAnalysisUIError("Unable to load read-only analysis task.") from error
    if task is None:
        return None
    render_task_status(st, TaskStatusViewModel.from_state(task.state))
    if st.button("启动或继续只读分析"):
        try:
            task = service.start_or_resume_readonly_analysis(task.state.task_id)
        except Exception as error:
            raise CaseAnalysisUIError("Unable to start read-only analysis.") from error
        st.session_state[_SELECTED_TASK_KEY] = task.state.task_id
        render_task_status(st, TaskStatusViewModel.from_state(task.state))
    if task.analysis_result is not None:
        render_report_view(st, ReportViewModel.from_analysis_result(task.analysis_result))
    return task
