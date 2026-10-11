'''【100】Streamlit 启动入口：加载已注册的只读案例分析服务并渲染最小案例分析页面。'''

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from app.local_demo_services import LocalDemoServices
from app.pages.approvals import ApprovalUIError, ApprovalUIService, render_approvals
from app.pages.audit_trail import AuditTrailService, render_audit_trail
from app.pages.case_analysis import CaseAnalysisService, CaseAnalysisUIError, render_case_analysis
from app.pages.task_history import TaskHistoryService, render_task_history


def main() -> None:
    """Render injected production services or a safe local sample mode when run directly."""
    _register_local_demo_services()
    pages = _available_pages()
    page_name = st.sidebar.selectbox("功能页面", options=list(pages))
    try:
        pages[page_name]()
    except CaseAnalysisUIError:
        st.error("只读案例分析暂时不可用，请检查任务状态后重试。")
    except ApprovalUIError:
        st.error("审批请求已变化或暂时不可用，请刷新后重新确认。")


def _register_local_demo_services() -> None:
    """Install session-local demo services only when a deployment host supplied none."""
    if any(
        key in st.session_state
        for key in (
            "case_analysis_service",
            "approval_ui_service",
            "task_history_service",
            "audit_trail_service",
        )
    ):
        return
    services = LocalDemoServices()
    st.session_state["case_analysis_service"] = services
    st.session_state["approval_ui_service"] = services
    st.session_state["task_history_service"] = services
    st.session_state["audit_trail_service"] = services


def _available_pages() -> dict[str, Callable[[], object]]:
    """Expose only injected-service pages; the UI imports no database or model implementation."""
    pages: dict[str, Callable[[], object]] = {}
    case_service = st.session_state.get("case_analysis_service")
    if case_service is not None:
        pages["案例分析"] = lambda: render_case_analysis(
            st, _as_case_analysis_service(case_service)
        )
    approval_service = st.session_state.get("approval_ui_service")
    if approval_service is not None:
        pages["人工审批"] = lambda: render_approvals(st, _as_approval_ui_service(approval_service))
    history_service = st.session_state.get("task_history_service")
    if history_service is not None:
        pages["任务历史"] = lambda: render_task_history(
            st, _as_task_history_service(history_service)
        )
    audit_service = st.session_state.get("audit_trail_service")
    if audit_service is not None:
        pages["审计记录"] = lambda: render_audit_trail(st, _as_audit_trail_service(audit_service))
    return pages


def _as_case_analysis_service(service: object) -> CaseAnalysisService:
    """Keep the entry point typed without importing databases, models, or application internals."""
    return service  # type: ignore[return-value]


def _as_approval_ui_service(service: object) -> ApprovalUIService:
    """Keep the Streamlit entry typed without importing persistence or execution implementations."""
    return service  # type: ignore[return-value]


def _as_task_history_service(service: object) -> TaskHistoryService:
    """Keep the Streamlit entry typed without importing persistence or execution implementations."""
    return service  # type: ignore[return-value]


def _as_audit_trail_service(service: object) -> AuditTrailService:
    """Keep the Streamlit entry typed without importing persistence or execution implementations."""
    return service  # type: ignore[return-value]


if __name__ == "__main__":
    main()
