'''【136】本地样例服务测试：验证直接启动时可创建并完成不依赖外部凭据的演示分析。'''

from __future__ import annotations

from app.local_demo_services import LocalDemoServices
from financial_risk_agent.core.enums import WorkflowStatus


def test_local_demo_service_creates_and_completes_a_sample_analysis() -> None:
    service = LocalDemoServices()

    sample = service.list_sample_cases()[0]
    created = service.create_task(sample.case_id)
    completed = service.start_or_resume_readonly_analysis(created.state.task_id)

    assert created.state.status is WorkflowStatus.CREATED
    assert completed.state.status is WorkflowStatus.COMPLETED
    assert completed.analysis_result is not None
    assert completed.analysis_result.report is not None
    assert service.list_task_history() == [completed.state]
