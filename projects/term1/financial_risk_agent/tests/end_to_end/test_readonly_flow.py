'''【135】端到端测试：验证只读案例流程不会产生审批决定、执行凭证或案例修改。'''

from __future__ import annotations

from app.pages.case_analysis import render_case_analysis
from tests.end_to_end.test_readonly_case_analysis import FakeCaseAnalysisService, FakeStreamlit


def test_readonly_case_flow_never_exposes_a_mutation_path() -> None:
    st = FakeStreamlit()
    service = FakeCaseAnalysisService()

    task = render_case_analysis(st, service)

    assert task is not None
    assert service.created_case_ids == ["sample-001"]
    assert service.started_task_ids == ["task-ui-e2e-001"]
    assert set(st.button_labels) == {"创建只读分析任务", "启动或继续只读分析"}
    assert all("批准" not in label and "执行" not in label for label in st.button_labels)
