'''【088】工作流测试：验证 AnalysisTeam 的工具预算、零重试和超时失败会安全停止。'''

from __future__ import annotations

import time

from financial_risk_agent.orchestration.team import AnalysisTeam, AnalysisTeamAgents
from financial_risk_agent.orchestration.termination import AnalysisTeamLimits
from tests.workflow.test_analysis_team import RecordingAgents, _input


class FailingIntake:
    allowed_tool_names = ()

    def __init__(self) -> None:
        self.calls = 0

    def run(self, _):
        self.calls += 1
        raise RuntimeError("unsafe detail")


class SlowIntake:
    allowed_tool_names = ()

    def run(self, _):
        time.sleep(0.05)
        raise AssertionError("the timeout should have stopped the workflow")


class OverBudgetIntake:
    allowed_tool_names = ("first", "second")

    def run(self, _):
        raise AssertionError("tool budget must stop before invoking the agent")


def _agents_with_intake(intake):
    recordings = RecordingAgents([])
    return AnalysisTeamAgents(
        intake=intake,
        graph=recordings.graph,
        rgcn=recordings.rgcn,
        compliance=recordings.compliance,
        critic=recordings.critic,
        report=recordings.report,
    )


def test_failure_has_no_retry_and_keeps_only_a_safe_error_record() -> None:
    failing = FailingIntake()
    result = AnalysisTeam(_agents_with_intake(failing)).run(_input())

    assert failing.calls == 1
    assert not result.completed
    assert result.error_records[0].code == "AGENT_EXECUTION_FAILED"
    assert "unsafe detail" not in result.error_records[0].message
    assert len(result.turns) == 1


def test_tool_budget_and_timeout_stop_before_next_agent_turn() -> None:
    over_budget = AnalysisTeam(
        _agents_with_intake(OverBudgetIntake()),
        limits=AnalysisTeamLimits(max_tools_per_turn=1),
    ).run(_input())
    timed_out = AnalysisTeam(
        _agents_with_intake(SlowIntake()),
        limits=AnalysisTeamLimits(agent_timeout_seconds=0.001),
    ).run(_input())

    assert over_budget.error_records[0].code == "TOOL_CALL_LIMIT_EXCEEDED"
    assert over_budget.turns[0].declared_tool_count == 2
    assert timed_out.error_records[0].code == "AGENT_TIMEOUT"
    assert len(timed_out.turns) == 1
