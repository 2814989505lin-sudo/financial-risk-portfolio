'''【087】工作流测试：验证 AnalysisTeam 严格按六个固定 Agent 阶段传递结构化产物。'''

from __future__ import annotations

from dataclasses import dataclass

from financial_risk_agent.agents.compliance import ComplianceAgentResult
from financial_risk_agent.agents.critic import CriticAgentResult
from financial_risk_agent.agents.graph import GraphAgentResult
from financial_risk_agent.agents.intake import IntakeAgentResult
from financial_risk_agent.agents.report import ReportDraft
from financial_risk_agent.agents.rgcn import RGCNAgentResult
from financial_risk_agent.core.schemas import InputSnapshot, RiskPrediction
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.domain.knowledge_graph.models import (
    GraphLineage,
    KnowledgeGraphArtifact,
    KnowledgeGraphEntity,
)
from financial_risk_agent.domain.risk.models import GraphFeatureContract
from financial_risk_agent.orchestration.selector import (
    FIXED_ANALYSIS_SEQUENCE,
    AnalysisStep,
    FixedAnalysisSelector,
)
from financial_risk_agent.orchestration.team import (
    AnalysisTeam,
    AnalysisTeamAgents,
    AnalysisTeamInput,
)


def _input() -> AnalysisTeamInput:
    return AnalysisTeamInput(
        task_id="task-team-001",
        input_snapshot=InputSnapshot(
            source_type="fixture",
            article_id="article-001",
            title="Fixture article",
            content_ref="fixture://article-001",
            content_hash=content_hash("fixture content"),
        ),
        content="fixture content",
        graph_contract=GraphFeatureContract(
            graph_schema_version="knowledge_graph.v1",
            input_feature_dimension=11,
            node_type_vocabulary_hash="b" * 64,
            relation_type_vocabulary_hash="c" * 64,
        ),
    )


def _artifact() -> KnowledgeGraphArtifact:
    return KnowledgeGraphArtifact(
        artifact_id="graph-001",
        entities=[
            KnowledgeGraphEntity(
                entity_id="entity-001",
                entity_type="ORG",
                name="Acme",
                source_document_refs=["article-001"],
            )
        ],
        lineage=GraphLineage(
            source_asset_id="fixture-asset",
            source_asset_hash="a" * 64,
            source_schema_version="knowledge_graph.v1",
            source_document_id="article-001",
        ),
    )


@dataclass
class RecordingAgents:
    calls: list[str]

    @property
    def intake(self):
        calls = self.calls

        class Intake:
            allowed_tool_names = ()

            def run(self, snapshot):
                calls.append("intake")
                return IntakeAgentResult(input_snapshot=snapshot, article_ref=snapshot.content_ref)

        return Intake()

    @property
    def graph(self):
        calls = self.calls
        artifact = _artifact()

        class Graph:
            allowed_tool_names = ("extract_graph",)

            def run(self, **_):
                calls.append("graph")
                return GraphAgentResult(artifact=artifact)

        return Graph()

    @property
    def rgcn(self):
        calls = self.calls
        artifact = _artifact()

        class RGCN:
            allowed_tool_names = ("predict_risk_rgcn",)

            def run(self, **_):
                calls.append("rgcn")
                return RGCNAgentResult(
                    prediction=RiskPrediction(
                        model_version="fixture-model",
                        graph_artifact_hash=artifact.artifact_hash,
                        credit_probability=0.1,
                        market_probability=0.1,
                        operational_probability=0.1,
                        fraud_probability=0.1,
                    )
                )

        return RGCN()

    @property
    def compliance(self):
        calls = self.calls

        class Compliance:
            allowed_tool_names = ("llm_structured_completion",)

            def run(self, **_):
                calls.append("compliance")
                return ComplianceAgentResult.model_validate(
                    {"assessment": {"assessment_id": "assessment-001", "summary": "No findings."}}
                )

        return Compliance()

    @property
    def critic(self):
        calls = self.calls

        class Critic:
            allowed_tool_names = ()

            def run(self, **_):
                calls.append("critic")
                return CriticAgentResult(accepted=True)

        return Critic()

    @property
    def report(self):
        calls = self.calls

        class Report:
            allowed_tool_names = ("llm_structured_completion",)

            def run(self, **_):
                calls.append("report")
                return ReportDraft(
                    report_id="report:task-team-001",
                    article_ref="fixture://article-001",
                    executive_summary="Completed.",
                    critic_accepted=True,
                )

        return Report()


def test_selector_and_team_execute_the_exact_six_turn_sequence() -> None:
    selector = FixedAnalysisSelector()
    assert selector.next_step(()) is AnalysisStep.INTAKE
    assert selector.next_step(FIXED_ANALYSIS_SEQUENCE) is None

    calls: list[str] = []
    recordings = RecordingAgents(calls)
    team = AnalysisTeam(
        AnalysisTeamAgents(
            intake=recordings.intake,
            graph=recordings.graph,
            rgcn=recordings.rgcn,
            compliance=recordings.compliance,
            critic=recordings.critic,
            report=recordings.report,
        )
    )

    result = team.run(_input())

    assert calls == ["intake", "graph", "rgcn", "compliance", "critic", "report"]
    assert [turn.step for turn in result.turns] == list(FIXED_ANALYSIS_SEQUENCE)
    assert result.completed
    assert result.graph and result.graph.artifact.artifact_hash
    assert result.rgcn and result.rgcn.prediction.prediction_hash
    assert result.report and result.report.critic_accepted
