'''【098】报告视图组件：将分析检查点转换为图谱、风险、证据与报告草稿的只读展示。'''

from __future__ import annotations

from pydantic import Field

from financial_risk_agent.core.schemas import StrictModel
from financial_risk_agent.orchestration.team import AnalysisTeamResult


class RiskProbabilityView(StrictModel):
    """One fixed risk category probability presented without model checkpoint content."""

    category: str = Field(min_length=1, max_length=100)
    probability: float = Field(ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)


class EvidenceView(StrictModel):
    """One retained evidence locator suitable for display without the original article body."""

    source_ref: str = Field(min_length=1, max_length=500)
    locator: str = Field(min_length=1, max_length=500)
    excerpt_hash: str = Field(min_length=64, max_length=64)


class ReportViewModel(StrictModel):
    """Read-only report, graph, prediction, and evidence data for the case-analysis page."""

    graph_entity_count: int = Field(ge=0)
    graph_relation_count: int = Field(ge=0)
    risk_probabilities: list[RiskProbabilityView] = Field(default_factory=list)
    evidence: list[EvidenceView] = Field(default_factory=list)
    report_summary: str | None = Field(default=None, max_length=10_000)
    conclusions: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)
    critic_accepted: bool | None = None

    @classmethod
    def from_analysis_result(cls, result: AnalysisTeamResult) -> ReportViewModel:
        """Build an incomplete-safe display model from a structured analysis checkpoint."""
        graph = result.graph.artifact if result.graph is not None else None
        prediction = result.rgcn.prediction if result.rgcn is not None else None
        evidence = (
            [
                EvidenceView(
                    source_ref=reference.source_ref,
                    locator=reference.locator,
                    excerpt_hash=reference.excerpt_hash,
                )
                for reference in result.compliance.assessment.evidence_refs
            ]
            if result.compliance is not None
            else []
        )
        report = result.report
        return cls(
            graph_entity_count=len(graph.entities) if graph is not None else 0,
            graph_relation_count=len(graph.relations) if graph is not None else 0,
            risk_probabilities=(
                [
                    RiskProbabilityView(
                        category=category,
                        probability=probability,
                        threshold=prediction.thresholds.get(category),
                    )
                    for category, probability in (
                        ("credit", prediction.credit_probability),
                        ("market", prediction.market_probability),
                        ("operational", prediction.operational_probability),
                        ("fraud", prediction.fraud_probability),
                    )
                ]
                if prediction is not None
                else []
            ),
            evidence=evidence,
            report_summary=report.executive_summary if report is not None else None,
            conclusions=report.conclusions if report is not None else [],
            recommended_next_steps=report.recommended_next_steps if report is not None else [],
            critic_accepted=report.critic_accepted if report is not None else None,
        )


def render_report_view(st, view_model: ReportViewModel) -> None:
    """Render existing structured analysis results without execution controls."""
    st.subheader("图谱摘要")
    st.write(f"实体数量：{view_model.graph_entity_count}")
    st.write(f"关系数量：{view_model.graph_relation_count}")
    st.subheader("风险概率")
    for risk in view_model.risk_probabilities:
        threshold = f"，阈值 {risk.threshold:.2f}" if risk.threshold is not None else ""
        st.write(f"{risk.category}：{risk.probability:.2%}{threshold}")
    st.subheader("证据")
    if view_model.evidence:
        for evidence in view_model.evidence:
            st.write(f"{evidence.source_ref} · {evidence.locator} · {evidence.excerpt_hash}")
    else:
        st.write("当前没有可展示的证据引用。")
    st.subheader("报告草稿")
    if view_model.report_summary is None:
        st.write("分析尚未生成报告草稿。")
        return
    st.write(view_model.report_summary)
    st.write("结论：")
    st.write(view_model.conclusions)
    st.write("建议的下一步：")
    st.write(view_model.recommended_next_steps)
    st.write(f"Critic 复核通过：{view_model.critic_accepted}")
