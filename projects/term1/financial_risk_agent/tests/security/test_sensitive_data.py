'''【134】安全测试：验证新增标签工具、提案和审计模型拒绝敏感字段且不会序列化凭据。'''

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import dotenv_values
from pydantic import ValidationError

from financial_risk_agent.services.audit import AuditEventDraft
from financial_risk_agent.tools.mutations.update_case_label import UpdateCaseLabelArguments
from tests.integration.test_update_case_label import _label_proposal


def test_label_mutation_contracts_reject_sensitive_fields_and_serialized_credentials() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        UpdateCaseLabelArguments(
            case_id="case-001",
            new_label="ELEVATED_RISK",
            expected_version=1,
            api_key="must-not-be-accepted",
        )
    with pytest.raises(ValueError, match="credentials or secrets"):
        proposal = _label_proposal(validated_arguments={"api_key": "must-not-be-stored"})
        proposal.model_dump_json()
    with pytest.raises(ValueError, match="credentials or secrets"):
        AuditEventDraft(event_type="MUTATION_EXECUTED", metadata={"token": "must-not-log"})

    serialized = _label_proposal().model_dump_json()
    assert "api_key" not in serialized
    assert "token" not in serialized


def test_environment_template_does_not_ship_llm_endpoint_or_credentials() -> None:
    template_path = Path(__file__).resolve().parents[2] / ".env.example"

    template = dotenv_values(template_path)

    assert template["FINANCIAL_RISK_AGENT_LLM_BASE_URL"] == ""
    assert template["FINANCIAL_RISK_AGENT_LLM_API_KEY"] == ""
