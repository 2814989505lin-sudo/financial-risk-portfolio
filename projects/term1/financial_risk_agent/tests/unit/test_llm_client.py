'''【074】单元测试：验证离线 OpenAI 兼容客户端的重试、结构化校验和启动自检。'''

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import Field

from financial_risk_agent.config.model_capabilities import ModelCapabilities
from financial_risk_agent.config.settings import Settings
from financial_risk_agent.core.schemas import StrictModel
from financial_risk_agent.integrations.llm_client import (
    HTTPResponse,
    LLMConnectionError,
    LLMMessage,
    LLMRequest,
    LLMStructuredOutputError,
    OpenAICompatibleLLMClient,
)
from financial_risk_agent.services.startup_check import StartupChecker, StartupCheckStatus


@dataclass
class FakeTransport:
    responses: deque[HTTPResponse | Exception]
    requests: list[dict[str, Any]]

    def post(self, **kwargs: Any) -> HTTPResponse:
        self.requests.append(kwargs)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class ProbeOutput(StrictModel):
    status: str = Field(pattern=r"^ok$")


def _settings() -> Settings:
    return Settings(
        app_env="test",
        llm_base_url="https://llm.example.test/v1",
        llm_api_key="unit-test-secret",
        llm_model="qwen3.7-max",
        llm_timeout_seconds=0.1,
        llm_max_retries=1,
    )


def _response(
    *, content: str = "pong", tool_calls: list[dict[str, Any]] | None = None
) -> HTTPResponse:
    return HTTPResponse(
        status_code=200,
        payload={
            "model": "qwen3.7-max",
            "choices": [{"message": {"content": content, "tool_calls": tool_calls or []}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


def test_client_uses_configured_provider_default_model_and_bounded_retry() -> None:
    transport = FakeTransport(
        responses=deque([LLMConnectionError("offline"), _response()]), requests=[]
    )
    client = OpenAICompatibleLLMClient(
        _settings(),
        ModelCapabilities(model_name="qwen3.7-max"),
        transport=transport,
        retry_delay_seconds=0,
    )

    response = client.complete(LLMRequest(messages=[LLMMessage(role="user", content="ping")]))

    assert response.content == "pong"
    assert len(transport.requests) == 2
    assert transport.requests[0]["url"] == "https://llm.example.test/v1/chat/completions"
    assert transport.requests[0]["payload"]["model"] == "qwen3.7-max"
    assert transport.requests[0]["headers"]["Authorization"] == "Bearer unit-test-secret"


def test_client_validates_structured_output_without_exposing_provider_failure_details() -> None:
    transport = FakeTransport(responses=deque([_response(content='{"status":"ok"}')]), requests=[])
    client = OpenAICompatibleLLMClient(
        _settings(), ModelCapabilities(model_name="qwen3.7-max"), transport=transport
    )

    response = client.complete(
        LLMRequest(messages=[LLMMessage(role="user", content="json")]), response_model=ProbeOutput
    )
    assert isinstance(response.structured_output, ProbeOutput)
    assert transport.requests[0]["payload"]["response_format"]["type"] == "json_schema"

    invalid_client = OpenAICompatibleLLMClient(
        _settings(),
        ModelCapabilities(model_name="qwen3.7-max"),
        transport=FakeTransport(responses=deque([_response(content="not-json")]), requests=[]),
    )
    with pytest.raises(LLMStructuredOutputError) as error:
        invalid_client.complete(
            LLMRequest(messages=[LLMMessage(role="user", content="json")]),
            response_model=ProbeOutput,
        )
    assert "unit-test-secret" not in str(error.value)


def test_startup_check_distinguishes_connection_tool_and_structured_output_failures() -> None:
    tool_call = {
        "id": "call-001",
        "function": {"name": "startup_probe", "arguments": "{}"},
    }
    client = OpenAICompatibleLLMClient(
        _settings(),
        ModelCapabilities(model_name="qwen3.7-max"),
        transport=FakeTransport(
            responses=deque(
                [_response(), _response(tool_calls=[tool_call]), _response(content="not-json")]
            ),
            requests=[],
        ),
    )

    report = StartupChecker(client).run()

    assert report.connection.status is StartupCheckStatus.PASSED
    assert report.tool_calling.status is StartupCheckStatus.PASSED
    assert report.structured_output.status is StartupCheckStatus.FAILED
    assert report.structured_output.error_code == "STRUCTURED_OUTPUT_FAILED"
