'''【075】集成测试：仅在显式测试 Provider 配置存在时验证真实 OpenAI 兼容接口契约。'''

from __future__ import annotations

import os

import pytest

from financial_risk_agent.config.model_capabilities import configured_model_capabilities
from financial_risk_agent.config.settings import Settings
from financial_risk_agent.integrations.llm_client import (
    LLMMessage,
    LLMRequest,
    OpenAICompatibleLLMClient,
)

TEST_BASE_URL = os.getenv("FINANCIAL_RISK_AGENT_TEST_LLM_BASE_URL")
TEST_API_KEY = os.getenv("FINANCIAL_RISK_AGENT_TEST_LLM_API_KEY")
TEST_MODEL = os.getenv("FINANCIAL_RISK_AGENT_TEST_LLM_MODEL", "qwen3.7-max")

pytestmark = pytest.mark.skipif(
    not TEST_BASE_URL or not TEST_API_KEY,
    reason="requires FINANCIAL_RISK_AGENT_TEST_LLM_BASE_URL and TEST_LLM_API_KEY",
)


def test_configured_provider_satisfies_basic_chat_contract() -> None:
    settings = Settings(
        app_env="test",
        llm_base_url=TEST_BASE_URL,
        llm_api_key=TEST_API_KEY,
        llm_model=TEST_MODEL,
        llm_timeout_seconds=10,
        llm_max_retries=1,
    )
    client = OpenAICompatibleLLMClient(settings, configured_model_capabilities(TEST_MODEL))

    response = client.complete(
        LLMRequest(messages=[LLMMessage(role="user", content="Reply with ok.")])
    )

    assert response.model
    assert response.content
