'''【045】单元测试：验证环境变量配置、密钥隐藏和配置优先级。'''

from __future__ import annotations

import pytest

from financial_risk_agent.config.settings import ConfigurationError, Settings


def test_loads_safe_defaults_without_llm_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINANCIAL_RISK_AGENT_APP_ENV", raising=False)
    monkeypatch.delenv("FINANCIAL_RISK_AGENT_DATABASE_PATH", raising=False)
    monkeypatch.delenv("FINANCIAL_RISK_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("FINANCIAL_RISK_AGENT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("FINANCIAL_RISK_AGENT_LLM_MODEL", raising=False)

    settings = Settings.from_environment()

    assert settings.app_env == "development"
    assert str(settings.database_path) == "var/runtime/financial_risk_agent.sqlite3"
    assert settings.llm_base_url is None
    assert settings.llm_api_key is None
    assert settings.llm_model == "qwen3.7-max"


def test_rejects_unknown_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCIAL_RISK_AGENT_APP_ENV", "unsafe")

    with pytest.raises(ConfigurationError, match="APP_ENV"):
        Settings.from_environment()


def test_llm_settings_are_required_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINANCIAL_RISK_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("FINANCIAL_RISK_AGENT_LLM_API_KEY", raising=False)

    settings = Settings.from_environment()

    with pytest.raises(ConfigurationError, match="LLM_BASE_URL"):
        settings.require_llm_configuration()


def test_safe_diagnostics_exclude_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCIAL_RISK_AGENT_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("FINANCIAL_RISK_AGENT_LLM_API_KEY", "test-secret-value")

    settings = Settings.from_environment()

    assert "test-secret-value" not in repr(settings)
    assert settings.to_safe_dict() == {
        "app_env": "development",
        "database_path": "var/runtime/financial_risk_agent.sqlite3",
        "llm_base_url": "https://llm.example.test/v1",
        "llm_model": "qwen3.7-max",
        "llm_timeout_seconds": "15.0",
        "llm_max_retries": "2",
    }


def test_loads_values_from_explicit_env_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FINANCIAL_RISK_AGENT_APP_ENV=test\n"
        "FINANCIAL_RISK_AGENT_DATABASE_PATH=var/test.sqlite3\n"
        "FINANCIAL_RISK_AGENT_LLM_BASE_URL=https://llm.example.test/v1\n"
        "FINANCIAL_RISK_AGENT_LLM_API_KEY=local-secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FINANCIAL_RISK_AGENT_APP_ENV", raising=False)
    monkeypatch.delenv("FINANCIAL_RISK_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("FINANCIAL_RISK_AGENT_LLM_API_KEY", raising=False)

    settings = Settings.from_environment(env_file=env_file)

    assert settings.app_env == "test"
    assert str(settings.database_path) == "var/test.sqlite3"
    assert settings.llm_base_url == "https://llm.example.test/v1"
    assert settings.llm_api_key == "local-secret"


def test_process_environment_overrides_env_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FINANCIAL_RISK_AGENT_APP_ENV=production\n", encoding="utf-8")
    monkeypatch.setenv("FINANCIAL_RISK_AGENT_APP_ENV", "test")

    settings = Settings.from_environment(env_file=env_file)

    assert settings.app_env == "test"
