'''【068】单元测试：验证只读工具注册表的白名单、参数模型和注册边界。'''

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from financial_risk_agent.core.enums import MutationToolName
from financial_risk_agent.tools.registry import (
    ToolAlreadyRegisteredError,
    ToolCallParameters,
    ToolNotRegisteredError,
    ToolRegistrationError,
    ToolRegistry,
)


class EchoParameters(ToolCallParameters):
    message: str = Field(min_length=1, max_length=100)


class LooseParameters(BaseModel):
    message: str


def echo(*, message: str) -> dict[str, str]:
    return {"message": message}


def test_registry_only_exposes_explicit_unique_readonly_tools() -> None:
    registry = ToolRegistry()
    registry.register(name="echo", parameters_model=EchoParameters, handler=echo)

    assert registry.names == ("echo",)
    assert (
        registry.get("echo").parameters_model.model_validate({"message": "safe"}).message == "safe"
    )
    with pytest.raises(ToolNotRegisteredError):
        registry.get("unknown_tool")
    with pytest.raises(ToolAlreadyRegisteredError):
        registry.register(name="echo", parameters_model=EchoParameters, handler=echo)


def test_registry_rejects_mutation_tools_and_non_strict_parameter_models() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolRegistrationError, match="Mutation"):
        registry.register(
            name=MutationToolName.UPDATE_CASE_STATUS.value,
            parameters_model=EchoParameters,
            handler=echo,
        )
    with pytest.raises(ToolRegistrationError, match="ToolCallParameters"):
        registry.register(name="unsafe", parameters_model=LooseParameters, handler=echo)
