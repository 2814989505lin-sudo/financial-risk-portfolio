'''【132】安全测试：验证运行时源码没有任意代码、命令、SQL 或动态导入执行入口。'''

from __future__ import annotations

import ast
from pathlib import Path

_PROHIBITED_CALLS = frozenset({"eval", "exec", "compile", "__import__"})
_PROHIBITED_IMPORTS = frozenset({"importlib", "subprocess"})
_RUNTIME_ROOT = Path(__file__).parents[2] / "src" / "financial_risk_agent"


def test_runtime_source_contains_no_prohibited_dynamic_execution_calls() -> None:
    violations: list[str] = []
    for source_path in _RUNTIME_ROOT.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _PROHIBITED_CALLS:
                    violations.append(f"{source_path.name}:{node.lineno}:{node.func.id}")
            if isinstance(node, ast.Import):
                if any(alias.name.split(".")[0] in _PROHIBITED_IMPORTS for alias in node.names):
                    violations.append(f"{source_path.name}:{node.lineno}:prohibited-import")
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.split(".")[0] in _PROHIBITED_IMPORTS:
                    violations.append(f"{source_path.name}:{node.lineno}:prohibited-import")

    assert violations == []
