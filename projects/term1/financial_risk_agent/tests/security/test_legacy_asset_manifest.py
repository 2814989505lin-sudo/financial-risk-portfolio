'''【039】安全测试：验证历史资产清单哈希、旧代码凭据清理和运行时隔离。'''

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/legacy_assets.json"
LEGACY_SOURCE_ROOTS = (
    WORKSPACE_ROOT / "SourceCode_LINYi_IndustrialProject20",
    WORKSPACE_ROOT / "HSBC_10_26",
    WORKSPACE_ROOT / "HSBC",
)
LITERAL_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|password)\s*=\s*['\"][^'\"]{8,}['\"]"),
)


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_is_complete_and_hashes_match() -> None:
    manifest = load_manifest()

    assert manifest["schema_version"] == "1.0"
    assert manifest["hash_algorithm"] == "sha256"
    assert len(manifest["assets"]) >= 9

    for asset in manifest["assets"]:
        path = PROJECT_ROOT / asset["relative_path"]
        assert path.is_file(), f"Missing asset: {asset['relative_path']}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == asset["sha256"], f"Hash mismatch: {asset['asset_id']}"


def test_manifest_contains_no_sensitive_fields() -> None:
    sensitive_names = {"api_key", "apikey", "password", "secret", "token", "credential"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                assert not any(name in key.lower() for name in sensitive_names)
                walk(nested_value)
        elif isinstance(value, list):
            for nested_value in value:
                walk(nested_value)

    walk(load_manifest())


def test_legacy_sources_have_no_literal_credentials_and_parse() -> None:
    for root in LEGACY_SOURCE_ROOTS:
        for path in root.rglob("*.py"):
            if ".conda" in path.parts or "__pycache__" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            for pattern in LITERAL_SECRET_PATTERNS:
                assert not pattern.search(source), f"Literal credential found in {path}"

        for path in root.rglob("*.ipynb"):
            source = path.read_text(encoding="utf-8")
            for pattern in LITERAL_SECRET_PATTERNS:
                assert not pattern.search(source), f"Literal credential found in {path}"


def test_new_application_source_has_no_legacy_runtime_imports() -> None:
    forbidden_markers = ("SourceCode_LINYi_IndustrialProject20", "HSBC_10_26", "../HSBC")

    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(marker in source for marker in forbidden_markers), path
