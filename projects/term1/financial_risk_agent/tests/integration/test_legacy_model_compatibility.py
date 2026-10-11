'''【059】集成测试：验证可信 RGCN 模型包兼容性，并拒绝未封装的旧 checkpoint。'''

from __future__ import annotations

from pathlib import Path

import pytest

from financial_risk_agent.domain.risk.models import GraphFeatureContract
from financial_risk_agent.integrations.legacy.model_bundle import (
    LegacyModelBundleAdapter,
    ModelBundleCompatibilityError,
    ModelBundleRejected,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests/fixtures/rgcn"


def test_verified_fixture_bundle_is_compatible_with_fixed_graph_contract() -> None:
    adapter = LegacyModelBundleAdapter()
    bundle = adapter.load(FIXTURE_ROOT / "trusted_bundle")
    graph_contract = GraphFeatureContract.model_validate_json(
        (FIXTURE_ROOT / "compatible_graph_contract.json").read_text(encoding="utf-8")
    )

    first = adapter.check_graph_compatibility(bundle, graph_contract)
    repeated = adapter.check_graph_compatibility(bundle, graph_contract)

    assert first == repeated
    assert first.model_version == "fixture-rgcn-v1"
    assert first.compatibility_hash


def test_incompatible_graph_and_unpackaged_legacy_checkpoint_are_rejected() -> None:
    adapter = LegacyModelBundleAdapter()
    bundle = adapter.load(FIXTURE_ROOT / "trusted_bundle")
    graph_contract = GraphFeatureContract.model_validate_json(
        (FIXTURE_ROOT / "compatible_graph_contract.json").read_text(encoding="utf-8")
    )
    incompatible_contract = GraphFeatureContract.model_validate(
        {
            **graph_contract.model_dump(),
            "input_feature_dimension": 12,
            "contract_hash": None,
        }
    )

    with pytest.raises(ModelBundleCompatibilityError, match="feature dimension"):
        adapter.check_graph_compatibility(bundle, incompatible_contract)
    with pytest.raises(ModelBundleRejected, match="not eligible"):
        adapter.load(PROJECT_ROOT / "artifacts/models/legacy-rgcn-checkpoint-v1")
