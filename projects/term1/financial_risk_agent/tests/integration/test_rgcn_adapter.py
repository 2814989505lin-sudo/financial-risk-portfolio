'''【064】集成测试：验证 RGCN 适配器对缺失、篡改与不兼容模型输入的安全失败。'''

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from financial_risk_agent.domain.risk.models import GraphFeatureContract
from financial_risk_agent.integrations.legacy.model_bundle import ModelBundleIntegrityError
from financial_risk_agent.integrations.rgcn_adapter import (
    RGCNInferenceAdapter,
    RGCNInferenceCompatibilityError,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures/rgcn"


def test_incompatible_graph_contract_fails_before_inference() -> None:
    adapter = RGCNInferenceAdapter.from_bundle_directory(FIXTURE_ROOT / "trusted_bundle")
    contract = GraphFeatureContract.model_validate_json(
        (FIXTURE_ROOT / "compatible_graph_contract.json").read_text(encoding="utf-8")
    )
    incompatible_contract = GraphFeatureContract.model_validate(
        {**contract.model_dump(), "input_feature_dimension": 12, "contract_hash": None}
    )

    with pytest.raises(RGCNInferenceCompatibilityError, match="feature dimension"):
        adapter.predict(_minimal_artifact(), incompatible_contract)


def test_missing_or_tampered_model_file_fails_before_adapter_registration(tmp_path: Path) -> None:
    missing_bundle = tmp_path / "missing"
    shutil.copytree(FIXTURE_ROOT / "trusted_bundle", missing_bundle)
    (missing_bundle / "checkpoint.bin").unlink()
    with pytest.raises(ModelBundleIntegrityError, match="missing"):
        RGCNInferenceAdapter.from_bundle_directory(missing_bundle)

    tampered_bundle = tmp_path / "tampered"
    shutil.copytree(FIXTURE_ROOT / "trusted_bundle", tampered_bundle)
    (tampered_bundle / "checkpoint.bin").write_text("tampered", encoding="utf-8")
    with pytest.raises(ModelBundleIntegrityError, match="SHA-256"):
        RGCNInferenceAdapter.from_bundle_directory(tampered_bundle)


def _minimal_artifact():
    from tests.unit.test_risk_prediction import _artifact

    return _artifact()
