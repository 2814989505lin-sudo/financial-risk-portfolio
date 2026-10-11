'''【058】单元测试：验证 RGCN 模型包元数据、固定风险顺序、阈值和训练集词表约束。'''

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_risk_agent.core.schemas import RiskPrediction
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.domain.risk.models import RGCNModelManifest

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures/rgcn"


def load_manifest_payload() -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / "trusted_bundle/manifest.json").read_text(encoding="utf-8"))


def test_complete_model_manifest_has_fixed_categories_and_training_only_vocabulary() -> None:
    manifest = RGCNModelManifest.model_validate(load_manifest_payload())

    assert [category.value for category in manifest.risk_category_order] == [
        "credit",
        "market",
        "operational",
        "fraud",
    ]
    assert manifest.feature_contract.vocabulary_source == "training_only"
    assert manifest.calibration_dataset_hash == manifest.validation_dataset_hash
    assert manifest.manifest_hash


def test_model_manifest_rejects_wrong_category_order_or_missing_threshold() -> None:
    wrong_order = load_manifest_payload()
    wrong_order["risk_category_order"] = ["market", "credit", "operational", "fraud"]
    with pytest.raises(ValidationError, match="risk_category_order"):
        RGCNModelManifest.model_validate(wrong_order)

    missing_threshold = load_manifest_payload()
    thresholds = dict(missing_threshold["thresholds"])
    thresholds.pop("fraud")
    missing_threshold["thresholds"] = thresholds
    with pytest.raises(ValidationError, match="thresholds"):
        RGCNModelManifest.model_validate(missing_threshold)


def test_risk_prediction_can_reference_a_verified_model_bundle_without_credentials() -> None:
    prediction = RiskPrediction(
        model_version="fixture-rgcn-v1",
        model_bundle_ref="model:fixture-rgcn-v1",
        model_bundle_manifest_hash="d" * 64,
        graph_artifact_hash=content_hash("graph-001"),
        credit_probability=0.1,
        market_probability=0.2,
        operational_probability=0.3,
        fraud_probability=0.4,
    )

    assert prediction.model_bundle_ref == "model:fixture-rgcn-v1"
    assert prediction.model_bundle_manifest_hash == "d" * 64
