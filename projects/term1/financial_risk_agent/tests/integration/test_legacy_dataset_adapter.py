'''【034】集成测试：验证清单登记的 Guardian 新闻数据只读转换且源文件不变。'''

from __future__ import annotations

import hashlib
from pathlib import Path

from financial_risk_agent.integrations.legacy.dataset_adapter import (
    LegacyDatasetAdapter,
    load_dataset_asset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/legacy_assets.json"


def test_reads_manifest_registered_guardian_article_without_mutating_source() -> None:
    asset = load_dataset_asset(MANIFEST_PATH, "guardian-news-val-v1")
    before_hash = hashlib.sha256(asset.path.read_bytes()).hexdigest()
    adapter = LegacyDatasetAdapter(asset)

    article = adapter.load_article("tv-and-radio/2026/jun/01/john-oliver-trump-pardons-maga")

    assert article.snapshot.article_id == "tv-and-radio/2026/jun/01/john-oliver-trump-pardons-maga"
    assert article.snapshot.source_type == "guardian"
    assert article.snapshot.source_metadata["asset_sha256"] == asset.sha256
    assert article.snapshot.content_hash
    assert hashlib.sha256(asset.path.read_bytes()).hexdigest() == before_hash
