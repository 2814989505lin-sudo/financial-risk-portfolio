'''【040】单元测试：验证新闻标准化、内容哈希及异常数据拒绝规则。'''

from __future__ import annotations

import hashlib
import json

import pytest

from financial_risk_agent.core.security import content_hash
from financial_risk_agent.integrations.legacy.dataset_adapter import (
    ArticleNotFoundError,
    DatasetAsset,
    DatasetEncodingError,
    DatasetSchemaError,
    LegacyDatasetAdapter,
)


def make_asset(
    tmp_path, rows: list[dict[str, str]], *, name: str = "articles.json"
) -> DatasetAsset:
    path = tmp_path / name
    path.write_text(json.dumps(rows), encoding="utf-8")
    return DatasetAsset(
        asset_id="unit-articles-v1",
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        schema_version="unit_articles.v1",
        source_type="sample",
    )


def test_normalizes_article_into_a_traceable_input_snapshot(tmp_path) -> None:
    adapter = LegacyDatasetAdapter(
        make_asset(
            tmp_path,
            [
                {
                    "article_id": "article-001",
                    "title": "Payment disruption",
                    "content": "A supplier delayed its settlement.",
                    "source_url": "sample://article-001",
                    "section": "business",
                }
            ],
        )
    )

    first = adapter.load_article("article-001")
    second = adapter.load_article("article-001")

    assert first.snapshot == second.snapshot
    assert first.snapshot.content_hash == content_hash("A supplier delayed its settlement.")
    assert first.snapshot.source_metadata["asset_id"] == "unit-articles-v1"
    assert first.import_ref == "article:unit-articles-v1:article-001"


def test_rejects_unknown_article_and_missing_content(tmp_path) -> None:
    adapter = LegacyDatasetAdapter(
        make_asset(
            tmp_path,
            [{"article_id": "article-001", "title": "Missing body", "content": ""}],
        )
    )

    with pytest.raises(ArticleNotFoundError, match="not found"):
        adapter.load_article("missing")
    with pytest.raises(DatasetSchemaError, match="non-empty content"):
        adapter.load_article("article-001")


def test_rejects_invalid_dataset_encoding(tmp_path) -> None:
    path = tmp_path / "articles.csv"
    path.write_bytes(b"\xff\xfeinvalid")
    adapter = LegacyDatasetAdapter(
        DatasetAsset(
            asset_id="invalid-encoding-v1",
            path=path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            schema_version="unit_articles.v1",
        )
    )

    with pytest.raises(DatasetEncodingError, match="UTF-8"):
        adapter.load_article("article-001")
