'''【044】单元测试：验证确定性哈希与敏感数据拦截规则。'''

from __future__ import annotations

import pytest

from financial_risk_agent.core.security import (
    assert_no_sensitive_data,
    assert_sha256_digest,
    content_hash,
    is_sha256_digest,
    proposal_hash,
)


def test_content_hash_is_deterministic_for_equivalent_mappings() -> None:
    first = {"article": "article-001", "risk": ["credit", "fraud"]}
    second = {"risk": ["credit", "fraud"], "article": "article-001"}

    assert content_hash(first) == content_hash(second)
    assert is_sha256_digest(content_hash(first))


def test_proposal_hash_changes_when_approved_content_changes() -> None:
    first = proposal_hash({"tool_name": "update_case_status", "target": "case-001"})
    second = proposal_hash({"tool_name": "update_case_status", "target": "case-002"})

    assert first != second


def test_sensitive_mapping_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="credentials or secrets"):
        assert_no_sensitive_data({"nested": {"access_token": "not-allowed"}})


def test_invalid_digest_is_rejected() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        assert_sha256_digest("not-a-hash", "content_hash")
