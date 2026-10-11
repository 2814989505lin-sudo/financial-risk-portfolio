'''【032】集成测试：验证文章加载与搜索工具的快照持久化和审计行为。'''

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from financial_risk_agent.core.schemas import InputSnapshot
from financial_risk_agent.core.security import content_hash
from financial_risk_agent.core.state import WorkflowState
from financial_risk_agent.integrations.legacy.dataset_adapter import (
    LegacyDatasetAdapter,
    load_dataset_asset,
)
from financial_risk_agent.services.audit import AuditService
from financial_risk_agent.storage.database import SQLiteDatabase
from financial_risk_agent.storage.repositories.article_repository import ArticleRepository
from financial_risk_agent.storage.repositories.audit_repository import AuditRepository
from financial_risk_agent.storage.repositories.task_repository import TaskRepository
from financial_risk_agent.tools.readonly.load_article import LoadArticleTool
from financial_risk_agent.tools.readonly.search_dataset import SearchDatasetTool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/legacy_assets.json"


def create_task(database: SQLiteDatabase) -> str:
    task = WorkflowState(
        task_id="task-article-tools-001",
        session_id="session-001",
        correlation_id="correlation-001",
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
        input=InputSnapshot(
            source_type="sample",
            article_id="bootstrap-001",
            title="Bootstrap task input",
            content_ref="sample://bootstrap-001",
            content_hash=content_hash("Bootstrap task input"),
        ),
    )
    TaskRepository(database).create(task)
    return task.task_id


def test_article_tools_persist_snapshots_and_audit_each_dataset_read(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "tasks.sqlite3")
    task_id = create_task(database)
    adapter = LegacyDatasetAdapter(load_dataset_asset(MANIFEST_PATH, "sample-articles-v1"))
    audit_service = AuditService(AuditRepository(database))
    article_repository = ArticleRepository(database)
    load_tool = LoadArticleTool(adapter, article_repository, audit_service)
    search_tool = SearchDatasetTool(adapter, audit_service)

    results = search_tool.search(task_id=task_id, query="invoice")
    stored = load_tool.load(task_id=task_id, article_id="sample-fraud-002")

    assert [result.article_id for result in results] == ["sample-fraud-002"]
    assert article_repository.get("sample-articles-v1", "sample-fraud-002") == stored
    assert stored.snapshot.content_hash == content_hash(
        "The finance team paused payment after detecting mismatched beneficiary details "
        "on an invoice."
    )
    events = AuditRepository(database).list_for_task(task_id)
    assert [event.event_type for event in events] == ["DATASET_SEARCHED", "ARTICLE_LOADED"]
    assert audit_service.verify_task_chain(task_id)
