import json
import os
from typing import Any, Dict, Iterable, List


def build_manual_label_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for row in rows:
        outputs.append(
            {
                "article_id": row.get("article_id", ""),
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "source": row.get("source", ""),
                "publish_time": row.get("publish_time", ""),
                "text": row.get("text", ""),
                "company_mentions": row.get("company_mentions", []),
                "split": "manual_review",
                "label_source": "manual_review",
                "labels": [],
                "is_in_scope": "",
                "review_notes": "",
                "llm_is_in_scope": row.get("is_in_scope", ""),
                "candidate_labels": row.get("candidate_labels", []),
                "screen_confidence": row.get("screen_confidence", 0.0),
                "screen_reason": row.get("screen_reason", ""),
                "canonical_company_name": row.get("canonical_company_name", ""),
                "matched_alias": row.get("matched_alias", ""),
                "global_id": row.get("global_id", ""),
                "alignment_status": row.get("alignment_status", ""),
            }
        )
    return outputs


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
