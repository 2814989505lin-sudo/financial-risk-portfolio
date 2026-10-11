import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


def derive_feature_month(publish_time: str) -> str:
    text = str(publish_time or "").strip()
    return text[:7] if len(text) >= 7 else ""


def make_group_id(global_id: str, feature_month: str, risk_dim: str) -> str:
    return f"{global_id}__{feature_month}__{risk_dim}"


def is_qualified_event(event: Dict[str, Any], min_confidence: float) -> bool:
    return (
        isinstance(event, dict)
        and str(event.get("alignment_status", "")).strip() == "resolved"
        and bool(str(event.get("global_id", "")).strip())
        and bool(str(event.get("taxonomy", "")).strip())
        and normalize_confidence(event.get("confidence")) >= min_confidence
    )


def build_group_event(article: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    canonical_company_name = str(event.get("canonical_company_name", "")).strip()
    if not canonical_company_name:
        canonical_company_name = str((article.get("company_mentions") or [""])[0]).strip()
    return {
        "article_id": str(article.get("article_id", "")).strip(),
        "url": str(article.get("url", "")).strip(),
        "title": str(article.get("title", "")).strip(),
        "source": str(article.get("source", "")).strip(),
        "publish_time": str(article.get("publish_time", "")).strip(),
        "event_type": str(event.get("event_type", "")).strip(),
        "taxonomy": str(event.get("taxonomy", "")).strip(),
        "event_polarity": str(event.get("event_polarity", "")).strip(),
        "arguments": event.get("arguments", {}) if isinstance(event.get("arguments"), dict) else {},
        "evidence": event.get("evidence", []) if isinstance(event.get("evidence"), list) else [],
        "confidence": normalize_confidence(event.get("confidence")),
        "global_id": str(event.get("global_id", "")).strip(),
        "canonical_company_name": canonical_company_name,
        "matched_alias": str(event.get("matched_alias", "")).strip(),
        "alignment_status": str(event.get("alignment_status", "")).strip(),
        "annotation_model": str(event.get("annotation_model", "")).strip(),
        "label_source": str(event.get("label_source", "")).strip(),
    }


def aggregate_events(
    article_rows: Iterable[Dict[str, Any]],
    min_confidence: float = 0.5,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    article_ids_by_group: Dict[Tuple[str, str, str], set[str]] = defaultdict(set)

    for article in article_rows:
        feature_month = derive_feature_month(str(article.get("publish_time", "")))
        if not feature_month:
            continue
        for event in article.get("events", []) or []:
            if not is_qualified_event(event, min_confidence=min_confidence):
                continue
            global_id = str(event.get("global_id", "")).strip()
            risk_dim = str(event.get("taxonomy", "")).strip()
            key = (global_id, feature_month, risk_dim)
            group = grouped.get(key)
            if group is None:
                canonical_company_name = str(event.get("canonical_company_name", "")).strip()
                if not canonical_company_name:
                    canonical_company_name = str((article.get("company_mentions") or [""])[0]).strip()
                group = {
                    "group_id": make_group_id(global_id, feature_month, risk_dim),
                    "global_id": global_id,
                    "canonical_company_name": canonical_company_name,
                    "feature_month": feature_month,
                    "risk_dim": risk_dim,
                    "events": [],
                }
                grouped[key] = group
            group["events"].append(build_group_event(article, event))
            article_id = str(article.get("article_id", "")).strip()
            if article_id:
                article_ids_by_group[key].add(article_id)

    results: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        group = grouped[key]
        events = list(group.get("events", []))
        negative_count = sum(1 for event in events if event.get("event_polarity") == "negative")
        positive_count = sum(1 for event in events if event.get("event_polarity") == "positive")
        results.append(
            {
                **group,
                "event_count": len(events),
                "article_count": len(article_ids_by_group.get(key, set())),
                "negative_event_count": negative_count,
                "positive_event_count": positive_count,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate resolved events by company-month-risk dimension.")
    parser.add_argument("--input", required=True, help="Input news_events.jsonl path")
    parser.add_argument("--output", required=True, help="Output aggregated_events.jsonl path")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min confidence for accepted events")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    aggregated = aggregate_events(rows, min_confidence=args.min_confidence)
    write_jsonl(args.output, aggregated)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "group_count": len(aggregated),
                "event_count": sum(row["event_count"] for row in aggregated),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
