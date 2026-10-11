import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from event_taxonomy import build_filter_terms, iter_event_definitions


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?\n])")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_sentences(text: str) -> List[str]:
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_PATTERN.split(str(text or "")) if piece.strip()]
    return pieces if pieces else [str(text or "").strip()]


def build_event_definition_maps(schema: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    event_map: Dict[str, Dict[str, Any]] = {}
    for item in iter_event_definitions(schema):
        event_map[item["event_type"]] = dict(item)
    return event_map, build_filter_terms(schema)


def sentence_match_score(sentence: str, terms: List[str]) -> int:
    score = 0
    for term in terms:
        if term and term in sentence:
            score += len(term)
    return score


def find_best_evidence(title: str, text: str, terms: List[str]) -> str:
    candidates = [str(title or "").strip()] + split_sentences(text)
    best_sentence = ""
    best_score = 0
    for sentence in candidates:
        score = sentence_match_score(sentence, terms)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence


def infer_events_for_row(
    row: Dict[str, Any],
    event_map: Dict[str, Dict[str, Any]],
    filter_terms: Dict[str, List[str]],
    max_events_per_article: int,
) -> List[Dict[str, Any]]:
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    haystack = f"{title}\n{text}"
    scored: List[Tuple[int, str, str]] = []
    for event_type, terms in filter_terms.items():
        score = 0
        for term in terms:
            if not term:
                continue
            if term in title:
                score += 3
            if term in text:
                score += 1
        if score > 0:
            evidence = find_best_evidence(title, text, terms)
            scored.append((score, event_type, evidence))

    scored.sort(key=lambda item: (-item[0], item[1]))
    events: List[Dict[str, Any]] = []
    seen_event_types = set()
    company_name = str(row.get("canonical_company_name", "") or row.get("matched_alias", "")).strip()
    publish_time = str(row.get("publish_time", "")).strip()
    publish_month = publish_time[:7] if len(publish_time) >= 7 else publish_time

    for score, event_type, evidence in scored:
        if event_type in seen_event_types:
            continue
        seen_event_types.add(event_type)
        event_def = event_map[event_type]
        arguments = {name: "" for name in event_def.get("arguments", [])}
        if "公司名称" in arguments:
            arguments["公司名称"] = company_name
        if "发生时间" in arguments:
            arguments["发生时间"] = publish_month
        events.append(
            {
                "event_type": event_type,
                "taxonomy": event_def["taxonomy"],
                "event_polarity": event_def["event_polarity"],
                "evidence": [evidence] if evidence else [],
                "arguments": arguments,
                "confidence": min(0.95, 0.55 + min(score, 8) * 0.05),
            }
        )
        if len(events) >= max_events_per_article:
            break
    return events


def build_event_gold_examples(
    rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    max_examples: int,
    max_events_per_article: int,
) -> List[Dict[str, Any]]:
    event_map, filter_terms = build_event_definition_maps(schema)
    examples: List[Dict[str, Any]] = []
    for row in rows[:max_examples]:
        examples.append(
            {
                "article_id": str(row.get("article_id", "")).strip(),
                "title": str(row.get("title", "")).strip(),
                "source": str(row.get("source", "")).strip(),
                "publish_time": str(row.get("publish_time", "")).strip(),
                "company_mentions": list(row.get("company_mentions", []) or []),
                "text": str(row.get("text", "")).strip(),
                "canonical_company_name": str(row.get("canonical_company_name", "")).strip(),
                "matched_alias": str(row.get("matched_alias", "")).strip(),
                "global_id": str(row.get("global_id", "")).strip(),
                "alignment_status": str(row.get("alignment_status", "")).strip(),
                "label_source": "ai_preannotation",
                "events": infer_events_for_row(
                    row=row,
                    event_map=event_map,
                    filter_terms=filter_terms,
                    max_events_per_article=max_events_per_article,
                ),
                "annotator_notes": "AI preannotation generated from schema keyword matching; requires human review before being treated as human gold.",
            }
        )
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event-level AI preannotation few-shot examples from manual review rows.")
    parser.add_argument("--input", required=True, help="manual_review.jsonl path")
    parser.add_argument("--schema", required=True, help="Schema yaml path")
    parser.add_argument("--output", required=True, help="Output event_gold_examples.jsonl path")
    parser.add_argument("--max-examples", type=int, default=100, help="Maximum number of examples to keep")
    parser.add_argument("--max-events-per-article", type=int, default=4, help="Maximum inferred events per article")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.schema, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    rows = load_jsonl(args.input)
    examples = build_event_gold_examples(
        rows=rows,
        schema=schema,
        max_examples=args.max_examples,
        max_events_per_article=args.max_events_per_article,
    )
    write_jsonl(args.output, examples)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "example_count": len(examples),
                "label_source": "ai_preannotation",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
