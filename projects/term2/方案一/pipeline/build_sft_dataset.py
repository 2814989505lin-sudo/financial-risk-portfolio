import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

from llm_extract_openai import build_chat_messages


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


def normalize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "article_id": row.get("article_id", ""),
        "title": row.get("title", ""),
        "source": row.get("source", ""),
        "publish_time": row.get("publish_time", ""),
        "text": row.get("text", ""),
        "company_mentions": list(row.get("company_mentions", []) or row.get("mentioned_companies", []) or []),
    }


def filter_qualified_events(events: Sequence[Dict[str, Any]], min_confidence: float) -> List[Dict[str, Any]]:
    qualified: List[Dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        confidence = float(event.get("confidence", 0.0) or 0.0)
        if event.get("alignment_status") != "resolved" or confidence < min_confidence:
            continue
        qualified.append(event)
    return qualified


def split_silver_positive_rows(
    rows: Sequence[Dict[str, Any]],
    val_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: str(row.get("article_id", "")))
    if len(ordered) <= 1 or val_ratio <= 0:
        return ordered, []
    val_count = max(1, int(round(len(ordered) * val_ratio)))
    if val_count >= len(ordered):
        val_count = len(ordered) - 1
    val_rows = ordered[:val_count]
    train_rows = ordered[val_count:]
    return train_rows, val_rows


def build_training_example(
    record: Dict[str, Any],
    assistant_events: Sequence[Dict[str, Any]],
    prompt_template: str,
    schema: Dict[str, Any],
    assistant_model: str,
) -> Dict[str, Any]:
    messages = build_chat_messages(
        record=normalize_record(record),
        prompt_template=prompt_template,
        schema=schema,
        few_shot_examples=[],
        alignment_candidates=[],
        annotation_model=assistant_model,
    )
    assistant_payload = json.dumps({"events": list(assistant_events)}, ensure_ascii=False)
    return {
        "messages": [
            *messages,
            {"role": "assistant", "content": assistant_payload},
        ]
    }


def build_sft_datasets(
    gold_rows: Sequence[Dict[str, Any]],
    silver_rows: Sequence[Dict[str, Any]],
    prompt_template: str,
    schema: Dict[str, Any],
    assistant_model: str,
    min_confidence: float,
    silver_val_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train_examples: List[Dict[str, Any]] = []
    val_examples: List[Dict[str, Any]] = []

    silver_positive_rows: List[Dict[str, Any]] = []
    for row in silver_rows:
        qualified_events = filter_qualified_events(row.get("events", []), min_confidence=min_confidence)
        if not qualified_events:
            continue
        silver_positive_rows.append(
            {
                "record": row,
                "assistant_events": qualified_events,
            }
        )
    silver_train, silver_val = split_silver_positive_rows(silver_positive_rows, val_ratio=silver_val_ratio)

    for item in silver_train:
        train_examples.append(
            build_training_example(
                record=item["record"],
                assistant_events=item["assistant_events"],
                prompt_template=prompt_template,
                schema=schema,
                assistant_model=assistant_model,
            )
        )
    for item in silver_val:
        val_examples.append(
            build_training_example(
                record=item["record"],
                assistant_events=item["assistant_events"],
                prompt_template=prompt_template,
                schema=schema,
                assistant_model=assistant_model,
            )
        )

    skipped_manual_review_positive = 0
    for row in gold_rows:
        is_in_scope = bool(row.get("is_in_scope"))
        labels = list(row.get("labels", []) or [])
        if is_in_scope and labels:
            skipped_manual_review_positive += 1
            continue
        if not is_in_scope:
            train_examples.append(
                build_training_example(
                    record=row,
                    assistant_events=[],
                    prompt_template=prompt_template,
                    schema=schema,
                    assistant_model=assistant_model,
                )
            )

    stats = {
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "silver_positive_rows": len(silver_positive_rows),
        "silver_train_rows": len(silver_train),
        "silver_val_rows": len(silver_val),
        "manual_review_negative_rows": sum(1 for row in gold_rows if not bool(row.get("is_in_scope"))),
        "skipped_manual_review_positive_without_event_annotations": skipped_manual_review_positive,
    }
    return train_examples, val_examples, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenAI fine-tuning dataset for event extraction SFT.")
    parser.add_argument("--schema", required=True, help="Schema yaml path")
    parser.add_argument("--prompt", required=True, help="Prompt template path")
    parser.add_argument("--gold-path", required=True, help="Manual review JSONL path")
    parser.add_argument("--silver-events-path", required=True, help="Silver event extraction JSONL path")
    parser.add_argument("--output", required=True, help="Training dataset JSONL path")
    parser.add_argument("--val-output", required=True, help="Validation dataset JSONL path")
    parser.add_argument(
        "--assistant-model",
        default="gpt-4.1-mini-2025-04-14",
        help="Model name inserted in prompt examples",
    )
    parser.add_argument("--min-confidence", type=float, default=0.75, help="Min confidence for positive silver events")
    parser.add_argument("--silver-val-ratio", type=float, default=0.2, help="Validation ratio for silver positives")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.schema, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    gold_rows = load_jsonl(args.gold_path)
    silver_rows = load_jsonl(args.silver_events_path)
    train_examples, val_examples, stats = build_sft_datasets(
        gold_rows=gold_rows,
        silver_rows=silver_rows,
        prompt_template=prompt_template,
        schema=schema,
        assistant_model=args.assistant_model,
        min_confidence=args.min_confidence,
        silver_val_ratio=args.silver_val_ratio,
    )

    write_jsonl(args.output, train_examples)
    write_jsonl(args.val_output, val_examples)
    print(
        json.dumps(
            {
                "train_output": args.output,
                "val_output": args.val_output,
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
