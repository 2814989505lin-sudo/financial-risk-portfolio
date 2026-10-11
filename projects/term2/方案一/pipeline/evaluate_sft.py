import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import yaml

from event_taxonomy import flatten_event_labels
from global_id_alignment import (
    build_record_alignment_context,
    extract_company_mentions_from_text,
    load_alias_map,
)
from llm_extract_openai import extract_with_chunking, get_client, load_few_shot_examples
from run_docfee_news import derive_training_labels, normalize_events


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


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_article_level_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    exact_match = 0
    json_success = 0
    valid_event_type_articles = 0
    negative_total = 0
    negative_correct = 0
    tp = 0
    fp = 0
    fn = 0
    predicted_label_count = 0

    for row in rows:
        gold = set(row.get("gold_labels", []) or [])
        predicted = set(row.get("predicted_labels", []) or [])
        if gold == predicted:
            exact_match += 1
        if row.get("json_success"):
            json_success += 1
        if not row.get("invalid_event_types"):
            valid_event_type_articles += 1
        if not gold:
            negative_total += 1
            if not predicted:
                negative_correct += 1
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        predicted_label_count += len(predicted)

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall) if precision and recall else 0.0
    return {
        "article_count": total,
        "exact_match_accuracy": safe_ratio(exact_match, total),
        "json_success_rate": safe_ratio(json_success, total),
        "valid_event_type_rate": safe_ratio(valid_event_type_articles, total),
        "negative_accuracy": safe_ratio(negative_correct, negative_total),
        "label_precision": precision,
        "label_recall": recall,
        "label_f1": f1,
        "avg_predicted_label_count": safe_ratio(predicted_label_count, total),
    }


def prepare_record(
    row: Dict[str, Any],
    alias_map: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    company_mentions = list(row.get("company_mentions", []) or [])
    if alias_map and not company_mentions:
        company_mentions = extract_company_mentions_from_text(
            title=row.get("title", ""),
            text=row.get("text", ""),
            alias_map=alias_map,
        )
    record = dict(row)
    record["company_mentions"] = company_mentions
    alignment_context = build_record_alignment_context(company_mentions, alias_map) if alias_map else {
        "candidate_companies": [],
        "mention_candidates": [],
    }
    return record, alignment_context


def evaluate_model(
    client: Any,
    model: str,
    gold_rows: Sequence[Dict[str, Any]],
    schema: Dict[str, Any],
    prompt_template: str,
    max_input_tokens: int,
    min_confidence: float,
    alias_map: Dict[str, List[Dict[str, Any]]],
    few_shot_examples: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    allowed_event_types = flatten_event_labels(schema)
    allowed_event_type_set: Set[str] = set(allowed_event_types)
    rows: List[Dict[str, Any]] = []

    for gold_row in gold_rows:
        record, alignment_context = prepare_record(gold_row, alias_map=alias_map)
        json_success = False
        invalid_event_types: List[str] = []
        predicted_labels: List[str] = []
        events: List[Dict[str, Any]] = []
        error = ""
        try:
            raw_events, _debug = extract_with_chunking(
                client=client,
                record=record,
                prompt_template=prompt_template,
                schema=schema,
                model=model,
                max_input_tokens=max_input_tokens,
                temperature=0.0,
                few_shot_examples=few_shot_examples,
                alignment_candidates=alignment_context.get("candidate_companies", []),
            )
            json_success = True
            for event in raw_events:
                event_type = str(event.get("event_type", "")).strip() if isinstance(event, dict) else ""
                if event_type and event_type not in allowed_event_type_set:
                    invalid_event_types.append(event_type)
            events = normalize_events(
                events=raw_events,
                record=record,
                allowed_event_types=allowed_event_types,
                model_name=model,
                alias_map=alias_map,
            )
            predicted_labels = derive_training_labels(events, min_confidence=min_confidence)
        except Exception as exc:
            error = str(exc)

        rows.append(
            {
                "article_id": gold_row.get("article_id", ""),
                "title": gold_row.get("title", ""),
                "gold_labels": list(gold_row.get("labels", []) or []),
                "predicted_labels": predicted_labels,
                "json_success": json_success,
                "invalid_event_types": invalid_event_types,
                "event_count": len(events),
                "error": error,
            }
        )

    return rows, summarize_article_level_metrics(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned extraction model against baseline on manual review rows.")
    parser.add_argument("--model", required=True, help="Fine-tuned model id")
    parser.add_argument("--baseline-model", required=True, help="Baseline model id")
    parser.add_argument("--gold-path", required=True, help="Manual review JSONL path")
    parser.add_argument("--schema", required=True, help="Schema yaml path")
    parser.add_argument("--prompt", required=True, help="Prompt template path")
    parser.add_argument("--output-dir", required=True, help="Evaluation output directory")
    parser.add_argument("--few-shot-path", default=None, help="Optional few-shot JSON path")
    parser.add_argument("--alias-map", default=None, help="Optional alias map JSON path")
    parser.add_argument("--split", default="manual_review", help="Manual review split to evaluate")
    parser.add_argument("--max-input-tokens", type=int, default=None, help="Max prompt text tokens")
    parser.add_argument("--min-confidence", type=float, default=0.75, help="Min confidence for derived labels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.schema, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    defaults = schema.get("defaults", {})
    max_input_tokens = args.max_input_tokens if args.max_input_tokens is not None else int(defaults.get("max_input_tokens", 12000))
    gold_rows = []
    for row in load_jsonl(args.gold_path):
        row_split = str(row.get("split", "")).strip()
        if row_split and row_split != args.split:
            continue
        gold_rows.append(row)
    client = get_client()
    alias_map = load_alias_map(args.alias_map) if args.alias_map else {}
    few_shot_examples = load_few_shot_examples(args.few_shot_path)

    candidate_rows, candidate_summary = evaluate_model(
        client=client,
        model=args.model,
        gold_rows=gold_rows,
        schema=schema,
        prompt_template=prompt_template,
        max_input_tokens=max_input_tokens,
        min_confidence=args.min_confidence,
        alias_map=alias_map,
        few_shot_examples=few_shot_examples,
    )
    baseline_rows, baseline_summary = evaluate_model(
        client=client,
        model=args.baseline_model,
        gold_rows=gold_rows,
        schema=schema,
        prompt_template=prompt_template,
        max_input_tokens=max_input_tokens,
        min_confidence=args.min_confidence,
        alias_map=alias_map,
        few_shot_examples=few_shot_examples,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    write_jsonl(os.path.join(args.output_dir, "candidate_predictions.jsonl"), candidate_rows)
    write_jsonl(os.path.join(args.output_dir, "baseline_predictions.jsonl"), baseline_rows)
    summary = {
        "candidate_model": args.model,
        "baseline_model": args.baseline_model,
        "split": args.split,
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "delta": {
            "exact_match_accuracy": candidate_summary["exact_match_accuracy"] - baseline_summary["exact_match_accuracy"],
            "label_f1": candidate_summary["label_f1"] - baseline_summary["label_f1"],
            "negative_accuracy": candidate_summary["negative_accuracy"] - baseline_summary["negative_accuracy"],
            "json_success_rate": candidate_summary["json_success_rate"] - baseline_summary["json_success_rate"],
        },
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
