import argparse
import csv
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import yaml
from tqdm import tqdm

from event_taxonomy import derive_article_labels, derive_taxonomy_polarity_pairs, flatten_event_labels
from global_id_alignment import (
    build_record_alignment_context,
    extract_company_mentions_from_text,
    finalize_alignment_fields,
    load_alias_map,
)
from llm_extract_openai import extract_multi_model, get_multi_clients, load_few_shot_examples
from prepare_inputs import load_or_prepare_inputs
from sample_review_pack import build_review_pack


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pretty_json(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def taxonomy_polarity_key(taxonomy: str, polarity: str) -> str:
    return f"{taxonomy}_{polarity}"


def label_column_name(taxonomy: str, polarity: str) -> str:
    return f"{taxonomy}_{polarity}_labels"


def derive_event_labels(
    events: List[Dict[str, Any]],
    taxonomy_polarity_pairs: List[Tuple[str, str]],
) -> Dict[str, List[str]]:
    labels_by_bucket: Dict[str, List[str]] = {
        taxonomy_polarity_key(taxonomy, polarity): []
        for taxonomy, polarity in taxonomy_polarity_pairs
    }
    for event in events:
        event_type = event.get("event_type", "")
        taxonomy = event.get("taxonomy", "")
        polarity = event.get("event_polarity", "")
        bucket = taxonomy_polarity_key(taxonomy, polarity)
        if bucket not in labels_by_bucket:
            labels_by_bucket[bucket] = []
        if event_type and event_type not in labels_by_bucket[bucket]:
            labels_by_bucket[bucket].append(event_type)

    return labels_by_bucket


def write_summary_csv(
    rows: List[Dict[str, Any]],
    csv_path: str,
    taxonomy_polarity_pairs: List[Tuple[str, str]],
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    taxonomy_columns = [
        label_column_name(taxonomy, polarity) for taxonomy, polarity in taxonomy_polarity_pairs
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "article_id",
                "url",
                "title",
                *taxonomy_columns,
                "event_count",
            ],
        )
        writer.writeheader()

        for row in rows:
            event_labels = derive_event_labels(row.get("events", []), taxonomy_polarity_pairs)
            summary_row = {
                "article_id": row.get("article_id", ""),
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "event_count": len(row.get("events", [])),
            }
            for taxonomy, polarity in taxonomy_polarity_pairs:
                summary_row[label_column_name(taxonomy, polarity)] = "|".join(
                    event_labels.get(taxonomy_polarity_key(taxonomy, polarity), [])
                )
            writer.writerow(summary_row)


def normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


def should_log_progress(completed: int, total: int, every: int) -> bool:
    if total <= 0 or completed <= 0:
        return False
    safe_every = max(1, every)
    return completed % safe_every == 0 or completed == total


def resolve_batch_window(total_records: int, start_offset: int, max_records: int | None) -> tuple[int, int]:
    safe_total = max(0, int(total_records))
    safe_start = max(0, min(int(start_offset), safe_total))
    if max_records is None:
        return safe_start, safe_total
    safe_limit = max(0, int(max_records))
    return safe_start, min(safe_total, safe_start + safe_limit)


def resolve_default_few_shot_path(input_path: str, explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path
    candidate = os.path.join(os.path.dirname(input_path) or ".", "event_gold_examples.jsonl")
    if os.path.exists(candidate):
        return candidate
    return None


def build_checkpoint_payload(
    *,
    input_path: str,
    output_path: str,
    total_input_records: int,
    start_offset: int,
    end_offset: int,
    processed_records: int,
) -> Dict[str, Any]:
    next_offset = min(max(0, int(processed_records)), max(0, int(total_input_records)))
    remaining = max(0, int(total_input_records) - next_offset)
    return {
        "input_path": input_path,
        "output_path": output_path,
        "total_input_records": int(total_input_records),
        "start_offset": int(start_offset),
        "end_offset_exclusive": int(end_offset),
        "processed_records": int(processed_records),
        "next_offset": next_offset,
        "remaining_records": remaining,
        "is_complete": remaining == 0,
    }


def write_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_failed_record(
    *,
    idx: int,
    record: Dict[str, Any],
    exc: Exception,
    stage: str,
    retry_attempt: int = 0,
) -> Dict[str, Any]:
    return {
        "idx": int(idx),
        "article_id": str(record.get("article_id", "")),
        "url": str(record.get("url", "")),
        "title": str(record.get("title", "")),
        "source": str(record.get("source", "")),
        "publish_time": str(record.get("publish_time", "")),
        "stage": stage,
        "retry_attempt": int(retry_attempt),
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
    }


def flush_contiguous_results(
    *,
    pending_results: Dict[int, Dict[str, Any]],
    next_write_idx: int,
    output_path: str,
    failed_output_path: str,
) -> tuple[int, int, List[Dict[str, Any]]]:
    flushed_rows: List[Dict[str, Any]] = []
    flushed_failed_rows: List[Dict[str, Any]] = []
    current_idx = next_write_idx
    while current_idx in pending_results:
        item = pending_results.pop(current_idx)
        if item.get("_result_kind") == "failed":
            failed_row = dict(item.get("failed_row", {}))
            failed_row["status"] = "pending_retry"
            flushed_failed_rows.append(failed_row)
        else:
            flushed_rows.append(dict(item.get("output_row", {})))
        current_idx += 1
    append_jsonl_rows(output_path, flushed_rows)
    append_jsonl_rows(failed_output_path, flushed_failed_rows)
    return current_idx, len(flushed_rows) + len(flushed_failed_rows), flushed_failed_rows


def normalize_events(
    events: List[Dict[str, Any]],
    record: Dict[str, Any],
    allowed_event_types: List[str],
    model_name: str,
    alias_map: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    allowed_set = set(allowed_event_types)
    company_mentions = record.get("company_mentions", []) or []
    normalized: List[Dict[str, Any]] = []
    for raw_event in events:
        if not isinstance(raw_event, dict):
            continue
        event_type = str(raw_event.get("event_type", "")).strip()
        if event_type not in allowed_set:
            continue
        event = {
            "event_type": event_type,
            "taxonomy": str(raw_event.get("taxonomy", "")).strip(),
            "event_polarity": str(raw_event.get("event_polarity", "")).strip(),
            "arguments": raw_event.get("arguments", {}) if isinstance(raw_event.get("arguments"), dict) else {},
            "evidence": raw_event.get("evidence", []) if isinstance(raw_event.get("evidence"), list) else [],
            "confidence": normalize_confidence(raw_event.get("confidence")),
            "global_id": str(raw_event.get("global_id", "")).strip(),
            "canonical_company_name": str(raw_event.get("canonical_company_name", "")).strip(),
            "matched_alias": str(raw_event.get("matched_alias", "")).strip(),
            "alignment_status": str(raw_event.get("alignment_status", "")).strip(),
            "annotation_model": model_name,
            "annotation_models": raw_event.get("annotation_models", []) if isinstance(raw_event.get("annotation_models"), list) else [model_name],
            "model_agreement": int(raw_event.get("model_agreement", 1) or 1),
            "per_model_confidence": raw_event.get("per_model_confidence", {}) if isinstance(raw_event.get("per_model_confidence"), dict) else {},
            "label_source": "silver",
        }
        event = finalize_alignment_fields(event, company_mentions=company_mentions, alias_map=alias_map)
        normalized.append(event)
    return normalized


def derive_training_labels(
    events: List[Dict[str, Any]],
    min_confidence: float,
) -> List[str]:
    accepted = [
        event
        for event in events
        if event.get("alignment_status") == "resolved" and normalize_confidence(event.get("confidence")) >= min_confidence
    ]
    return derive_article_labels(accepted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DocFEE-style news event extraction pipeline")
    parser.add_argument("--input", required=True, help="Input normalized article JSONL or original raw JSON file")
    parser.add_argument("--schema", required=True, help="Schema yaml path")
    parser.add_argument("--model", default="kimi-k2.5", help="Fallback single model name")
    parser.add_argument(
        "--models",
        default="aliyun",
        help="Comma-separated providers to use for extraction, e.g. aliyun",
    )
    parser.add_argument(
        "--prompt",
        default="prompt/prompt.news.docfee.v1.txt",
        help="Prompt template path",
    )
    parser.add_argument(
        "--few-shot-path",
        default=None,
        help="Optional JSON/JSONL file with few-shot examples; if omitted, auto-load sibling event_gold_examples.jsonl when present",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--pretty-output",
        default=None,
        help="Optional pretty JSON output path for human review; default: <output>.pretty.json",
    )
    parser.add_argument("--review-rate", type=float, default=0.1, help="Review sample rate")
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Max input tokens per extraction; over-limit docs will be chunked",
    )
    parser.add_argument(
        "--intermediate-dir",
        default="data",
        help="Directory for intermediate JSONL files",
    )
    parser.add_argument("--alias-map", default=None, help="Optional local company alias map JSON path")
    parser.add_argument("--min-confidence", type=float, default=0.75, help="Min confidence for silver labels")
    parser.add_argument("--max-workers", type=int, default=2, help="Number of concurrent extraction workers")
    parser.add_argument("--progress-log-every", type=int, default=10, help="Emit a JSON progress log every N completed records")
    parser.add_argument("--start-offset", type=int, default=0, help="Start offset within the input records for batched extraction")
    parser.add_argument("--max-records", type=int, default=None, help="Maximum number of records to process in this batch")
    parser.add_argument("--checkpoint-path", default=None, help="Path to write batch checkpoint metadata JSON")
    parser.add_argument("--resume-checkpoint", default=None, help="Optional checkpoint JSON to resume from its next_offset")
    parser.add_argument("--retry-failed-at-end", action="store_true", default=True, help="Retry failed records once after the main pass completes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.schema, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    taxonomy_polarity_pairs = derive_taxonomy_polarity_pairs(schema)
    allowed_event_types = flatten_event_labels(schema)

    defaults = schema.get("defaults", {})
    max_input_tokens = (
        args.max_input_tokens if args.max_input_tokens is not None else int(defaults.get("max_input_tokens", 12000))
    )

    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    os.makedirs(args.intermediate_dir, exist_ok=True)
    prepared_path = os.path.join(args.intermediate_dir, "news_input.jsonl")
    resolved_few_shot_path = resolve_default_few_shot_path(args.input, args.few_shot_path)
    few_shot_examples = load_few_shot_examples(resolved_few_shot_path)

    all_records, prepared_path = load_or_prepare_inputs(input_path=args.input, output_path=prepared_path)
    total_input_records = len(all_records)

    start_offset = args.start_offset
    end_offset_override = None
    processed_records_before_resume = 0
    if args.resume_checkpoint:
        with open(args.resume_checkpoint, "r", encoding="utf-8") as f:
            checkpoint_payload = json.load(f)
        start_offset = int(checkpoint_payload.get("next_offset", start_offset) or 0)
        end_offset_override = int(checkpoint_payload.get("end_offset_exclusive", 0) or 0)
        processed_records_before_resume = int(checkpoint_payload.get("processed_records", 0) or 0)

    batch_start, batch_end = resolve_batch_window(
        total_records=total_input_records,
        start_offset=start_offset if args.resume_checkpoint else args.start_offset,
        max_records=None if args.resume_checkpoint else args.max_records,
    )
    if end_offset_override is not None:
        batch_end = max(batch_start, min(total_input_records, end_offset_override))
    records = all_records[batch_start:batch_end]
    checkpoint_path = args.checkpoint_path or (args.output + ".checkpoint.json")
    failed_output_path = args.output + ".failed.jsonl"
    recovered_output_path = args.output + ".retry_recovered.jsonl"
    if args.resume_checkpoint:
        batch_origin_start = int(checkpoint_payload.get("start_offset", args.start_offset) or 0)
    else:
        batch_origin_start = args.start_offset
    batch_total_records = max(0, batch_end - batch_origin_start)

    if args.resume_checkpoint:
        output_mode = "a"
    else:
        output_mode = "w"
    if output_mode == "w":
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8"):
            pass
        with open(failed_output_path, "w", encoding="utf-8"):
            pass
        with open(recovered_output_path, "w", encoding="utf-8"):
            pass

    alias_map = load_alias_map(args.alias_map) if args.alias_map else {}
    thread_local = threading.local()
    configured_model_specs = list((defaults.get("models") or []))
    selected_providers = [item.strip() for item in str(args.models or "").split(",") if item.strip()]
    active_model_specs = [
        spec for spec in configured_model_specs if str(spec.get("provider", "")).strip() in selected_providers
    ]
    if not active_model_specs:
        active_model_specs = [{"provider": "openai", "model": args.model, "env_key": "OPENAI_API_KEY"}]

    print(
        json.dumps(
            {
                "stage": "run_docfee_news",
                "status": "start",
                "records": len(records),
                "total_input_records": total_input_records,
                "start_offset": batch_start,
                "end_offset_exclusive": batch_end,
                "batch_origin_start": batch_origin_start,
                "models": selected_providers,
                "max_workers": max(1, args.max_workers),
                "retry_failed_at_end": bool(args.retry_failed_at_end),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if not records:
        checkpoint_payload = build_checkpoint_payload(
            input_path=args.input,
            output_path=args.output,
            total_input_records=total_input_records,
            start_offset=batch_origin_start,
            end_offset=batch_end,
            processed_records=processed_records_before_resume,
        )
        write_checkpoint(checkpoint_path, checkpoint_payload)
        print(json.dumps({"stage": "run_docfee_news", "status": "no_records", "checkpoint_path": checkpoint_path}, ensure_ascii=False), flush=True)
        return

    def get_thread_model_clients():
        model_clients = getattr(thread_local, "model_clients", None)
        if model_clients is None:
            model_clients = get_multi_clients(active_model_specs)
            thread_local.model_clients = model_clients
        return model_clients

    def process_record(idx: int, record: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        company_mentions = list(record.get("company_mentions", []) or [])
        if alias_map and not company_mentions:
            company_mentions = extract_company_mentions_from_text(
                title=record.get("title", ""),
                text=record.get("text", ""),
                alias_map=alias_map,
            )
        enriched_record = dict(record)
        enriched_record["company_mentions"] = company_mentions
        alignment_context = build_record_alignment_context(company_mentions, alias_map) if alias_map else {
            "candidate_companies": [],
            "mention_candidates": [],
        }
        model_clients = get_thread_model_clients()
        fused_model_name = "+".join(model for _provider, model, _client in model_clients)
        events, _debug = extract_multi_model(
            model_clients=model_clients,
            record=enriched_record,
            prompt_template=prompt_template,
            schema=schema,
            max_input_tokens=max_input_tokens,
            temperature=0.0,
            few_shot_examples=few_shot_examples,
            alignment_candidates=alignment_context.get("candidate_companies", []),
        )
        normalized_events = normalize_events(
            events=events,
            record=enriched_record,
            allowed_event_types=allowed_event_types,
            model_name=fused_model_name,
            alias_map=alias_map,
        )
        labels = derive_training_labels(normalized_events, min_confidence=args.min_confidence)
        hard_negative = bool(enriched_record.get("hard_negative", False))
        training_eligible = bool(labels) or hard_negative
        review_reasons: List[str] = []
        if not training_eligible:
            review_reasons.append("no_resolved_events_above_threshold")
        if any(event.get("alignment_status") in {"ambiguous", "unresolved"} for event in normalized_events):
            review_reasons.append("contains_unresolved_alignment")

        output_row = {
            "article_id": enriched_record.get("article_id", ""),
            "url": enriched_record.get("url", ""),
            "title": enriched_record.get("title", ""),
            "source": enriched_record.get("source", ""),
            "publish_time": enriched_record.get("publish_time", ""),
            "text": enriched_record.get("text", ""),
            "company_mentions": company_mentions,
            "split": enriched_record.get("split", ""),
            "label_source": "silver",
            "hard_negative": hard_negative,
            "matched_labels": enriched_record.get("matched_labels", []),
            "matched_terms": enriched_record.get("matched_terms", []),
            "labels": labels,
            "training_eligible": training_eligible,
            "review_reasons": review_reasons,
            "events": normalized_events,
        }
        return idx, output_row

    results: List[Dict[str, Any] | None] = [None] * len(records)
    pending_results: Dict[int, Dict[str, Any]] = {}
    failed_records_for_retry: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    completed = processed_records_before_resume
    next_write_idx = processed_records_before_resume
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(process_record, idx, record): (idx, record)
            for idx, record in enumerate(records, start=processed_records_before_resume)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting", ncols=100):
            idx, record = futures[future]
            try:
                result_idx, output_row = future.result()
                local_idx = result_idx - processed_records_before_resume
                results[local_idx] = output_row
                pending_results[result_idx] = {
                    "_result_kind": "success",
                    "output_row": output_row,
                }
            except Exception as exc:
                failed_row = build_failed_record(
                    idx=idx,
                    record=record,
                    exc=exc,
                    stage="initial_extract",
                    retry_attempt=0,
                )
                failed_records_for_retry.append((idx, record, failed_row))
                pending_results[idx] = {
                    "_result_kind": "failed",
                    "failed_row": failed_row,
                }
            next_write_idx, flushed_count, _flushed_failed = flush_contiguous_results(
                pending_results=pending_results,
                next_write_idx=next_write_idx,
                output_path=args.output,
                failed_output_path=failed_output_path,
            )
            completed = next_write_idx
            if flushed_count:
                checkpoint_payload = build_checkpoint_payload(
                    input_path=args.input,
                    output_path=args.output,
                    total_input_records=total_input_records,
                    start_offset=batch_origin_start,
                    end_offset=batch_end,
                    processed_records=completed,
                )
                write_checkpoint(checkpoint_path, checkpoint_payload)
            if should_log_progress(completed=completed, total=batch_total_records, every=args.progress_log_every):
                print(
                    json.dumps(
                        {
                            "stage": "run_docfee_news",
                            "status": "progress",
                            "completed": completed,
                            "total": batch_total_records,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    recovered_rows: List[Dict[str, Any]] = []
    final_failed_rows: List[Dict[str, Any]] = []
    if args.retry_failed_at_end and failed_records_for_retry:
        print(
            json.dumps(
                {
                    "stage": "run_docfee_news",
                    "status": "retry_failed_start",
                    "failed_records": len(failed_records_for_retry),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        for idx, record, failed_row in failed_records_for_retry:
            try:
                _result_idx, output_row = process_record(idx, record)
                output_row["retry_recovered"] = True
                append_jsonl_rows(args.output, [output_row])
                recovered_rows.append(
                    {
                        "idx": idx,
                        "article_id": output_row.get("article_id", ""),
                        "status": "recovered",
                    }
                )
            except Exception as exc:
                failed_retry_row = dict(failed_row)
                failed_retry_row["status"] = "final_failed"
                failed_retry_row["retry_attempt"] = 1
                failed_retry_row["final_error_type"] = exc.__class__.__name__
                failed_retry_row["final_error_message"] = str(exc)
                final_failed_rows.append(failed_retry_row)

        append_jsonl_rows(recovered_output_path, recovered_rows)
        if final_failed_rows:
            append_jsonl_rows(failed_output_path, final_failed_rows)
        print(
            json.dumps(
                {
                    "stage": "run_docfee_news",
                    "status": "retry_failed_done",
                    "recovered": len(recovered_rows),
                    "final_failed": len(final_failed_rows),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    pretty_output_path = args.pretty_output
    if not pretty_output_path:
        base, _ = os.path.splitext(args.output)
        pretty_output_path = base + ".pretty.json"
    finalized_rows = load_or_prepare_inputs(input_path=args.output, output_path=prepared_path)[0]
    write_pretty_json(pretty_output_path, finalized_rows)

    # Phase E: summary csv
    output_dir = os.path.dirname(args.output) or "."
    summary_path = os.path.join(output_dir, "summary.csv")
    write_summary_csv(finalized_rows, summary_path, taxonomy_polarity_pairs)

    # Phase F: review sample pack
    sample_path = os.path.join(output_dir, "review_sample.jsonl")
    report_path = os.path.join(output_dir, "review_report.md")
    review_stat = build_review_pack(
        input_path=args.output,
        sample_path=sample_path,
        report_path=report_path,
        sample_rate=args.review_rate,
        seed=42,
    )

    checkpoint_payload = build_checkpoint_payload(
        input_path=args.input,
        output_path=args.output,
        total_input_records=total_input_records,
        start_offset=batch_origin_start,
        end_offset=batch_end,
        processed_records=next_write_idx,
    )
    write_checkpoint(checkpoint_path, checkpoint_payload)

    print(
        json.dumps(
            {
                "records": len(finalized_rows),
                "output": args.output,
                "pretty_output": pretty_output_path,
                "summary_csv": summary_path,
                "review": review_stat,
                "prepared": prepared_path,
                "few_shot_path": resolved_few_shot_path,
                "models": selected_providers,
                "checkpoint_path": checkpoint_path,
                "next_offset": checkpoint_payload["next_offset"],
                "remaining_records": checkpoint_payload["remaining_records"],
                "failed_output": failed_output_path,
                "retry_recovered_output": recovered_output_path,
                "recovered_count": len(recovered_rows),
                "final_failed_count": len(final_failed_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
