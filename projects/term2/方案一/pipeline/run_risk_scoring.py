import argparse
import json
import os
from typing import Any, Callable, Dict, List

import torch
import yaml

from aggregate_events import aggregate_events, load_jsonl, write_jsonl
from event_text_repr import build_group_text_representation, embed_texts
from llm_extract_openai import chat_complete_with_retry, get_client_for_model
from risk_analysis_gen import build_risk_analysis_messages, generate_risk_analysis
from risk_analysis_repr import build_analysis_representations
from risk_scorer import (
    RISK_DIM_ORDER,
    build_company_month_labels,
    build_labeled_company_month_rows,
    build_company_month_outputs,
    parse_hidden_dims,
    predict_company_month_scores,
    score_group_vectors,
    train_company_month_mlp,
    write_scores_csv,
)


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def maybe_load_jsonl(path: str, reuse_existing: bool) -> List[Dict[str, Any]] | None:
    if reuse_existing and os.path.exists(path):
        return load_jsonl(path)
    return None


def append_jsonl_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_stage_checkpoint_payload(
    *,
    stage: str,
    total_rows: int,
    next_offset: int,
    output_path: str,
) -> Dict[str, Any]:
    safe_total = max(0, int(total_rows))
    safe_next = max(0, min(int(next_offset), safe_total))
    return {
        "stage": stage,
        "total_rows": safe_total,
        "next_offset": safe_next,
        "remaining_rows": max(0, safe_total - safe_next),
        "is_complete": safe_next >= safe_total,
        "output_path": output_path,
    }


def write_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def should_log_stage_progress(completed: int, total: int, every: int) -> bool:
    if total <= 0 or completed <= 0:
        return False
    safe_every = max(1, every)
    return completed % safe_every == 0 or completed == total


def run_resumable_jsonl_stage(
    *,
    stage_name: str,
    rows: List[Dict[str, Any]],
    output_path: str,
    checkpoint_path: str,
    process_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    resume_long_ops: bool,
    progress_log_every: int,
) -> List[Dict[str, Any]]:
    total_rows = len(rows)
    start_offset = 0
    if resume_long_ops and os.path.exists(checkpoint_path) and os.path.exists(output_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        start_offset = int(payload.get("next_offset", 0) or 0)
    else:
        ensure_parent_dir(output_path)
        with open(output_path, "w", encoding="utf-8"):
            pass

    print(
        json.dumps(
            {
                "stage": stage_name,
                "status": "start",
                "rows": total_rows,
                "start_offset": start_offset,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if start_offset > 0:
        print(
            json.dumps(
                {
                    "stage": stage_name,
                    "status": "resume",
                    "next_offset": start_offset,
                    "total_rows": total_rows,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    completed = start_offset
    for idx in range(start_offset, total_rows):
        output_row = process_fn(rows[idx])
        append_jsonl_rows(output_path, [output_row])
        completed = idx + 1
        write_checkpoint(
            checkpoint_path,
            build_stage_checkpoint_payload(
                stage=stage_name,
                total_rows=total_rows,
                next_offset=completed,
                output_path=output_path,
            ),
        )
        if should_log_stage_progress(completed=completed, total=total_rows, every=progress_log_every):
            print(
                json.dumps(
                    {
                        "stage": stage_name,
                        "status": "progress",
                        "completed": completed,
                        "total": total_rows,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    write_checkpoint(
        checkpoint_path,
        build_stage_checkpoint_payload(
            stage=stage_name,
            total_rows=total_rows,
            next_offset=total_rows,
            output_path=output_path,
        ),
    )
    print(json.dumps({"stage": stage_name, "status": "done", "rows": total_rows}, ensure_ascii=False), flush=True)
    return load_jsonl(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end risk scoring pipeline.")
    parser.add_argument("--input", required=True, help="Input news_events.jsonl path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--schema", required=True, help="Schema yaml path")
    parser.add_argument("--prompt", required=True, help="Risk analysis prompt template path")
    parser.add_argument("--model", default="kimi-k2.5", help="Chat model name for risk analysis")
    parser.add_argument("--embedding-model", default="text-embedding-v4", help="Embedding model")
    parser.add_argument("--mode", choices=["inference", "train"], default="inference", help="Run mode")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min confidence for accepted events")
    parser.add_argument("--checkpoint-path", default=None, help="Company-month MLP checkpoint path for train output or inference input")
    parser.add_argument("--horizon-months", type=int, default=3, help="Future label horizon in months for company-month supervision")
    parser.add_argument("--mlp-hidden-dims", default="64,32", help="Comma-separated hidden dims for company-month MLP")
    parser.add_argument("--mlp-dropout", type=float, default=0.2, help="Dropout for company-month MLP")
    parser.add_argument("--mlp-seed", type=int, default=42, help="Random seed for company-month MLP training")
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3, help="Learning rate for company-month MLP training")
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4, help="Weight decay for company-month MLP training")
    parser.add_argument("--mlp-batch-size", type=int, default=64, help="Batch size for company-month MLP training")
    parser.add_argument("--mlp-max-epochs", type=int, default=100, help="Max epochs for company-month MLP training")
    parser.add_argument("--mlp-patience", type=int, default=10, help="Early stopping patience for company-month MLP training")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio by global_id for company-month MLP")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio by global_id for company-month MLP")
    parser.add_argument(
        "--resume-long-ops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume long-running LLM/embedding stages from stage checkpoints when available",
    )
    parser.add_argument("--stage-progress-log-every", type=int, default=10, help="Emit a JSON progress log every N completed groups for long-running stages")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing intermediate JSONL files in output-dir when available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.schema, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    risk_dims = list((schema.get("event_taxonomy") or {}).keys()) or RISK_DIM_ORDER

    aggregated_path = os.path.join(args.output_dir, "aggregated_events.jsonl")
    event_text_repr_path = os.path.join(args.output_dir, "event_text_repr.jsonl")
    risk_analysis_path = os.path.join(args.output_dir, "risk_analysis.jsonl")
    risk_analysis_repr_path = os.path.join(args.output_dir, "risk_analysis_repr.jsonl")
    risk_group_vectors_path = os.path.join(args.output_dir, "risk_group_vectors.jsonl")
    risk_vectors_path = os.path.join(args.output_dir, "risk_vectors.jsonl")
    risk_scores_path = os.path.join(args.output_dir, "risk_scores.csv")
    company_month_labels_path = os.path.join(args.output_dir, "company_month_labels.jsonl")
    company_month_train_summary_path = os.path.join(args.output_dir, "company_month_mlp_summary.json")
    company_month_checkpoint_path = args.checkpoint_path or os.path.join(args.output_dir, "company_month_mlp.pt")
    event_text_checkpoint_path = os.path.join(args.output_dir, "event_text_repr.checkpoint.json")
    risk_analysis_checkpoint_path = os.path.join(args.output_dir, "risk_analysis.checkpoint.json")
    risk_analysis_repr_checkpoint_path = os.path.join(args.output_dir, "risk_analysis_repr.checkpoint.json")

    input_rows = load_jsonl(args.input)

    aggregated_rows = maybe_load_jsonl(aggregated_path, args.reuse_existing)
    if aggregated_rows is None:
        aggregated_rows = aggregate_events(input_rows, min_confidence=args.min_confidence)
        write_jsonl(aggregated_path, aggregated_rows)

    client = get_client_for_model(args.embedding_model)

    def build_analysis_repr_row(row: Dict[str, Any]) -> Dict[str, Any]:
        embedding = embed_texts(
            client=client,
            texts=[str(row.get("analysis_text", "")).strip()],
            model=args.embedding_model,
        )[0]
        return {
            "group_id": row.get("group_id", ""),
            "global_id": row.get("global_id", ""),
            "canonical_company_name": row.get("canonical_company_name", ""),
            "feature_month": row.get("feature_month", ""),
            "risk_dim": row.get("risk_dim", ""),
            "event_count": row.get("event_count", 0),
            "embedding_model": args.embedding_model,
            "embedding_dim": len(embedding),
            "h_analyze": embedding,
        }

    event_text_rows = maybe_load_jsonl(event_text_repr_path, args.reuse_existing)
    if event_text_rows is None:
        event_text_rows = run_resumable_jsonl_stage(
            stage_name="event_text_repr",
            rows=aggregated_rows,
            output_path=event_text_repr_path,
            checkpoint_path=event_text_checkpoint_path,
            resume_long_ops=args.resume_long_ops,
            progress_log_every=args.stage_progress_log_every,
            process_fn=lambda group: build_group_text_representation(
                group=group,
                client=client,
                embedding_model=args.embedding_model,
                num_heads=args.num_heads,
            ),
        )

    risk_analysis_rows = maybe_load_jsonl(risk_analysis_path, args.reuse_existing)
    if risk_analysis_rows is None:
        analysis_client = get_client_for_model(args.model)
        risk_analysis_rows = run_resumable_jsonl_stage(
            stage_name="risk_analysis",
            rows=aggregated_rows,
            output_path=risk_analysis_path,
            checkpoint_path=risk_analysis_checkpoint_path,
            resume_long_ops=args.resume_long_ops,
            progress_log_every=args.stage_progress_log_every,
            process_fn=lambda group: {
                "group_id": group.get("group_id", ""),
                "global_id": group.get("global_id", ""),
                "canonical_company_name": group.get("canonical_company_name", ""),
                "feature_month": group.get("feature_month", ""),
                "risk_dim": group.get("risk_dim", ""),
                "event_count": group.get("event_count", 0),
                "analysis_model": args.model,
                "analysis_text": chat_complete_with_retry(
                    client=analysis_client,
                    model=args.model,
                    messages=build_risk_analysis_messages(prompt_template, group),
                    temperature=0.0,
                ).strip(),
            },
        )

    risk_analysis_repr_rows = maybe_load_jsonl(risk_analysis_repr_path, args.reuse_existing)
    if risk_analysis_repr_rows is None:
        risk_analysis_repr_rows = run_resumable_jsonl_stage(
            stage_name="risk_analysis_repr",
            rows=risk_analysis_rows,
            output_path=risk_analysis_repr_path,
            checkpoint_path=risk_analysis_repr_checkpoint_path,
            resume_long_ops=args.resume_long_ops,
            progress_log_every=args.stage_progress_log_every,
            process_fn=build_analysis_repr_row,
        )

    group_vector_rows = score_group_vectors(
        aggregated_rows=aggregated_rows,
        event_text_rows=event_text_rows,
        analysis_rows=risk_analysis_repr_rows,
        checkpoint_path=None,
    )
    risk_vector_rows, _proxy_score_rows = build_company_month_outputs(group_vector_rows, risk_dims=risk_dims)
    write_jsonl(risk_group_vectors_path, group_vector_rows)
    write_jsonl(risk_vectors_path, risk_vector_rows)

    training_summary: Dict[str, Any] | None = None
    if args.mode == "train":
        company_month_label_rows = build_company_month_labels(
            risk_vector_rows=risk_vector_rows,
            aggregated_rows=aggregated_rows,
            horizon_months=args.horizon_months,
            risk_dims=risk_dims,
        )
        write_jsonl(company_month_labels_path, company_month_label_rows)
        labeled_rows = build_labeled_company_month_rows(
            risk_vector_rows=risk_vector_rows,
            label_rows=company_month_label_rows,
            risk_dims=risk_dims,
        )
        training_result = train_company_month_mlp(
            rows=labeled_rows,
            seed=args.mlp_seed,
            hidden_dims=parse_hidden_dims(args.mlp_hidden_dims),
            dropout=args.mlp_dropout,
            learning_rate=args.mlp_learning_rate,
            weight_decay=args.mlp_weight_decay,
            batch_size=args.mlp_batch_size,
            max_epochs=args.mlp_max_epochs,
            patience=args.mlp_patience,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        ensure_parent_dir(company_month_checkpoint_path)
        torch.save(training_result["checkpoint"], company_month_checkpoint_path)
        score_rows = predict_company_month_scores(
            risk_vector_rows=risk_vector_rows,
            checkpoint_payload=training_result["checkpoint"],
            risk_dims=risk_dims,
        )
        training_summary = {
            "labels_path": company_month_labels_path,
            "checkpoint_path": company_month_checkpoint_path,
            "hidden_dims": parse_hidden_dims(args.mlp_hidden_dims),
            "dropout": args.mlp_dropout,
            "seed": args.mlp_seed,
            "label_horizon_months": args.horizon_months,
            "label_count": len(company_month_label_rows),
            "best_epoch": training_result["best_epoch"],
            "param_count": training_result["param_count"],
            "train_metrics": training_result["train_metrics"],
            "val_metrics": training_result["val_metrics"],
            "test_metrics": training_result["test_metrics"],
        }
        with open(company_month_train_summary_path, "w", encoding="utf-8") as f:
            json.dump(training_summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
    else:
        if not os.path.exists(company_month_checkpoint_path):
            raise FileNotFoundError(
                f"Inference mode requires an existing company-month MLP checkpoint: {company_month_checkpoint_path}"
            )
        checkpoint_payload = torch.load(company_month_checkpoint_path, map_location="cpu")
        score_rows = predict_company_month_scores(
            risk_vector_rows=risk_vector_rows,
            checkpoint_payload=checkpoint_payload,
            risk_dims=risk_dims,
        )

    write_scores_csv(risk_scores_path, score_rows, risk_dims=risk_dims)

    summary = {
        "input": args.input,
        "output_dir": args.output_dir,
        "mode": args.mode,
        "risk_analysis_model": args.model,
        "embedding_model": args.embedding_model,
        "aggregated_group_count": len(aggregated_rows),
        "company_month_count": len(risk_vector_rows),
        "artifacts": {
            "aggregated_events": aggregated_path,
            "event_text_repr": event_text_repr_path,
            "risk_analysis": risk_analysis_path,
            "risk_analysis_repr": risk_analysis_repr_path,
            "event_text_repr_checkpoint": event_text_checkpoint_path,
            "risk_analysis_checkpoint": risk_analysis_checkpoint_path,
            "risk_analysis_repr_checkpoint": risk_analysis_repr_checkpoint_path,
            "risk_group_vectors": risk_group_vectors_path,
            "risk_vectors": risk_vectors_path,
            "risk_scores": risk_scores_path,
        },
    }
    if training_summary is not None:
        summary["artifacts"]["company_month_labels"] = company_month_labels_path
        summary["artifacts"]["company_month_mlp_checkpoint"] = company_month_checkpoint_path
        summary["artifacts"]["company_month_mlp_summary"] = company_month_train_summary_path
        summary["company_month_training"] = training_summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    ensure_parent_dir(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
