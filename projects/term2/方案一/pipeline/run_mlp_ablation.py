import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn

from aggregate_events import load_jsonl, write_jsonl
from risk_scorer import (
    CompanyMonthRiskMLP,
    RISK_DIM_ORDER,
    build_company_month_feature_vector,
    count_trainable_parameters,
)


@dataclass
class AblationConfig:
    name: str
    hidden_dims: List[int]
    dropout: float


def parse_seed_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def default_ablation_configs() -> List[AblationConfig]:
    return [
        AblationConfig(name="D1", hidden_dims=[64], dropout=0.3),
        AblationConfig(name="D2", hidden_dims=[64, 32], dropout=0.3),
        AblationConfig(name="D3", hidden_dims=[128, 64], dropout=0.3),
        AblationConfig(name="D4", hidden_dims=[128, 64, 32], dropout=0.3),
    ]


def load_company_month_labels(path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    labels: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in load_jsonl(path):
        global_id = str(row.get("global_id", "")).strip()
        feature_month = str(row.get("feature_month", "")).strip()
        if not global_id or not feature_month:
            continue
        labels[(global_id, feature_month)] = {
            "y_credit": float(row.get("y_credit", 0.0) or 0.0),
            "y_fraud": float(row.get("y_fraud", 0.0) or 0.0),
            "y_operational": float(row.get("y_operational", 0.0) or 0.0),
            "y_market": float(row.get("y_market", 0.0) or 0.0),
        }
    return labels


def build_labeled_rows(
    risk_vector_rows: Sequence[Dict[str, Any]],
    labels_map: Dict[Tuple[str, str], Dict[str, float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in risk_vector_rows:
        key = (str(row.get("global_id", "")).strip(), str(row.get("feature_month", "")).strip())
        labels = labels_map.get(key)
        if labels is None:
            continue
        rows.append(
            {
                "global_id": key[0],
                "feature_month": key[1],
                "canonical_company_name": row.get("canonical_company_name", ""),
                "feature_vector": build_company_month_feature_vector(row, risk_dims=RISK_DIM_ORDER),
                **labels,
            }
        )
    if not rows:
        raise ValueError("No overlapping rows found between risk_vectors and labels.")
    return rows


def split_rows_by_global_id(
    rows: Sequence[Dict[str, Any]],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("global_id", "")), []).append(dict(row))
    global_ids = sorted(grouped.keys())
    if len(global_ids) < 3:
        raise ValueError("MLP 消融实验至少需要 3 个不同的 global_id，才能切分 train/val/test。")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio 和 val_ratio 必须为正，且 train_ratio + val_ratio < 1。")
    rng = random.Random(seed)
    rng.shuffle(global_ids)

    total = len(global_ids)
    train_count = max(1, int(round(total * train_ratio)))
    val_count = max(1, int(round(total * val_ratio))) if total >= 3 else max(0, total - train_count)
    if train_count + val_count >= total and total >= 3:
        val_count = 1
        train_count = total - val_count - 1
    test_count = max(0, total - train_count - val_count)

    train_ids = set(global_ids[:train_count])
    val_ids = set(global_ids[train_count:train_count + val_count])
    test_ids = set(global_ids[train_count + val_count:train_count + val_count + test_count])

    train_rows = [row for gid in sorted(train_ids) for row in grouped[gid]]
    val_rows = [row for gid in sorted(val_ids) for row in grouped[gid]]
    test_rows = [row for gid in sorted(test_ids) for row in grouped[gid]]
    return train_rows, val_rows, test_rows


def extract_xy(rows: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("Cannot extract features from an empty split.")
    x = torch.tensor([list(row.get("feature_vector", []) or []) for row in rows], dtype=torch.float32)
    y = torch.tensor(
        [
            [
                float(row.get("y_credit", 0.0)),
                float(row.get("y_fraud", 0.0)),
                float(row.get("y_operational", 0.0)),
                float(row.get("y_market", 0.0)),
            ]
            for row in rows
        ],
        dtype=torch.float32,
    )
    return x, y


def standardize_features(
    train_x: torch.Tensor,
    other_tensors: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    normalized_train = (train_x - mean) / std
    normalized_others = [((tensor - mean) / std) for tensor in other_tensors]
    return normalized_train, normalized_others, mean, std


def binary_roc_auc(y_true: List[float], y_score: List[float]) -> float | None:
    positives = sum(1 for value in y_true if value >= 0.5)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_score, label) in enumerate(ranked, start=1):
        if label >= 0.5:
            rank_sum += rank
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def binary_average_precision(y_true: List[float], y_score: List[float]) -> float | None:
    positives = sum(1 for value in y_true if value >= 0.5)
    if positives == 0:
        return None
    ranked = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for idx, (_score, label) in enumerate(ranked, start=1):
        if label >= 0.5:
            true_positives += 1
            precision_sum += true_positives / idx
    return float(precision_sum / positives)


def binary_f1(y_true: List[float], y_score: List[float], threshold: float = 0.5) -> float | None:
    tp = fp = fn = 0
    for truth, score in zip(y_true, y_score):
        pred = 1 if score >= threshold else 0
        actual = 1 if truth >= 0.5 else 0
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 1:
            fn += 1
    if tp == 0 and fp == 0 and fn == 0:
        return None
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def evaluate_predictions(y_true: torch.Tensor, logits: torch.Tensor) -> Dict[str, Any]:
    probabilities = torch.sigmoid(logits).cpu().tolist()
    labels = y_true.cpu().tolist()
    metrics: Dict[str, Any] = {}
    aurocs: List[float] = []
    auprcs: List[float] = []
    f1s: List[float] = []

    for idx, risk_dim in enumerate(RISK_DIM_ORDER):
        dim_true = [row[idx] for row in labels]
        dim_score = [row[idx] for row in probabilities]
        dim_auroc = binary_roc_auc(dim_true, dim_score)
        dim_auprc = binary_average_precision(dim_true, dim_score)
        dim_f1 = binary_f1(dim_true, dim_score)
        metrics[f"{risk_dim}_auroc"] = dim_auroc
        metrics[f"{risk_dim}_auprc"] = dim_auprc
        metrics[f"{risk_dim}_f1"] = dim_f1
        if dim_auroc is not None:
            aurocs.append(dim_auroc)
        if dim_auprc is not None:
            auprcs.append(dim_auprc)
        if dim_f1 is not None:
            f1s.append(dim_f1)

    metrics["macro_auroc"] = sum(aurocs) / len(aurocs) if aurocs else None
    metrics["macro_auprc"] = sum(auprcs) / len(auprcs) if auprcs else None
    metrics["macro_f1"] = sum(f1s) / len(f1s) if f1s else None
    return metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def iterate_minibatches(x: torch.Tensor, y: torch.Tensor, batch_size: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    total = x.size(0)
    indices = torch.randperm(total)
    for start in range(0, total, batch_size):
        batch_indices = indices[start:start + batch_size]
        yield x[batch_indices], y[batch_indices]


def train_single_run(
    *,
    config: AblationConfig,
    seed: int,
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> Dict[str, Any]:
    set_seed(seed)
    train_x, train_y = extract_xy(train_rows)
    val_x, val_y = extract_xy(val_rows)
    test_x, test_y = extract_xy(test_rows)
    train_x, [val_x, test_x], mean, std = standardize_features(train_x, [val_x, test_x])

    model = CompanyMonthRiskMLP(
        input_dim=train_x.size(1),
        hidden_dims=config.hidden_dims,
        output_dim=len(RISK_DIM_ORDER),
        dropout=config.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_metrics = None
    best_epoch = -1
    best_val = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        for batch_x, batch_y in iterate_minibatches(train_x, train_y, batch_size=batch_size):
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_metrics = evaluate_predictions(val_y, val_logits)
        current_val = float(val_metrics.get("macro_auprc") or float("-inf"))
        if current_val > best_val:
            best_val = current_val
            best_epoch = epoch + 1
            best_metrics = val_metrics
            best_state = {
                "model_state_dict": model.state_dict(),
                "input_dim": train_x.size(1),
                "hidden_dims": list(config.hidden_dims),
                "output_dim": len(RISK_DIM_ORDER),
                "dropout": config.dropout,
                "feature_mean": mean.tolist(),
                "feature_std": std.tolist(),
                "seed": seed,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training failed to produce a valid checkpoint.")

    model.load_state_dict(best_state["model_state_dict"])
    model.eval()
    with torch.no_grad():
        train_logits = model(train_x)
        test_logits = model(test_x)

    return {
        "config_name": config.name,
        "hidden_dims": list(config.hidden_dims),
        "dropout": config.dropout,
        "seed": seed,
        "param_count": count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "train_metrics": evaluate_predictions(train_y, train_logits),
        "val_metrics": best_metrics,
        "test_metrics": evaluate_predictions(test_y, test_logits),
        "checkpoint": best_state,
    }


def aggregate_run_metrics(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        return {}
    keys = ["macro_auroc", "macro_auprc", "macro_f1"]
    summary: Dict[str, Any] = {
        "config_name": runs[0]["config_name"],
        "hidden_dims": runs[0]["hidden_dims"],
        "dropout": runs[0]["dropout"],
        "param_count": runs[0]["param_count"],
        "seed_count": len(runs),
    }
    for split in ("train_metrics", "val_metrics", "test_metrics"):
        for key in keys:
            values = [run[split].get(key) for run in runs if run[split].get(key) is not None]
            summary[f"{split}_{key}_mean"] = sum(values) / len(values) if values else None
    summary["best_by_val_macro_auprc"] = summary.get("val_metrics_macro_auprc_mean")
    return summary


def choose_best_summary(summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [summary for summary in summaries if summary.get("best_by_val_macro_auprc") is not None]
    if not valid:
        raise ValueError("No valid ablation summaries available.")
    valid.sort(
        key=lambda item: (
            float(item.get("best_by_val_macro_auprc") or float("-inf")),
            -int(item.get("param_count", 0) or 0),
        ),
        reverse=True,
    )
    best_score = float(valid[0]["best_by_val_macro_auprc"])
    tied = [item for item in valid if math.isclose(float(item["best_by_val_macro_auprc"]), best_score, rel_tol=1e-9)]
    tied.sort(key=lambda item: int(item.get("param_count", 0) or 0))
    return tied[0]


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled MLP ablation experiments for company-month risk prediction.")
    parser.add_argument("--risk-vectors", required=True, help="Input risk_vectors.jsonl path")
    parser.add_argument("--labels", required=True, help="Input company_month_labels.jsonl path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated random seeds")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    risk_vector_rows = load_jsonl(args.risk_vectors)
    labels_map = load_company_month_labels(args.labels)
    labeled_rows = build_labeled_rows(risk_vector_rows, labels_map)
    seeds = parse_seed_list(args.seeds)
    configs = default_ablation_configs()

    all_runs: List[Dict[str, Any]] = []
    all_run_results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for config in configs:
        config_runs: List[Dict[str, Any]] = []
        for seed in seeds:
            train_rows, val_rows, test_rows = split_rows_by_global_id(
                labeled_rows,
                seed=seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
            run_result = train_single_run(
                config=config,
                seed=seed,
                train_rows=train_rows,
                val_rows=val_rows,
                test_rows=test_rows,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
            config_runs.append(run_result)
            all_run_results.append(run_result)
            all_runs.append(
                {
                    "config_name": run_result["config_name"],
                    "hidden_dims": run_result["hidden_dims"],
                    "dropout": run_result["dropout"],
                    "seed": run_result["seed"],
                    "param_count": run_result["param_count"],
                    "best_epoch": run_result["best_epoch"],
                    "train_metrics": run_result["train_metrics"],
                    "val_metrics": run_result["val_metrics"],
                    "test_metrics": run_result["test_metrics"],
                }
            )
        summary_rows.append(aggregate_run_metrics(config_runs))

    best_summary = choose_best_summary(summary_rows)
    best_run = max(
        (
            run
            for run in all_runs
            if run["config_name"] == best_summary["config_name"]
        ),
        key=lambda item: float(item["val_metrics"].get("macro_auprc") or float("-inf")),
    )
    best_checkpoint_payload = next(
        result["checkpoint"]
        for result in all_run_results
        if result["config_name"] == best_run["config_name"] and result["seed"] == best_run["seed"]
    )

    runs_path = os.path.join(args.output_dir, "mlp_ablation_runs.jsonl")
    summary_path = os.path.join(args.output_dir, "mlp_ablation_summary.json")
    checkpoint_path = os.path.join(args.output_dir, "best_company_month_mlp.pt")

    write_jsonl(runs_path, all_runs)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "risk_vectors": args.risk_vectors,
                "labels": args.labels,
                "seeds": seeds,
                "fixed_hyperparams": {
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "max_epochs": args.max_epochs,
                    "patience": args.patience,
                },
                "configs": [asdict(config) for config in configs],
                "summaries": summary_rows,
                "best_config": best_summary,
                "best_run": best_run,
                "artifacts": {
                    "runs": runs_path,
                    "summary": summary_path,
                    "best_checkpoint": checkpoint_path,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")
    save_checkpoint(checkpoint_path, best_checkpoint_payload)

    print(
        json.dumps(
            {
                "runs": runs_path,
                "summary": summary_path,
                "best_checkpoint": checkpoint_path,
                "best_config": best_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
