import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn

from aggregate_events import load_jsonl, write_jsonl
from risk_scorer import (
    CompanyMonthRiskMLP,
    RISK_DIM_ORDER,
    binary_average_precision,
    binary_f1,
    binary_roc_auc,
    build_company_month_feature_vector,
    count_trainable_parameters,
    iterate_minibatches,
    set_seed,
    split_rows_by_global_id,
    standardize_features,
)


class LinearMultiLabelHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class SplitTensors:
    train_x_raw: torch.Tensor
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x_raw: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x_raw: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


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


def split_zr(detail: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    zr = list(detail.get("z_r", []) or [])
    if not zr:
        return [], []
    midpoint = len(zr) // 2
    return zr[:midpoint], zr[midpoint:]


def build_stats_only_feature_vector(row: Dict[str, Any], risk_dims: Sequence[str] | None = None) -> List[float]:
    ordered_risk_dims = list(risk_dims or row.get("risk_dim_order") or RISK_DIM_ORDER)
    dimension_details = row.get("dimension_details", {}) if isinstance(row.get("dimension_details"), dict) else {}
    stats: List[float] = []
    total_event_count = 0.0
    active_dim_count = 0.0
    for risk_dim in ordered_risk_dims:
        detail = dimension_details.get(risk_dim, {}) if isinstance(dimension_details.get(risk_dim), dict) else {}
        score = float(detail.get("score", 0.0) or 0.0)
        event_count = float(detail.get("event_count", 0) or 0)
        is_active = 1.0 if event_count > 0 else 0.0
        total_event_count += event_count
        active_dim_count += is_active
        stats.extend([score, event_count, is_active])
    stats.extend([total_event_count, active_dim_count])
    return stats


def build_h_text_only_feature_vector(row: Dict[str, Any], risk_dims: Sequence[str] | None = None) -> List[float]:
    ordered_risk_dims = list(risk_dims or row.get("risk_dim_order") or RISK_DIM_ORDER)
    dimension_details = row.get("dimension_details", {}) if isinstance(row.get("dimension_details"), dict) else {}
    vector: List[float] = []
    for risk_dim in ordered_risk_dims:
        detail = dimension_details.get(risk_dim, {}) if isinstance(dimension_details.get(risk_dim), dict) else {}
        h_text, _ = split_zr(detail)
        vector.extend(h_text)
    return vector


def build_h_analyze_only_feature_vector(row: Dict[str, Any], risk_dims: Sequence[str] | None = None) -> List[float]:
    ordered_risk_dims = list(risk_dims or row.get("risk_dim_order") or RISK_DIM_ORDER)
    dimension_details = row.get("dimension_details", {}) if isinstance(row.get("dimension_details"), dict) else {}
    vector: List[float] = []
    for risk_dim in ordered_risk_dims:
        detail = dimension_details.get(risk_dim, {}) if isinstance(dimension_details.get(risk_dim), dict) else {}
        _, h_analyze = split_zr(detail)
        vector.extend(h_analyze)
    return vector


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
                "full_feature_vector": build_company_month_feature_vector(row, risk_dims=RISK_DIM_ORDER),
                "stats_only_feature_vector": build_stats_only_feature_vector(row, risk_dims=RISK_DIM_ORDER),
                "h_text_only_feature_vector": build_h_text_only_feature_vector(row, risk_dims=RISK_DIM_ORDER),
                "h_analyze_only_feature_vector": build_h_analyze_only_feature_vector(row, risk_dims=RISK_DIM_ORDER),
                **labels,
            }
        )
    if not rows:
        raise ValueError("No overlapping rows found between risk_vectors and labels.")
    return rows


def extract_xy(rows: Sequence[Dict[str, Any]], feature_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("Cannot extract features from an empty split.")
    x = torch.tensor([list(row.get(feature_key, []) or []) for row in rows], dtype=torch.float32)
    y = torch.tensor(
        [
            [
                float(row.get("y_credit", 0.0) or 0.0),
                float(row.get("y_fraud", 0.0) or 0.0),
                float(row.get("y_operational", 0.0) or 0.0),
                float(row.get("y_market", 0.0) or 0.0),
            ]
            for row in rows
        ],
        dtype=torch.float32,
    )
    return x, y


def prepare_split_tensors(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_key: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> SplitTensors:
    train_rows, val_rows, test_rows = split_rows_by_global_id(
        rows,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    train_x_raw, train_y = extract_xy(train_rows, feature_key)
    val_x_raw, val_y = extract_xy(val_rows, feature_key)
    test_x_raw, test_y = extract_xy(test_rows, feature_key)
    train_x, [val_x, test_x], _, _ = standardize_features(train_x_raw, [val_x_raw, test_x_raw])
    return SplitTensors(
        train_x_raw=train_x_raw,
        train_x=train_x,
        train_y=train_y,
        val_x_raw=val_x_raw,
        val_x=val_x,
        val_y=val_y,
        test_x_raw=test_x_raw,
        test_x=test_x,
        test_y=test_y,
    )


def evaluate_probability_predictions(
    y_true: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    thresholds: Sequence[float] | None = None,
) -> Dict[str, Any]:
    labels = y_true.cpu().tolist()
    probs = probabilities.cpu().tolist()
    threshold_values = list(thresholds or [0.5] * len(RISK_DIM_ORDER))

    metrics: Dict[str, Any] = {}
    aurocs: List[float] = []
    auprcs: List[float] = []
    f1s: List[float] = []
    for idx, risk_dim in enumerate(RISK_DIM_ORDER):
        dim_true = [row[idx] for row in labels]
        dim_score = [row[idx] for row in probs]
        dim_auroc = binary_roc_auc(dim_true, dim_score)
        dim_auprc = binary_average_precision(dim_true, dim_score)
        dim_f1 = binary_f1(dim_true, dim_score, threshold=threshold_values[idx])
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


def train_linear_baseline(
    *,
    split_tensors: SplitTensors,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)
    model = LinearMultiLabelHead(input_dim=split_tensors.train_x.size(1), output_dim=len(RISK_DIM_ORDER))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_val = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        for batch_x, batch_y in iterate_minibatches(split_tensors.train_x, split_tensors.train_y, batch_size=batch_size):
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probabilities = torch.sigmoid(model(split_tensors.val_x))
            val_metrics = evaluate_probability_predictions(split_tensors.val_y, val_probabilities)
        current_val = float(val_metrics.get("macro_auprc") or float("-inf"))
        if best_state is None or current_val > best_val:
            best_val = current_val
            best_state = model.state_dict()
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Linear baseline training failed to produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_probabilities = torch.sigmoid(model(split_tensors.train_x))
        val_probabilities = torch.sigmoid(model(split_tensors.val_x))
        test_probabilities = torch.sigmoid(model(split_tensors.test_x))
    return {
        "best_epoch": best_epoch,
        "param_count": count_trainable_parameters(model),
        "train_metrics": evaluate_probability_predictions(split_tensors.train_y, train_probabilities),
        "val_metrics": evaluate_probability_predictions(split_tensors.val_y, val_probabilities),
        "test_metrics": evaluate_probability_predictions(split_tensors.test_y, test_probabilities),
        "test_probabilities": test_probabilities,
        "val_probabilities": val_probabilities,
    }


def evaluate_prior_baseline(split_tensors: SplitTensors) -> Dict[str, Any]:
    train_prior = split_tensors.train_y.mean(dim=0, keepdim=True)
    train_probabilities = train_prior.repeat(split_tensors.train_y.size(0), 1)
    val_probabilities = train_prior.repeat(split_tensors.val_y.size(0), 1)
    test_probabilities = train_prior.repeat(split_tensors.test_y.size(0), 1)
    return {
        "best_epoch": 0,
        "param_count": 0,
        "train_metrics": evaluate_probability_predictions(split_tensors.train_y, train_probabilities),
        "val_metrics": evaluate_probability_predictions(split_tensors.val_y, val_probabilities),
        "test_metrics": evaluate_probability_predictions(split_tensors.test_y, test_probabilities),
        "test_probabilities": test_probabilities,
        "val_probabilities": val_probabilities,
        "train_prior": train_prior.squeeze(0).tolist(),
    }


def load_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return checkpoint


def evaluate_checkpoint_baseline(
    *,
    split_tensors: SplitTensors,
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    model = CompanyMonthRiskMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        output_dim=int(checkpoint.get("output_dim", len(RISK_DIM_ORDER))),
        dropout=float(checkpoint.get("dropout", 0.0) or 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = torch.tensor(list(checkpoint["feature_mean"]), dtype=torch.float32)
    std = torch.tensor(list(checkpoint["feature_std"]), dtype=torch.float32)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))

    train_x = (split_tensors.train_x_raw - mean) / std
    val_x = (split_tensors.val_x_raw - mean) / std
    test_x = (split_tensors.test_x_raw - mean) / std

    with torch.no_grad():
        train_probabilities = torch.sigmoid(model(train_x))
        val_probabilities = torch.sigmoid(model(val_x))
        test_probabilities = torch.sigmoid(model(test_x))
    return {
        "best_epoch": None,
        "param_count": count_trainable_parameters(model),
        "train_metrics": evaluate_probability_predictions(split_tensors.train_y, train_probabilities),
        "val_metrics": evaluate_probability_predictions(split_tensors.val_y, val_probabilities),
        "test_metrics": evaluate_probability_predictions(split_tensors.test_y, test_probabilities),
        "test_probabilities": test_probabilities,
        "val_probabilities": val_probabilities,
    }


def build_threshold_grid(step: float) -> List[float]:
    if step <= 0 or step >= 1:
        raise ValueError("threshold step must be between 0 and 1.")
    values: List[float] = []
    current = step
    while current < 1.0:
        values.append(round(current, 6))
        current += step
    if 0.5 not in values:
        values.append(0.5)
    return sorted(set(values))


def select_best_thresholds(
    val_y: torch.Tensor,
    val_probabilities: torch.Tensor,
    *,
    grid: Sequence[float],
) -> Dict[str, Any]:
    labels = val_y.cpu().tolist()
    probs = val_probabilities.cpu().tolist()
    selected: Dict[str, float] = {}
    for idx, risk_dim in enumerate(RISK_DIM_ORDER):
        dim_true = [row[idx] for row in labels]
        dim_score = [row[idx] for row in probs]
        best_threshold = 0.5
        best_f1 = float("-inf")
        for threshold in grid:
            current_f1 = binary_f1(dim_true, dim_score, threshold=threshold)
            current_f1 = float(current_f1 if current_f1 is not None else 0.0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        selected[risk_dim] = best_threshold
    thresholds = [selected[risk_dim] for risk_dim in RISK_DIM_ORDER]
    return {
        "threshold_by_risk_dim": selected,
        "thresholds": thresholds,
        "val_metrics_at_best_thresholds": evaluate_probability_predictions(val_y, val_probabilities, thresholds=thresholds),
    }


def summarize_label_distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    positive_counts = {
        "y_credit": sum(int(row.get("y_credit", 0) or 0) for row in rows),
        "y_fraud": sum(int(row.get("y_fraud", 0) or 0) for row in rows),
        "y_operational": sum(int(row.get("y_operational", 0) or 0) for row in rows),
        "y_market": sum(int(row.get("y_market", 0) or 0) for row in rows),
    }
    all_zero_count = sum(
        1
        for row in rows
        if sum(int(row.get(key, 0) or 0) for key in positive_counts.keys()) == 0
    )
    multi_positive_count = sum(
        1
        for row in rows
        if sum(int(row.get(key, 0) or 0) for key in positive_counts.keys()) >= 2
    )
    return {
        "label_count": total,
        "positive_count": positive_counts,
        "positive_rate": {key: (value / total if total else 0.0) for key, value in positive_counts.items()},
        "all_zero_count": all_zero_count,
        "multi_positive_count": multi_positive_count,
    }


def run_risk_validation(
    *,
    risk_vectors_path: str,
    labels_path: str,
    checkpoint_path: str,
    output_dir: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    threshold_step: float,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    risk_vector_rows = load_jsonl(risk_vectors_path)
    labels_map = load_company_month_labels(labels_path)
    labeled_rows = build_labeled_rows(risk_vector_rows, labels_map)

    full_split = prepare_split_tensors(
        labeled_rows,
        feature_key="full_feature_vector",
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    stats_split = prepare_split_tensors(
        labeled_rows,
        feature_key="stats_only_feature_vector",
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    h_text_split = prepare_split_tensors(
        labeled_rows,
        feature_key="h_text_only_feature_vector",
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    h_analyze_split = prepare_split_tensors(
        labeled_rows,
        feature_key="h_analyze_only_feature_vector",
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    baseline_runs: List[Dict[str, Any]] = []
    baseline_runs.append(
        {
            "baseline_name": "prior",
            **evaluate_prior_baseline(full_split),
        }
    )
    baseline_runs.append(
        {
            "baseline_name": "stats_only_linear",
            **train_linear_baseline(
                split_tensors=stats_split,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed,
            ),
        }
    )
    baseline_runs.append(
        {
            "baseline_name": "h_text_only_linear",
            **train_linear_baseline(
                split_tensors=h_text_split,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed,
            ),
        }
    )
    baseline_runs.append(
        {
            "baseline_name": "h_analyze_only_linear",
            **train_linear_baseline(
                split_tensors=h_analyze_split,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed,
            ),
        }
    )
    checkpoint_payload = load_checkpoint(checkpoint_path)
    full_model_result = {
        "baseline_name": "full_mlp",
        **evaluate_checkpoint_baseline(split_tensors=full_split, checkpoint=checkpoint_payload),
    }
    baseline_runs.append(full_model_result)

    threshold_grid = build_threshold_grid(threshold_step)
    threshold_selection = select_best_thresholds(
        full_split.val_y,
        full_model_result["val_probabilities"],
        grid=threshold_grid,
    )
    tuned_thresholds = threshold_selection["thresholds"]
    threshold_analysis = {
        "default_threshold": 0.5,
        "threshold_grid_size": len(threshold_grid),
        **threshold_selection,
        "test_metrics_at_default_threshold": evaluate_probability_predictions(
            full_split.test_y,
            full_model_result["test_probabilities"],
            thresholds=[0.5] * len(RISK_DIM_ORDER),
        ),
        "test_metrics_at_best_thresholds": evaluate_probability_predictions(
            full_split.test_y,
            full_model_result["test_probabilities"],
            thresholds=tuned_thresholds,
        ),
    }

    runs_path = os.path.join(output_dir, "baseline_runs.jsonl")
    cleaned_runs: List[Dict[str, Any]] = []
    for run in baseline_runs:
        cleaned = {
            key: value
            for key, value in run.items()
            if key not in {"test_probabilities", "val_probabilities"}
        }
        cleaned_runs.append(cleaned)
    write_jsonl(runs_path, cleaned_runs)

    summary = {
        "risk_vectors_path": risk_vectors_path,
        "labels_path": labels_path,
        "checkpoint_path": checkpoint_path,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "label_distribution": summarize_label_distribution(labeled_rows),
        "baseline_summary": cleaned_runs,
        "threshold_analysis": threshold_analysis,
        "artifacts": {
            "baseline_runs": runs_path,
            "summary": os.path.join(output_dir, "risk_validation_summary.json"),
        },
    }
    write_json(summary["artifacts"]["summary"], summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline comparison and threshold analysis for company-month risk scoring.")
    parser.add_argument("--risk-vectors", required=True, help="risk_vectors.jsonl path")
    parser.add_argument("--labels", required=True, help="company_month_labels.jsonl path")
    parser.add_argument("--checkpoint-path", required=True, help="Trained company_month_mlp.pt path")
    parser.add_argument("--output-dir", required=True, help="Output directory for validation artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio by global_id")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio by global_id")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for linear baselines")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for linear baselines")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for linear baselines")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs for linear baselines")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience for linear baselines")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="Threshold sweep step size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_risk_validation(
        risk_vectors_path=args.risk_vectors,
        labels_path=args.labels,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        threshold_step=args.threshold_step,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
