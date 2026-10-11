import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from aggregate_events import load_jsonl, write_jsonl
from risk_scorer import (
    CompanyMonthRiskMLP,
    RISK_DIM_ORDER,
    binary_average_precision,
    build_labeled_company_month_rows,
    extract_company_month_xy,
    split_rows_by_global_id,
    standardize_features,
    train_company_month_mlp,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def split_rows_by_feature_month(
    rows: Sequence[Dict[str, Any]],
    calibration_ratio: float,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    months = sorted({str(row.get("feature_month", "")).strip() for row in rows if str(row.get("feature_month", "")).strip()})
    if len(months) < 2:
        raise ValueError("Reliability backtest requires at least two distinct feature months.")
    if calibration_ratio <= 0 or calibration_ratio >= 1:
        raise ValueError("calibration_ratio must be between 0 and 1.")
    calibration_month_count = max(1, int(round(len(months) * calibration_ratio)))
    calibration_month_count = min(calibration_month_count, len(months) - 1)
    calibration_months = months[:calibration_month_count]
    prediction_months = months[calibration_month_count:]
    calibration_rows = [row for row in rows if str(row.get("feature_month", "")).strip() in calibration_months]
    prediction_rows = [row for row in rows if str(row.get("feature_month", "")).strip() in prediction_months]
    if not calibration_rows or not prediction_rows:
        raise ValueError("Temporal split produced an empty calibration or prediction segment.")
    return calibration_months, prediction_months, calibration_rows, prediction_rows


def load_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return checkpoint


def normalize_with_checkpoint(rows: Sequence[Dict[str, Any]], checkpoint: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = extract_company_month_xy(rows)
    mean = torch.tensor(list(checkpoint["feature_mean"]), dtype=torch.float32)
    std = torch.tensor(list(checkpoint["feature_std"]), dtype=torch.float32)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    x = (x - mean) / std
    return x, y


def predict_probabilities(
    rows: Sequence[Dict[str, Any]],
    checkpoint: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    model = CompanyMonthRiskMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        output_dim=int(checkpoint.get("output_dim", len(RISK_DIM_ORDER))),
        dropout=float(checkpoint.get("dropout", 0.0) or 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    x, y = normalize_with_checkpoint(rows, checkpoint)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x))
    return probabilities, y


def binary_brier_score(y_true: List[float], y_prob: List[float]) -> float:
    if not y_true:
        return 0.0
    return float(sum((prob - truth) ** 2 for truth, prob in zip(y_true, y_prob)) / len(y_true))


def calibration_bins(
    y_true: List[float],
    y_prob: List[float],
    *,
    n_bins: int = 5,
) -> List[Dict[str, Any]]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    bins: List[Dict[str, Any]] = []
    width = 1.0 / n_bins
    for idx in range(n_bins):
        left = idx * width
        right = 1.0 if idx == n_bins - 1 else (idx + 1) * width
        if idx == n_bins - 1:
            mask = [left <= prob <= right for prob in y_prob]
        else:
            mask = [left <= prob < right for prob in y_prob]
        indices = [i for i, keep in enumerate(mask) if keep]
        if not indices:
            bins.append(
                {
                    "bin_index": idx,
                    "bin_start": left,
                    "bin_end": right,
                    "count": 0,
                    "avg_predicted_probability": None,
                    "observed_positive_rate": None,
                    "absolute_gap": None,
                }
            )
            continue
        preds = [y_prob[i] for i in indices]
        truths = [y_true[i] for i in indices]
        avg_pred = sum(preds) / len(preds)
        obs_rate = sum(truths) / len(truths)
        bins.append(
            {
                "bin_index": idx,
                "bin_start": left,
                "bin_end": right,
                "count": len(indices),
                "avg_predicted_probability": avg_pred,
                "observed_positive_rate": obs_rate,
                "absolute_gap": abs(avg_pred - obs_rate),
            }
        )
    return bins


def expected_calibration_error(
    y_true: List[float],
    y_prob: List[float],
    *,
    n_bins: int = 5,
) -> float:
    bins = calibration_bins(y_true, y_prob, n_bins=n_bins)
    total = len(y_true)
    if total == 0:
        return 0.0
    error = 0.0
    for bucket in bins:
        if not bucket["count"]:
            continue
        error += (bucket["count"] / total) * float(bucket["absolute_gap"] or 0.0)
    return float(error)


def evaluate_probability_metrics(y_true: torch.Tensor, probabilities: torch.Tensor) -> Dict[str, Any]:
    labels = y_true.cpu().tolist()
    probs = probabilities.cpu().tolist()
    metrics: Dict[str, Any] = {}
    auprcs: List[float] = []
    briers: List[float] = []
    eces: List[float] = []
    for idx, risk_dim in enumerate(RISK_DIM_ORDER):
        dim_true = [row[idx] for row in labels]
        dim_prob = [row[idx] for row in probs]
        dim_auprc = binary_average_precision(dim_true, dim_prob)
        dim_brier = binary_brier_score(dim_true, dim_prob)
        dim_ece = expected_calibration_error(dim_true, dim_prob, n_bins=5)
        metrics[f"{risk_dim}_auprc"] = dim_auprc
        metrics[f"{risk_dim}_brier"] = dim_brier
        metrics[f"{risk_dim}_ece"] = dim_ece
        metrics[f"{risk_dim}_mean_predicted_probability"] = sum(dim_prob) / len(dim_prob) if dim_prob else 0.0
        metrics[f"{risk_dim}_observed_positive_rate"] = sum(dim_true) / len(dim_true) if dim_true else 0.0
        if dim_auprc is not None:
            auprcs.append(dim_auprc)
        briers.append(dim_brier)
        eces.append(dim_ece)
    metrics["macro_auprc"] = sum(auprcs) / len(auprcs) if auprcs else None
    metrics["macro_brier"] = sum(briers) / len(briers) if briers else None
    metrics["macro_ece"] = sum(eces) / len(eces) if eces else None
    return metrics


def bootstrap_confidence_interval(
    y_true: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    metric_name: str,
    bootstrap_samples: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    labels = y_true.cpu().tolist()
    probs = probabilities.cpu().tolist()
    if not labels:
        return {"metric_name": metric_name, "sample_count": 0, "lower": None, "upper": None, "mean": None}
    rng = random.Random(seed)
    values: List[float] = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(labels)) for _ in range(len(labels))]
        boot_y = torch.tensor([labels[i] for i in indices], dtype=torch.float32)
        boot_p = torch.tensor([probs[i] for i in indices], dtype=torch.float32)
        metrics = evaluate_probability_metrics(boot_y, boot_p)
        value = metrics.get(metric_name)
        if value is not None:
            values.append(float(value))
    if not values:
        return {"metric_name": metric_name, "sample_count": 0, "lower": None, "upper": None, "mean": None}
    values.sort()
    lower = values[int(0.025 * (len(values) - 1))]
    upper = values[int(0.975 * (len(values) - 1))]
    return {
        "metric_name": metric_name,
        "sample_count": len(values),
        "lower": lower,
        "upper": upper,
        "mean": sum(values) / len(values),
    }


def select_best_thresholds_from_validation(
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
        dim_prob = [row[idx] for row in probs]
        best_threshold = 0.5
        best_auprc = float("-inf")
        for threshold in grid:
            masked_prob = [1.0 if prob >= threshold else 0.0 for prob in dim_prob]
            current = binary_average_precision(dim_true, masked_prob)
            current = float(current if current is not None else float("-inf"))
            if current > best_auprc:
                best_auprc = current
                best_threshold = threshold
        selected[risk_dim] = best_threshold
    return {
        "threshold_by_risk_dim": selected,
        "thresholds": [selected[risk_dim] for risk_dim in RISK_DIM_ORDER],
    }


def monthly_backtest_rows(rows: Sequence[Dict[str, Any]], probabilities: torch.Tensor) -> List[Dict[str, Any]]:
    probs = probabilities.cpu().tolist()
    month_map: Dict[str, List[Tuple[Dict[str, Any], List[float]]]] = {}
    for row, prob_row in zip(rows, probs):
        feature_month = str(row.get("feature_month", "")).strip()
        month_map.setdefault(feature_month, []).append((row, prob_row))
    outputs: List[Dict[str, Any]] = []
    for feature_month in sorted(month_map.keys()):
        batch = month_map[feature_month]
        month_row = {"feature_month": feature_month, "sample_count": len(batch)}
        for idx, risk_dim in enumerate(RISK_DIM_ORDER):
            predicted = [item[1][idx] for item in batch]
            observed = [float(item[0][f"y_{risk_dim.replace('_risk', '')}"]) for item in batch]
            month_row[f"{risk_dim}_mean_predicted_probability"] = sum(predicted) / len(predicted) if predicted else 0.0
            month_row[f"{risk_dim}_observed_positive_rate"] = sum(observed) / len(observed) if observed else 0.0
        outputs.append(month_row)
    return outputs


def build_prediction_rows(rows: Sequence[Dict[str, Any]], probabilities: torch.Tensor) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for row, prob_row in zip(rows, probabilities.cpu().tolist()):
        output = {
            "global_id": row.get("global_id", ""),
            "canonical_company_name": row.get("canonical_company_name", ""),
            "feature_month": row.get("feature_month", ""),
        }
        for idx, risk_dim in enumerate(RISK_DIM_ORDER):
            prefix = risk_dim.replace("_risk", "")
            output[f"p_{prefix}"] = float(prob_row[idx])
            output[f"y_{prefix}"] = float(row.get(f"y_{prefix}", 0.0) or 0.0)
        outputs.append(output)
    return outputs


def run_reliability_backtest(
    *,
    risk_vectors_path: str,
    labels_path: str,
    output_dir: str,
    calibration_ratio: float,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    hidden_dims: List[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    bootstrap_samples: int,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    risk_vector_rows = load_jsonl(risk_vectors_path)
    label_rows = load_jsonl(labels_path)
    labeled_rows = build_labeled_company_month_rows(risk_vector_rows, label_rows, risk_dims=RISK_DIM_ORDER)

    calibration_months, prediction_months, calibration_rows, prediction_rows = split_rows_by_feature_month(
        labeled_rows,
        calibration_ratio=calibration_ratio,
    )

    train_result = train_company_month_mlp(
        rows=calibration_rows,
        seed=seed,
        hidden_dims=hidden_dims,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    checkpoint = train_result["checkpoint"]
    checkpoint_path = os.path.join(output_dir, "backtest_company_month_mlp.pt")
    torch.save(checkpoint, checkpoint_path)

    train_rows, val_rows, _test_rows = split_rows_by_global_id(
        calibration_rows,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    _train_probs, _train_y = predict_probabilities(train_rows, checkpoint)
    val_probabilities, val_y = predict_probabilities(val_rows, checkpoint)
    prediction_probabilities, prediction_y = predict_probabilities(prediction_rows, checkpoint)

    threshold_grid = [round(step / 100.0, 2) for step in range(1, 100)]
    threshold_selection = select_best_thresholds_from_validation(
        val_y,
        val_probabilities,
        grid=threshold_grid,
    )

    prediction_metrics = evaluate_probability_metrics(prediction_y, prediction_probabilities)
    bootstrap_summary = {
        "macro_auprc": bootstrap_confidence_interval(
            prediction_y,
            prediction_probabilities,
            metric_name="macro_auprc",
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "macro_brier": bootstrap_confidence_interval(
            prediction_y,
            prediction_probabilities,
            metric_name="macro_brier",
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "macro_ece": bootstrap_confidence_interval(
            prediction_y,
            prediction_probabilities,
            metric_name="macro_ece",
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    }

    reliability_bins: List[Dict[str, Any]] = []
    labels = prediction_y.cpu().tolist()
    probs = prediction_probabilities.cpu().tolist()
    for idx, risk_dim in enumerate(RISK_DIM_ORDER):
        dim_true = [row[idx] for row in labels]
        dim_prob = [row[idx] for row in probs]
        for bucket in calibration_bins(dim_true, dim_prob, n_bins=5):
            reliability_bins.append({"risk_dim": risk_dim, **bucket})

    prediction_rows_output = build_prediction_rows(prediction_rows, prediction_probabilities)
    monthly_backtest = monthly_backtest_rows(prediction_rows, prediction_probabilities)

    reliability_bins_path = os.path.join(output_dir, "reliability_bins.jsonl")
    prediction_rows_path = os.path.join(output_dir, "prediction_rows.jsonl")
    monthly_backtest_path = os.path.join(output_dir, "monthly_backtest.jsonl")
    summary_path = os.path.join(output_dir, "reliability_summary.json")

    write_jsonl(reliability_bins_path, reliability_bins)
    write_jsonl(prediction_rows_path, prediction_rows_output)
    write_jsonl(monthly_backtest_path, monthly_backtest)

    summary = {
        "risk_vectors_path": risk_vectors_path,
        "labels_path": labels_path,
        "calibration_ratio": calibration_ratio,
        "seed": seed,
        "calibration_months": calibration_months,
        "prediction_months": prediction_months,
        "calibration_row_count": len(calibration_rows),
        "prediction_row_count": len(prediction_rows),
        "model": {
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "param_count": train_result["param_count"],
            "best_epoch": train_result["best_epoch"],
        },
        "calibration_training_metrics": {
            "train_metrics": train_result["train_metrics"],
            "val_metrics": train_result["val_metrics"],
            "test_metrics": train_result["test_metrics"],
        },
        "threshold_calibration": threshold_selection,
        "prediction_metrics": prediction_metrics,
        "bootstrap_confidence_intervals": bootstrap_summary,
        "artifacts": {
            "checkpoint_path": checkpoint_path,
            "reliability_bins": reliability_bins_path,
            "prediction_rows": prediction_rows_path,
            "monthly_backtest": monthly_backtest_path,
            "summary": summary_path,
        },
    }
    write_json(summary_path, summary)
    return summary


def parse_hidden_dims(raw: str) -> List[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal reliability backtest for company-month risk scoring.")
    parser.add_argument("--risk-vectors", required=True, help="risk_vectors.jsonl path")
    parser.add_argument("--labels", required=True, help="company_month_labels.jsonl path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--calibration-ratio", type=float, default=0.7, help="Share of months used for calibration/training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio within calibration period")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio within calibration period")
    parser.add_argument("--mlp-hidden-dims", default="64,32", help="MLP hidden dims")
    parser.add_argument("--mlp-dropout", type=float, default=0.3, help="MLP dropout")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--bootstrap-samples", type=int, default=200, help="Bootstrap sample count for confidence intervals")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_reliability_backtest(
        risk_vectors_path=args.risk_vectors,
        labels_path=args.labels,
        output_dir=args.output_dir,
        calibration_ratio=args.calibration_ratio,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        hidden_dims=parse_hidden_dims(args.mlp_hidden_dims),
        dropout=args.mlp_dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
