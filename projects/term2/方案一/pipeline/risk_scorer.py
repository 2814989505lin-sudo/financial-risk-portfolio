import argparse
import csv
import json
import math
import os
import random
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn

from aggregate_events import load_jsonl, write_jsonl


RISK_DIM_ORDER = ["credit_risk", "fraud_risk", "operational_risk", "market_risk"]


class RiskScorerMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CompanyMonthRiskMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parse_hidden_dims(raw: str) -> List[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def combine_vectors(h_text: List[float], h_analyze: List[float]) -> List[float]:
    return list(h_text) + list(h_analyze)


def shift_feature_month(feature_month: str, offset: int) -> str:
    base = datetime.strptime(feature_month, "%Y-%m")
    year = base.year + ((base.month - 1 + offset) // 12)
    month = ((base.month - 1 + offset) % 12) + 1
    return f"{year:04d}-{month:02d}"


def build_row_map(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("group_id", "")).strip(): row for row in rows if str(row.get("group_id", "")).strip()}


def compute_proxy_score(group: Dict[str, Any]) -> float:
    negative_conf = 0.0
    positive_conf = 0.0
    for event in group.get("events", []) or []:
        confidence = float(event.get("confidence", 0.0) or 0.0)
        if event.get("event_polarity") == "negative":
            negative_conf += confidence
        elif event.get("event_polarity") == "positive":
            positive_conf += confidence
    net_signal = max(negative_conf - 0.5 * positive_conf, 0.0)
    return 1.0 - math.exp(-net_signal)


def score_group_vectors(
    aggregated_rows: List[Dict[str, Any]],
    event_text_rows: List[Dict[str, Any]],
    analysis_rows: List[Dict[str, Any]],
    checkpoint_path: str | None = None,
) -> List[Dict[str, Any]]:
    aggregated_map = build_row_map(aggregated_rows)
    event_text_map = build_row_map(event_text_rows)
    analysis_map = build_row_map(analysis_rows)
    group_ids = sorted(set(aggregated_map) & set(event_text_map) & set(analysis_map))

    model: RiskScorerMLP | None = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        input_dim = int(checkpoint["input_dim"])
        model = RiskScorerMLP(input_dim=input_dim)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    outputs: List[Dict[str, Any]] = []
    for group_id in group_ids:
        aggregated_row = aggregated_map[group_id]
        event_text_row = event_text_map[group_id]
        analysis_row = analysis_map[group_id]
        z_r = combine_vectors(
            h_text=list(event_text_row.get("h_text", []) or []),
            h_analyze=list(analysis_row.get("h_analyze", []) or []),
        )
        score_source = "proxy_without_trained_mlp"
        score = compute_proxy_score(aggregated_row)
        if model is not None:
            with torch.no_grad():
                tensor = torch.tensor(z_r, dtype=torch.float32).unsqueeze(0)
                score = float(model(tensor).squeeze(0).item())
            score_source = "trained_mlp"
        outputs.append(
            {
                "group_id": group_id,
                "global_id": aggregated_row.get("global_id", ""),
                "canonical_company_name": aggregated_row.get("canonical_company_name", ""),
                "feature_month": aggregated_row.get("feature_month", ""),
                "risk_dim": aggregated_row.get("risk_dim", ""),
                "event_count": aggregated_row.get("event_count", 0),
                "negative_event_count": aggregated_row.get("negative_event_count", 0),
                "positive_event_count": aggregated_row.get("positive_event_count", 0),
                "embedding_dim": int(event_text_row.get("embedding_dim", 0) or 0),
                "z_r": z_r,
                "score": score,
                "score_source": score_source,
            }
        )
    return outputs


def build_company_month_outputs(
    group_vector_rows: List[Dict[str, Any]],
    risk_dims: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered_risk_dims = list(risk_dims or RISK_DIM_ORDER)
    vector_dim = len(group_vector_rows[0].get("z_r", []) or []) if group_vector_rows else 0
    company_month_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in group_vector_rows:
        key = (str(row.get("global_id", "")), str(row.get("feature_month", "")))
        bucket = company_month_map.get(key)
        if bucket is None:
            bucket = {
                "global_id": row.get("global_id", ""),
                "canonical_company_name": row.get("canonical_company_name", ""),
                "feature_month": row.get("feature_month", ""),
                "dimension_details": {},
            }
            company_month_map[key] = bucket
        risk_dim = str(row.get("risk_dim", ""))
        bucket["dimension_details"][risk_dim] = {
            "score": float(row.get("score", 0.0) or 0.0),
            "event_count": int(row.get("event_count", 0) or 0),
            "score_source": str(row.get("score_source", "")),
            "z_r": list(row.get("z_r", []) or []),
        }

    risk_vector_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    for key in sorted(company_month_map.keys()):
        bucket = company_month_map[key]
        dimension_details = bucket["dimension_details"]
        full_vector: List[float] = []
        score_row = {
            "global_id": bucket["global_id"],
            "canonical_company_name": bucket["canonical_company_name"],
            "feature_month": bucket["feature_month"],
        }
        normalized_details: Dict[str, Dict[str, Any]] = {}
        for risk_dim in ordered_risk_dims:
            detail = dimension_details.get(risk_dim)
            if detail is None:
                detail = {
                    "score": 0.0,
                    "event_count": 0,
                    "score_source": "missing_dim",
                    "z_r": [0.0] * vector_dim,
                }
            normalized_details[risk_dim] = detail
            full_vector.extend(detail["z_r"])
            score_row[risk_dim] = detail["score"]
        risk_vector_rows.append(
            {
                "global_id": bucket["global_id"],
                "canonical_company_name": bucket["canonical_company_name"],
                "feature_month": bucket["feature_month"],
                "risk_dim_order": ordered_risk_dims,
                "vector_dim_per_risk_dim": vector_dim,
                "risk_vector": full_vector,
                "dimension_details": normalized_details,
            }
        )
        score_rows.append(score_row)
    return risk_vector_rows, score_rows


def build_company_month_feature_vector(
    row: Dict[str, Any],
    risk_dims: List[str] | None = None,
) -> List[float]:
    ordered_risk_dims = list(risk_dims or row.get("risk_dim_order") or RISK_DIM_ORDER)
    dimension_details = row.get("dimension_details", {}) if isinstance(row.get("dimension_details"), dict) else {}
    risk_vector = list(row.get("risk_vector", []) or [])
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
    return risk_vector + stats


def build_company_month_labels(
    risk_vector_rows: Sequence[Dict[str, Any]],
    aggregated_rows: Sequence[Dict[str, Any]],
    *,
    horizon_months: int = 3,
    risk_dims: List[str] | None = None,
) -> List[Dict[str, Any]]:
    ordered_risk_dims = list(risk_dims or RISK_DIM_ORDER)
    negative_months: set[Tuple[str, str, str]] = set()
    observed_months = {
        str(row.get("feature_month", "")).strip()
        for row in risk_vector_rows
        if str(row.get("feature_month", "")).strip()
    }
    max_observed_month = max(observed_months) if observed_months else ""
    for row in aggregated_rows:
        global_id = str(row.get("global_id", "")).strip()
        feature_month = str(row.get("feature_month", "")).strip()
        risk_dim = str(row.get("risk_dim", "")).strip()
        if not global_id or not feature_month or risk_dim not in ordered_risk_dims:
            continue
        if int(row.get("negative_event_count", 0) or 0) > 0:
            negative_months.add((global_id, feature_month, risk_dim))

    labels: List[Dict[str, Any]] = []
    for row in risk_vector_rows:
        global_id = str(row.get("global_id", "")).strip()
        feature_month = str(row.get("feature_month", "")).strip()
        if not global_id or not feature_month:
            continue
        if max_observed_month and shift_feature_month(feature_month, horizon_months) > max_observed_month:
            continue
        label_row = {
            "global_id": global_id,
            "canonical_company_name": row.get("canonical_company_name", ""),
            "feature_month": feature_month,
        }
        for risk_dim in ordered_risk_dims:
            label_row[f"y_{risk_dim.replace('_risk', '')}"] = 1.0 if any(
                (global_id, shift_feature_month(feature_month, offset), risk_dim) in negative_months
                for offset in range(1, horizon_months + 1)
            ) else 0.0
        labels.append(label_row)
    return labels


def build_labeled_company_month_rows(
    risk_vector_rows: Sequence[Dict[str, Any]],
    label_rows: Sequence[Dict[str, Any]],
    risk_dims: List[str] | None = None,
) -> List[Dict[str, Any]]:
    labels_map = {
        (str(row.get("global_id", "")).strip(), str(row.get("feature_month", "")).strip()): row
        for row in label_rows
    }
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
                "feature_vector": build_company_month_feature_vector(row, risk_dims=risk_dims),
                "y_credit": float(labels.get("y_credit", 0.0) or 0.0),
                "y_fraud": float(labels.get("y_fraud", 0.0) or 0.0),
                "y_operational": float(labels.get("y_operational", 0.0) or 0.0),
                "y_market": float(labels.get("y_market", 0.0) or 0.0),
            }
        )
    if not rows:
        raise ValueError("No overlapping rows found between risk_vectors and company-month labels.")
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
        raise ValueError("MLP 训练至少需要 3 个不同的 global_id，才能切分 train/val/test。")
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


def extract_company_month_xy(rows: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("Cannot extract features from an empty split.")
    x = torch.tensor([list(row.get("feature_vector", []) or []) for row in rows], dtype=torch.float32)
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


def train_company_month_mlp(
    *,
    rows: Sequence[Dict[str, Any]],
    seed: int,
    hidden_dims: List[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, Any]:
    set_seed(seed)
    train_rows, val_rows, test_rows = split_rows_by_global_id(rows, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    train_x, train_y = extract_company_month_xy(train_rows)
    val_x, val_y = extract_company_month_xy(val_rows)
    test_x, test_y = extract_company_month_xy(test_rows)
    train_x, [val_x, test_x], mean, std = standardize_features(train_x, [val_x, test_x])

    model = CompanyMonthRiskMLP(
        input_dim=train_x.size(1),
        hidden_dims=hidden_dims,
        output_dim=len(RISK_DIM_ORDER),
        dropout=dropout,
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
        macro_auprc = val_metrics.get("macro_auprc")
        current_val = float(macro_auprc) if macro_auprc is not None else float("-inf")
        if current_val > best_val:
            best_val = current_val
            best_epoch = epoch + 1
            best_metrics = val_metrics
            best_state = {
                "model_state_dict": model.state_dict(),
                "input_dim": train_x.size(1),
                "hidden_dims": list(hidden_dims),
                "output_dim": len(RISK_DIM_ORDER),
                "dropout": dropout,
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
        raise RuntimeError("Training failed to produce a valid company-month MLP checkpoint.")

    model.load_state_dict(best_state["model_state_dict"])
    model.eval()
    with torch.no_grad():
        train_logits = model(train_x)
        test_logits = model(test_x)

    return {
        "seed": seed,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "param_count": count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "train_metrics": evaluate_predictions(train_y, train_logits),
        "val_metrics": best_metrics,
        "test_metrics": evaluate_predictions(test_y, test_logits),
        "checkpoint": best_state,
    }


def predict_company_month_scores(
    risk_vector_rows: Sequence[Dict[str, Any]],
    checkpoint_payload: Dict[str, Any],
    risk_dims: List[str] | None = None,
) -> List[Dict[str, Any]]:
    ordered_risk_dims = list(risk_dims or RISK_DIM_ORDER)
    model = CompanyMonthRiskMLP(
        input_dim=int(checkpoint_payload["input_dim"]),
        hidden_dims=list(checkpoint_payload["hidden_dims"]),
        output_dim=int(checkpoint_payload.get("output_dim", len(ordered_risk_dims))),
        dropout=float(checkpoint_payload.get("dropout", 0.0) or 0.0),
    )
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()

    mean = torch.tensor(list(checkpoint_payload["feature_mean"]), dtype=torch.float32)
    std = torch.tensor(list(checkpoint_payload["feature_std"]), dtype=torch.float32)

    rows: List[Dict[str, Any]] = []
    for row in risk_vector_rows:
        feature_vector = torch.tensor(build_company_month_feature_vector(row, risk_dims=ordered_risk_dims), dtype=torch.float32)
        normalized = ((feature_vector - mean) / std).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(normalized)).squeeze(0).tolist()
        score_row = {
            "global_id": row.get("global_id", ""),
            "canonical_company_name": row.get("canonical_company_name", ""),
            "feature_month": row.get("feature_month", ""),
        }
        for idx, risk_dim in enumerate(ordered_risk_dims):
            score_row[risk_dim] = float(probabilities[idx])
        rows.append(score_row)
    return rows


def write_scores_csv(path: str, rows: List[Dict[str, Any]], risk_dims: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["global_id", "canonical_company_name", "feature_month", *risk_dims],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine H_text and H_analyze into risk vectors and scores.")
    parser.add_argument("--aggregated-events", required=True, help="aggregated_events.jsonl path")
    parser.add_argument("--event-text-repr", required=True, help="event_text_repr.jsonl path")
    parser.add_argument("--analysis-repr", required=True, help="risk_analysis_repr.jsonl path")
    parser.add_argument("--group-output", required=True, help="Output group-level risk vectors JSONL path")
    parser.add_argument("--risk-vectors-output", required=True, help="Output company-month risk_vectors.jsonl path")
    parser.add_argument("--risk-scores-output", required=True, help="Output risk_scores.csv path")
    parser.add_argument("--checkpoint-path", default=None, help="Optional trained MLP checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregated_rows = load_jsonl(args.aggregated_events)
    event_text_rows = load_jsonl(args.event_text_repr)
    analysis_rows = load_jsonl(args.analysis_repr)
    group_rows = score_group_vectors(
        aggregated_rows=aggregated_rows,
        event_text_rows=event_text_rows,
        analysis_rows=analysis_rows,
        checkpoint_path=args.checkpoint_path,
    )
    risk_vector_rows, score_rows = build_company_month_outputs(group_rows, risk_dims=RISK_DIM_ORDER)
    write_jsonl(args.group_output, group_rows)
    write_jsonl(args.risk_vectors_output, risk_vector_rows)
    write_scores_csv(args.risk_scores_output, score_rows, risk_dims=RISK_DIM_ORDER)
    print(
        json.dumps(
            {
                "group_output": args.group_output,
                "risk_vectors_output": args.risk_vectors_output,
                "risk_scores_output": args.risk_scores_output,
                "group_count": len(group_rows),
                "company_month_count": len(risk_vector_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
