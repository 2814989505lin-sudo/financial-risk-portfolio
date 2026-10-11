import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
from scipy.sparse import save_npz

from aggregate_events import load_jsonl, write_jsonl
from community_detection import detect_leiden_communities
from community_evolution import track_community_evolution
from community_graph import adjacency_to_sparse, build_knn_adjacency
from community_similarity import compute_cosine_similarity_matrix


IDENTITY_FIELDS = {"global_id", "canonical_company_name", "feature_month"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_score_input(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], []
    feature_columns = [column for column in rows[0].keys() if column not in IDENTITY_FIELDS]
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        normalized_rows.append(
            {
                "global_id": str(row.get("global_id", "")).strip(),
                "canonical_company_name": str(row.get("canonical_company_name", "")).strip(),
                "feature_month": str(row.get("feature_month", "")).strip(),
                "vector": [float(row.get(column, 0.0) or 0.0) for column in feature_columns],
                "feature_names": feature_columns,
            }
        )
    return normalized_rows, feature_columns


def load_repr_input(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows = load_jsonl(path)
    normalized_rows: List[Dict[str, Any]] = []
    feature_names: List[str] = []
    for row in rows:
        vector = [float(value) for value in list(row.get("risk_vector", []) or [])]
        if not feature_names:
            feature_names = [f"repr_{index}" for index in range(len(vector))]
        normalized_rows.append(
            {
                "global_id": str(row.get("global_id", "")).strip(),
                "canonical_company_name": str(row.get("canonical_company_name", "")).strip(),
                "feature_month": str(row.get("feature_month", "")).strip(),
                "vector": vector,
                "feature_names": feature_names,
            }
        )
    return normalized_rows, feature_names


def load_feature_rows(path: str, input_mode: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    if input_mode == "repr":
        return load_repr_input(path)
    return load_score_input(path)


def group_rows_by_month(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        month = str(row.get("feature_month", "")).strip()
        if month:
            grouped[month].append(row)
    return dict(grouped)


def save_similarity_artifacts(
    output_dir: str,
    adjacency_by_month: Dict[str, np.ndarray],
) -> Dict[str, str]:
    if not adjacency_by_month:
        return {}
    if len(adjacency_by_month) == 1:
        feature_month, adjacency = next(iter(adjacency_by_month.items()))
        output_path = os.path.join(output_dir, "similarity_matrix.npz")
        save_npz(output_path, adjacency_to_sparse(adjacency))
        return {feature_month: output_path}

    matrix_dir = os.path.join(output_dir, "similarity_matrices")
    ensure_dir(matrix_dir)
    output_paths: Dict[str, str] = {}
    for feature_month, adjacency in sorted(adjacency_by_month.items()):
        output_path = os.path.join(matrix_dir, f"{feature_month}.npz")
        save_npz(output_path, adjacency_to_sparse(adjacency))
        output_paths[feature_month] = output_path
    return output_paths


def write_month_assignment_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def build_month_checkpoint_payload(
    input_path: str,
    input_mode: str,
    output_dir: str,
    k: int,
    resolution: float,
    total_months: int,
    completed_months: List[str],
    current_month: str | None,
) -> Dict[str, Any]:
    return {
        "input": input_path,
        "input_mode": input_mode,
        "output_dir": output_dir,
        "k": k,
        "resolution": resolution,
        "total_months": total_months,
        "completed_months": completed_months,
        "completed_month_count": len(completed_months),
        "remaining_month_count": max(total_months - len(completed_months), 0),
        "current_month": current_month,
        "is_complete": len(completed_months) >= total_months,
    }


def summarize_month_assignments(
    feature_month: str,
    month_rows: List[Dict[str, Any]],
    assignment_rows: List[Dict[str, Any]],
    modularity: float,
    similarity_matrix_path: str,
    graph_stats: Dict[str, Any],
) -> Dict[str, Any]:
    companies_by_community: Dict[int, List[str]] = defaultdict(list)
    for row in assignment_rows:
        companies_by_community[int(row["community_id"])].append(str(row.get("canonical_company_name", "")))
    communities: List[Dict[str, Any]] = []
    for community_id in sorted(companies_by_community.keys()):
        companies = sorted(name for name in companies_by_community[community_id] if name)
        communities.append(
            {
                "community_id": community_id,
                "company_count": len(companies_by_community[community_id]),
                "sample_companies": companies[:5],
            }
        )
    zero_vector_count = sum(1 for row in assignment_rows if int(row["community_id"]) == -1)
    return {
        "feature_month": feature_month,
        "company_count": len(month_rows),
        "community_count": len(communities),
        "zero_vector_count": zero_vector_count,
        "modularity": modularity,
        "similarity_matrix_path": similarity_matrix_path,
        "graph_stats": graph_stats,
        "communities": communities,
    }


def build_report(
    input_mode: str,
    month_summaries: List[Dict[str, Any]],
    k: int,
    resolution: float,
) -> str:
    lines = [
        "# 风险社区报告",
        "",
        f"- 输入模式：`{input_mode}`",
        f"- KNN K 值：`{k}`",
        f"- Leiden resolution：`{resolution}`",
        "",
    ]
    for summary in month_summaries:
        lines.extend(
            [
                f"## {summary['feature_month']}",
                "",
                f"- 公司数：`{summary['company_count']}`",
                f"- 社区数：`{summary['community_count']}`",
                f"- 全零向量公司数：`{summary['zero_vector_count']}`",
                f"- 模块度：`{summary['modularity']:.6f}`",
                "",
            ]
        )
        for community in summary["communities"]:
            sample_text = "、".join(community["sample_companies"]) if community["sample_companies"] else "无"
            lines.append(
                f"- 社区 `{community['community_id']}`：`{community['company_count']}` 家公司；示例：{sample_text}"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_community_pipeline(
    input_path: str,
    input_mode: str,
    output_dir: str,
    k: int,
    resolution: float,
    checkpoint_path: str | None = None,
    resume: bool = True,
    progress_log_every: int = 1,
) -> Dict[str, Any]:
    feature_rows, feature_names = load_feature_rows(input_path, input_mode=input_mode)
    rows_by_month = group_rows_by_month(feature_rows)

    checkpoint_path = checkpoint_path or os.path.join(output_dir, "community.checkpoint.json")
    month_assignments_dir = os.path.join(output_dir, "community_assignments_by_month")
    month_summaries_dir = os.path.join(output_dir, "community_month_summaries")
    ensure_dir(month_assignments_dir)
    ensure_dir(month_summaries_dir)

    completed_months: Set[str] = set()
    if resume and os.path.exists(checkpoint_path):
        checkpoint = load_json(checkpoint_path)
        completed_months = {
            str(month).strip()
            for month in list(checkpoint.get("completed_months", []) or [])
            if str(month).strip()
        }
        if completed_months:
            print(
                json.dumps(
                    {
                        "stage": "community",
                        "status": "resume",
                        "completed_month_count": len(completed_months),
                        "completed_months": sorted(completed_months),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    adjacency_by_month: Dict[str, np.ndarray] = {}
    month_summaries_by_month: Dict[str, Dict[str, Any]] = {}

    sorted_months = sorted(rows_by_month.keys())
    print(
        json.dumps(
            {
                "stage": "community",
                "status": "start",
                "month_count": len(sorted_months),
                "input_mode": input_mode,
                "resume": bool(resume),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    processed_month_count = 0
    for feature_month in sorted_months:
        month_rows = rows_by_month[feature_month]
        month_assignment_path = os.path.join(month_assignments_dir, f"{feature_month}.jsonl")
        month_summary_path = os.path.join(month_summaries_dir, f"{feature_month}.json")

        if feature_month in completed_months and os.path.exists(month_assignment_path) and os.path.exists(month_summary_path):
            month_summaries_by_month[feature_month] = load_json(month_summary_path)
            processed_month_count += 1
            if processed_month_count % max(progress_log_every, 1) == 0 or processed_month_count == len(sorted_months):
                print(
                    json.dumps(
                        {
                            "stage": "community",
                            "status": "progress",
                            "completed_months": processed_month_count,
                            "total_months": len(sorted_months),
                            "last_month": feature_month,
                            "resumed": True,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            continue

        matrix = np.asarray([row["vector"] for row in month_rows], dtype=float)
        vector_norms = np.linalg.norm(matrix, axis=1)
        zero_mask = vector_norms <= 1e-12
        active_indices = [index for index, is_zero in enumerate(zero_mask.tolist()) if not is_zero]

        memberships: Dict[int, int] = {}
        graph_stats: Dict[str, Any] = {"node_count": len(month_rows), "effective_k": 0, "edge_count": 0}
        modularity = 0.0

        if active_indices:
            active_matrix = matrix[active_indices]
            similarity = compute_cosine_similarity_matrix(active_matrix)
            adjacency, graph_stats = build_knn_adjacency(similarity, k=k)
            adjacency_by_month[feature_month] = adjacency
            detection = detect_leiden_communities(adjacency, resolution=resolution)
            modularity = float(detection["modularity"])
            for relative_index, membership in enumerate(detection["membership"]):
                memberships[active_indices[relative_index]] = int(membership)
        else:
            adjacency_by_month[feature_month] = np.zeros((len(month_rows), len(month_rows)), dtype=float)

        month_assignment_rows: List[Dict[str, Any]] = []
        for node_index, row in enumerate(month_rows):
            community_id = memberships.get(node_index, -1)
            month_assignment_rows.append(
                {
                    "global_id": row["global_id"],
                    "canonical_company_name": row["canonical_company_name"],
                    "feature_month": feature_month,
                    "community_id": int(community_id),
                    "node_index": node_index,
                    "vector_norm": float(vector_norms[node_index]),
                    "is_zero_vector": bool(zero_mask[node_index]),
                }
            )
        month_summary = summarize_month_assignments(
            feature_month=feature_month,
            month_rows=month_rows,
            assignment_rows=month_assignment_rows,
            modularity=modularity,
            similarity_matrix_path="",
            graph_stats=graph_stats,
        )
        write_month_assignment_rows(month_assignment_path, month_assignment_rows)
        write_json(month_summary_path, month_summary)
        month_summaries_by_month[feature_month] = month_summary
        completed_months.add(feature_month)
        processed_month_count += 1
        write_json(
            checkpoint_path,
            build_month_checkpoint_payload(
                input_path=input_path,
                input_mode=input_mode,
                output_dir=output_dir,
                k=k,
                resolution=resolution,
                total_months=len(sorted_months),
                completed_months=sorted(completed_months),
                current_month=feature_month,
            ),
        )
        if processed_month_count % max(progress_log_every, 1) == 0 or processed_month_count == len(sorted_months):
            print(
                json.dumps(
                    {
                        "stage": "community",
                        "status": "progress",
                        "completed_months": processed_month_count,
                        "total_months": len(sorted_months),
                        "last_month": feature_month,
                        "resumed": False,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    all_assignment_rows: List[Dict[str, Any]] = []
    month_summaries: List[Dict[str, Any]] = []
    for feature_month in sorted_months:
        month_assignment_path = os.path.join(month_assignments_dir, f"{feature_month}.jsonl")
        month_summary_path = os.path.join(month_summaries_dir, f"{feature_month}.json")
        if os.path.exists(month_assignment_path):
            all_assignment_rows.extend(load_jsonl(month_assignment_path))
        if os.path.exists(month_summary_path):
            month_summaries.append(load_json(month_summary_path))

    similarity_paths = save_similarity_artifacts(output_dir=output_dir, adjacency_by_month=adjacency_by_month)
    for summary in month_summaries:
        summary["similarity_matrix_path"] = similarity_paths.get(summary["feature_month"], "")

    evolution_rows = track_community_evolution(all_assignment_rows)
    report_text = build_report(input_mode=input_mode, month_summaries=month_summaries, k=k, resolution=resolution)

    assignments_path = os.path.join(output_dir, "community_assignments.jsonl")
    summary_path = os.path.join(output_dir, "community_summary.json")
    evolution_path = os.path.join(output_dir, "community_evolution.jsonl")
    report_path = os.path.join(output_dir, "community_report.md")

    write_jsonl(assignments_path, all_assignment_rows)
    write_jsonl(evolution_path, evolution_rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": input_path,
                "input_mode": input_mode,
                "k": k,
                "resolution": resolution,
                "feature_names": feature_names,
                "month_count": len(month_summaries),
                "months": month_summaries,
                "artifacts": {
                    "community_assignments": assignments_path,
                    "community_evolution": evolution_path,
                    "community_report": report_path,
                    "similarity_matrices": similarity_paths,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    write_json(
        checkpoint_path,
        build_month_checkpoint_payload(
            input_path=input_path,
            input_mode=input_mode,
            output_dir=output_dir,
            k=k,
            resolution=resolution,
            total_months=len(sorted_months),
            completed_months=sorted(completed_months),
            current_month=None,
        ),
    )

    return {
        "input": input_path,
        "input_mode": input_mode,
        "output_dir": output_dir,
        "company_month_count": len(feature_rows),
        "month_count": len(month_summaries),
        "assignment_count": len(all_assignment_rows),
        "artifacts": {
            "community_assignments": assignments_path,
            "community_summary": summary_path,
            "community_evolution": evolution_path,
            "community_report": report_path,
            "similarity_matrices": similarity_paths,
            "community_checkpoint": checkpoint_path,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KNN + Leiden risk community discovery.")
    parser.add_argument("--input", required=True, help="risk_scores.csv or risk_vectors.jsonl path")
    parser.add_argument("--input-mode", choices=["score", "repr"], default="score", help="Input feature mode")
    parser.add_argument("--output-dir", default="方案一/output/community", help="Output directory")
    parser.add_argument("--k", type=int, default=3, help="KNN K")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution")
    parser.add_argument("--checkpoint-path", default=None, help="Optional path for community stage checkpoint JSON")
    parser.add_argument("--progress-log-every", type=int, default=1, help="Log community stage progress every N months")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume month-level community processing from checkpoint and per-month artifacts when available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    summary = run_community_pipeline(
        input_path=args.input,
        input_mode=args.input_mode,
        output_dir=args.output_dir,
        k=args.k,
        resolution=args.resolution,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
        progress_log_every=args.progress_log_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
