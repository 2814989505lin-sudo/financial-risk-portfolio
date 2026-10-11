import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from aggregate_events import load_jsonl, write_jsonl


RISK_DIMS = ["credit_risk", "fraud_risk", "operational_risk", "market_risk"]
SUPPORT_STATUS_ORDER = ["supported", "mixed", "contradicted", "no_external_evidence"]

QUERY_KEYWORDS = {
    ("credit_risk", "increase"): ["评级下调", "信用评级", "违约", "展期失败", "债务重组", "流动性危机"],
    ("credit_risk", "decrease"): ["展期成功", "融资支持", "债务偿还", "流动性改善"],
    ("fraud_risk", "increase"): ["立案调查", "财务造假", "信披违规", "欺诈发行"],
    ("fraud_risk", "decrease"): ["处罚落地", "整改完成", "结案"],
    ("operational_risk", "increase"): ["停产", "事故", "处罚", "合规违规", "大额亏损"],
    ("operational_risk", "decrease"): ["复工复产", "恢复生产", "经营改善"],
    ("market_risk", "increase"): ["股价大跌", "债券暴跌", "目标价下调", "波动加剧"],
    ("market_risk", "decrease"): ["股价反弹", "目标价上调", "市场情绪修复"],
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_risk_scores(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "global_id": str(row.get("global_id", "")).strip(),
                "canonical_company_name": str(row.get("canonical_company_name", "")).strip(),
                "feature_month": str(row.get("feature_month", "")).strip(),
                **{risk_dim: float(row.get(risk_dim, 0.0) or 0.0) for risk_dim in RISK_DIMS},
            }
        )
    return normalized


def load_external_evidence(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = load_jsonl(path)
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        event_date = str(row.get("event_date", "")).strip()
        event_month = str(row.get("event_month", "")).strip() or event_date[:7]
        normalized.append(
            {
                "global_id": str(row.get("global_id", "")).strip(),
                "canonical_company_name": str(row.get("canonical_company_name", "")).strip(),
                "event_date": event_date,
                "event_month": event_month,
                "risk_dim": str(row.get("risk_dim", "")).strip(),
                "polarity": str(row.get("polarity", "")).strip(),
                "evidence_type": str(row.get("evidence_type", "")).strip(),
                "source_name": str(row.get("source_name", "")).strip(),
                "source_title": str(row.get("source_title", "")).strip(),
                "source_url": str(row.get("source_url", "")).strip(),
                "evidence_summary": str(row.get("evidence_summary", "")).strip(),
            }
        )
    return normalized


def month_to_index(month: str) -> int:
    year, month_value = month.split("-")
    return int(year) * 12 + int(month_value)


def index_to_month(index: int) -> str:
    year = index // 12
    month_value = index % 12
    if month_value == 0:
        year -= 1
        month_value = 12
    return f"{year:04d}-{month_value:02d}"


def expand_month_window(start_month: str, window_months: int) -> List[str]:
    start_index = month_to_index(start_month)
    return [index_to_month(start_index + offset) for offset in range(window_months + 1)]


def load_internal_group_evidence(path: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = (
            str(row.get("global_id", "")).strip(),
            str(row.get("feature_month", "")).strip(),
            str(row.get("risk_dim", "")).strip(),
        )
        events = list(row.get("events", []) or [])
        grouped[key] = {
            "group_id": row.get("group_id", ""),
            "event_count": int(row.get("event_count", 0) or 0),
            "negative_event_count": int(row.get("negative_event_count", 0) or 0),
            "positive_event_count": int(row.get("positive_event_count", 0) or 0),
            "sample_titles": [str(event.get("title", "")).strip() for event in events[:3] if str(event.get("title", "")).strip()],
            "sample_event_types": [str(event.get("event_type", "")).strip() for event in events[:5] if str(event.get("event_type", "")).strip()],
            "sample_evidence": [
                str(item).strip()
                for event in events[:3]
                for item in list(event.get("evidence", []) or [])[:2]
                if str(item).strip()
            ][:5],
        }
    return grouped


def load_migrations(path: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    migrations: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in load_jsonl(path):
        from_month = str(row.get("from_month", "")).strip()
        to_month = str(row.get("to_month", "")).strip()
        for migration in list(row.get("company_migrations", []) or []):
            key = (str(migration.get("global_id", "")).strip(), from_month, to_month)
            migrations[key] = {
                "from_community_id": migration.get("from_community_id"),
                "to_community_id": migration.get("to_community_id"),
            }
    return migrations


def build_transition_cases(
    score_rows: Sequence[Dict[str, Any]],
    migrations: Dict[Tuple[str, str, str], Dict[str, Any]],
    internal_evidence: Dict[Tuple[str, str, str], Dict[str, Any]],
    *,
    min_score_delta: float,
    max_month_gap: int,
) -> List[Dict[str, Any]]:
    rows_by_gid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        rows_by_gid[str(row.get("global_id", "")).strip()].append(row)

    cases: List[Dict[str, Any]] = []
    for global_id, rows in rows_by_gid.items():
        rows = sorted(rows, key=lambda item: str(item.get("feature_month", "")))
        for left, right in zip(rows, rows[1:]):
            from_month = str(left["feature_month"])
            to_month = str(right["feature_month"])
            month_gap = month_to_index(to_month) - month_to_index(from_month)
            migration = migrations.get((global_id, from_month, to_month))
            if month_gap < 1:
                continue
            if month_gap > max_month_gap and migration is None:
                continue
            deltas = {risk_dim: float(right[risk_dim]) - float(left[risk_dim]) for risk_dim in RISK_DIMS}
            dominant_risk_dim = max(RISK_DIMS, key=lambda dim: abs(deltas[dim]))
            dominant_delta = float(deltas[dominant_risk_dim])
            dominant_change_direction = "increase" if dominant_delta >= 0 else "decrease"
            has_large_delta = abs(dominant_delta) >= min_score_delta
            if not migration and not has_large_delta:
                continue

            case = {
                "case_id": f"{global_id}__{from_month}__{to_month}",
                "global_id": global_id,
                "canonical_company_name": str(right.get("canonical_company_name", "")).strip(),
                "from_month": from_month,
                "to_month": to_month,
                "month_gap": month_gap,
                "has_migration": migration is not None,
                "from_community_id": migration.get("from_community_id") if migration else None,
                "to_community_id": migration.get("to_community_id") if migration else None,
                "score_from": {risk_dim: float(left[risk_dim]) for risk_dim in RISK_DIMS},
                "score_to": {risk_dim: float(right[risk_dim]) for risk_dim in RISK_DIMS},
                "delta_by_risk_dim": deltas,
                "dominant_risk_dim": dominant_risk_dim,
                "dominant_delta": dominant_delta,
                "dominant_change_direction": dominant_change_direction,
                "has_large_delta": has_large_delta,
                "internal_evidence_from": internal_evidence.get((global_id, from_month, dominant_risk_dim), {}),
                "internal_evidence_to": internal_evidence.get((global_id, to_month, dominant_risk_dim), {}),
            }
            cases.append(case)
    return sorted(
        cases,
        key=lambda item: (not bool(item["has_migration"]), -abs(float(item["dominant_delta"]))),
    )


def match_external_evidence(
    case: Dict[str, Any],
    evidence_rows: Sequence[Dict[str, Any]],
    *,
    post_window_months: int,
) -> List[Dict[str, Any]]:
    candidate_months = set(expand_month_window(str(case["to_month"]), post_window_months))
    global_id = str(case["global_id"])
    company_name = str(case["canonical_company_name"])
    risk_dim = str(case["dominant_risk_dim"])

    matched: List[Dict[str, Any]] = []
    for row in evidence_rows:
        same_company = (
            (row.get("global_id") and str(row.get("global_id")) == global_id)
            or (
                not row.get("global_id")
                and str(row.get("canonical_company_name", "")).strip()
                and str(row.get("canonical_company_name", "")).strip() == company_name
            )
        )
        if not same_company:
            continue
        if str(row.get("event_month", "")).strip() not in candidate_months:
            continue
        if str(row.get("risk_dim", "")).strip() != risk_dim:
            continue
        matched.append(row)
    return matched


def classify_support_status(case: Dict[str, Any], matched_evidence: Sequence[Dict[str, Any]]) -> str:
    if not matched_evidence:
        return "no_external_evidence"
    desired_polarity = "negative" if str(case["dominant_change_direction"]) == "increase" else "positive"
    polarities = {str(row.get("polarity", "")).strip() for row in matched_evidence}
    if desired_polarity in polarities and len(polarities) == 1:
        return "supported"
    if desired_polarity in polarities and len(polarities) > 1:
        return "mixed"
    if desired_polarity not in polarities and polarities:
        return "contradicted"
    return "no_external_evidence"


def build_query_text(case: Dict[str, Any]) -> str:
    keywords = QUERY_KEYWORDS.get((str(case["dominant_risk_dim"]), str(case["dominant_change_direction"])), [])
    return " ".join(
        [
            str(case["canonical_company_name"]),
            str(case["to_month"]),
            *keywords,
        ]
    ).strip()


def summarize_cases(validated_cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    status_counter = defaultdict(int)
    migration_case_count = 0
    large_delta_case_count = 0
    for case in validated_cases:
        status_counter[str(case.get("support_status", ""))] += 1
        if bool(case.get("has_migration")):
            migration_case_count += 1
        if bool(case.get("has_large_delta")):
            large_delta_case_count += 1
    cases_with_external = len(validated_cases) - status_counter["no_external_evidence"]
    return {
        "case_count": len(validated_cases),
        "migration_case_count": migration_case_count,
        "large_delta_case_count": large_delta_case_count,
        "supported_count": status_counter["supported"],
        "mixed_count": status_counter["mixed"],
        "contradicted_count": status_counter["contradicted"],
        "no_external_evidence_count": status_counter["no_external_evidence"],
        "support_rate_on_cases_with_external_evidence": (
            status_counter["supported"] / cases_with_external if cases_with_external else None
        ),
    }


def write_query_sheet(path: str, cases: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "canonical_company_name",
                "from_month",
                "to_month",
                "dominant_risk_dim",
                "dominant_change_direction",
                "query_text",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "canonical_company_name": case["canonical_company_name"],
                    "from_month": case["from_month"],
                    "to_month": case["to_month"],
                    "dominant_risk_dim": case["dominant_risk_dim"],
                    "dominant_change_direction": case["dominant_change_direction"],
                    "query_text": build_query_text(case),
                }
            )


def run_external_evidence_validation(
    *,
    risk_scores_path: str,
    community_evolution_path: str,
    aggregated_events_path: str,
    external_evidence_path: str,
    output_dir: str,
    min_score_delta: float,
    post_window_months: int,
    max_month_gap: int,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    score_rows = load_risk_scores(risk_scores_path)
    migrations = load_migrations(community_evolution_path)
    internal_evidence = load_internal_group_evidence(aggregated_events_path)
    external_rows = load_external_evidence(external_evidence_path)

    candidate_cases = build_transition_cases(
        score_rows,
        migrations,
        internal_evidence,
        min_score_delta=min_score_delta,
        max_month_gap=max_month_gap,
    )

    validated_cases: List[Dict[str, Any]] = []
    unresolved_cases: List[Dict[str, Any]] = []
    for case in candidate_cases:
        matched_evidence = match_external_evidence(case, external_rows, post_window_months=post_window_months)
        support_status = classify_support_status(case, matched_evidence)
        enriched_case = {
            **case,
            "support_status": support_status,
            "matched_external_evidence": matched_evidence,
        }
        validated_cases.append(enriched_case)
        if support_status == "no_external_evidence":
            unresolved_cases.append(enriched_case)

    candidate_cases_path = os.path.join(output_dir, "candidate_cases.jsonl")
    validated_cases_path = os.path.join(output_dir, "validated_cases.jsonl")
    query_sheet_path = os.path.join(output_dir, "evidence_query_sheet.csv")
    summary_path = os.path.join(output_dir, "external_validation_summary.json")

    write_jsonl(candidate_cases_path, candidate_cases)
    write_jsonl(validated_cases_path, validated_cases)
    write_query_sheet(query_sheet_path, unresolved_cases)

    summary = {
        "risk_scores_path": risk_scores_path,
        "community_evolution_path": community_evolution_path,
        "aggregated_events_path": aggregated_events_path,
        "external_evidence_path": external_evidence_path,
        "min_score_delta": min_score_delta,
        "post_window_months": post_window_months,
        "max_month_gap": max_month_gap,
        "metrics": summarize_cases(validated_cases),
        "artifacts": {
            "candidate_cases": candidate_cases_path,
            "validated_cases": validated_cases_path,
            "query_sheet": query_sheet_path,
            "summary": summary_path,
        },
    }
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate risk score changes and community migrations with external evidence.")
    parser.add_argument("--risk-scores", required=True, help="risk_scores.csv path")
    parser.add_argument("--community-evolution", required=True, help="community_evolution.jsonl path")
    parser.add_argument("--aggregated-events", required=True, help="aggregated_events.jsonl path")
    parser.add_argument("--external-evidence", required=True, help="External evidence JSONL/CSV path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--min-score-delta", type=float, default=0.2, help="Minimum dominant score delta to keep non-migration cases")
    parser.add_argument("--post-window-months", type=int, default=1, help="How many months after to_month to include when matching external evidence")
    parser.add_argument("--max-month-gap", type=int, default=1, help="Maximum allowed calendar month gap for non-migration score-change cases")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_external_evidence_validation(
        risk_scores_path=args.risk_scores,
        community_evolution_path=args.community_evolution,
        aggregated_events_path=args.aggregated_events,
        external_evidence_path=args.external_evidence,
        output_dir=args.output_dir,
        min_score_delta=args.min_score_delta,
        post_window_months=args.post_window_months,
        max_month_gap=args.max_month_gap,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
