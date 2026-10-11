import argparse
import json
import os
from typing import Any, Dict, List

from aggregate_events import load_jsonl, write_jsonl
from run_community import run_community_pipeline


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def safe_slug(value: float) -> str:
    return str(value).replace(".", "p")


def summarize_community_outputs(summary_path: str, evolution_path: str) -> Dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as f:
        community_summary = json.load(f)
    evolution_rows = load_jsonl(evolution_path)

    months = list(community_summary.get("months", []) or [])
    month_count = len(months)
    avg_company_count = sum(float(row.get("company_count", 0) or 0) for row in months) / max(month_count, 1)
    avg_community_count = sum(float(row.get("community_count", 0) or 0) for row in months) / max(month_count, 1)
    modularities = [float(row.get("modularity", 0.0) or 0.0) for row in months]

    nonzero_match_count = 0
    nonzero_match_months = 0
    migration_months = 0
    migration_count = 0
    jaccards: List[float] = []
    for row in evolution_rows:
        month_has_nonzero = False
        for match in row.get("community_matches", []) or []:
            jaccard = float(match.get("jaccard", 0.0) or 0.0)
            if jaccard > 0:
                nonzero_match_count += 1
                jaccards.append(jaccard)
                month_has_nonzero = True
        if month_has_nonzero:
            nonzero_match_months += 1
        company_migrations = list(row.get("company_migrations", []) or [])
        if company_migrations:
            migration_months += 1
            migration_count += len(company_migrations)

    return {
        "month_count": month_count,
        "avg_company_count": avg_company_count,
        "avg_community_count": avg_community_count,
        "avg_modularity": sum(modularities) / len(modularities) if modularities else 0.0,
        "min_modularity": min(modularities) if modularities else 0.0,
        "max_modularity": max(modularities) if modularities else 0.0,
        "nonzero_match_count": nonzero_match_count,
        "nonzero_match_months": nonzero_match_months,
        "avg_nonzero_jaccard": (sum(jaccards) / len(jaccards)) if jaccards else 0.0,
        "migration_months": migration_months,
        "migration_count": migration_count,
    }


def run_sensitivity_grid(
    *,
    input_path: str,
    input_mode: str,
    output_dir: str,
    ks: List[int],
    resolutions: List[float],
    progress_log_every: int,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    runs: List[Dict[str, Any]] = []
    for k in ks:
        for resolution in resolutions:
            run_slug = f"k{k}_r{safe_slug(resolution)}"
            run_output_dir = os.path.join(output_dir, run_slug)
            run_summary = run_community_pipeline(
                input_path=input_path,
                input_mode=input_mode,
                output_dir=run_output_dir,
                k=k,
                resolution=resolution,
                checkpoint_path=os.path.join(run_output_dir, "community.checkpoint.json"),
                resume=False,
                progress_log_every=progress_log_every,
            )
            summarized = summarize_community_outputs(
                summary_path=run_summary["artifacts"]["community_summary"],
                evolution_path=run_summary["artifacts"]["community_evolution"],
            )
            runs.append(
                {
                    "k": k,
                    "resolution": resolution,
                    "output_dir": run_output_dir,
                    **summarized,
                    "artifacts": run_summary["artifacts"],
                }
            )

    runs_path = os.path.join(output_dir, "community_sensitivity_runs.jsonl")
    write_jsonl(runs_path, runs)
    best_by_modularity = max(runs, key=lambda row: float(row.get("avg_modularity", 0.0) or 0.0)) if runs else None
    best_by_evolution = max(runs, key=lambda row: int(row.get("nonzero_match_months", 0) or 0)) if runs else None
    summary = {
        "input": input_path,
        "input_mode": input_mode,
        "grid": {
            "ks": ks,
            "resolutions": resolutions,
            "run_count": len(runs),
        },
        "best_by_avg_modularity": best_by_modularity,
        "best_by_nonzero_match_months": best_by_evolution,
        "artifacts": {
            "runs": runs_path,
            "summary": os.path.join(output_dir, "community_sensitivity_summary.json"),
        },
    }
    write_json(summary["artifacts"]["summary"], summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KNN/Leiden sensitivity analysis.")
    parser.add_argument("--input", required=True, help="risk_scores.csv or risk_vectors.jsonl path")
    parser.add_argument("--input-mode", choices=["score", "repr"], default="score", help="Input feature mode")
    parser.add_argument("--output-dir", required=True, help="Output directory for sensitivity runs")
    parser.add_argument("--ks", default="3,5,10", help="Comma-separated KNN K values")
    parser.add_argument("--resolutions", default="0.5,1.0,1.5", help="Comma-separated Leiden resolutions")
    parser.add_argument("--progress-log-every", type=int, default=1000, help="Progress logging cadence per run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_sensitivity_grid(
        input_path=args.input,
        input_mode=args.input_mode,
        output_dir=args.output_dir,
        ks=parse_int_list(args.ks),
        resolutions=parse_float_list(args.resolutions),
        progress_log_every=args.progress_log_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
