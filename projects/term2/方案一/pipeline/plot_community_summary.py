import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_month_df(summary: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in summary.get("months", []):
        rows.append(
            {
                "feature_month": item["feature_month"],
                "company_count": item.get("company_count"),
                "community_count": item.get("community_count"),
                "modularity": item.get("modularity"),
                "zero_vector_count": item.get("zero_vector_count"),
                "node_count": (item.get("graph_stats") or {}).get("node_count"),
                "edge_count": (item.get("graph_stats") or {}).get("edge_count"),
                "effective_k": (item.get("graph_stats") or {}).get("effective_k"),
            }
        )
    return pd.DataFrame(rows)


def build_evolution_df(evolution_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in evolution_rows:
        matches = item.get("community_matches") or []
        valid_matches = [m for m in matches if m.get("to_community_id") is not None]
        jaccards = [float(m.get("jaccard") or 0.0) for m in valid_matches]
        rows.append(
            {
                "from_month": item["from_month"],
                "to_month": item["to_month"],
                "match_count_total": len(matches),
                "match_count_linked": len(valid_matches),
                "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
                "max_jaccard": float(np.max(jaccards)) if jaccards else 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_trend_chart(month_df: pd.DataFrame, output_path: str) -> None:
    x = np.arange(len(month_df))
    labels = month_df["feature_month"].tolist()

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    line1 = ax1.plot(
        x,
        month_df["community_count"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="Community count",
    )
    ax1.set_xlabel("Feature Month")
    ax1.set_ylabel("Community Count", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        x,
        month_df["modularity"],
        color="#d62728",
        marker="s",
        linewidth=2,
        label="Modularity",
    )
    ax2.set_ylabel("Modularity", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(bottom=0)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, loc="upper left")
    plt.title("Monthly Community Count and Modularity Trend")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_evolution_heatmap(
    month_df: pd.DataFrame,
    evolution_df: pd.DataFrame,
    output_path: str,
    value_col: str = "mean_jaccard",
) -> None:
    months = month_df["feature_month"].tolist()
    matrix = np.full((len(months), len(months)), np.nan)

    month_to_idx = {month: idx for idx, month in enumerate(months)}
    for _, row in evolution_df.iterrows():
        i = month_to_idx.get(row["from_month"])
        j = month_to_idx.get(row["to_month"])
        if i is None or j is None:
            continue
        matrix[i, j] = row[value_col]

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="white")
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=np.nanmax(np.nan_to_num(matrix)))

    ax.set_xticks(range(len(months)))
    ax.set_yticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_yticklabels(months)
    ax.set_xlabel("To Month")
    ax.set_ylabel("From Month")
    ax.set_title("Adjacent-Month Community Matching Heatmap (Mean Jaccard)")

    for i in range(len(months)):
        for j in range(len(months)):
            value = matrix[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="black")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Jaccard")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot community summary figures.")
    parser.add_argument(
        "--summary-json",
        default="方案一/output/community/community_summary.json",
        help="Path to community_summary.json",
    )
    parser.add_argument(
        "--evolution-jsonl",
        default="方案一/output/community/community_evolution.jsonl",
        help="Path to community_evolution.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="方案一/output/community",
        help="Directory to save generated tables and figures",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary = load_json(args.summary_json)
    evolution_rows = load_jsonl(args.evolution_jsonl)

    month_df = build_month_df(summary)
    evolution_df = build_evolution_df(evolution_rows)

    month_csv = os.path.join(args.output_dir, "community_month_trend_table.csv")
    evolution_csv = os.path.join(args.output_dir, "community_evolution_summary_table.csv")
    trend_png = os.path.join(args.output_dir, "community_month_trend.png")
    heatmap_png = os.path.join(args.output_dir, "community_evolution_heatmap.png")

    month_df.to_csv(month_csv, index=False, encoding="utf-8-sig")
    evolution_df.to_csv(evolution_csv, index=False, encoding="utf-8-sig")
    save_trend_chart(month_df, trend_png)
    save_evolution_heatmap(month_df, evolution_df, heatmap_png)

    print(f"saved: {month_csv}")
    print(f"saved: {evolution_csv}")
    print(f"saved: {trend_png}")
    print(f"saved: {heatmap_png}")
    print(month_df[['feature_month', 'community_count', 'modularity']].to_string(index=False))


if __name__ == "__main__":
    main()
