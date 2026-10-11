import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataframe(summary: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in summary.get("summaries", []):
        rows.append(
            {
                "config_name": item["config_name"],
                "hidden_dims": str(item.get("hidden_dims", [])),
                "param_count": item.get("param_count"),
                "val_macro_auprc_mean": item.get("val_metrics_macro_auprc_mean"),
                "test_macro_auprc_mean": item.get("test_metrics_macro_auprc_mean"),
                "train_macro_auprc_mean": item.get("train_metrics_macro_auprc_mean"),
                "val_macro_auroc_mean": item.get("val_metrics_macro_auroc_mean"),
                "test_macro_auroc_mean": item.get("test_metrics_macro_auroc_mean"),
                "seed_count": item.get("seed_count"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("config_name").reset_index(drop=True)
    return df


def save_bar_chart(df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    x = range(len(df))
    width = 0.36
    val_scores = df["val_macro_auprc_mean"].tolist()
    test_scores = df["test_macro_auprc_mean"].tolist()

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        val_scores,
        width=width,
        label="Validation macro-AUPRC",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        test_scores,
        width=width,
        label="Test macro-AUPRC",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["config_name"].tolist())
    ax.set_ylabel("macro-AUPRC")
    ax.set_xlabel("MLP Configuration")
    ax.set_title("MLP Ablation Summary: Validation vs Test macro-AUPRC")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.002,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MLP ablation summary.")
    parser.add_argument(
        "--summary-json",
        default="方案一/output/risk_scoring/mlp_ablation/mlp_ablation_summary.json",
        help="Path to mlp_ablation_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="方案一/output/risk_scoring/mlp_ablation",
        help="Directory to save the generated csv and figure",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary_json)
    df = build_dataframe(summary)
    if df.empty:
        raise ValueError("No ablation summaries found in the summary json.")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "mlp_ablation_summary_table.csv")
    png_path = os.path.join(args.output_dir, "mlp_ablation_summary_chart.png")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_bar_chart(df, png_path)

    print(f"saved: {csv_path}")
    print(f"saved: {png_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
