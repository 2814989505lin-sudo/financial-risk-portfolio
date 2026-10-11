import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_records(rows: List[Dict[str, Any]], sample_rate: float, seed: int = 42) -> List[Dict[str, Any]]:
    if not rows:
        return []
    n = max(1, int(round(len(rows) * sample_rate)))
    n = min(n, len(rows))
    rng = random.Random(seed)
    sampled = rng.sample(rows, n)

    out: List[Dict[str, Any]] = []
    for row in sampled:
        out.append(
            {
                "article_id": row.get("article_id", ""),
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "pred": row,
                "review": {
                    "error_type": "",
                    "comments": "",
                    "is_accepted": "",
                },
            }
        )
    return out


def render_report(rows: List[Dict[str, Any]], sample_rows: List[Dict[str, Any]]) -> str:
    total = len(rows)
    non_empty = sum(1 for r in rows if r.get("events"))
    event_counter: Counter = Counter()
    for row in rows:
        for event in row.get("events", []):
            event_counter[event.get("event_type", "")] += 1

    top_events = "\n".join([f"- {k}: {v}" for k, v in event_counter.most_common(20)]) or "- 无"

    report = f"""# Review Report

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 运行概览
- 总文章数: {total}
- 非空事件文章数: {non_empty}
- 抽样复核条数: {len(sample_rows)}

## 事件类型分布Top
{top_events}

## 人工复核标注建议
- `error_type` 建议枚举: 误抽 / 漏抽 / 字段错位 / 其他
- `is_accepted` 建议枚举: yes / no

## 下一版Prompt改进建议（自动生成）
- 若误抽偏多: 增加 few-shot 示例中的负样本或边界样本。
- 若漏抽偏多: 增加同类事件的人工示例，并强化“同一文章可输出多个事件”。
- 若字段错位偏多: 在 prompt 明确字段含义，并补充字段级示例。
"""
    return report


def build_review_pack(
    input_path: str,
    sample_path: str,
    report_path: str,
    sample_rate: float,
    seed: int = 42,
) -> Dict[str, Any]:
    rows = read_jsonl(input_path)
    sample_rows = sample_records(rows, sample_rate=sample_rate, seed=seed)
    write_jsonl(sample_path, sample_rows)

    report = render_report(rows, sample_rows)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "total": len(rows),
        "sample_count": len(sample_rows),
        "sample_path": sample_path,
        "report_path": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate review sample and markdown report")
    parser.add_argument("--input", required=True, help="Input result JSONL")
    parser.add_argument("--sample-output", required=True, help="Sample JSONL output")
    parser.add_argument("--report-output", required=True, help="Markdown report output")
    parser.add_argument("--sample-rate", type=float, default=0.1, help="Sampling rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stat = build_review_pack(
        input_path=args.input,
        sample_path=args.sample_output,
        report_path=args.report_output,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )
    print(json.dumps(stat, ensure_ascii=False, indent=2))
