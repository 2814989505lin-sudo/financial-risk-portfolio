import argparse
import hashlib
import html
import json
import os
import re
from typing import Any, Dict, List, Tuple


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_to_text(item) for item in value)
    if isinstance(value, dict):
        # Keep deterministic ordering for reproducibility.
        return "\n".join(f"{k}: {_to_text(v)}" for k, v in sorted(value.items()))
    return str(value)


def normalize_text(text: Any) -> str:
    text = _to_text(text)
    text = html.unescape(text or "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def make_article_id(url: str, idx: int) -> str:
    digest = hashlib.md5((url or f"row-{idx}").encode("utf-8")).hexdigest()[:10]
    return f"news_{idx:06d}_{digest}"


def build_records(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta_map: Dict[str, Dict[str, Any]] = {
        m.get("url", ""): m for m in raw.get("meta", []) if m.get("url")
    }

    records: List[Dict[str, Any]] = []
    for idx, content in enumerate(raw.get("contents", []), start=1):
        url = content.get("url", "")
        meta = meta_map.get(url, {})
        text = normalize_text(content.get("text", ""))

        record = {
            "article_id": make_article_id(url, idx),
            "url": url,
            "title": meta.get("title", ""),
            "source": meta.get("source", ""),
            "publish_time": meta.get("publish_time", ""),
            "company_mentions": meta.get("company_mentions", []) or [],
            "text": text,
        }
        records.append(record)

    return records


def write_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_inputs(input_path: str, output_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = build_records(raw)
    write_jsonl(records, output_path)
    return records


def read_jsonl(input_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_or_prepare_inputs(input_path: str, output_path: str | None = None) -> Tuple[List[Dict[str, Any]], str]:
    if input_path.endswith(".jsonl"):
        return read_jsonl(input_path), input_path

    default_output = output_path or os.path.join(os.path.dirname(input_path) or ".", "news_input.jsonl")
    return prepare_inputs(input_path, default_output), default_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare normalized news JSONL input.")
    parser.add_argument("--input", required=True, help="Path to original raw JSON input")
    parser.add_argument(
        "--output",
        default="data/news_input.jsonl",
        help="Output normalized JSONL path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = prepare_inputs(args.input, args.output)
    print(f"Prepared {len(rows)} records -> {args.output}")
