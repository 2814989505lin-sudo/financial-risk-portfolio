import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from global_id_alignment import load_alias_map, resolve_company_identity
from manual_label_pack import build_manual_label_rows
from prepare_inputs import normalize_text


def make_article_id(row: Dict[str, Any], idx: int) -> str:
    payload = "|".join(
        [
            str(row.get("url", "")),
            str(row.get("title", "")),
            str(row.get("publish_time", "")),
            str(row.get("source", "")),
        ]
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    return f"news_{idx:06d}_{digest}"


def title_hash(title: Any) -> str:
    normalized = str(title or "").strip().lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def text_hash(text: Any) -> str:
    normalized = normalize_text(text or "")
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def chinese_ratio(text: Any) -> float:
    raw = str(text or "")
    if not raw:
        return 0.0
    chinese_chars = sum(1 for ch in raw if "\u4e00" <= ch <= "\u9fff")
    return chinese_chars / max(1, len(raw))


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_csv_value(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() == "nan":
        return ""
    return text


def infer_dataset_source(path: str) -> str:
    name = Path(path).name.lower()
    if "flash_news" in name:
        return "flash_news"
    if "deep_mining" in name:
        return "deep_mining"
    return Path(path).stem


def load_risk_csv_rows(input_paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for input_path in input_paths:
        source_name = infer_dataset_source(input_path)
        with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                company_name = _clean_csv_value(raw_row.get("相关公司"))
                title = _clean_csv_value(raw_row.get("标题")) or _clean_csv_value(raw_row.get("title"))
                body = normalize_text(raw_row.get("正文", "") or raw_row.get("content", ""))
                publish_time = _clean_csv_value(raw_row.get("时间")) or _clean_csv_value(raw_row.get("pub_time"))
                source = _clean_csv_value(raw_row.get("src")) or source_name
                if not body:
                    continue
                rows.append(
                    {
                        "title": title,
                        "content": body,
                        "pub_time": publish_time,
                        "src": source,
                        "url": "",
                        "company_mentions": [company_name] if company_name else [],
                        "source_file": input_path,
                    }
                )
    if not rows:
        raise FileNotFoundError("No valid rows loaded from input CSV files.")
    return rows


def normalize_news_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    title = str(row.get("title", "")).strip()
    text = normalize_text(row.get("content", row.get("text", "")))
    publish_time = str(row.get("pub_time", row.get("publish_time", ""))).strip()
    source = str(row.get("src", "") or row.get("source", "")).strip() or "未知来源"
    normalized = {
        "article_id": make_article_id(
            {
                "title": title,
                "publish_time": publish_time,
                "source": source,
                "url": row.get("url", ""),
            },
            idx,
        ),
        "url": str(row.get("url", "")).strip(),
        "title": title,
        "source": source,
        "publish_time": publish_time,
        "text": text,
        "company_mentions": list(row.get("company_mentions", []) or []),
        "split": "",
        "label_source": "",
    }
    if "news_id" in row:
        normalized["news_id"] = str(row.get("news_id", "")).strip()
    if "source_file" in row:
        normalized["source_file"] = str(row.get("source_file", "")).strip()
    normalized["title_hash"] = title_hash(title)
    normalized["text_hash"] = text_hash(text)
    return normalized


def dedupe_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        primary_company = str((row.get("company_mentions") or [""])[0]).strip()
        key = (
            str(row.get("url", "")).strip(),
            str(row.get("title_hash", "")).strip(),
            str(row.get("publish_time", "")).strip(),
            primary_company,
            str(row.get("text_hash", "")).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped


def filter_rows(rows: Sequence[Dict[str, Any]], min_chars: int = 80, min_chinese_ratio: float = 0.2) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if len(text) < min_chars:
            continue
        if chinese_ratio(text) < min_chinese_ratio:
            continue
        kept.append(dict(row))
    return kept


def _bucket_key(row: Dict[str, Any]) -> str:
    labels = row.get("candidate_labels") or []
    return labels[0] if labels else "__all__"


def stratified_sample(rows: Sequence[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_bucket_key(row), []).append(dict(row))
    rng = random.Random(seed)
    for bucket_rows in groups.values():
        rng.shuffle(bucket_rows)

    selected: List[Dict[str, Any]] = []
    buckets = sorted(groups.keys())
    while buckets and len(selected) < count:
        next_buckets: List[str] = []
        for bucket in buckets:
            bucket_rows = groups[bucket]
            if not bucket_rows:
                continue
            selected.append(bucket_rows.pop())
            if bucket_rows:
                next_buckets.append(bucket)
            if len(selected) >= count:
                break
        buckets = next_buckets
    return selected


def mark_rows_all_in_scope(
    rows: Sequence[Dict[str, Any]],
    alias_map: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        company_mentions = list(normalized.get("company_mentions", []) or [])
        primary_company = str(company_mentions[0]).strip() if company_mentions else ""
        resolved = resolve_company_identity(
            company_name=primary_company,
            matched_alias=primary_company,
            alias_map=alias_map,
        )
        normalized["is_in_scope"] = True
        normalized["candidate_labels"] = []
        normalized["mentioned_companies"] = company_mentions
        normalized["canonical_company_name"] = str(resolved.get("canonical_company_name", "")).strip()
        normalized["matched_alias"] = str(resolved.get("matched_alias", "")).strip()
        normalized["global_id"] = str(resolved.get("global_id", "")).strip()
        normalized["alignment_status"] = str(resolved.get("alignment_status", "")).strip() or "unresolved"
        normalized["screen_confidence"] = 1.0
        normalized["screen_reason"] = "all_balanced_rows_kept"
        outputs.append(normalized)
    return outputs


def assign_manual_review(
    rows: Sequence[Dict[str, Any]],
    manual_gold_total: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    assigned = stratified_sample(rows, manual_gold_total, seed)
    assigned_ids = {row["article_id"] for row in assigned}
    remaining_rows = [dict(row) for row in rows if row["article_id"] not in assigned_ids]

    for row in assigned:
        row["split"] = "manual_review"
        row["label_source"] = "manual_review"
    for row in remaining_rows:
        row["split"] = "silver_pool"
        row["label_source"] = "silver"

    return assigned, remaining_rows


def select_silver_pool(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = [dict(row) for row in rows]
    selected.sort(key=lambda row: str(row.get("article_id", "")))
    return selected


def build_corpus(
    rows: Sequence[Dict[str, Any]],
    manual_gold_total: int,
    seed: int,
    alias_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    print(json.dumps({"stage": "normalize", "status": "start", "rows": len(rows)}, ensure_ascii=False), flush=True)
    normalized = [normalize_news_row(row, idx + 1) for idx, row in enumerate(rows)]
    print(json.dumps({"stage": "normalize", "status": "done", "rows": len(normalized)}, ensure_ascii=False), flush=True)

    print(json.dumps({"stage": "dedupe", "status": "start", "rows": len(normalized)}, ensure_ascii=False), flush=True)
    deduped = dedupe_rows(normalized)
    print(json.dumps({"stage": "dedupe", "status": "done", "rows": len(deduped)}, ensure_ascii=False), flush=True)

    print(json.dumps({"stage": "filter", "status": "start", "rows": len(deduped)}, ensure_ascii=False), flush=True)
    filtered = filter_rows(deduped)
    print(json.dumps({"stage": "filter", "status": "done", "rows": len(filtered)}, ensure_ascii=False), flush=True)

    print(json.dumps({"stage": "corpus_select", "status": "pass_through", "rows": len(filtered)}, ensure_ascii=False), flush=True)
    print(json.dumps({"stage": "scope_assign", "status": "start", "rows": len(filtered)}, ensure_ascii=False), flush=True)
    scoped_rows = mark_rows_all_in_scope(filtered, alias_map or {})
    print(json.dumps({"stage": "scope_assign", "status": "done", "rows": len(scoped_rows)}, ensure_ascii=False), flush=True)

    print(json.dumps({"stage": "manual_review", "status": "start", "rows": len(scoped_rows)}, ensure_ascii=False), flush=True)
    manual_rows, silver_rows = assign_manual_review(
        scoped_rows,
        manual_gold_total=manual_gold_total,
        seed=seed,
    )
    print(
        json.dumps(
            {
                "stage": "manual_review",
                "status": "done",
                "manual_rows": len(manual_rows),
                "silver_candidate_rows": len(silver_rows),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    print(json.dumps({"stage": "silver_pool", "status": "start", "rows": len(silver_rows)}, ensure_ascii=False), flush=True)
    silver_rows = select_silver_pool(silver_rows)
    print(json.dumps({"stage": "silver_pool", "status": "done", "rows": len(silver_rows)}, ensure_ascii=False), flush=True)

    return {
        "manual_review": build_manual_label_rows(manual_rows),
        "silver_pool": silver_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manual review and silver pool files from normalized news CSV files.")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--alias-map", default=None, help="Optional local alias map JSON path")
    parser.add_argument(
        "--input-csvs",
        nargs="+",
        required=True,
        help="One or more normalized news CSV files, e.g. sina_major_news_last_2y_balanced_3k.csv",
    )
    parser.add_argument("--manual-gold-total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_risk_csv_rows(args.input_csvs)
    alias_map = load_alias_map(args.alias_map) if args.alias_map else {}
    corpus = build_corpus(
        rows=rows,
        manual_gold_total=args.manual_gold_total,
        seed=args.seed,
        alias_map=alias_map,
    )
    outputs = {
        "manual_review": os.path.join(args.output_dir, "manual_review.jsonl"),
        "silver_pool": os.path.join(args.output_dir, "silver_pool.jsonl"),
    }
    write_jsonl(outputs["manual_review"], corpus["manual_review"])
    write_jsonl(outputs["silver_pool"], corpus["silver_pool"])

    print(
        json.dumps(
            {
                "outputs": outputs,
                "counts": {key: len(value) for key, value in corpus.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
