import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


COMPANY_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "股份",
    "公司",
    "集团",
)


def normalize_company_name(name: Any) -> str:
    text = str(name or "").strip()
    text = re.sub(r"[\s\u3000]+", "", text)
    text = text.replace("（", "(").replace("）", ")")
    changed = True
    while changed:
        changed = False
        for suffix in COMPANY_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                changed = True
                break
    return text.lower()


def canonicalize_company_name(name: Any) -> str:
    text = str(name or "").strip()
    text = re.sub(r"[\s\u3000]+", "", text)
    text = text.replace("（", "(").replace("）", ")")
    changed = True
    while changed:
        changed = False
        for suffix in COMPANY_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                changed = True
                break
    return text


def make_global_id(name: Any) -> str:
    normalized = normalize_company_name(name)
    if not normalized:
        return ""
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"cnco_{digest}"


def _iter_alias_values(row: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []
    for key in ("matched_alias", "name", "fullname", "short_name", "canonical_company_name"):
        value = str(row.get(key, "")).strip()
        if value and value not in aliases:
            aliases.append(value)
    raw_aliases = row.get("aliases", [])
    if isinstance(raw_aliases, list):
        for alias in raw_aliases:
            value = str(alias or "").strip()
            if value and value not in aliases:
                aliases.append(value)
    return aliases


def build_alias_map(company_rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    alias_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_alias(alias: Any, canonical_company_name: Any, global_id: Any, alias_type: str) -> None:
        canonical_name = canonicalize_company_name(canonical_company_name)
        normalized = normalize_company_name(alias)
        if not canonical_name or not normalized:
            return
        stable_global_id = str(global_id or "").strip() or make_global_id(canonical_name)
        payload = {
            "global_id": stable_global_id,
            "canonical_company_name": canonical_name,
            "matched_alias": str(alias or "").strip(),
            "alias_type": alias_type,
        }
        if payload not in alias_map[normalized]:
            alias_map[normalized].append(payload)

    for row in company_rows or []:
        canonical_name = (
            row.get("canonical_company_name")
            or row.get("canonical_name")
            or row.get("name")
            or row.get("fullname")
            or ""
        )
        canonical_name = canonicalize_company_name(canonical_name)
        if not canonical_name:
            continue
        stable_global_id = str(row.get("global_id") or "").strip() or make_global_id(canonical_name)
        add_alias(canonical_name, canonical_name, stable_global_id, "canonical_name")
        for alias in _iter_alias_values(row):
            add_alias(alias, canonical_name, stable_global_id, "company_master")

    return dict(alias_map)


def save_alias_map(path: str, alias_map: Dict[str, List[Dict[str, Any]]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(alias_map, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_alias_map(path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def build_record_alignment_context(
    company_mentions: Iterable[Any],
    alias_map: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    candidate_companies: Dict[str, Dict[str, Any]] = {}
    mention_candidates: List[Dict[str, Any]] = []

    for mention in company_mentions or []:
        mention_text = str(mention or "").strip()
        normalized = normalize_company_name(mention_text)
        matches = list(alias_map.get(normalized, []))
        mention_candidates.append(
            {
                "mention": mention_text,
                "normalized_mention": normalized,
                "candidates": matches,
            }
        )
        for match in matches:
            global_id = str(match.get("global_id", "")).strip()
            if not global_id:
                continue
            bucket = candidate_companies.setdefault(
                global_id,
                {
                    "global_id": global_id,
                    "canonical_company_name": match.get("canonical_company_name", ""),
                    "matched_aliases": [],
                },
            )
            alias = str(match.get("matched_alias", "")).strip()
            if alias and alias not in bucket["matched_aliases"]:
                bucket["matched_aliases"].append(alias)

    return {
        "mention_candidates": mention_candidates,
        "candidate_companies": list(candidate_companies.values()),
    }


def extract_company_mentions_from_text(
    title: Any,
    text: Any,
    alias_map: Dict[str, List[Dict[str, Any]]],
    max_mentions: int = 20,
) -> List[str]:
    haystack = f"{title or ''}\n{text or ''}"
    mentions: List[str] = []
    for alias_entries in alias_map.values():
        for entry in alias_entries:
            alias = str(entry.get("matched_alias", "")).strip()
            if not alias or alias in mentions:
                continue
            if alias in haystack:
                mentions.append(alias)
                if len(mentions) >= max_mentions:
                    return mentions
    return mentions


def _unique_match_for_name(company_name: Any, alias_map: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    normalized = normalize_company_name(company_name)
    deduped: Dict[str, Dict[str, Any]] = {}
    for item in alias_map.get(normalized, []):
        global_id = str(item.get("global_id", "")).strip()
        if global_id and global_id not in deduped:
            deduped[global_id] = item
    return list(deduped.values())


def resolve_company_identity(
    company_name: Any,
    matched_alias: Any = "",
    alias_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    global_id: Any = "",
    canonical_company_name: Any = "",
    alignment_status: Any = "",
) -> Dict[str, str]:
    alias_map = alias_map or {}
    explicit_global_id = str(global_id or "").strip()
    explicit_canonical_name = canonicalize_company_name(canonical_company_name)
    explicit_alias = str(matched_alias or "").strip()
    explicit_status = str(alignment_status or "").strip().lower()

    if explicit_global_id and explicit_canonical_name:
        return {
            "global_id": explicit_global_id,
            "canonical_company_name": explicit_canonical_name,
            "matched_alias": explicit_alias or explicit_canonical_name,
            "alignment_status": "resolved",
        }

    for lookup_value in (explicit_alias, explicit_canonical_name, company_name):
        if not lookup_value:
            continue
        matches = _unique_match_for_name(lookup_value, alias_map)
        if len(matches) == 1:
            match = matches[0]
            return {
                "global_id": str(match.get("global_id", "")).strip(),
                "canonical_company_name": str(match.get("canonical_company_name", "")).strip(),
                "matched_alias": explicit_alias or str(lookup_value).strip() or str(match.get("matched_alias", "")).strip(),
                "alignment_status": "resolved",
            }
        if len(matches) > 1:
            return {
                "global_id": "",
                "canonical_company_name": "",
                "matched_alias": explicit_alias or str(lookup_value).strip(),
                "alignment_status": "ambiguous",
            }

    fallback_name = explicit_canonical_name or canonicalize_company_name(company_name)
    if fallback_name:
        return {
            "global_id": make_global_id(fallback_name),
            "canonical_company_name": fallback_name,
            "matched_alias": explicit_alias or str(company_name or fallback_name).strip(),
            "alignment_status": "resolved",
        }

    return {
        "global_id": "",
        "canonical_company_name": "",
        "matched_alias": explicit_alias or str(company_name or "").strip(),
        "alignment_status": explicit_status if explicit_status in {"ambiguous", "unresolved"} else "unresolved",
    }


def finalize_alignment_fields(
    event: Dict[str, Any],
    company_mentions: Iterable[Any],
    alias_map: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    finalized = dict(event)
    arguments = finalized.get("arguments", {}) if isinstance(finalized.get("arguments"), dict) else {}
    company_name = str(arguments.get("公司名称", "")).strip()
    if not company_name:
        mentions = [str(item or "").strip() for item in company_mentions or [] if str(item or "").strip()]
        company_name = mentions[0] if mentions else ""

    finalized.update(
        resolve_company_identity(
            company_name=company_name,
            matched_alias=finalized.get("matched_alias", ""),
            alias_map=alias_map,
            global_id=finalized.get("global_id", ""),
            canonical_company_name=finalized.get("canonical_company_name", ""),
            alignment_status=finalized.get("alignment_status", ""),
        )
    )
    return finalized


def _load_company_rows(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("companies", [])
    return [item for item in payload if isinstance(item, dict)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local company alias map from JSON/JSONL.")
    parser.add_argument("--input", required=True, help="Company master JSON or JSONL path")
    parser.add_argument("--output", required=True, help="Alias map JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alias_map = build_alias_map(_load_company_rows(args.input))
    save_alias_map(args.output, alias_map)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "alias_count": len(alias_map),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
