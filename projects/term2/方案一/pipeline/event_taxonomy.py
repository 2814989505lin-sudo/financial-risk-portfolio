import re
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_RISK_KEYWORDS = {
    "credit_risk": ["债务", "违约", "评级", "破产", "清算", "流动性", "重组", "逾期"],
    "fraud_risk": ["造假", "虚假", "舞弊", "占用", "挪用", "贿赂", "诈骗", "骗贷"],
    "operational_risk": ["停工", "停产", "事故", "故障", "攻击", "泄露", "处罚", "中断"],
    "market_risk": ["股价", "利率", "汇率", "价格", "利差", "流动性", "浮亏", "景气度"],
}

SPLIT_PATTERN = re.compile(r"[\\/、,，\s]+")
GENERIC_PREFIXES = ("重大",)
GENERIC_SUFFIXES = ("事件",)


def iter_event_definitions(schema: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for taxonomy, polarity_groups in (schema.get("event_taxonomy") or {}).items():
        for group_name, event_map in (polarity_groups or {}).items():
            event_polarity = "positive" if group_name == "positive_events" else "negative"
            for event_type, event_def in (event_map or {}).items():
                yield {
                    "event_type": event_type,
                    "taxonomy": taxonomy,
                    "event_polarity": event_polarity,
                    "arguments": list((event_def or {}).get("arguments", [])),
                }


def schema_for_prompt(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        item["event_type"]: {
            "taxonomy": item["taxonomy"],
            "event_polarity": item["event_polarity"],
            "arguments": item["arguments"],
        }
        for item in iter_event_definitions(schema)
    }


def flatten_event_labels(schema: Dict[str, Any]) -> List[str]:
    return [item["event_type"] for item in iter_event_definitions(schema)]


def derive_taxonomy_polarity_pairs(schema: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for item in iter_event_definitions(schema):
        pair = (item["taxonomy"], item["event_polarity"])
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def derive_article_labels(events: List[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    for event in events:
        event_type = str(event.get("event_type", "")).strip()
        if event_type and event_type not in labels:
            labels.append(event_type)
    return labels


def _keyword_variants(event_type: str) -> List[str]:
    pieces = [event_type]
    pieces.extend(part for part in SPLIT_PATTERN.split(event_type) if part)
    variants: List[str] = []
    for piece in pieces:
        text = piece.strip()
        if not text:
            continue
        variants.append(text)
        for prefix in GENERIC_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                variants.append(text[len(prefix) :])
        for suffix in GENERIC_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix):
                variants.append(text[: -len(suffix)])
    deduped: List[str] = []
    for item in variants:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def build_filter_terms(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    terms: Dict[str, List[str]] = {}
    for item in iter_event_definitions(schema):
        event_type = item["event_type"]
        taxonomy = item["taxonomy"]
        merged = _keyword_variants(event_type) + list(DEFAULT_RISK_KEYWORDS.get(taxonomy, []))
        deduped: List[str] = []
        for term in merged:
            if term and term not in deduped:
                deduped.append(term)
        terms[event_type] = deduped
    return terms
