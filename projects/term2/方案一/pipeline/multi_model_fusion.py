import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple
LEGAL_SUFFIXES = [
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "集团股份有限公司",
    "集团有限公司",
    "控股股份有限公司",
    "控股有限公司",
]


def normalize_company_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s\-\(\)（）,，.。·]", "", text)
    for suffix in LEGAL_SUFFIXES:
        normalized_suffix = suffix.lower()
        if text.endswith(normalized_suffix):
            text = text[: -len(normalized_suffix)]
            break
    return text


def event_signature(event: Dict[str, Any]) -> Tuple[str, str, str, str]:
    arguments = event.get("arguments", {}) if isinstance(event.get("arguments"), dict) else {}
    company_name = (
        arguments.get("公司名称")
        or event.get("canonical_company_name")
        or event.get("matched_alias")
        or ""
    )
    return (
        str(event.get("event_type", "")).strip(),
        str(event.get("taxonomy", "")).strip(),
        str(event.get("event_polarity", "")).strip(),
        normalize_company_name(company_name),
    )


def normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


def choose_majority_value(values: List[Any], preferred_model_order: List[str]) -> Any:
    non_empty_pairs = [(model_name, value) for model_name, value in values if str(value or "").strip()]
    if not non_empty_pairs:
        return ""
    counts = Counter(value for _, value in non_empty_pairs)
    top_count = max(counts.values())
    winners = {value for value, count in counts.items() if count == top_count}
    for model_name in preferred_model_order:
        for candidate_model, candidate_value in non_empty_pairs:
            if candidate_model == model_name and candidate_value in winners:
                return candidate_value
    return non_empty_pairs[0][1]


def merge_arguments(
    grouped_events: List[Tuple[str, Dict[str, Any]]],
    preferred_model_order: List[str],
) -> Dict[str, Any]:
    argument_values: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
    for model_name, event in grouped_events:
        arguments = event.get("arguments", {}) if isinstance(event.get("arguments"), dict) else {}
        for key, value in arguments.items():
            argument_values[str(key)].append((model_name, value))

    merged: Dict[str, Any] = {}
    for key, values in argument_values.items():
        merged[key] = choose_majority_value(values, preferred_model_order)
    return merged


def merge_evidence(grouped_events: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for _model_name, event in grouped_events:
        evidence_list = event.get("evidence", []) if isinstance(event.get("evidence"), list) else []
        for item in evidence_list:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return merged


def choose_field(
    grouped_events: List[Tuple[str, Dict[str, Any]]],
    field_name: str,
    preferred_model_order: List[str],
) -> str:
    values = [(model_name, str(event.get(field_name, "")).strip()) for model_name, event in grouped_events]
    chosen = choose_majority_value(values, preferred_model_order)
    return str(chosen or "").strip()


def fuse_events(
    model_results: List[Tuple[str, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    preferred_model_order = [model_name for model_name, _events in model_results]

    for model_name, events in model_results:
        for event in events:
            if not isinstance(event, dict):
                continue
            signature = event_signature(event)
            grouped[signature].append((model_name, event))

    fused: List[Dict[str, Any]] = []
    for grouped_events in grouped.values():
        first_event = grouped_events[0][1]
        annotation_models = [model_name for model_name, _event in grouped_events]
        per_model_confidence = {
            model_name: normalize_confidence(event.get("confidence"))
            for model_name, event in grouped_events
        }
        confidence_values = list(per_model_confidence.values())
        model_agreement = len(annotation_models)
        final_confidence = sum(confidence_values) / max(1, len(confidence_values))

        fused_event = {
            "event_type": str(first_event.get("event_type", "")).strip(),
            "taxonomy": str(first_event.get("taxonomy", "")).strip(),
            "event_polarity": str(first_event.get("event_polarity", "")).strip(),
            "arguments": merge_arguments(grouped_events, preferred_model_order),
            "evidence": merge_evidence(grouped_events),
            "confidence": round(final_confidence, 6),
            "global_id": choose_field(grouped_events, "global_id", preferred_model_order),
            "canonical_company_name": choose_field(grouped_events, "canonical_company_name", preferred_model_order),
            "matched_alias": choose_field(grouped_events, "matched_alias", preferred_model_order),
            "alignment_status": choose_field(grouped_events, "alignment_status", preferred_model_order),
            "annotation_models": annotation_models,
            "model_agreement": model_agreement,
            "per_model_confidence": per_model_confidence,
            "label_source": "silver",
        }
        fused.append(fused_event)

    fused.sort(
        key=lambda event: (
            str(event.get("taxonomy", "")),
            str(event.get("event_polarity", "")),
            str(event.get("event_type", "")),
            str((event.get("arguments") or {}).get("公司名称", "")),
        )
    )
    return fused
