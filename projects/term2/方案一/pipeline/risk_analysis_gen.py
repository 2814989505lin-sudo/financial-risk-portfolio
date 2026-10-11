import argparse
import json
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

from aggregate_events import load_jsonl, write_jsonl
from llm_extract_openai import chat_complete_with_retry, get_client_for_model


def render_group_events(group: Dict[str, Any]) -> str:
    rendered: List[str] = []
    for idx, event in enumerate(group.get("events", []) or [], start=1):
        arguments = event.get("arguments", {}) if isinstance(event.get("arguments"), dict) else {}
        evidence = event.get("evidence", []) if isinstance(event.get("evidence"), list) else []
        argument_text = "；".join(
            f"{key}:{value}"
            for key, value in arguments.items()
            if str(value or "").strip()
        )
        evidence_text = " ".join(str(item).strip() for item in evidence if str(item).strip())
        parts = [
            f"{idx}. 事件类型：{event.get('event_type', '')}",
            f"极性：{event.get('event_polarity', '')}",
            f"置信度：{event.get('confidence', 0.0)}",
            f"标题：{event.get('title', '')}",
            f"发布时间：{event.get('publish_time', '')}",
        ]
        if argument_text:
            parts.append(f"事件要素：{argument_text}")
        if evidence_text:
            parts.append(f"证据：{evidence_text}")
        rendered.append("\n".join(parts))
    return "\n\n".join(rendered)


def render_prompt(template: str, group: Dict[str, Any]) -> str:
    prompt = template
    prompt = prompt.replace("{{GLOBAL_ID}}", str(group.get("global_id", "")))
    prompt = prompt.replace("{{CANONICAL_COMPANY_NAME}}", str(group.get("canonical_company_name", "")))
    prompt = prompt.replace("{{FEATURE_MONTH}}", str(group.get("feature_month", "")))
    prompt = prompt.replace("{{RISK_DIM}}", str(group.get("risk_dim", "")))
    prompt = prompt.replace("{{EVENT_COUNT}}", str(group.get("event_count", 0)))
    prompt = prompt.replace("{{NEGATIVE_EVENT_COUNT}}", str(group.get("negative_event_count", 0)))
    prompt = prompt.replace("{{POSITIVE_EVENT_COUNT}}", str(group.get("positive_event_count", 0)))
    prompt = prompt.replace("{{EVENT_SUMMARY_BLOCK}}", render_group_events(group))
    return prompt


def build_risk_analysis_messages(template: str, group: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = render_prompt(template=template, group=group)
    return [
        {
            "role": "user",
            "content": prompt,
        }
    ]


def generate_risk_analysis(
    groups: Iterable[Dict[str, Any]],
    prompt_template: str,
    model: str,
) -> List[Dict[str, Any]]:
    client = get_client_for_model(model)
    outputs: List[Dict[str, Any]] = []
    for group in tqdm(list(groups), desc="RiskAnalysis", ncols=100):
        messages = build_risk_analysis_messages(prompt_template, group)
        analysis_text = chat_complete_with_retry(
            client=client,
            model=model,
            messages=messages,
            temperature=0.0,
        ).strip()
        outputs.append(
            {
                "group_id": group.get("group_id", ""),
                "global_id": group.get("global_id", ""),
                "canonical_company_name": group.get("canonical_company_name", ""),
                "feature_month": group.get("feature_month", ""),
                "risk_dim": group.get("risk_dim", ""),
                "event_count": group.get("event_count", 0),
                "analysis_model": model,
                "analysis_text": analysis_text,
            }
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate risk analysis text for aggregated risk groups.")
    parser.add_argument("--input", required=True, help="Input aggregated_events.jsonl path")
    parser.add_argument("--prompt", required=True, help="Risk analysis prompt template path")
    parser.add_argument("--output", required=True, help="Output risk_analysis.jsonl path")
    parser.add_argument("--model", default="kimi-k2.5", help="Chat model name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = load_jsonl(args.input)
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    outputs = generate_risk_analysis(
        groups=groups,
        prompt_template=prompt_template,
        model=args.model,
    )
    write_jsonl(args.output, outputs)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "group_count": len(outputs),
                "model": args.model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
