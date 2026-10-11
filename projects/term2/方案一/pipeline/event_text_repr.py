import argparse
import json
import math
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F
from openai import OpenAI
from tqdm import tqdm

from aggregate_events import load_jsonl, write_jsonl
from llm_extract_openai import get_client_for_model

MAX_EMBED_BATCH_SIZE = 10


def render_event_text(event: Dict[str, Any]) -> str:
    arguments = event.get("arguments", {}) if isinstance(event.get("arguments"), dict) else {}
    evidence = event.get("evidence", []) if isinstance(event.get("evidence"), list) else []
    argument_text = "；".join(
        f"{key}:{value}"
        for key, value in arguments.items()
        if str(value or "").strip()
    )
    evidence_text = " ".join(str(item).strip() for item in evidence if str(item).strip())
    parts = [
        f"标题：{event.get('title', '')}",
        f"事件类型：{event.get('event_type', '')}",
        f"风险维度：{event.get('taxonomy', '')}",
        f"事件极性：{event.get('event_polarity', '')}",
        f"发布时间：{event.get('publish_time', '')}",
    ]
    if argument_text:
        parts.append(f"事件要素：{argument_text}")
    if evidence_text:
        parts.append(f"证据：{evidence_text}")
    return "\n".join(parts)


def embed_texts(
    client: OpenAI,
    texts: List[str],
    model: str,
    batch_size: int = 64,
) -> List[List[float]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    effective_batch_size = min(batch_size, MAX_EMBED_BATCH_SIZE)
    embeddings: List[List[float]] = []
    for start in range(0, len(texts), effective_batch_size):
        batch = texts[start : start + effective_batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float",
        )
        embeddings.extend([list(item.embedding) for item in response.data])
    return embeddings


def parameter_free_multihead_self_attention(
    event_matrix: torch.Tensor,
    num_heads: int = 4,
) -> torch.Tensor:
    if event_matrix.ndim != 2:
        raise ValueError("event_matrix must be a 2D tensor")
    event_count, embed_dim = event_matrix.shape
    if embed_dim % num_heads != 0:
        raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
    if event_count == 0:
        raise ValueError("event_matrix must contain at least one event")

    head_dim = embed_dim // num_heads
    normalized = F.normalize(event_matrix, p=2, dim=-1)
    heads = normalized.view(event_count, num_heads, head_dim).permute(1, 0, 2)
    scores = torch.matmul(heads, heads.transpose(1, 2)) / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    attended = torch.matmul(weights, heads)
    return attended.permute(1, 0, 2).contiguous().view(event_count, embed_dim)


def build_group_text_representation(
    group: Dict[str, Any],
    client: OpenAI,
    embedding_model: str,
    num_heads: int = 4,
) -> Dict[str, Any]:
    events = list(group.get("events", []) or [])
    event_texts = [render_event_text(event) for event in events]
    embeddings = embed_texts(client=client, texts=event_texts, model=embedding_model)
    event_matrix = torch.tensor(embeddings, dtype=torch.float32)
    attended = parameter_free_multihead_self_attention(event_matrix, num_heads=num_heads)
    pooled = attended.mean(dim=0)
    return {
        "group_id": group.get("group_id", ""),
        "global_id": group.get("global_id", ""),
        "canonical_company_name": group.get("canonical_company_name", ""),
        "feature_month": group.get("feature_month", ""),
        "risk_dim": group.get("risk_dim", ""),
        "event_count": group.get("event_count", len(events)),
        "embedding_model": embedding_model,
        "embedding_dim": pooled.shape[0],
        "num_heads": num_heads,
        "h_text": pooled.tolist(),
    }


def build_group_text_representations(
    groups: Iterable[Dict[str, Any]],
    client: OpenAI,
    embedding_model: str,
    num_heads: int = 4,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for group in tqdm(list(groups), desc="EventTextRepr", ncols=100):
        outputs.append(
            build_group_text_representation(
                group=group,
                client=client,
                embedding_model=embedding_model,
                num_heads=num_heads,
            )
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build H_text representations for aggregated event groups.")
    parser.add_argument("--input", required=True, help="Input aggregated_events.jsonl path")
    parser.add_argument("--output", required=True, help="Output event_text_repr.jsonl path")
    parser.add_argument("--embedding-model", default="text-embedding-v4", help="Embedding model")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = get_client_for_model(args.embedding_model)
    groups = load_jsonl(args.input)
    outputs = build_group_text_representations(
        groups=groups,
        client=client,
        embedding_model=args.embedding_model,
        num_heads=args.num_heads,
    )
    write_jsonl(args.output, outputs)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "group_count": len(outputs),
                "embedding_model": args.embedding_model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
