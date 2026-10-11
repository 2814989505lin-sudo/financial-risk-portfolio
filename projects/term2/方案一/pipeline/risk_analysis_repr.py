import argparse
import json
from typing import Any, Dict, List

from tqdm import tqdm

from aggregate_events import load_jsonl, write_jsonl
from event_text_repr import embed_texts
from llm_extract_openai import get_client_for_model


def build_analysis_representations(
    rows: List[Dict[str, Any]],
    embedding_model: str,
) -> List[Dict[str, Any]]:
    client = get_client_for_model(embedding_model)
    texts = [str(row.get("analysis_text", "")).strip() for row in rows]
    embeddings = embed_texts(client=client, texts=texts, model=embedding_model)
    outputs: List[Dict[str, Any]] = []
    for row, embedding in tqdm(list(zip(rows, embeddings)), desc="RiskAnalysisRepr", ncols=100):
        outputs.append(
            {
                "group_id": row.get("group_id", ""),
                "global_id": row.get("global_id", ""),
                "canonical_company_name": row.get("canonical_company_name", ""),
                "feature_month": row.get("feature_month", ""),
                "risk_dim": row.get("risk_dim", ""),
                "event_count": row.get("event_count", 0),
                "embedding_model": embedding_model,
                "embedding_dim": len(embedding),
                "h_analyze": embedding,
            }
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build H_analyze representations for generated risk analyses.")
    parser.add_argument("--input", required=True, help="Input risk_analysis.jsonl path")
    parser.add_argument("--output", required=True, help="Output risk_analysis_repr.jsonl path")
    parser.add_argument("--embedding-model", default="text-embedding-v4", help="Embedding model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    outputs = build_analysis_representations(rows=rows, embedding_model=args.embedding_model)
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
