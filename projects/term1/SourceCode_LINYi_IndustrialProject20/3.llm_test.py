from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import pandas as pd
import json
import re
import io
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from time import sleep
from llm_config import BAILIAN_BASE_URL, BAILIAN_MODEL, get_bailian_api_key

model = ChatOpenAI(
    temperature=0,
    model=BAILIAN_MODEL,
    timeout=180,
    max_retries=1,
    api_key=get_bailian_api_key(),
    base_url=BAILIAN_BASE_URL,
    extra_body={"enable_thinking": False},
)

ALLOWED_TYPES = {"PERSON", "ORG", "GPE", "PRODUCT", "LAW", "NORP", "MONEY"}

RISK_LABELS = ["credit_risk", "market_risk", "operational_risk", "fraud_risk"]

schema_example = (
    '{\n'
    '  "entities": [ {"name": "HSBC", "type": "ORG", "aliases":["HSBC Holdings"], "confidence": 0.95} ],\n'
    '  "relations": [ {"subject": "HSBC", "predicate": "acquired", "object": "SVB UK", "evidence": "...", "confidence": 0.86} ]\n'
    '}'
)

kg_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an IE system. Extract high-precision entities and binary relations from the input news. "
        "Return ONLY valid JSON with keys: entities (list) and relations (list). "
        "Entity schema: name, type, aliases(optional), confidence(optional); type in [PERSON, ORG, GPE, PRODUCT, LAW, NORP, MONEY]. "
        "Relation schema: subject, predicate, object, subject_type(optional), object_type(optional), evidence(optional), confidence(optional). "
        "Keep entities unique, concise, and deduplicate obvious synonyms via aliases. "
        "Return at most 12 entities and at most 12 relations."
    ),
    (
        "user",
        "Text: {text}\n"
        "Return a JSON object like:\n{schema_example}\n"
        "Return JSON only, no explanations."
    ),
]).partial(schema_example=schema_example)

kg_chain = kg_prompt | model

class Entity(BaseModel):
    name: str
    type: str | None = Field(default=None)
    aliases: list[str] | None = None
    confidence: float | None = None

class Relation(BaseModel):
    subject: str
    predicate: str
    object: str
    subject_type: str | None = None
    object_type: str | None = None
    evidence: str | None = None
    confidence: float | None = None

class Extraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

structured_chain = None

def read_csv_smart(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1", "ISO-8859-1", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="cp1252", errors="replace") as f:
        txt = f.read()
    return pd.read_csv(io.StringIO(txt))

def normalize_predicate(p: str) -> str:
    p = (p or "").strip().lower()
    p = re.sub(r"\s+", "_", p)
    return p or "related_to"

def _s(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def extract_all_risk_types(row: Dict) -> List[str]:
    risks: List[str] = []

    def is_true(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return v.strip().lower() in {"true", "1", "yes", "y"}
        return False

    if is_true(row.get("has_credit_risk")):
        risks.append("credit_risk")
    if is_true(row.get("has_market_risk")):
        risks.append("market_risk")
    if is_true(row.get("has_operational_risk")):
        risks.append("operational_risk")
    if is_true(row.get("has_fraud_risk")):
        risks.append("fraud_risk")

    return risks

def risk_list_to_multihot(all_risk_types: List[str]) -> List[int]:
    s = set(all_risk_types)
    return [1 if lbl in s else 0 for lbl in RISK_LABELS]

ENTITY_TYPES = sorted(ALLOWED_TYPES)
ENTITY_TYPE_TO_IDX = {t: i for i, t in enumerate(ENTITY_TYPES)}

def entity_type_one_hot(etype: str) -> List[int]:
    vec = [0] * len(ENTITY_TYPES)
    if etype in ENTITY_TYPE_TO_IDX:
        vec[ENTITY_TYPE_TO_IDX[etype]] = 1
    return vec

def parse_llm_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise

def extract_entities_relations(text: str):
    last_error = None
    for attempt in range(1, 3):
        try:
            resp = kg_chain.invoke({"text": text[:3500], "schema_example": schema_example})
            content = getattr(resp, "content", resp)
            data = parse_llm_json(str(content))
            ents = data.get("entities", []) or []
            rels = data.get("relations", []) or []
            return ents, rels
        except Exception as e:
            last_error = e
            if attempt < 2:
                print(f"[RETRY] LLM call failed once: {e}", flush=True)
                sleep(2)
    raise RuntimeError(f"LLM call failed: {last_error}")

def build_graph_for_doc(
    doc_id,
    title: str,
    text: str,
    url: str | None,
    published_utc,
    all_risk_types: List[str],
    ents: List[Dict],
    rels: List[Dict]
) -> Optional[Dict]:
    local_name_to_idx: Dict[str, int] = {}
    local_entities: List[Dict] = []

    for e in ents:
        name = (e.get("name") or e.get("entity") or "").strip()
        etype = (e.get("type") or e.get("label") or "").strip().upper() or None

        if not name or etype not in ALLOWED_TYPES:
            continue

        if name in local_name_to_idx:
            continue

        idx = len(local_entities)
        local_name_to_idx[name] = idx
        local_entities.append({"name": name, "type": etype})

    if not local_entities:
        return None

    node_names = [e["name"] for e in local_entities]
    node_types = [e["type"] for e in local_entities]

    edges_src: List[int] = []
    edges_dst: List[int] = []
    edge_types: List[str] = []
    edge_seen: Set[Tuple[int, str, int]] = set()

    for r in rels:
        s_name = (r.get("subject") or "").strip()
        o_name = (r.get("object") or "").strip()
        if not s_name or not o_name:
            continue

        if s_name not in local_name_to_idx or o_name not in local_name_to_idx:
            continue

        s_idx = local_name_to_idx[s_name]
        o_idx = local_name_to_idx[o_name]

        pred = normalize_predicate(r.get("predicate") or "related_to")

        key = (s_idx, pred, o_idx)
        if key in edge_seen:
            continue

        edge_seen.add(key)
        edges_src.append(s_idx)
        edges_dst.append(o_idx)
        edge_types.append(pred)

    x: List[List[int]] = [entity_type_one_hot(t) for t in node_types]

    y_vec: List[int] = risk_list_to_multihot(all_risk_types)

    graph = {
        "doc_id": doc_id,
        "title": title,
        "url": url,
        "published_utc": published_utc,
        "risk_types": list(all_risk_types),
        "y": y_vec,
        "node_names": node_names,
        "node_types": node_types,
        "x": x,
        "edge_index": [edges_src, edges_dst],
        "edge_type": edge_types,
    }
    return graph

def build_graph_dataset(df: pd.DataFrame, output_path: str | None = None) -> List[Dict]:
    graphs: List[Dict] = []
    processed_doc_ids: Set[str] = set()

    checkpoint_path = Path(output_path) if output_path else None
    if checkpoint_path and checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        with checkpoint_path.open("r", encoding="utf-8") as f:
            graphs = json.load(f)
        processed_doc_ids = {str(g.get("doc_id")) for g in graphs if g.get("doc_id") is not None}
        print(f"[RESUME] loaded {len(graphs)} existing graphs from {checkpoint_path}", flush=True)

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        doc_id = row.get("id")
        doc_key = str(doc_id)
        if doc_key in processed_doc_ids:
            continue

        title = _s(row.get("title"))
        text = _s(row.get("text"))
        url = _s(row.get("url")) or None
        published_utc = row.get("published_utc")

        all_risk_types = extract_all_risk_types(dict(row))
        if not all_risk_types:
            continue

        full_text = f"{title}\n\n{text}".strip()
        if not full_text:
            continue

        print(f"[GRAPH] {i}/{total} id={doc_id} title={title[:60]}", flush=True)

        try:
            ents, rels = extract_entities_relations(full_text)
        except Exception as e:
            print(f"[SKIP DOC] id={doc_id} url={url} err={e}", flush=True)
            continue

        g = build_graph_for_doc(
            doc_id=doc_id,
            title=title,
            text=text,
            url=url,
            published_utc=published_utc,
            all_risk_types=all_risk_types,
            ents=ents,
            rels=rels,
        )
        if g is not None:
            graphs.append(g)
            processed_doc_ids.add(doc_key)
            if checkpoint_path:
                with checkpoint_path.open("w", encoding="utf-8") as f:
                    json.dump(graphs, f, ensure_ascii=False, indent=2)

    return graphs

def main():
    csv_path = "guardian_news_test.csv"
    df_raw = read_csv_smart(csv_path)

    def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
        m = {c.lower(): c for c in df.columns}

        def pick(*names):
            for n in names:
                if n in m:
                    return m[n]
            return None

        keep: Dict[str, pd.Series] = {}
        col_id = pick("id")
        col_tit = pick("title", "webtitle", "webTitle")
        col_txt = pick("text", "bodytext", "bodyText", "content")
        col_url = pick("url", "weburl", "webUrl")
        col_pub = pick("published_utc", "webpublicationdate", "webPublicationDate")

        if col_id:
            keep["id"] = df[col_id]
        if col_tit:
            keep["title"] = df[col_tit]
        if col_txt:
            keep["text"] = df[col_txt]
        if col_url:
            keep["url"] = df[col_url]
        if col_pub:
            keep["published_utc"] = df[col_pub]

        for col in df.columns:
            cl = col.lower()
            if (
                cl.startswith("has_")
                or "risk" in cl
                or "label" in cl
                or cl.startswith("ai_")
            ):
                keep[col] = df[col]

        return pd.DataFrame(keep)

    df = norm_cols(df_raw)

    output_path = "graph_dataset_test.json"
    graphs = build_graph_dataset(df, output_path=output_path)

    preview = graphs[0] if graphs else None
    print({
        "num_graphs": len(graphs),
        "preview_graph": preview,
    })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graphs, f, ensure_ascii=False, indent=2)

    print(f"圖數據構建完成！總圖數: {len(graphs)}，保存至: {output_path}")

if __name__ == "__main__":
    main()
