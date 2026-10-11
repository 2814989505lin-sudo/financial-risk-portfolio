import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from openai import OpenAI
from event_taxonomy import schema_for_prompt as taxonomy_schema_for_prompt
from multi_model_fusion import fuse_events

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


ARTICLE_SECTION_MARKER = "文章信息："
OUTPUT_SECTION_MARKER = "仅输出 JSON"


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken is None:
        return len(text)
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def split_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int = 200) -> List[str]:
    if count_tokens(text) <= max_tokens or tiktoken is None:
        return [text]

    encoder = tiktoken.get_encoding("cl100k_base")
    token_ids = encoder.encode(text)

    chunks: List[str] = []
    step = max(1, max_tokens - overlap_tokens)
    for start in range(0, len(token_ids), step):
        piece = token_ids[start : start + max_tokens]
        if not piece:
            continue
        chunks.append(encoder.decode(piece))
        if start + max_tokens >= len(token_ids):
            break
    return chunks


def parse_json_response(content: str) -> Dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return {"events": []}

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    if fenced:
        text = fenced.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                payload, _end = decoder.raw_decode(text[idx:])
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                continue
        raise


def schema_for_prompt(schema: Dict[str, Any]) -> Dict[str, Any]:
    return taxonomy_schema_for_prompt(schema)


def load_few_shot_examples(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return []

    if str(path).endswith(".jsonl"):
        examples: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    examples.append(item)
        return examples

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        payload = payload.get("examples", [])

    if not isinstance(payload, list):
        raise ValueError("Few-shot file must be a JSON list or an object with an 'examples' list.")

    examples: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            examples.append(item)
    return examples


def render_few_shot_examples(examples: List[Dict[str, Any]]) -> str:
    if not examples:
        return "无"

    rendered: List[str] = []
    for idx, example in enumerate(examples, start=1):
        rendered.append(
            "\n".join(
                [
                    f"示例{idx}输入：",
                    f"- title: {example.get('title', '')}",
                    f"- source: {example.get('source', '')}",
                    f"- publish_time: {example.get('publish_time', '')}",
                    f"- company_mentions: {json.dumps(example.get('company_mentions', []), ensure_ascii=False)}",
                    "TEXT:",
                    str(example.get("text", "")),
                    "",
                    f"示例{idx}输出：",
                    json.dumps({"events": example.get("events", [])}, ensure_ascii=False, indent=2),
                ]
            )
        )
    return "\n\n".join(rendered)


def render_prompt_fields(
    template: str,
    schema: Dict[str, Any],
    few_shot_examples: List[Dict[str, Any]] | None = None,
    alignment_candidates: List[Dict[str, Any]] | None = None,
    annotation_model: str | None = None,
    record: Dict[str, Any] | None = None,
) -> str:
    prompt = template
    prompt = prompt.replace(
        "{{SCHEMA_JSON}}",
        json.dumps(schema_for_prompt(schema), ensure_ascii=False, indent=2),
    )
    prompt = prompt.replace("{{TAXONOMY_VALUES}}", "|".join(schema.get("event_taxonomy", {}).keys()))
    prompt = prompt.replace("{{POLARITY_VALUES}}", "positive|negative")
    prompt = prompt.replace("{{FEW_SHOT_EXAMPLES}}", render_few_shot_examples(few_shot_examples or []))
    prompt = prompt.replace("{{ANNOTATION_MODEL}}", str(annotation_model or ""))
    record = record or {}
    prompt = prompt.replace("{{ARTICLE_ID}}", str(record.get("article_id", "")))
    prompt = prompt.replace("{{TITLE}}", str(record.get("title", "")))
    prompt = prompt.replace("{{SOURCE}}", str(record.get("source", "")))
    prompt = prompt.replace("{{PUBLISH_TIME}}", str(record.get("publish_time", "")))
    prompt = prompt.replace(
        "{{COMPANY_MENTIONS}}",
        json.dumps(record.get("company_mentions", []), ensure_ascii=False),
    )
    prompt = prompt.replace(
        "{{ALIGNMENT_CANDIDATES_JSON}}",
        json.dumps(alignment_candidates or [], ensure_ascii=False, indent=2),
    )
    prompt = prompt.replace("{{TEXT}}", str(record.get("text", "")))
    return prompt


def split_prompt_template(prompt_template: str) -> Tuple[str, str]:
    if ARTICLE_SECTION_MARKER not in prompt_template or OUTPUT_SECTION_MARKER not in prompt_template:
        raise ValueError("Prompt template must contain article and output section markers.")
    before_article, after_article = prompt_template.split(ARTICLE_SECTION_MARKER, 1)
    article_block, output_block = after_article.split(OUTPUT_SECTION_MARKER, 1)
    system_template = f"{before_article.strip()}\n\n{OUTPUT_SECTION_MARKER}{output_block}".strip()
    user_template = f"{ARTICLE_SECTION_MARKER}{article_block}".strip()
    return system_template, user_template


def build_system_prompt(
    prompt_template: str,
    schema: Dict[str, Any],
    few_shot_examples: List[Dict[str, Any]] | None = None,
    annotation_model: str | None = None,
) -> str:
    system_template, _user_template = split_prompt_template(prompt_template)
    return render_prompt_fields(
        template=system_template,
        schema=schema,
        few_shot_examples=few_shot_examples,
        annotation_model=annotation_model,
    )


def build_user_prompt(
    record: Dict[str, Any],
    prompt_template: str,
    schema: Dict[str, Any],
    alignment_candidates: List[Dict[str, Any]] | None = None,
) -> str:
    _system_template, user_template = split_prompt_template(prompt_template)
    return render_prompt_fields(
        template=user_template,
        schema=schema,
        alignment_candidates=alignment_candidates,
        record=record,
    )


def build_chat_messages(
    record: Dict[str, Any],
    prompt_template: str,
    schema: Dict[str, Any],
    few_shot_examples: List[Dict[str, Any]] | None = None,
    alignment_candidates: List[Dict[str, Any]] | None = None,
    annotation_model: str | None = None,
) -> List[Dict[str, str]]:
    system_prompt = build_system_prompt(
        prompt_template=prompt_template,
        schema=schema,
        few_shot_examples=few_shot_examples,
        annotation_model=annotation_model,
    )
    user_prompt = build_user_prompt(
        record=record,
        prompt_template=prompt_template,
        schema=schema,
        alignment_candidates=alignment_candidates,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_prompt(
    record: Dict[str, Any],
    prompt_template: str,
    schema: Dict[str, Any],
    few_shot_examples: List[Dict[str, Any]] | None = None,
    alignment_candidates: List[Dict[str, Any]] | None = None,
    annotation_model: str | None = None,
) -> str:
    if ARTICLE_SECTION_MARKER not in prompt_template or OUTPUT_SECTION_MARKER not in prompt_template:
        return render_prompt_fields(
            template=prompt_template,
            schema=schema,
            few_shot_examples=few_shot_examples,
            alignment_candidates=alignment_candidates,
            annotation_model=annotation_model,
            record=record,
        )
    messages = build_chat_messages(
        record=record,
        prompt_template=prompt_template,
        schema=schema,
        few_shot_examples=few_shot_examples,
        alignment_candidates=alignment_candidates,
        annotation_model=annotation_model,
    )
    return "\n\n".join(message["content"] for message in messages)


def read_env_file(path: str | Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    file_path = Path(path)
    if not file_path.exists():
        return env
    for raw_line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            env[key] = value
    return env


def find_project_env_file() -> Path | None:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def get_api_key(provider: str, env_key: str | None = None) -> str:
    provider_candidates = {
        "openai": ["OPENAI_API_KEY"],
        "minimax": ["MINIMAX_API_KEY"],
        "zhipu": ["ZHIPU_API_KEY"],
        "ling": ["LING_STUDIO_API_KEY", "Ling_STUDIO_API_KEy"],
        "aliyun": ["ALIYUN_API_KEY"],
    }
    provider_name = str(provider or "").strip().lower()
    candidate_keys: List[str] = []
    if env_key:
        candidate_keys.append(str(env_key).strip())
    candidate_keys.extend(provider_candidates.get(provider_name, []))
    candidate_keys = [key for key in candidate_keys if key]
    if not candidate_keys:
        return ""
    for candidate_key in candidate_keys:
        api_key = os.getenv(candidate_key, "").strip()
        if api_key:
            return api_key
    env_path = find_project_env_file()
    if env_path is None:
        return ""
    env_values = read_env_file(env_path)
    for candidate_key in candidate_keys:
        api_key = env_values.get(candidate_key, "").strip()
        if api_key:
            return api_key
    return ""


def get_openai_api_key() -> str:
    return get_api_key("openai", "OPENAI_API_KEY")


def get_client() -> OpenAI:
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required but not set.")
    return OpenAI(api_key=api_key, timeout=60.0)


def infer_provider_from_model(model: str) -> str:
    lowered = str(model or "").strip().lower()
    if lowered.startswith("text-embedding-v"):
        return "aliyun"
    if lowered.startswith("qwen"):
        return "aliyun"
    if lowered.startswith("kimi-"):
        return "aliyun"
    if lowered == "minimax-m2.5":
        return "aliyun"
    if lowered == "glm-4.5":
        return "aliyun"
    if lowered.startswith("glm-"):
        return "zhipu"
    if lowered.startswith("ling-"):
        return "ling"
    if lowered.startswith("minimax-"):
        return "minimax"
    return "openai"


def get_client_for_model(model: str) -> OpenAI:
    provider = infer_provider_from_model(model)
    if provider == "aliyun":
        return OpenAI(
            api_key=get_api_key("aliyun", "ALIYUN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=60.0,
        )
    if provider == "zhipu":
        return OpenAI(
            api_key=get_api_key("zhipu", "ZHIPU_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4",
            timeout=60.0,
        )
    if provider == "ling":
        return OpenAI(
            api_key=get_api_key("ling"),
            base_url="https://api.tbox.cn/api/llm/v1",
            timeout=60.0,
        )
    if provider == "minimax":
        return OpenAI(
            api_key=get_api_key("minimax", "MINIMAX_API_KEY"),
            base_url="https://api.minimax.chat/v1",
            timeout=60.0,
        )
    return get_client()


def get_multi_clients(model_specs: Sequence[Dict[str, Any]] | None = None) -> List[Tuple[str, str, OpenAI]]:
    specs = list(model_specs or [])
    if not specs:
        specs = [
            {"provider": "aliyun", "model": "kimi-k2.5", "env_key": "ALIYUN_API_KEY", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
            {"provider": "aliyun", "model": "MiniMax-M2.5", "env_key": "ALIYUN_API_KEY", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
        ]

    clients: List[Tuple[str, str, OpenAI]] = []
    for spec in specs:
        provider = str(spec.get("provider", "")).strip()
        model = str(spec.get("model", "")).strip()
        env_key = str(spec.get("env_key", "")).strip() or None
        base_url = str(spec.get("base_url", "")).strip() or None
        api_key = get_api_key(provider, env_key=env_key)
        if not provider or not model:
            continue
        if not api_key:
            raise RuntimeError(f"{env_key or provider} is required but not set.")
        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": 60.0}
        if base_url:
            kwargs["base_url"] = base_url
        clients.append((provider, model, OpenAI(**kwargs)))
    return clients


def supports_json_object_response_format(model: str) -> bool:
    lowered = str(model or "").strip().lower()
    if lowered.startswith("ft:"):
        return False
    if lowered.startswith("minimax-"):
        return False
    if lowered.startswith("kimi-"):
        return False
    if lowered.startswith("qwen"):
        return False
    if lowered == "glm-4.5":
        return False
    if lowered.startswith("glm-"):
        return False
    if lowered.startswith("ling-"):
        return False
    return True


def supports_custom_temperature(model: str) -> bool:
    lowered = str(model or "").strip().lower()
    return not lowered.startswith("gpt-5")


def requires_stream_mode(model: str) -> bool:
    lowered = str(model or "").strip().lower()
    return lowered == "glm-4.5"


def default_max_output_tokens(model: str) -> int | None:
    lowered = str(model or "").strip().lower()
    if lowered.startswith("qwen3.5-flash-2026-02-23"):
        return 512
    if lowered == "glm-4.5":
        return 512
    if lowered.startswith("kimi-"):
        return 512
    if lowered == "minimax-m2.5":
        return 512
    return None


def call_with_retry(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_retries: int = 3,
    request_timeout: float = 60.0,
) -> str:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            max_output_tokens = default_max_output_tokens(model)
            if max_output_tokens is not None:
                request_kwargs["max_tokens"] = max_output_tokens
            if supports_custom_temperature(model):
                request_kwargs["temperature"] = temperature
            if supports_json_object_response_format(model):
                request_kwargs["response_format"] = {"type": "json_object"}
            request_kwargs["timeout"] = request_timeout
            if requires_stream_mode(model):
                stream = client.chat.completions.create(stream=True, **request_kwargs)
                content_parts: List[str] = []
                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta.content
                    except Exception:
                        delta = None
                    if delta:
                        content_parts.append(delta)
                return "".join(content_parts)
            response = client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 10))
    raise last_err  # type: ignore[misc]


def chat_complete_with_retry(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            max_output_tokens = default_max_output_tokens(model)
            if max_output_tokens is not None:
                request_kwargs["max_tokens"] = max_output_tokens
            if supports_custom_temperature(model):
                request_kwargs["temperature"] = temperature
            response = client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 10))
    raise last_err  # type: ignore[misc]


def extract_once(
    client: OpenAI,
    model: str,
    prompt: str | None = None,
    messages: List[Dict[str, str]] | None = None,
    temperature: float = 0.0,
) -> Tuple[List[Dict[str, Any]], str]:
    if messages is None:
        messages = [{"role": "user", "content": prompt or ""}]
    content = call_with_retry(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
    )
    payload = parse_json_response(content)
    events = payload.get("events", [])
    if not isinstance(events, list):
        events = []
    return events, content


def dedupe_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    merged: List[Dict[str, Any]] = []
    for event in events:
        signature = (
            event.get("event_type", ""),
            event.get("taxonomy", ""),
            event.get("event_polarity", ""),
            tuple(sorted((event.get("arguments") or {}).items())),
        )
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(event)
    return merged


def extract_with_chunking(
    client: OpenAI,
    record: Dict[str, Any],
    prompt_template: str,
    schema: Dict[str, Any],
    model: str,
    max_input_tokens: int,
    temperature: float = 0.0,
    few_shot_examples: List[Dict[str, Any]] | None = None,
    alignment_candidates: List[Dict[str, Any]] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    text = record.get("text", "")
    chunks = split_text_by_tokens(text, max_input_tokens)

    all_events: List[Dict[str, Any]] = []
    raw_outputs: List[str] = []
    for idx, chunk in enumerate(chunks):
        chunk_record = dict(record)
        chunk_record["text"] = chunk
        if len(chunks) > 1:
            chunk_record["title"] = f"{record.get('title', '')} [chunk {idx + 1}/{len(chunks)}]"

        messages = build_chat_messages(
            chunk_record,
            prompt_template,
            schema,
            few_shot_examples=few_shot_examples,
            alignment_candidates=alignment_candidates,
            annotation_model=model,
        )
        events, raw = extract_once(client=client, model=model, messages=messages, temperature=temperature)
        all_events.extend(events)
        raw_outputs.append(raw)

    return dedupe_events(all_events), {"chunk_count": len(chunks), "raw_outputs": raw_outputs}


def extract_multi_model(
    model_clients: Sequence[Tuple[str, str, OpenAI]],
    record: Dict[str, Any],
    prompt_template: str,
    schema: Dict[str, Any],
    max_input_tokens: int,
    temperature: float = 0.0,
    few_shot_examples: List[Dict[str, Any]] | None = None,
    alignment_candidates: List[Dict[str, Any]] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    indexed_results: List[Tuple[str, List[Dict[str, Any]]]] = [("", []) for _ in model_clients]
    model_debug: Dict[str, Any] = {}

    def run_single_model(
        provider: str,
        model: str,
        client: OpenAI,
    ) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
        events, debug = extract_with_chunking(
            client=client,
            record=record,
            prompt_template=prompt_template,
            schema=schema,
            model=model,
            max_input_tokens=max_input_tokens,
            temperature=temperature,
            few_shot_examples=few_shot_examples,
            alignment_candidates=alignment_candidates,
        )
        return provider, model, events, debug

    with ThreadPoolExecutor(max_workers=max(1, min(len(model_clients), 4))) as executor:
        future_to_index = {
            executor.submit(run_single_model, provider, model, client): idx
            for idx, (provider, model, client) in enumerate(model_clients)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            provider, model, events, debug = future.result()
            indexed_results[idx] = (model, events)
            model_debug[provider] = {
                "model": model,
                "event_count": len(events),
                "chunk_count": debug.get("chunk_count", 0),
            }

    per_model_events = [item for item in indexed_results if item[0]]
    fused_events = fuse_events(per_model_events)
    return fused_events, {"models": model_debug, "model_count": len(model_clients)}
