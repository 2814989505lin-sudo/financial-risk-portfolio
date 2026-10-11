import os
import sys
import time
import json
import tempfile
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import llm_extract_openai as module  # noqa: E402
from llm_extract_openai import default_max_output_tokens, get_api_key, get_multi_clients, infer_provider_from_model, load_few_shot_examples, parse_json_response  # noqa: E402


class TestLlmExtractOpenAI(unittest.TestCase):
    def test_parse_json_response_extracts_first_valid_object_from_mixed_text(self):
        content = """
<think>
先分析一下
</think>
{
  "is_in_scope": true,
  "candidate_labels": ["债务违约"]
}

补充说明：
{
  "ignored": true
}
""".strip()

        payload = parse_json_response(content)
        self.assertEqual(payload["is_in_scope"], True)
        self.assertEqual(payload["candidate_labels"], ["债务违约"])

    def test_infer_provider_from_model_supports_ling(self):
        self.assertEqual(infer_provider_from_model("Ling-1T"), "ling")
        self.assertEqual(infer_provider_from_model("glm-4-flash-250414"), "zhipu")
        self.assertEqual(infer_provider_from_model("qwen3.5-flash-2026-02-23"), "aliyun")
        self.assertEqual(infer_provider_from_model("glm-4.5"), "aliyun")
        self.assertEqual(infer_provider_from_model("kimi-k2.5"), "aliyun")
        self.assertEqual(infer_provider_from_model("MiniMax-M2.5"), "aliyun")

    def test_default_max_output_tokens_for_aliyun_models(self):
        self.assertEqual(default_max_output_tokens("qwen3.5-flash-2026-02-23"), 512)
        self.assertEqual(default_max_output_tokens("glm-4.5"), 512)
        self.assertEqual(default_max_output_tokens("kimi-k2.5"), 512)
        self.assertEqual(default_max_output_tokens("MiniMax-M2.5"), 512)
        self.assertIsNone(default_max_output_tokens("glm-4-flash-250414"))

    def test_get_api_key_accepts_legacy_ling_env_name(self):
        original_standard = os.environ.get("LING_STUDIO_API_KEY")
        original_legacy = os.environ.get("Ling_STUDIO_API_KEy")
        os.environ.pop("LING_STUDIO_API_KEY", None)
        os.environ["Ling_STUDIO_API_KEy"] = "legacy-test-key"
        try:
            self.assertEqual(get_api_key("ling"), "legacy-test-key")
        finally:
            if original_standard is None:
                os.environ.pop("LING_STUDIO_API_KEY", None)
            else:
                os.environ["LING_STUDIO_API_KEY"] = original_standard
            if original_legacy is None:
                os.environ.pop("Ling_STUDIO_API_KEy", None)
            else:
                os.environ["Ling_STUDIO_API_KEy"] = original_legacy

    def test_extract_multi_model_runs_models_in_parallel(self):
        original = module.extract_with_chunking

        def fake_extract_with_chunking(**kwargs):
            time.sleep(0.2)
            model = kwargs["model"]
            return ([{"event_type": f"evt-{model}", "taxonomy": "credit_risk", "event_polarity": "negative"}], {"chunk_count": 1})

        module.extract_with_chunking = fake_extract_with_chunking
        try:
            start = time.time()
            events, debug = module.extract_multi_model(
                model_clients=[
                    ("zhipu", "glm-4-flash-250414", object()),
                    ("ling", "Ling-1T", object()),
                ],
                record={"text": "测试正文", "title": "标题"},
                prompt_template="测试模板",
                schema={"event_taxonomy": {}},
                max_input_tokens=1000,
            )
            elapsed = time.time() - start
        finally:
            module.extract_with_chunking = original

        self.assertLess(elapsed, 0.35)
        self.assertEqual(debug["model_count"], 2)
        self.assertEqual(len(events), 2)

    def test_get_multi_clients_default_specs_are_aliyun_pair(self):
        original_openai = module.OpenAI
        original_get_api_key = module.get_api_key

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        module.OpenAI = FakeOpenAI
        module.get_api_key = lambda provider, env_key=None: f"fake-{provider}-{env_key or ''}"
        try:
            clients = get_multi_clients(
                [
                    {"provider": "aliyun", "model": "kimi-k2.5", "env_key": "ALIYUN_API_KEY", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
                    {"provider": "aliyun", "model": "MiniMax-M2.5", "env_key": "ALIYUN_API_KEY", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
                ]
            )
        finally:
            module.OpenAI = original_openai
            module.get_api_key = original_get_api_key

        self.assertEqual([provider for provider, _model, _client in clients], ["aliyun", "aliyun"])
        self.assertEqual([model for _provider, model, _client in clients], ["kimi-k2.5", "MiniMax-M2.5"])

    def test_load_few_shot_examples_supports_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "few_shot.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"title": "t1", "text": "x", "events": []}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"title": "t2", "text": "y", "events": [{"event_type": "债务违约"}]}, ensure_ascii=False) + "\n")
            rows = load_few_shot_examples(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["title"], "t2")


if __name__ == "__main__":
    unittest.main()
