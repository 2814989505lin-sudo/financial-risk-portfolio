import os
import sys
import tempfile
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from llm_extract_openai import build_prompt, load_few_shot_examples, schema_for_prompt
from run_docfee_news import derive_event_labels


TEST_SCHEMA = {
    "event_taxonomy": {
        "credit_risk": {
            "negative_events": {
                "债务违约": {
                    "arguments": ["公司名称", "发生时间"],
                }
            }
        },
        "fraud_risk": {
            "negative_events": {
                "财务造假": {
                    "arguments": ["公司名称", "发生时间"],
                }
            }
        },
    }
}


class TestDynamicTaxonomy(unittest.TestCase):
    def test_schema_for_prompt_preserves_taxonomy_names(self):
        prompt_schema = schema_for_prompt(TEST_SCHEMA)
        self.assertEqual(prompt_schema["债务违约"]["taxonomy"], "credit_risk")
        self.assertEqual(prompt_schema["债务违约"]["event_polarity"], "negative")
        self.assertEqual(prompt_schema["财务造假"]["taxonomy"], "fraud_risk")
        self.assertEqual(prompt_schema["财务造假"]["event_polarity"], "negative")

    def test_event_labels_are_dynamic(self):
        labels = derive_event_labels(
            [
                {"event_type": "债务违约", "taxonomy": "credit_risk", "event_polarity": "negative"},
                {"event_type": "财务造假", "taxonomy": "fraud_risk", "event_polarity": "negative"},
            ],
            [("credit_risk", "negative"), ("fraud_risk", "negative")],
        )
        self.assertEqual(labels["credit_risk_negative"], ["债务违约"])
        self.assertEqual(labels["fraud_risk_negative"], ["财务造假"])

    def test_build_prompt_includes_few_shot_examples(self):
        prompt_template = "\n".join(
            [
                "Schema:",
                "{{SCHEMA_JSON}}",
                "Examples:",
                "{{FEW_SHOT_EXAMPLES}}",
                "Article:",
                "{{TITLE}}",
                "{{TEXT}}",
            ]
        )
        record = {"title": "当前文章", "text": "正文"}
        few_shot_examples = [
            {
                "title": "示例文章",
                "source": "测试来源",
                "publish_time": "2025-01-01",
                "company_mentions": ["测试公司"],
                "text": "示例正文",
                "events": [
                    {
                        "event_type": "债务违约",
                        "taxonomy": "credit_risk",
                        "event_polarity": "negative",
                        "arguments": {"公司名称": "测试公司", "发生时间": "2025-01-01"},
                        "evidence": ["示例正文"],
                        "confidence": 0.9,
                    }
                ],
            }
        ]

        prompt = build_prompt(
            record=record,
            prompt_template=prompt_template,
            schema=TEST_SCHEMA,
            few_shot_examples=few_shot_examples,
        )

        self.assertIn("示例1输入：", prompt)
        self.assertIn("示例文章", prompt)
        self.assertIn('"event_type": "债务违约"', prompt)
        self.assertNotIn("{{FEW_SHOT_EXAMPLES}}", prompt)

    def test_build_prompt_without_few_shot_examples(self):
        prompt = build_prompt(
            record={"title": "当前文章", "text": "正文"},
            prompt_template="Examples:\n{{FEW_SHOT_EXAMPLES}}",
            schema=TEST_SCHEMA,
            few_shot_examples=[],
        )
        self.assertEqual(prompt, "Examples:\n无")

    def test_load_few_shot_examples_from_file(self):
        payload = [
            {
                "title": "示例文章",
                "source": "测试来源",
                "publish_time": "2025-01-01",
                "company_mentions": ["测试公司"],
                "text": "示例正文",
                "events": [],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "few_shot.json")
            with open(path, "w", encoding="utf-8") as f:
                import json

                json.dump(payload, f, ensure_ascii=False)

            examples = load_few_shot_examples(path)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["title"], "示例文章")


if __name__ == "__main__":
    unittest.main()
