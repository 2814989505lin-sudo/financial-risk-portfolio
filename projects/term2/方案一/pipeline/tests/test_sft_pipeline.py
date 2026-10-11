import os
import sys
import tempfile
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from build_sft_dataset import build_sft_datasets  # noqa: E402
from evaluate_sft import summarize_article_level_metrics  # noqa: E402
from llm_extract_openai import (  # noqa: E402
    build_chat_messages,
    read_env_file,
    supports_json_object_response_format,
)


TEST_PROMPT = """
你是事件抽取专家。

Schema:
{{SCHEMA_JSON}}

人工标注 few-shot 示例：
{{FEW_SHOT_EXAMPLES}}

文章信息：
- article_id: {{ARTICLE_ID}}
- title: {{TITLE}}
- source: {{SOURCE}}
- publish_time: {{PUBLISH_TIME}}
- company_mentions: {{COMPANY_MENTIONS}}

候选公司规范名（可选，仅用于归一提示）：
{{ALIGNMENT_CANDIDATES_JSON}}

TEXT:
{{TEXT}}

仅输出 JSON：
{
  "events": []
}
""".strip()

TEST_SCHEMA = {
    "event_taxonomy": {
        "operational_risk": {
            "negative_events": {
                "重大安全事故": {"arguments": ["公司名称"]},
                "复工复产": {"arguments": ["公司名称"]},
            }
        }
    }
}


class TestSFTPipeline(unittest.TestCase):
    def test_build_chat_messages_splits_system_and_user(self):
        messages = build_chat_messages(
            record={
                "article_id": "news_1",
                "title": "标题",
                "source": "来源",
                "publish_time": "2025-01-01",
                "company_mentions": ["测试公司"],
                "text": "正文",
            },
            prompt_template=TEST_PROMPT,
            schema=TEST_SCHEMA,
            few_shot_examples=[],
            alignment_candidates=[{"canonical_company_name": "测试公司"}],
            annotation_model="gpt-4o-mini",
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("仅输出 JSON", messages[0]["content"])
        self.assertIn("文章信息：", messages[1]["content"])
        self.assertIn("正文", messages[1]["content"])

    def test_supports_json_object_response_format_skips_ft_models(self):
        self.assertTrue(supports_json_object_response_format("gpt-4o-mini"))
        self.assertFalse(supports_json_object_response_format("ft:gpt-4o-mini-2024-07-18:org::abc"))

    def test_read_env_file_parses_openai_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("OPENAI_API_KEY=test-key\n")
                f.write("OTHER=value\n")

            values = read_env_file(env_path)

        self.assertEqual(values["OPENAI_API_KEY"], "test-key")
        self.assertEqual(values["OTHER"], "value")

    def test_build_sft_datasets_uses_silver_positive_and_manual_review_negative(self):
        gold_rows = [
            {
                "article_id": "g_neg_1",
                "title": "负样本1",
                "source": "来源",
                "publish_time": "2025-01-01",
                "company_mentions": [],
                "text": "正常文章",
                "split": "manual_review",
                "is_in_scope": False,
                "labels": [],
            },
            {
                "article_id": "g_neg_2",
                "title": "负样本2",
                "source": "来源",
                "publish_time": "2025-01-02",
                "company_mentions": [],
                "text": "正常文章",
                "split": "manual_review",
                "is_in_scope": False,
                "labels": [],
            },
            {
                "article_id": "g_pos_1",
                "title": "正样本1",
                "source": "来源",
                "publish_time": "2025-01-03",
                "company_mentions": [],
                "text": "只有文章级标签，没有事件级标注",
                "split": "manual_review",
                "is_in_scope": True,
                "labels": ["复工复产"],
            },
        ]
        silver_rows = [
            {
                "article_id": "s_pos_1",
                "title": "事件1",
                "source": "来源",
                "publish_time": "2025-01-04",
                "company_mentions": ["测试公司"],
                "text": "发生事故",
                "events": [
                    {
                        "event_type": "重大安全事故",
                        "taxonomy": "operational_risk",
                        "event_polarity": "negative",
                        "arguments": {"公司名称": "测试公司"},
                        "evidence": ["发生事故"],
                        "confidence": 0.9,
                        "alignment_status": "resolved",
                    }
                ],
            },
            {
                "article_id": "s_pos_2",
                "title": "事件2",
                "source": "来源",
                "publish_time": "2025-01-05",
                "company_mentions": ["测试公司"],
                "text": "开始复工",
                "events": [
                    {
                        "event_type": "复工复产",
                        "taxonomy": "operational_risk",
                        "event_polarity": "positive",
                        "arguments": {"公司名称": "测试公司"},
                        "evidence": ["开始复工"],
                        "confidence": 0.95,
                        "alignment_status": "resolved",
                    }
                ],
            },
        ]

        train_rows, val_rows, stats = build_sft_datasets(
            gold_rows=gold_rows,
            silver_rows=silver_rows,
            prompt_template=TEST_PROMPT,
            schema=TEST_SCHEMA,
            assistant_model="gpt-4o-mini",
            min_confidence=0.75,
            silver_val_ratio=0.5,
        )

        self.assertEqual(stats["skipped_manual_review_positive_without_event_annotations"], 1)
        self.assertEqual(stats["manual_review_negative_rows"], 2)
        self.assertEqual(len(train_rows), 3)
        self.assertEqual(len(val_rows), 1)
        self.assertIn('{"events": []}', [row["messages"][-1]["content"] for row in train_rows + val_rows])

    def test_summarize_article_level_metrics(self):
        summary = summarize_article_level_metrics(
            [
                {
                    "gold_labels": ["重大安全事故"],
                    "predicted_labels": ["重大安全事故"],
                    "json_success": True,
                    "invalid_event_types": [],
                },
                {
                    "gold_labels": [],
                    "predicted_labels": ["复工复产"],
                    "json_success": True,
                    "invalid_event_types": [],
                },
            ]
        )

        self.assertEqual(summary["article_count"], 2)
        self.assertAlmostEqual(summary["exact_match_accuracy"], 0.5)
        self.assertAlmostEqual(summary["negative_accuracy"], 0.0)
        self.assertGreater(summary["label_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
