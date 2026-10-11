import os
import sys
import tempfile
import unittest

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from build_event_gold_examples import build_event_gold_examples  # noqa: E402


class TestBuildEventGoldExamples(unittest.TestCase):
    def test_build_event_gold_examples_infers_schema_events(self):
        schema_path = os.path.join(os.path.dirname(PIPELINE_DIR), "config", "schema_news_v0.yaml")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = yaml.safe_load(f)

        rows = [
            {
                "article_id": "a1",
                "title": "评级下调并完成债券展期",
                "source": "新浪财经",
                "publish_time": "2024-04-10 09:30:00",
                "text": "公司评级由AA下调至A，同时公告债券展期成功。",
                "canonical_company_name": "万科企业",
                "matched_alias": "万科企业",
                "global_id": "gid_a1",
                "alignment_status": "resolved",
                "company_mentions": ["万科企业"],
            }
        ]
        examples = build_event_gold_examples(rows, schema=schema, max_examples=100, max_events_per_article=4)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["label_source"], "ai_preannotation")
        event_types = [item["event_type"] for item in examples[0]["events"]]
        self.assertIn("主体信用评级下调", event_types)
        self.assertIn("债券展期成功", event_types)


if __name__ == "__main__":
    unittest.main()
