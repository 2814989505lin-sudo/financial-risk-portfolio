import csv
import json
import os
import sys
import tempfile
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from run_docfee_news import write_summary_csv
from sample_review_pack import build_review_pack, read_jsonl


class TestPipelineOutputs(unittest.TestCase):
    def test_write_summary_csv_uses_events_only(self):
        rows = [
            {
                "article_id": "news_1",
                "url": "https://example.com/1",
                "title": "标题1",
                "events": [
                    {
                        "event_type": "债务违约",
                        "taxonomy": "credit_risk",
                        "event_polarity": "negative",
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "summary.csv")
            write_summary_csv(rows, summary_path, [("credit_risk", "negative"), ("fraud_risk", "negative")])

            with open(summary_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                parsed_rows = list(reader)

        self.assertIn("credit_risk_negative_labels", fieldnames)
        self.assertIn("event_count", fieldnames)
        self.assertNotIn("json_valid", fieldnames)
        self.assertEqual(parsed_rows[0]["credit_risk_negative_labels"], "债务违约")
        self.assertEqual(parsed_rows[0]["event_count"], "1")

    def test_build_review_pack_supports_simplified_rows(self):
        rows = [
            {
                "article_id": "news_1",
                "url": "https://example.com/1",
                "title": "标题1",
                "source": "来源",
                "publish_time": "2025-01-01",
                "company_mentions": ["测试公司"],
                "events": [
                    {
                        "event_type": "债务违约",
                        "taxonomy": "credit_risk",
                        "event_polarity": "negative",
                        "arguments": {"公司名称": "测试公司"},
                        "evidence": ["测试公司发生债务违约"],
                        "confidence": 0.9,
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "news_events.jsonl")
            sample_path = os.path.join(tmpdir, "review_sample.jsonl")
            report_path = os.path.join(tmpdir, "review_report.md")
            with open(input_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            stat = build_review_pack(
                input_path=input_path,
                sample_path=sample_path,
                report_path=report_path,
                sample_rate=1.0,
                seed=42,
            )
            sample_rows = read_jsonl(sample_path)
            with open(report_path, "r", encoding="utf-8") as f:
                report = f.read()

        self.assertEqual(stat["sample_count"], 1)
        self.assertEqual(sample_rows[0]["pred"]["events"][0]["event_type"], "债务违约")
        self.assertNotIn("validation", sample_rows[0]["pred"])
        self.assertIn("事件类型分布Top", report)
        self.assertNotIn("校验失败文章数", report)


if __name__ == "__main__":
    unittest.main()
