import os
import sys
import tempfile
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from build_event_corpus import (  # noqa: E402
    build_corpus,
    dedupe_rows,
    load_risk_csv_rows,
    mark_rows_all_in_scope,
    normalize_news_row,
)
from global_id_alignment import make_global_id  # noqa: E402


class TestBuildEventCorpus(unittest.TestCase):
    def test_load_risk_csv_rows_preserves_company_and_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = os.path.join(tmpdir, "deep_mining_risk_events.csv")
            flash_path = os.path.join(tmpdir, "flash_news_risk_events_2021_to_2026.csv")
            with open(deep_path, "w", encoding="utf-8-sig") as f:
                f.write("时间,相关公司,标题,正文\n")
                f.write("2026-03-01 10:00:00,平安银行,深度风险新闻,这是一篇较长的风险正文。\n")
            with open(flash_path, "w", encoding="utf-8-sig") as f:
                f.write("时间,相关公司,标题,正文\n")
                f.write("2021-01-01 09:00:00,新宁物流,,快讯正文一\n")

            rows = load_risk_csv_rows([deep_path, flash_path])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["src"], "deep_mining")
            self.assertEqual(rows[0]["company_mentions"], ["平安银行"])
            self.assertEqual(rows[1]["src"], "flash_news")
            self.assertEqual(rows[1]["title"], "")

    def test_load_risk_csv_rows_supports_sina_balanced_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sina_path = os.path.join(tmpdir, "sina_major_news_last_2y_balanced_3k.csv")
            with open(sina_path, "w", encoding="utf-8-sig") as f:
                f.write("title,content,pub_time,src\n")
                f.write("标题A,正文A正文A正文A正文A正文A正文A正文A正文A正文A正文A,2025-01-01 10:00:00,新浪财经\n")

            rows = load_risk_csv_rows([sina_path])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["title"], "标题A")
            self.assertEqual(rows[0]["src"], "新浪财经")
            self.assertEqual(rows[0]["pub_time"], "2025-01-01 10:00:00")
            self.assertEqual(rows[0]["company_mentions"], [])

    def test_dedupe_rows_does_not_collapse_blank_title_flash_rows(self):
        rows = [
            normalize_news_row(
                {
                    "title": "",
                    "content": "快讯正文一",
                    "pub_time": "2021-01-01 09:00:00",
                    "src": "flash_news",
                    "company_mentions": ["公司A"],
                },
                1,
            ),
            normalize_news_row(
                {
                    "title": "",
                    "content": "快讯正文二",
                    "pub_time": "2021-01-01 09:00:00",
                    "src": "flash_news",
                    "company_mentions": ["公司A"],
                },
                2,
            ),
        ]

        deduped = dedupe_rows(rows)
        self.assertEqual(len(deduped), 2)

    def test_mark_rows_all_in_scope_sets_positive_scope_fields(self):
        alias_map = {
            "威马汽车": [
                {
                    "canonical_company_name": "威马汽车",
                    "global_id": make_global_id("威马汽车"),
                    "matched_alias": "威马汽车",
                }
            ]
        }
        rows = [
            {
                "article_id": "a1",
                "title": "标题",
                "source": "测试来源",
                "publish_time": "2025-01-01 10:00:00",
                "text": "正文",
                "company_mentions": ["威马汽车"],
            }
        ]

        scoped_rows = mark_rows_all_in_scope(rows, alias_map)
        self.assertEqual(len(scoped_rows), 1)
        self.assertTrue(scoped_rows[0]["is_in_scope"])
        self.assertEqual(scoped_rows[0]["candidate_labels"], [])
        self.assertEqual(scoped_rows[0]["canonical_company_name"], "威马汽车")
        self.assertEqual(scoped_rows[0]["global_id"], make_global_id("威马汽车"))
        self.assertEqual(scoped_rows[0]["screen_reason"], "all_balanced_rows_kept")

    def test_build_corpus_builds_manual_review_and_silver_pool(self):
        alias_map = {
            "威马汽车": [
                {
                    "canonical_company_name": "威马汽车",
                    "global_id": make_global_id("威马汽车"),
                    "matched_alias": "威马汽车",
                }
            ]
        }
        rows = []
        for idx in range(18):
            rows.append(
                {
                    "title": f"新闻{idx} 债务违约" if idx < 14 else f"新闻{idx} 行业观察",
                    "content": (
                        "公司发生债务违约，影响较大。后续违约处置仍在推进，债权人关注偿债安排。"
                        "公司发生债务违约，影响较大。后续违约处置仍在推进，债权人关注偿债安排。"
                        "公司发生债务违约，影响较大。后续违约处置仍在推进，债权人关注偿债安排。"
                        if idx < 14
                        else "宏观市场平稳，企业经营正常，未见异常事项。主营业务推进顺利，市场反馈稳定。"
                        "宏观市场平稳，企业经营正常，未见异常事项。主营业务推进顺利，市场反馈稳定。"
                        "宏观市场平稳，企业经营正常，未见异常事项。主营业务推进顺利，市场反馈稳定。"
                    ),
                    "pub_time": f"2025-01-{idx % 9 + 1:02d} 10:00:00",
                    "src": "测试来源",
                    "url": f"https://example.com/{idx}",
                    "company_mentions": ["威马汽车"] if idx < 14 else [],
                }
            )

        corpus = build_corpus(
            rows=rows,
            manual_gold_total=5,
            seed=42,
            alias_map=alias_map,
        )

        self.assertEqual(len(corpus["manual_review"]), 5)
        self.assertEqual(len(corpus["silver_pool"]), 13)
        split_names = {row["split"] for row in corpus["manual_review"]}
        self.assertEqual(split_names, {"manual_review"})
        manual_ids = {row["article_id"] for row in corpus["manual_review"]}
        silver_ids = {row["article_id"] for row in corpus["silver_pool"]}
        self.assertTrue(manual_ids.isdisjoint(silver_ids))
        self.assertEqual(corpus["manual_review"][0]["label_source"], "manual_review")


if __name__ == "__main__":
    unittest.main()
