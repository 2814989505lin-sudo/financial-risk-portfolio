import importlib.util
from pathlib import Path
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[3] / "数据抓取" / "长文抓取.py"
SPEC = importlib.util.spec_from_file_location("long_news_fetch", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class TestLongNewsFetch(unittest.TestCase):
    def test_sample_balanced_dataframe_returns_month_balanced_rows(self):
        df = pd.DataFrame(
            [
                {"title": "甲公司流动性紧张", "content": "公司甲流动性紧张", "pub_time": "2025-01-01 10:00:00", "src": "新浪财经"},
                {"title": "甲公司裁员", "content": "公司甲裁员并进行业务调整", "pub_time": "2025-01-02 10:00:00", "src": "新浪财经"},
                {"title": "乙公司债务违约", "content": "公司乙债务违约", "pub_time": "2025-01-03 10:00:00", "src": "新浪财经"},
                {"title": "丙公司复工复产", "content": "公司丙复工复产", "pub_time": "2025-02-01 10:00:00", "src": "新浪财经"},
                {"title": "丁公司订单下滑", "content": "公司丁订单下滑", "pub_time": "2025-02-02 10:00:00", "src": "新浪财经"},
                {"title": "戊公司重大安全事故", "content": "公司戊发生重大安全事故", "pub_time": "2025-03-01 10:00:00", "src": "新浪财经"},
                {"title": "己公司股价暴跌", "content": "公司己股价暴跌", "pub_time": "2025-03-02 10:00:00", "src": "新浪财经"},
                {"title": "庚公司融资受阻", "content": "公司庚融资受阻", "pub_time": "2025-03-03 10:00:00", "src": "新浪财经"},
            ]
        )

        sampled_df, summary = MODULE.sample_balanced_dataframe(df, target_total=6, seed=42)

        self.assertEqual(len(sampled_df), 6)
        self.assertEqual(summary["input_rows_after_time_filter"], 8)
        self.assertEqual(summary["output_rows"], 6)
        self.assertEqual(sum(summary["output_month_counts"].values()), 6)
        self.assertLessEqual(max(summary["output_month_counts"].values()) - min(summary["output_month_counts"].values()), 1)
        self.assertGreaterEqual(summary["output_unique_company_count"], 6)

    def test_append_unique_rows_respects_limit(self):
        df = pd.DataFrame(
            [
                {"title": "a1", "content": "c", "pub_time": "2025-01-01 10:00:00", "src": "新浪财经"},
                {"title": "a2", "content": "c", "pub_time": "2025-01-02 10:00:00", "src": "新浪财经"},
                {"title": "a3", "content": "c", "pub_time": "2025-01-03 10:00:00", "src": "新浪财经"},
            ]
        )
        target_rows = []
        seen_keys = set()

        MODULE.append_unique_rows(target_rows=target_rows, source_df=df, seen_keys=seen_keys, limit=2)

        self.assertEqual(len(target_rows), 2)
        self.assertEqual(len(seen_keys), 2)

    def test_compute_article_priority_prefers_strong_risk_terms(self):
        strong = MODULE.compute_article_priority(
            {
                "title": "某公司债务违约并评级下调",
                "content": "公司债务违约，评级下调，流动性危机加剧",
            }
        )
        early = MODULE.compute_article_priority(
            {
                "title": "某公司融资受阻",
                "content": "公司融资受阻，现金流恶化，订单下滑",
            }
        )
        change = MODULE.compute_article_priority(
            {
                "title": "某公司业务调整",
                "content": "公司业务调整，战略收缩并裁员",
            }
        )

        self.assertGreater(strong, early)
        self.assertGreater(early, change)

    def test_day_iterator_advances_one_day_each_time(self):
        steps = list(
            MODULE.day_iterator(
                MODULE.datetime(2025, 1, 1),
                MODULE.datetime(2025, 1, 4),
            )
        )

        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0][0].strftime("%Y-%m-%d"), "2025-01-01")
        self.assertEqual(steps[0][1].strftime("%Y-%m-%d"), "2025-01-02")
        self.assertEqual(steps[-1][0].strftime("%Y-%m-%d"), "2025-01-03")
        self.assertEqual(steps[-1][1].strftime("%Y-%m-%d"), "2025-01-04")

    def test_clean_title_text_skips_plain_text_without_html_parse_warning(self):
        self.assertEqual(MODULE.clean_title_text("plain/path/like/title"), "plain/path/like/title")

    def test_extract_company_key_prefers_company_like_name(self):
        company_key = MODULE.extract_company_key("华夏幸福股份有限公司债务重组", "正文")
        self.assertEqual(company_key, "华夏幸福")

    def test_sample_balanced_dataframe_prioritizes_company_coverage_within_month(self):
        df = pd.DataFrame(
            [
                {"title": "甲公司债务违约", "content": "甲公司债务违约", "pub_time": "2025-01-01 10:00:00", "src": "新浪财经"},
                {"title": "甲公司流动性紧张", "content": "甲公司流动性紧张", "pub_time": "2025-01-02 10:00:00", "src": "新浪财经"},
                {"title": "乙公司融资受阻", "content": "乙公司融资受阻", "pub_time": "2025-01-03 10:00:00", "src": "新浪财经"},
                {"title": "丙公司裁员", "content": "丙公司裁员", "pub_time": "2025-01-04 10:00:00", "src": "新浪财经"},
            ]
        )

        sampled_df, _summary = MODULE.sample_balanced_dataframe(df, target_total=3, seed=42)
        titles = set(sampled_df["title"].tolist())

        self.assertIn("甲公司债务违约", titles)
        self.assertIn("乙公司融资受阻", titles)
        self.assertIn("丙公司裁员", titles)
        self.assertNotIn("甲公司流动性紧张", titles)

    def test_fetch_adaptive_records_failed_leaf_window_and_returns_empty(self):
        class DummyPro:
            def major_news(self, **_kwargs):
                raise Exception("查询数据失败")

        failed_windows = []
        result = MODULE.fetch_adaptive(
            DummyPro(),
            MODULE.datetime(2025, 1, 1, 0, 0, 0),
            MODULE.datetime(2025, 1, 1, 3, 0, 0),
            failed_windows=failed_windows,
        )

        self.assertTrue(result.empty)
        self.assertEqual(len(failed_windows), 1)


if __name__ == "__main__":
    unittest.main()
