import csv
import os
import sys
import tempfile
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from run_community_sensitivity import run_sensitivity_grid  # noqa: E402


class TestCommunitySensitivity(unittest.TestCase):
    def test_run_sensitivity_grid_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "risk_scores.csv")
            with open(input_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "global_id",
                        "canonical_company_name",
                        "feature_month",
                        "credit_risk",
                        "fraud_risk",
                        "operational_risk",
                        "market_risk",
                    ],
                )
                writer.writeheader()
                for row in [
                    ("gid_a", "公司A", "2026-03", 0.9, 0.1, 0.0, 0.0),
                    ("gid_b", "公司B", "2026-03", 0.88, 0.12, 0.0, 0.0),
                    ("gid_c", "公司C", "2026-04", 0.1, 0.0, 0.9, 0.0),
                    ("gid_d", "公司D", "2026-04", 0.12, 0.0, 0.88, 0.0),
                ]:
                    writer.writerow(
                        {
                            "global_id": row[0],
                            "canonical_company_name": row[1],
                            "feature_month": row[2],
                            "credit_risk": row[3],
                            "fraud_risk": row[4],
                            "operational_risk": row[5],
                            "market_risk": row[6],
                        }
                    )

            output_dir = os.path.join(tmp_dir, "sensitivity")
            summary = run_sensitivity_grid(
                input_path=input_path,
                input_mode="score",
                output_dir=output_dir,
                ks=[2],
                resolutions=[1.0, 1.5],
                progress_log_every=1000,
            )
            self.assertEqual(summary["grid"]["run_count"], 2)
            self.assertTrue(os.path.exists(summary["artifacts"]["runs"]))
            self.assertTrue(os.path.exists(summary["artifacts"]["summary"]))
            self.assertIsNotNone(summary["best_by_avg_modularity"])


if __name__ == "__main__":
    unittest.main()
