import os
import sys
import unittest


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from run_mlp_ablation import (  # noqa: E402
    binary_average_precision,
    binary_f1,
    binary_roc_auc,
    build_labeled_rows,
    split_rows_by_global_id,
)


class TestMlpAblation(unittest.TestCase):
    def test_build_labeled_rows_keeps_only_overlapping_company_months(self):
        risk_vector_rows = [
            {
                "global_id": "gid_a",
                "feature_month": "2026-01",
                "canonical_company_name": "公司A",
                "risk_vector": [0.1, 0.2],
            },
            {
                "global_id": "gid_b",
                "feature_month": "2026-01",
                "canonical_company_name": "公司B",
                "risk_vector": [0.3, 0.4],
            },
        ]
        labels_map = {
            ("gid_a", "2026-01"): {
                "y_credit": 1.0,
                "y_fraud": 0.0,
                "y_operational": 0.0,
                "y_market": 1.0,
            }
        }

        rows = build_labeled_rows(risk_vector_rows, labels_map)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["global_id"], "gid_a")
        self.assertEqual(rows[0]["y_market"], 1.0)

    def test_split_rows_by_global_id_keeps_same_global_id_in_one_split(self):
        rows = [
            {"global_id": "gid_a", "feature_month": "2026-01", "risk_vector": [0.1], "y_credit": 1, "y_fraud": 0, "y_operational": 0, "y_market": 0},
            {"global_id": "gid_a", "feature_month": "2026-02", "risk_vector": [0.2], "y_credit": 0, "y_fraud": 0, "y_operational": 1, "y_market": 0},
            {"global_id": "gid_b", "feature_month": "2026-01", "risk_vector": [0.3], "y_credit": 0, "y_fraud": 1, "y_operational": 0, "y_market": 0},
            {"global_id": "gid_c", "feature_month": "2026-01", "risk_vector": [0.4], "y_credit": 0, "y_fraud": 0, "y_operational": 0, "y_market": 1},
        ]

        train_rows, val_rows, test_rows = split_rows_by_global_id(rows, seed=42)
        train_ids = {row["global_id"] for row in train_rows}
        val_ids = {row["global_id"] for row in val_rows}
        test_ids = {row["global_id"] for row in test_rows}

        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))
        self.assertEqual(train_ids | val_ids | test_ids, {"gid_a", "gid_b", "gid_c"})

    def test_split_rows_by_global_id_rejects_too_few_companies(self):
        rows = [
            {"global_id": "gid_a", "feature_month": "2026-01", "risk_vector": [0.1], "y_credit": 1, "y_fraud": 0, "y_operational": 0, "y_market": 0},
            {"global_id": "gid_b", "feature_month": "2026-01", "risk_vector": [0.2], "y_credit": 0, "y_fraud": 1, "y_operational": 0, "y_market": 0},
        ]
        with self.assertRaises(ValueError):
            split_rows_by_global_id(rows, seed=42)

    def test_binary_metrics_return_expected_bounds(self):
        y_true = [1.0, 0.0, 1.0, 0.0]
        y_score = [0.9, 0.2, 0.8, 0.1]

        auroc = binary_roc_auc(y_true, y_score)
        auprc = binary_average_precision(y_true, y_score)
        f1 = binary_f1(y_true, y_score, threshold=0.5)

        self.assertIsNotNone(auroc)
        self.assertIsNotNone(auprc)
        self.assertIsNotNone(f1)
        self.assertGreaterEqual(float(auroc), 0.0)
        self.assertLessEqual(float(auroc), 1.0)
        self.assertGreaterEqual(float(auprc), 0.0)
        self.assertLessEqual(float(auprc), 1.0)
        self.assertGreaterEqual(float(f1), 0.0)
        self.assertLessEqual(float(f1), 1.0)


if __name__ == "__main__":
    unittest.main()
