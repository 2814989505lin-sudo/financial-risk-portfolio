import json
import os
import sys
import tempfile
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from run_reliability_backtest import (  # noqa: E402
    calibration_bins,
    expected_calibration_error,
    run_reliability_backtest,
    split_rows_by_feature_month,
)


class TestReliabilityBacktest(unittest.TestCase):
    def test_split_rows_by_feature_month_is_temporal(self):
        rows = [
            {"feature_month": "2025-01"},
            {"feature_month": "2025-02"},
            {"feature_month": "2025-03"},
            {"feature_month": "2025-04"},
        ]
        calibration_months, prediction_months, calibration_rows, prediction_rows = split_rows_by_feature_month(rows, 0.5)
        self.assertEqual(calibration_months, ["2025-01", "2025-02"])
        self.assertEqual(prediction_months, ["2025-03", "2025-04"])
        self.assertEqual(len(calibration_rows), 2)
        self.assertEqual(len(prediction_rows), 2)

    def test_expected_calibration_error_is_non_negative(self):
        y_true = [0.0, 1.0, 0.0, 1.0]
        y_prob = [0.1, 0.9, 0.2, 0.8]
        bins = calibration_bins(y_true, y_prob, n_bins=2)
        self.assertEqual(len(bins), 2)
        ece = expected_calibration_error(y_true, y_prob, n_bins=2)
        self.assertGreaterEqual(ece, 0.0)

    def test_run_reliability_backtest_outputs_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            risk_vectors_path = os.path.join(tmp_dir, "risk_vectors.jsonl")
            labels_path = os.path.join(tmp_dir, "labels.jsonl")
            output_dir = os.path.join(tmp_dir, "backtest")

            risk_rows = []
            label_rows = []
            months = ["2025-01", "2025-02", "2025-03", "2025-04"]
            global_ids = ["gid_a", "gid_b", "gid_c", "gid_d", "gid_e", "gid_f"]
            for gid_idx, gid in enumerate(global_ids):
                for month_idx, month in enumerate(months):
                    base = float(gid_idx + 1)
                    credit_zr = [base, 0.0, float(month_idx + 1), 0.0]
                    zero_zr = [0.0, 0.0, 0.0, 0.0]
                    risk_rows.append(
                        {
                            "global_id": gid,
                            "canonical_company_name": gid,
                            "feature_month": month,
                            "risk_dim_order": ["credit_risk", "fraud_risk", "operational_risk", "market_risk"],
                            "vector_dim_per_risk_dim": 4,
                            "risk_vector": credit_zr + zero_zr + zero_zr + zero_zr,
                            "dimension_details": {
                                "credit_risk": {"score": 0.5 if gid_idx % 2 == 0 else 0.1, "event_count": 1.0, "z_r": credit_zr},
                                "fraud_risk": {"score": 0.0, "event_count": 0.0, "z_r": zero_zr},
                                "operational_risk": {"score": 0.0, "event_count": 0.0, "z_r": zero_zr},
                                "market_risk": {"score": 0.0, "event_count": 0.0, "z_r": zero_zr},
                            },
                        }
                    )
                    label_rows.append(
                        {
                            "global_id": gid,
                            "canonical_company_name": gid,
                            "feature_month": month,
                            "y_credit": 1.0 if gid in {"gid_a", "gid_c", "gid_e"} and month in {"2025-02", "2025-03", "2025-04"} else 0.0,
                            "y_fraud": 0.0,
                            "y_operational": 0.0,
                            "y_market": 0.0,
                        }
                    )

            with open(risk_vectors_path, "w", encoding="utf-8") as f:
                for row in risk_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with open(labels_path, "w", encoding="utf-8") as f:
                for row in label_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            summary = run_reliability_backtest(
                risk_vectors_path=risk_vectors_path,
                labels_path=labels_path,
                output_dir=output_dir,
                calibration_ratio=0.5,
                seed=42,
                train_ratio=0.6,
                val_ratio=0.2,
                hidden_dims=[8],
                dropout=0.1,
                learning_rate=1e-3,
                weight_decay=1e-4,
                batch_size=4,
                max_epochs=10,
                patience=2,
                bootstrap_samples=10,
            )

            self.assertEqual(summary["calibration_months"], ["2025-01", "2025-02"])
            self.assertEqual(summary["prediction_months"], ["2025-03", "2025-04"])
            self.assertTrue(os.path.exists(summary["artifacts"]["summary"]))
            self.assertTrue(os.path.exists(summary["artifacts"]["prediction_rows"]))
            self.assertTrue(os.path.exists(summary["artifacts"]["reliability_bins"]))
            self.assertTrue(os.path.exists(summary["artifacts"]["monthly_backtest"]))


if __name__ == "__main__":
    unittest.main()
