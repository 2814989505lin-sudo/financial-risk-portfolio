import json
import os
import sys
import tempfile
import unittest

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from run_risk_validation import (  # noqa: E402
    build_h_analyze_only_feature_vector,
    build_h_text_only_feature_vector,
    build_labeled_rows,
    build_stats_only_feature_vector,
    build_threshold_grid,
    run_risk_validation,
)


class TestRiskValidation(unittest.TestCase):
    def test_feature_extractors_split_zr_correctly(self):
        row = {
            "risk_dim_order": ["credit_risk"],
            "dimension_details": {
                "credit_risk": {
                    "score": 0.8,
                    "event_count": 2,
                    "z_r": [1.0, 2.0, 3.0, 4.0],
                }
            },
        }
        self.assertEqual(build_h_text_only_feature_vector(row, risk_dims=["credit_risk"]), [1.0, 2.0])
        self.assertEqual(build_h_analyze_only_feature_vector(row, risk_dims=["credit_risk"]), [3.0, 4.0])
        self.assertEqual(build_stats_only_feature_vector(row, risk_dims=["credit_risk"]), [0.8, 2.0, 1.0, 2.0, 1.0])

    def test_build_threshold_grid_includes_default_threshold(self):
        grid = build_threshold_grid(0.2)
        self.assertIn(0.5, grid)
        self.assertGreater(len(grid), 0)

    def test_run_risk_validation_outputs_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            risk_vectors_path = os.path.join(tmp_dir, "risk_vectors.jsonl")
            labels_path = os.path.join(tmp_dir, "labels.jsonl")
            checkpoint_path = os.path.join(tmp_dir, "company_month_mlp.pt")
            output_dir = os.path.join(tmp_dir, "validation")

            risk_rows = []
            label_rows = []
            for company_idx, gid in enumerate(["gid_a", "gid_b", "gid_c", "gid_d"]):
                for month_idx, month in enumerate(["2026-01", "2026-02"]):
                    credit_zr = [float(company_idx + 1), 0.0, float(month_idx + 1), 0.0]
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
                                "credit_risk": {"score": float(company_idx % 2), "event_count": float(month_idx + 1), "z_r": credit_zr},
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
                            "y_credit": 1.0 if gid in {"gid_a", "gid_c"} and month == "2026-02" else 0.0,
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

            checkpoint = {
                "model_state_dict": {
                    "layers.0.weight": torch.zeros((4, 30)),
                    "layers.0.bias": torch.zeros(4),
                },
                "input_dim": 30,
                "hidden_dims": [],
                "output_dim": 4,
                "dropout": 0.0,
                "feature_mean": [0.0] * 30,
                "feature_std": [1.0] * 30,
            }
            torch.save(checkpoint, checkpoint_path)

            summary = run_risk_validation(
                risk_vectors_path=risk_vectors_path,
                labels_path=labels_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                seed=42,
                train_ratio=0.5,
                val_ratio=0.25,
                learning_rate=1e-3,
                weight_decay=1e-4,
                batch_size=4,
                max_epochs=5,
                patience=2,
                threshold_step=0.2,
            )

            self.assertIn("baseline_summary", summary)
            self.assertIn("threshold_analysis", summary)
            self.assertTrue(os.path.exists(summary["artifacts"]["summary"]))
            self.assertTrue(os.path.exists(summary["artifacts"]["baseline_runs"]))
            baseline_names = [row["baseline_name"] for row in summary["baseline_summary"]]
            self.assertIn("prior", baseline_names)
            self.assertIn("full_mlp", baseline_names)


if __name__ == "__main__":
    unittest.main()
