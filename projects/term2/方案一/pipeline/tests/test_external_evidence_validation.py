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

from run_external_evidence_validation import (  # noqa: E402
    classify_support_status,
    match_external_evidence,
    run_external_evidence_validation,
)


class TestExternalEvidenceValidation(unittest.TestCase):
    def test_classify_support_status(self):
        increase_case = {"dominant_change_direction": "increase"}
        decrease_case = {"dominant_change_direction": "decrease"}

        self.assertEqual(classify_support_status(increase_case, []), "no_external_evidence")
        self.assertEqual(classify_support_status(increase_case, [{"polarity": "negative"}]), "supported")
        self.assertEqual(classify_support_status(increase_case, [{"polarity": "positive"}]), "contradicted")
        self.assertEqual(
            classify_support_status(increase_case, [{"polarity": "negative"}, {"polarity": "positive"}]),
            "mixed",
        )
        self.assertEqual(classify_support_status(decrease_case, [{"polarity": "positive"}]), "supported")

    def test_match_external_evidence_filters_company_month_and_risk_dim(self):
        case = {
            "global_id": "gid_a",
            "canonical_company_name": "测试公司",
            "to_month": "2025-02",
            "dominant_risk_dim": "credit_risk",
        }
        evidence_rows = [
            {"global_id": "gid_a", "canonical_company_name": "测试公司", "event_month": "2025-02", "risk_dim": "credit_risk"},
            {"global_id": "gid_a", "canonical_company_name": "测试公司", "event_month": "2025-03", "risk_dim": "credit_risk"},
            {"global_id": "gid_a", "canonical_company_name": "测试公司", "event_month": "2025-04", "risk_dim": "credit_risk"},
            {"global_id": "gid_a", "canonical_company_name": "测试公司", "event_month": "2025-02", "risk_dim": "market_risk"},
        ]
        matched = match_external_evidence(case, evidence_rows, post_window_months=1)
        self.assertEqual(len(matched), 2)

    def test_run_external_evidence_validation_outputs_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            risk_scores_path = os.path.join(tmp_dir, "risk_scores.csv")
            community_evolution_path = os.path.join(tmp_dir, "community_evolution.jsonl")
            aggregated_events_path = os.path.join(tmp_dir, "aggregated_events.jsonl")
            external_evidence_path = os.path.join(tmp_dir, "external_evidence.jsonl")
            output_dir = os.path.join(tmp_dir, "validation")

            with open(risk_scores_path, "w", encoding="utf-8", newline="") as f:
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
                writer.writerow(
                    {
                        "global_id": "gid_a",
                        "canonical_company_name": "测试公司",
                        "feature_month": "2025-01",
                        "credit_risk": "0.10",
                        "fraud_risk": "0.01",
                        "operational_risk": "0.01",
                        "market_risk": "0.01",
                    }
                )
                writer.writerow(
                    {
                        "global_id": "gid_a",
                        "canonical_company_name": "测试公司",
                        "feature_month": "2025-02",
                        "credit_risk": "0.55",
                        "fraud_risk": "0.01",
                        "operational_risk": "0.01",
                        "market_risk": "0.01",
                    }
                )

            with open(community_evolution_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "from_month": "2025-01",
                            "to_month": "2025-02",
                            "company_migrations": [
                                {"global_id": "gid_a", "from_community_id": 0, "to_community_id": 1}
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            with open(aggregated_events_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "group_id": "gid_a__2025-02__credit_risk",
                            "global_id": "gid_a",
                            "canonical_company_name": "测试公司",
                            "feature_month": "2025-02",
                            "risk_dim": "credit_risk",
                            "event_count": 1,
                            "negative_event_count": 1,
                            "positive_event_count": 0,
                            "events": [
                                {
                                    "title": "测试公司评级下调",
                                    "event_type": "主体信用评级下调",
                                    "evidence": ["评级由AA下调至A"],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            with open(external_evidence_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "global_id": "gid_a",
                            "canonical_company_name": "测试公司",
                            "event_date": "2025-02-15",
                            "event_month": "2025-02",
                            "risk_dim": "credit_risk",
                            "polarity": "negative",
                            "evidence_type": "rating_downgrade",
                            "source_name": "测试来源",
                            "source_title": "测试公司评级下调",
                            "source_url": "https://example.com/a",
                            "evidence_summary": "评级下调",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            summary = run_external_evidence_validation(
                risk_scores_path=risk_scores_path,
                community_evolution_path=community_evolution_path,
                aggregated_events_path=aggregated_events_path,
                external_evidence_path=external_evidence_path,
                output_dir=output_dir,
                min_score_delta=0.2,
                post_window_months=0,
                max_month_gap=1,
            )

            self.assertEqual(summary["metrics"]["case_count"], 1)
            self.assertEqual(summary["metrics"]["supported_count"], 1)
            self.assertTrue(os.path.exists(summary["artifacts"]["candidate_cases"]))
            self.assertTrue(os.path.exists(summary["artifacts"]["validated_cases"]))
            self.assertTrue(os.path.exists(summary["artifacts"]["query_sheet"]))
            self.assertTrue(os.path.exists(summary["artifacts"]["summary"]))


if __name__ == "__main__":
    unittest.main()
