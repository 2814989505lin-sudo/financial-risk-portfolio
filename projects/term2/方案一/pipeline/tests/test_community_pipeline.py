import csv
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from community_detection import detect_leiden_communities  # noqa: E402
from community_evolution import track_community_evolution  # noqa: E402
from community_graph import build_knn_adjacency  # noqa: E402
from community_similarity import compute_cosine_similarity_matrix  # noqa: E402
from run_community import parse_args, run_community_pipeline  # noqa: E402


class TestCommunityPipeline(unittest.TestCase):
    def test_parse_args_defaults_to_k_3(self):
        with patch.object(sys, "argv", ["run_community.py", "--input", "dummy.csv"]):
            args = parse_args()
        self.assertEqual(args.k, 3)
        self.assertEqual(args.resolution, 1.0)

    def test_compute_cosine_similarity_matrix_is_symmetric(self):
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
        similarity = compute_cosine_similarity_matrix(vectors)
        self.assertTrue(np.allclose(similarity, similarity.T))
        self.assertTrue(np.allclose(np.diag(similarity), np.ones(3)))
        self.assertTrue(np.all((similarity >= -1.0) & (similarity <= 1.0)))

    def test_build_knn_adjacency_keeps_positive_edges_only(self):
        similarity = np.asarray(
            [
                [1.0, 0.9, -0.1],
                [0.9, 1.0, 0.3],
                [-0.1, 0.3, 1.0],
            ],
            dtype=float,
        )
        adjacency, stats = build_knn_adjacency(similarity, k=1)
        self.assertEqual(stats["effective_k"], 1)
        self.assertTrue(np.allclose(adjacency, adjacency.T))
        self.assertEqual(float(adjacency[0, 2]), 0.0)
        self.assertGreater(float(adjacency[0, 1]), 0.0)

    def test_detect_leiden_communities_returns_membership(self):
        adjacency = np.asarray(
            [
                [0.0, 0.9, 0.0, 0.0],
                [0.9, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.8],
                [0.0, 0.0, 0.8, 0.0],
            ],
            dtype=float,
        )
        result = detect_leiden_communities(adjacency, resolution=1.0)
        self.assertEqual(len(result["membership"]), 4)
        self.assertGreaterEqual(result["community_count"], 1)

    def test_track_community_evolution_handles_single_month(self):
        rows = [
            {"global_id": "gid_a", "feature_month": "2026-03", "community_id": 0},
            {"global_id": "gid_b", "feature_month": "2026-03", "community_id": 1},
        ]
        self.assertEqual(track_community_evolution(rows), [])

    def test_run_community_pipeline_on_score_input(self):
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
                writer.writerow(
                    {
                        "global_id": "gid_a",
                        "canonical_company_name": "公司A",
                        "feature_month": "2026-03",
                        "credit_risk": 0.9,
                        "fraud_risk": 0.1,
                        "operational_risk": 0.0,
                        "market_risk": 0.0,
                    }
                )
                writer.writerow(
                    {
                        "global_id": "gid_b",
                        "canonical_company_name": "公司B",
                        "feature_month": "2026-03",
                        "credit_risk": 0.88,
                        "fraud_risk": 0.12,
                        "operational_risk": 0.0,
                        "market_risk": 0.0,
                    }
                )
                writer.writerow(
                    {
                        "global_id": "gid_c",
                        "canonical_company_name": "公司C",
                        "feature_month": "2026-03",
                        "credit_risk": 0.0,
                        "fraud_risk": 0.0,
                        "operational_risk": 0.0,
                        "market_risk": 0.0,
                    }
                )

            summary = run_community_pipeline(
                input_path=input_path,
                input_mode="score",
                output_dir=tmp_dir,
                k=2,
                resolution=1.0,
            )
            self.assertEqual(summary["assignment_count"], 3)

            assignment_path = summary["artifacts"]["community_assignments"]
            with open(assignment_path, "r", encoding="utf-8") as f:
                assignments = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(assignments), 3)
            zero_rows = [row for row in assignments if row["is_zero_vector"]]
            self.assertEqual(len(zero_rows), 1)
            self.assertEqual(zero_rows[0]["community_id"], -1)

    def test_run_community_pipeline_resumes_completed_month(self):
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
                writer.writerow(
                    {
                        "global_id": "gid_a",
                        "canonical_company_name": "公司A",
                        "feature_month": "2026-03",
                        "credit_risk": 0.9,
                        "fraud_risk": 0.1,
                        "operational_risk": 0.0,
                        "market_risk": 0.0,
                    }
                )
                writer.writerow(
                    {
                        "global_id": "gid_b",
                        "canonical_company_name": "公司B",
                        "feature_month": "2026-04",
                        "credit_risk": 0.8,
                        "fraud_risk": 0.2,
                        "operational_risk": 0.0,
                        "market_risk": 0.0,
                    }
                )

            month_assignments_dir = os.path.join(tmp_dir, "community_assignments_by_month")
            month_summaries_dir = os.path.join(tmp_dir, "community_month_summaries")
            os.makedirs(month_assignments_dir, exist_ok=True)
            os.makedirs(month_summaries_dir, exist_ok=True)

            with open(os.path.join(month_assignments_dir, "2026-03.jsonl"), "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "global_id": "gid_a",
                            "canonical_company_name": "公司A",
                            "feature_month": "2026-03",
                            "community_id": 0,
                            "node_index": 0,
                            "vector_norm": 1.0,
                            "is_zero_vector": False,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            with open(os.path.join(month_summaries_dir, "2026-03.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "feature_month": "2026-03",
                        "company_count": 1,
                        "community_count": 1,
                        "zero_vector_count": 0,
                        "modularity": 0.0,
                        "similarity_matrix_path": "",
                        "graph_stats": {"node_count": 1, "effective_k": 0, "edge_count": 0},
                        "communities": [{"community_id": 0, "company_count": 1, "sample_companies": ["公司A"]}],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            checkpoint_path = os.path.join(tmp_dir, "community.checkpoint.json")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "completed_months": ["2026-03"],
                        "completed_month_count": 1,
                        "remaining_month_count": 1,
                        "is_complete": False,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            original_compute = run_community_pipeline.__globals__["compute_cosine_similarity_matrix"]
            call_count = {"value": 0}

            def wrapped_compute(vectors):
                call_count["value"] += 1
                return original_compute(vectors)

            with patch("run_community.compute_cosine_similarity_matrix", side_effect=wrapped_compute):
                summary = run_community_pipeline(
                    input_path=input_path,
                    input_mode="score",
                    output_dir=tmp_dir,
                    k=2,
                    resolution=1.0,
                    checkpoint_path=checkpoint_path,
                    resume=True,
                    progress_log_every=1,
                )

            self.assertEqual(call_count["value"], 1)
            self.assertEqual(summary["month_count"], 2)
            self.assertEqual(summary["assignment_count"], 2)


if __name__ == "__main__":
    unittest.main()
