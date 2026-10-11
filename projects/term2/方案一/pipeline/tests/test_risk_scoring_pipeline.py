import os
import sys
import unittest
import json
import tempfile

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from aggregate_events import aggregate_events  # noqa: E402
from event_text_repr import embed_texts, parameter_free_multihead_self_attention  # noqa: E402
from risk_analysis_gen import build_risk_analysis_messages  # noqa: E402
from run_risk_scoring import run_resumable_jsonl_stage  # noqa: E402
from risk_scorer import (  # noqa: E402
    build_company_month_feature_vector,
    build_company_month_labels,
    build_company_month_outputs,
    score_group_vectors,
)


class TestRiskScoringPipeline(unittest.TestCase):
    def test_embed_texts_caps_batch_size_to_provider_limit(self):
        class FakeEmbeddingItem:
            def __init__(self, embedding):
                self.embedding = embedding

        class FakeEmbeddingsAPI:
            def __init__(self):
                self.batch_sizes = []

            def create(self, model, input, encoding_format):
                self.batch_sizes.append(len(input))
                data = [FakeEmbeddingItem([float(idx), float(idx + 1)]) for idx, _ in enumerate(input)]
                return type("Resp", (), {"data": data})()

        class FakeClient:
            def __init__(self):
                self.embeddings = FakeEmbeddingsAPI()

        client = FakeClient()
        texts = [f"text-{idx}" for idx in range(23)]
        embeddings = embed_texts(client=client, texts=texts, model="text-embedding-v4", batch_size=64)

        self.assertEqual(len(embeddings), 23)
        self.assertEqual(client.embeddings.batch_sizes, [10, 10, 3])

    def test_aggregate_events_filters_unresolved_and_groups_by_company_month_dim(self):
        article_rows = [
            {
                "article_id": "a1",
                "title": "公司A停工",
                "publish_time": "2026-03-10 09:00:00",
                "events": [
                    {
                        "event_type": "生产中断/停工停产",
                        "taxonomy": "operational_risk",
                        "event_polarity": "negative",
                        "confidence": 0.9,
                        "global_id": "gid_a",
                        "canonical_company_name": "公司A",
                        "alignment_status": "resolved",
                        "arguments": {"公司名称": "公司A"},
                        "evidence": ["发生停工"],
                    },
                    {
                        "event_type": "IT系统故障",
                        "taxonomy": "operational_risk",
                        "event_polarity": "negative",
                        "confidence": 0.9,
                        "global_id": "",
                        "canonical_company_name": "",
                        "alignment_status": "unresolved",
                    },
                ],
            },
            {
                "article_id": "a2",
                "title": "公司A复工",
                "publish_time": "2026-03-20 12:00:00",
                "events": [
                    {
                        "event_type": "复工复产",
                        "taxonomy": "operational_risk",
                        "event_polarity": "positive",
                        "confidence": 0.85,
                        "global_id": "gid_a",
                        "canonical_company_name": "公司A",
                        "alignment_status": "resolved",
                        "arguments": {"公司名称": "公司A"},
                        "evidence": ["恢复生产"],
                    }
                ],
            },
        ]

        groups = aggregate_events(article_rows, min_confidence=0.75)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["global_id"], "gid_a")
        self.assertEqual(groups[0]["feature_month"], "2026-03")
        self.assertEqual(groups[0]["risk_dim"], "operational_risk")
        self.assertEqual(groups[0]["event_count"], 2)
        self.assertEqual(groups[0]["negative_event_count"], 1)
        self.assertEqual(groups[0]["positive_event_count"], 1)

    def test_parameter_free_multihead_self_attention_preserves_shape(self):
        event_matrix = torch.randn(3, 8)
        attended = parameter_free_multihead_self_attention(event_matrix, num_heads=4)
        self.assertEqual(tuple(attended.shape), (3, 8))

    def test_build_risk_analysis_messages_renders_key_fields(self):
        template = "风险维度: {{RISK_DIM}}\n公司: {{CANONICAL_COMPANY_NAME}}\n{{EVENT_SUMMARY_BLOCK}}"
        group = {
            "risk_dim": "credit_risk",
            "canonical_company_name": "示例公司",
            "events": [
                {
                    "event_type": "债务违约",
                    "event_polarity": "negative",
                    "confidence": 0.9,
                    "title": "示例公司债务违约",
                    "publish_time": "2026-03-01 10:00:00",
                    "arguments": {"公司名称": "示例公司"},
                    "evidence": ["发生债务违约"],
                }
            ],
        }
        messages = build_risk_analysis_messages(template, group)
        self.assertEqual(len(messages), 1)
        self.assertIn("credit_risk", messages[0]["content"])
        self.assertIn("示例公司", messages[0]["content"])
        self.assertIn("债务违约", messages[0]["content"])

    def test_score_group_vectors_and_build_company_month_outputs(self):
        aggregated_rows = [
            {
                "group_id": "gid_a__2026-03__credit_risk",
                "global_id": "gid_a",
                "canonical_company_name": "公司A",
                "feature_month": "2026-03",
                "risk_dim": "credit_risk",
                "event_count": 2,
                "negative_event_count": 2,
                "positive_event_count": 0,
                "events": [
                    {"confidence": 0.9, "event_polarity": "negative"},
                    {"confidence": 0.8, "event_polarity": "negative"},
                ],
            }
        ]
        event_text_rows = [
            {
                "group_id": "gid_a__2026-03__credit_risk",
                "embedding_dim": 4,
                "h_text": [0.1, 0.2, 0.3, 0.4],
            }
        ]
        analysis_rows = [
            {
                "group_id": "gid_a__2026-03__credit_risk",
                "h_analyze": [0.5, 0.6, 0.7, 0.8],
            }
        ]

        group_vector_rows = score_group_vectors(
            aggregated_rows=aggregated_rows,
            event_text_rows=event_text_rows,
            analysis_rows=analysis_rows,
            checkpoint_path=None,
        )
        self.assertEqual(len(group_vector_rows), 1)
        self.assertEqual(len(group_vector_rows[0]["z_r"]), 8)
        self.assertGreater(group_vector_rows[0]["score"], 0.0)

        risk_vector_rows, score_rows = build_company_month_outputs(group_vector_rows, risk_dims=["credit_risk", "fraud_risk"])
        self.assertEqual(len(risk_vector_rows), 1)
        self.assertEqual(score_rows[0]["credit_risk"], group_vector_rows[0]["score"])
        self.assertEqual(score_rows[0]["fraud_risk"], 0.0)

    def test_build_company_month_feature_vector_appends_stats(self):
        row = {
            "risk_vector": [0.1, 0.2, 0.3],
            "dimension_details": {
                "credit_risk": {"score": 0.8, "event_count": 2},
                "fraud_risk": {"score": 0.0, "event_count": 0},
            },
        }
        feature_vector = build_company_month_feature_vector(row, risk_dims=["credit_risk", "fraud_risk"])
        self.assertEqual(feature_vector[:3], [0.1, 0.2, 0.3])
        self.assertEqual(feature_vector[3:], [0.8, 2.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0])

    def test_build_company_month_labels_looks_ahead_three_months(self):
        risk_vector_rows = [
            {
                "global_id": "gid_a",
                "canonical_company_name": "公司A",
                "feature_month": "2026-01",
                "risk_dim_order": ["credit_risk", "fraud_risk"],
                "risk_vector": [0.1, 0.2],
                "dimension_details": {
                    "credit_risk": {"score": 0.4, "event_count": 1, "z_r": [0.1, 0.2]},
                    "fraud_risk": {"score": 0.0, "event_count": 0, "z_r": [0.0, 0.0]},
                },
            },
            {
                "global_id": "gid_a",
                "canonical_company_name": "公司A",
                "feature_month": "2026-04",
                "risk_dim_order": ["credit_risk", "fraud_risk"],
                "risk_vector": [0.3, 0.4],
                "dimension_details": {
                    "credit_risk": {"score": 0.7, "event_count": 1, "z_r": [0.3, 0.4]},
                    "fraud_risk": {"score": 0.0, "event_count": 0, "z_r": [0.0, 0.0]},
                },
            },
            {
                "global_id": "gid_a",
                "canonical_company_name": "公司A",
                "feature_month": "2026-05",
                "risk_dim_order": ["credit_risk", "fraud_risk"],
                "risk_vector": [0.5, 0.6],
                "dimension_details": {
                    "credit_risk": {"score": 0.0, "event_count": 0, "z_r": [0.0, 0.0]},
                    "fraud_risk": {"score": 0.0, "event_count": 0, "z_r": [0.0, 0.0]},
                },
            },
        ]
        aggregated_rows = [
            {
                "global_id": "gid_a",
                "feature_month": "2026-02",
                "risk_dim": "credit_risk",
                "negative_event_count": 1,
            }
        ]

        labels = build_company_month_labels(
            risk_vector_rows=risk_vector_rows,
            aggregated_rows=aggregated_rows,
            horizon_months=3,
            risk_dims=["credit_risk", "fraud_risk"],
        )
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]["feature_month"], "2026-01")
        self.assertEqual(labels[0]["y_credit"], 1.0)
        self.assertEqual(labels[0]["y_fraud"], 0.0)

    def test_run_resumable_jsonl_stage_resumes_from_checkpoint(self):
        rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "stage.jsonl")
            checkpoint_path = os.path.join(tmpdir, "stage.checkpoint.json")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "a", "done": True}, ensure_ascii=False) + "\n")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stage": "demo_stage",
                        "total_rows": 3,
                        "next_offset": 1,
                        "remaining_rows": 2,
                        "is_complete": False,
                        "output_path": output_path,
                    },
                    f,
                    ensure_ascii=False,
                )

            outputs = run_resumable_jsonl_stage(
                stage_name="demo_stage",
                rows=rows,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                process_fn=lambda row: {"id": row["id"], "done": True},
                resume_long_ops=True,
                progress_log_every=1,
            )

            self.assertEqual([row["id"] for row in outputs], ["a", "b", "c"])
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["next_offset"], 3)
            self.assertTrue(payload["is_complete"])


if __name__ == "__main__":
    unittest.main()
