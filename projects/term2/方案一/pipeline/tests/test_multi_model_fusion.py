import os
import sys
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from multi_model_fusion import fuse_events  # noqa: E402


class TestMultiModelFusion(unittest.TestCase):
    def test_fuse_events_merges_matching_events(self):
        model_results = [
            (
                "gpt-4.1-mini-2025-04-14",
                [
                    {
                        "event_type": "债务违约",
                        "taxonomy": "credit_risk",
                        "event_polarity": "negative",
                        "arguments": {"公司名称": "威马汽车股份有限公司", "违约金额": "10亿元"},
                        "evidence": ["证据1"],
                        "confidence": 0.9,
                        "canonical_company_name": "威马汽车",
                        "matched_alias": "威马汽车股份有限公司",
                        "alignment_status": "resolved",
                    }
                ],
            ),
            (
                "MiniMax-M2.1",
                [
                    {
                        "event_type": "债务违约",
                        "taxonomy": "credit_risk",
                        "event_polarity": "negative",
                        "arguments": {"公司名称": "威马汽车", "违约金额": "10亿元"},
                        "evidence": ["证据2"],
                        "confidence": 0.8,
                        "canonical_company_name": "威马汽车",
                        "matched_alias": "威马汽车",
                        "alignment_status": "resolved",
                    }
                ],
            ),
            (
                "glm-4-flash-250414",
                [
                    {
                        "event_type": "债务违约",
                        "taxonomy": "credit_risk",
                        "event_polarity": "negative",
                        "arguments": {"公司名称": "威马汽车", "违约金额": "9亿元"},
                        "evidence": ["证据3"],
                        "confidence": 0.7,
                        "canonical_company_name": "威马汽车",
                        "matched_alias": "威马汽车",
                        "alignment_status": "resolved",
                    }
                ],
            ),
        ]

        fused = fuse_events(model_results)
        self.assertEqual(len(fused), 1)
        event = fused[0]
        self.assertEqual(event["model_agreement"], 3)
        self.assertEqual(event["annotation_models"], ["gpt-4.1-mini-2025-04-14", "MiniMax-M2.1", "glm-4-flash-250414"])
        self.assertAlmostEqual(event["confidence"], 0.8, places=6)
        self.assertEqual(event["arguments"]["违约金额"], "10亿元")
        self.assertEqual(set(event["evidence"]), {"证据1", "证据2", "证据3"})

    def test_fuse_events_keeps_original_confidence_for_solo_event(self):
        model_results = [
            (
                "gpt-4.1-mini-2025-04-14",
                [
                    {
                        "event_type": "财务造假",
                        "taxonomy": "fraud_risk",
                        "event_polarity": "negative",
                        "arguments": {"公司名称": "甲公司"},
                        "confidence": 0.9,
                        "alignment_status": "resolved",
                    }
                ],
            ),
            ("MiniMax-M2.1", []),
            ("glm-4-flash-250414", []),
        ]

        fused = fuse_events(model_results)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["model_agreement"], 1)
        self.assertAlmostEqual(fused[0]["confidence"], 0.9, places=6)


if __name__ == "__main__":
    unittest.main()
