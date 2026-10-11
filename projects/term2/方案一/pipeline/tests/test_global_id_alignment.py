import os
import sys
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from global_id_alignment import (  # noqa: E402
    build_alias_map,
    build_record_alignment_context,
    extract_company_mentions_from_text,
    finalize_alignment_fields,
    make_global_id,
    resolve_company_identity,
)


class TestGlobalIdAlignment(unittest.TestCase):
    def setUp(self):
        company_rows = [
            {
                "canonical_company_name": "平安银行",
                "aliases": ["平安银行股份有限公司", "深圳发展银行"],
            },
            {
                "canonical_company_name": "万科A",
                "aliases": ["万科企业股份有限公司"],
            },
            {
                "canonical_company_name": "招商证券",
                "aliases": ["招商证券股份有限公司"],
            },
        ]
        self.alias_map = build_alias_map(company_rows)

    def test_build_record_alignment_context_collects_candidates(self):
        context = build_record_alignment_context(["平安银行", "招商证券"], self.alias_map)
        global_ids = sorted(item["global_id"] for item in context["candidate_companies"])
        self.assertEqual(
            global_ids,
            sorted([make_global_id("平安银行"), make_global_id("招商证券")]),
        )

    def test_finalize_alignment_fields_resolves_unique_company(self):
        event = {
            "event_type": "债务违约",
            "arguments": {"公司名称": "平安银行股份有限公司"},
            "global_id": "",
            "canonical_company_name": "",
            "matched_alias": "",
            "alignment_status": "",
        }
        finalized = finalize_alignment_fields(event, ["平安银行股份有限公司"], self.alias_map)
        self.assertEqual(finalized["global_id"], make_global_id("平安银行"))
        self.assertEqual(finalized["canonical_company_name"], "平安银行")
        self.assertEqual(finalized["alignment_status"], "resolved")

    def test_resolve_company_identity_falls_back_to_hash(self):
        resolved = resolve_company_identity(
            company_name="威马汽车科技集团有限公司",
            matched_alias="威马汽车",
            alias_map={},
        )
        self.assertEqual(resolved["global_id"], make_global_id("威马汽车科技"))
        self.assertEqual(resolved["canonical_company_name"], "威马汽车科技")
        self.assertEqual(resolved["alignment_status"], "resolved")

    def test_finalize_alignment_fields_marks_unresolved(self):
        event = {
            "event_type": "债务违约",
            "arguments": {"公司名称": ""},
            "global_id": "",
            "canonical_company_name": "",
            "matched_alias": "",
            "alignment_status": "",
        }
        finalized = finalize_alignment_fields(event, [], self.alias_map)
        self.assertEqual(finalized["global_id"], "")
        self.assertEqual(finalized["alignment_status"], "unresolved")

    def test_extract_company_mentions_from_text(self):
        mentions = extract_company_mentions_from_text(
            title="平安银行获监管关注",
            text="招商证券发布研报，认为平安银行资产质量稳定。",
            alias_map=self.alias_map,
        )
        self.assertIn("平安银行", mentions)
        self.assertIn("招商证券", mentions)


if __name__ == "__main__":
    unittest.main()
