import os
import sys
import unittest
import json
import tempfile

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from run_docfee_news import (  # noqa: E402
    build_checkpoint_payload,
    build_failed_record,
    flush_contiguous_results,
    resolve_default_few_shot_path,
    resolve_batch_window,
    should_log_progress,
)


class TestRunDocfeeNews(unittest.TestCase):
    def test_should_log_progress_every_interval(self):
        self.assertFalse(should_log_progress(completed=0, total=100, every=10))
        self.assertFalse(should_log_progress(completed=9, total=100, every=10))
        self.assertTrue(should_log_progress(completed=10, total=100, every=10))
        self.assertTrue(should_log_progress(completed=100, total=100, every=10))

    def test_should_log_progress_always_logs_last_item(self):
        self.assertTrue(should_log_progress(completed=7, total=7, every=10))

    def test_resolve_batch_window_respects_limit_and_bounds(self):
        self.assertEqual(resolve_batch_window(total_records=6500, start_offset=0, max_records=2000), (0, 2000))
        self.assertEqual(resolve_batch_window(total_records=6500, start_offset=2000, max_records=4500), (2000, 6500))
        self.assertEqual(resolve_batch_window(total_records=6500, start_offset=7000, max_records=100), (6500, 6500))

    def test_build_checkpoint_payload_tracks_next_offset(self):
        payload = build_checkpoint_payload(
            input_path="in.jsonl",
            output_path="out.jsonl",
            total_input_records=6500,
            start_offset=0,
            end_offset=2000,
            processed_records=2000,
        )
        self.assertEqual(payload["next_offset"], 2000)
        self.assertEqual(payload["remaining_records"], 4500)
        self.assertFalse(payload["is_complete"])

    def test_build_checkpoint_payload_uses_processed_records_not_batch_end(self):
        payload = build_checkpoint_payload(
            input_path="in.jsonl",
            output_path="out.jsonl",
            total_input_records=2900,
            start_offset=0,
            end_offset=2900,
            processed_records=1472,
        )
        self.assertEqual(payload["next_offset"], 1472)
        self.assertEqual(payload["remaining_records"], 1428)
        self.assertFalse(payload["is_complete"])

    def test_build_failed_record_captures_error_metadata(self):
        record = {"article_id": "a1", "title": "标题", "url": "u"}
        exc = RuntimeError("bad response")
        failed = build_failed_record(idx=7, record=record, exc=exc, stage="initial_extract")
        self.assertEqual(failed["idx"], 7)
        self.assertEqual(failed["article_id"], "a1")
        self.assertEqual(failed["stage"], "initial_extract")
        self.assertEqual(failed["error_type"], "RuntimeError")
        self.assertIn("bad response", failed["error_message"])

    def test_flush_contiguous_results_splits_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.jsonl")
            failed_output_path = os.path.join(tmpdir, "out.failed.jsonl")
            next_idx, flushed_count, failed_rows = flush_contiguous_results(
                pending_results={
                    0: {"_result_kind": "success", "output_row": {"article_id": "a0"}},
                    1: {"_result_kind": "failed", "failed_row": {"idx": 1, "article_id": "a1"}},
                },
                next_write_idx=0,
                output_path=output_path,
                failed_output_path=failed_output_path,
            )
            self.assertEqual(next_idx, 2)
            self.assertEqual(flushed_count, 2)
            self.assertEqual(len(failed_rows), 1)
            with open(output_path, "r", encoding="utf-8") as f:
                success_rows = [json.loads(line) for line in f if line.strip()]
            with open(failed_output_path, "r", encoding="utf-8") as f:
                logged_failed_rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(success_rows), 1)
            self.assertEqual(success_rows[0]["article_id"], "a0")
            self.assertEqual(len(logged_failed_rows), 1)
            self.assertEqual(logged_failed_rows[0]["article_id"], "a1")
            self.assertEqual(logged_failed_rows[0]["status"], "pending_retry")

    def test_resolve_default_few_shot_path_prefers_sibling_generated_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "silver_pool.jsonl")
            auto_path = os.path.join(tmpdir, "event_gold_examples.jsonl")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write("")
            with open(auto_path, "w", encoding="utf-8") as f:
                f.write("")
            self.assertEqual(resolve_default_few_shot_path(input_path, None), auto_path)
            self.assertEqual(resolve_default_few_shot_path(input_path, "manual.json"), "manual.json")


if __name__ == "__main__":
    unittest.main()
