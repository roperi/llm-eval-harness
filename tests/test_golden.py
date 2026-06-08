from __future__ import annotations

import json
import os
import tempfile
import unittest

from llm_eval_harness.golden import GoldenSetError, load_golden_set


class GoldenLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self.golden_path = os.path.join(self.fixtures_dir, "golden_set_v1.json")

    def test_load_golden_succeeds(self) -> None:
        gs = load_golden_set(self.golden_path)
        self.assertEqual(gs.version, 1)
        self.assertTrue(len(gs.summary_records) > 0)
        self.assertEqual(gs.summary_records[0].episode_date, "2026-01-01")
        self.assertIn("John Smith", gs.summary_records[0].expected_spellings or {})

    def test_load_has_speaker_records(self) -> None:
        gs = load_golden_set(self.golden_path)
        self.assertTrue(len(gs.speaker_identity_records) > 0)
        self.assertEqual(gs.speaker_identity_records[0].expected_host_labels, ["SPEAKER_00"])

    def test_raises_on_missing_file(self) -> None:
        with self.assertRaises(GoldenSetError):
            load_golden_set("/nonexistent/path.json")

    def test_raises_on_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            tmp_path = f.name
        try:
            with self.assertRaises(GoldenSetError):
                load_golden_set(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_raises_on_missing_version(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            tmp_path = f.name
        try:
            with self.assertRaises(GoldenSetError):
                load_golden_set(tmp_path)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
