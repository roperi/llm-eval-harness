from __future__ import annotations

import unittest

from llm_eval_harness.golden import SpeakerIdentityGoldenRecord, SummaryGoldenRecord
from llm_eval_harness.scorers.speaker_id import (
    build_speaker_id_scorers,
    contract_validator,
    duplicate_label_detection,
    host_label_correctness,
)
from llm_eval_harness.scorers.summary import (
    build_summary_scorers,
    coverage_scorer,
    hallucinated_name_scorer,
    quote_attribution_scorer,
    required_fields_scorer,
)


class SummaryScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.golden = SummaryGoldenRecord(
            episode_date="2026-01-01",
            title="Test",
            transcript_path="fake.txt",
            expected_fields_present=["summary_text"],
            expected_spellings={"John Smith": "John Smith"},
            min_coverage_terms=["Smith", "interview"],
            forbidden_terms=["Smithh"],
            known_hosts=["John Smith", "Jane Doe"],
        )

    def test_required_fields_passes(self) -> None:
        output = {"summary_text": "John Smith hosted the show."}
        result = required_fields_scorer(output, self.golden)
        self.assertTrue(result.passed)

    def test_required_fields_fails_on_empty(self) -> None:
        result = required_fields_scorer({"summary_text": ""}, self.golden)
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "missing_field")

    def test_hallucinated_name_passes(self) -> None:
        result = hallucinated_name_scorer(
            {"summary_text": "John Smith hosted the show."}, self.golden
        )
        self.assertTrue(result.passed)

    def test_hallucinated_name_fails_on_misspelling(self) -> None:
        result = hallucinated_name_scorer(
            {"summary_text": "John Smithh hosted the show."}, self.golden
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "hallucinated_name")

    def test_hallucinated_name_fails_on_unexpected_name(self) -> None:
        result = hallucinated_name_scorer(
            {"summary_text": "Bob Wilson joined John Smith."}, self.golden
        )
        self.assertFalse(result.passed)

    def test_quote_attribution_passes(self) -> None:
        result = quote_attribution_scorer(
            {"summary_text": 'John said "hello everyone" and then Jane said "welcome."'},
            self.golden,
        )
        self.assertTrue(result.passed)

    def test_quote_attribution_fails_on_unattributed(self) -> None:
        result = quote_attribution_scorer(
            {"summary_text": 'The episode featured "hello everyone."'},
            self.golden,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "unattributed_quote")

    def test_coverage_passes(self) -> None:
        output = {"summary_text": "Smith and Doe conducted an interview."}
        result = coverage_scorer(output, self.golden)
        self.assertTrue(result.passed)

    def test_coverage_fails_on_missing_term(self) -> None:
        result = coverage_scorer({"summary_text": "The show featured a caller."}, self.golden)
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "low_coverage")

    def test_all_summary_scorers_have_names(self) -> None:
        for scorer in build_summary_scorers():
            r = scorer({"summary_text": "test"}, self.golden)
            self.assertTrue(r.name)


class SpeakerIdentityScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.golden = SpeakerIdentityGoldenRecord(
            episode_date="2026-01-01",
            diarized_turns=[{"speaker": "SPEAKER_00", "text": "hello", "start": 0.0, "end": 1.0}],
            expected_host_labels=["SPEAKER_00"],
            expected_unresolved_count_max=0,
            expected_known_speakers={"SPEAKER_00": "John Smith"},
            known_hosts=["John Smith", "Jane Doe"],
        )

    def test_contract_validator_passes(self) -> None:
        result = contract_validator(
            {
                "metadata": {
                    "mappings": [
                        {
                            "channel": "SPEAKER_00",
                            "resolved_label": "John Smith",
                            "status": "resolved",
                        }
                    ]
                }
            },
            self.golden,
        )
        self.assertTrue(result.passed)

    def test_contract_validator_fails_on_unresolved(self) -> None:
        result = contract_validator(
            {
                "metadata": {
                    "mappings": [
                        {
                            "channel": "SPEAKER_00",
                            "resolved_label": "Unknown Speaker 1",
                            "status": "unresolved",
                        }
                    ]
                }
            },
            self.golden,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "unresolved_speaker")

    def test_host_label_correctness_passes(self) -> None:
        result = host_label_correctness(
            {
                "metadata": {
                    "mappings": [
                        {
                            "channel": "SPEAKER_00",
                            "resolved_label": "John Smith",
                            "status": "resolved",
                        }
                    ]
                }
            },
            self.golden,
        )
        self.assertTrue(result.passed)

    def test_host_label_correctness_fails(self) -> None:
        result = host_label_correctness(
            {
                "metadata": {
                    "mappings": [
                        {
                            "channel": "SPEAKER_00",
                            "resolved_label": "John Smith",
                            "status": "unresolved",
                        }
                    ]
                }
            },
            self.golden,
        )
        self.assertFalse(result.passed)

    def test_duplicate_label_detects_duplicate(self) -> None:
        result = duplicate_label_detection(
            {
                "metadata": {
                    "mappings": [
                        {"channel": "SPEAKER_00", "resolved_label": "John Smith"},
                        {"channel": "SPEAKER_01", "resolved_label": "John Smith"},
                    ]
                }
            },
            self.golden,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "duplicate_label")

    def test_all_speaker_id_scorers_have_names(self) -> None:
        for scorer in build_speaker_id_scorers():
            r = scorer({"metadata": {"mappings": []}}, self.golden)
            self.assertTrue(r.name)


if __name__ == "__main__":
    unittest.main()
