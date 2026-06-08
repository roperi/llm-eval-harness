from __future__ import annotations

import unittest

from llm_eval_harness.contracts import EvalReport, JudgeDimension, ScorerResult, StageTrace


class ScorerResultTests(unittest.TestCase):
    def test_passed_result(self) -> None:
        r = ScorerResult(name="test", passed=True)
        self.assertTrue(r.passed)
        self.assertIsNone(r.failures)

    def test_failed_result_with_failures(self) -> None:
        r = ScorerResult(
            name="test", passed=False, failures=["bad"], failure_category="missing_field"
        )
        self.assertFalse(r.passed)
        self.assertEqual(r.failures, ["bad"])
        self.assertEqual(r.failure_category, "missing_field")


class StageTraceTests(unittest.TestCase):
    def test_trace_to_dict(self) -> None:
        t = StageTrace(
            stage="show_summary",
            model="gpt-4",
            latency_seconds=1.5,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            output_length_chars=200,
            validation_flags={"valid": True},
        )
        d = t.to_dict()
        self.assertEqual(d["stage"], "show_summary")
        self.assertEqual(d["model"], "gpt-4")
        self.assertEqual(d["latency_seconds"], 1.5)
        self.assertEqual(d["prompt_tokens"], 100)
        self.assertTrue(d["validation_flags"]["valid"])


class JudgeDimensionTests(unittest.TestCase):
    def test_passed_above_threshold(self) -> None:
        d = JudgeDimension(name="coherence", score=4.0, max_score=5.0)
        self.assertTrue(d.passed)

    def test_failed_below_threshold(self) -> None:
        d = JudgeDimension(name="coherence", score=2.0, max_score=5.0)
        self.assertFalse(d.passed)


class EvalReportTests(unittest.TestCase):
    def test_to_dict_shape(self) -> None:
        report = EvalReport(
            scorer_results=[ScorerResult(name="t1", passed=True)],
            judge_report=None,
            aggregate_pass_rate=1.0,
            total_scorers=1,
            passed_scorers=1,
            failure_taxonomy={},
        )
        d = report.to_dict()
        self.assertEqual(d["pass_rate"], 1.0)
        self.assertEqual(d["total_scorers"], 1)
        self.assertEqual(d["passed_scorers"], 1)
        self.assertEqual(d["failure_taxonomy"], {})
        self.assertIsNone(d["judge_report"])
        self.assertIn("ran_at", d)


if __name__ == "__main__":
    unittest.main()
