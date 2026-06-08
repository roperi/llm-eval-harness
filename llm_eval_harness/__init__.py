from __future__ import annotations

from typing import Any

from llm_eval_harness.contracts import EvalReport, ScorerResult
from llm_eval_harness.golden import load_golden_set
from llm_eval_harness.scorers import get_deterministic_scorers


def run_evals(
    golden_set_path: str,
    episode_metrics: list[dict[str, Any]],
    use_judge: bool = False,
    judge_model: Any = None,
) -> EvalReport:
    golden_records = load_golden_set(golden_set_path)
    golden_by_date = {r.episode_date: r for r in golden_records}

    scorers = get_deterministic_scorers()
    scorer_results: list[ScorerResult] = []
    failures_by_category: dict[str, int] = {}

    for ep in episode_metrics:
        date = ep.get("date") or ep.get("episode_date") or ""
        golden = golden_by_date.get(date)
        if golden is None:
            continue

        for scorer in scorers:
            result = scorer(ep, golden)
            scorer_results.append(result)
            if not result.passed and result.failure_category:
                failures_by_category[result.failure_category] = (
                    failures_by_category.get(result.failure_category, 0) + 1
                )

    total = len(scorer_results)
    passed = sum(1 for r in scorer_results if r.passed)
    pass_rate = passed / total if total > 0 else 1.0

    judge_report = None
    if use_judge and judge_model is not None:
        from llm_eval_harness.judge import run_judge

        judge_report = run_judge(episode_metrics, golden_records, judge_model=judge_model)
        if judge_report and not judge_report.passed:
            failures_by_category["judge_failure"] = failures_by_category.get("judge_failure", 0) + 1

    return EvalReport(
        scorer_results=scorer_results,
        judge_report=judge_report,
        aggregate_pass_rate=pass_rate,
        total_scorers=total,
        passed_scorers=passed,
        failure_taxonomy=failures_by_category,
    )


__all__ = ["run_evals", "EvalReport"]
