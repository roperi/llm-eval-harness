from __future__ import annotations

from typing import Any

from llm_eval_harness.contracts import JudgeDimension, JudgeModel, JudgeReport
from llm_eval_harness.golden import SummaryGoldenRecord


def run_judge(
    episode_metrics: list[dict[str, Any]],
    golden_records: list[Any],
    judge_model: JudgeModel | None = None,
) -> JudgeReport | None:
    if judge_model is None:
        return None

    summary_records = [r for r in golden_records if isinstance(r, SummaryGoldenRecord)]

    dimensions: list[JudgeDimension] = []
    for golden in summary_records:
        matching = [
            ep
            for ep in episode_metrics
            if (ep.get("date") or ep.get("episode_date")) == golden.episode_date
        ]
        if not matching:
            continue
        ep = matching[0]
        summary = str(ep.get("summary_text") or ep.get("summary") or "")
        if not summary.strip():
            continue

        rubric = _build_rubric(summary, golden)
        score = judge_model(rubric)
        dimensions.append(
            JudgeDimension(
                name=f"summary_coherence_{golden.episode_date}",
                score=score,
                rationale=rubric,
            )
        )

    if not dimensions:
        return JudgeReport(
            dimensions=[],
            overall_score=0.0,
            max_score=5.0,
            passed=True,
            details="No episodes with summaries to judge",
        )

    overall = sum(d.score for d in dimensions) / len(dimensions)
    passed = overall / 5.0 >= 0.6
    return JudgeReport(
        dimensions=dimensions,
        overall_score=overall,
        max_score=5.0,
        passed=passed,
    )


def _build_rubric(summary: str, golden: SummaryGoldenRecord) -> str:
    return (
        f"Summary: {summary[:500]}\n"
        f"Known hosts: {golden.known_hosts}\n"
        f"Expected spellings: {golden.expected_spellings}\n"
    )


__all__ = ["run_judge"]
