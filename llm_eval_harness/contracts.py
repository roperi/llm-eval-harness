from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ScorerResult:
    name: str
    passed: bool
    details: str | None = None
    failures: list[str] | None = None
    failure_category: str | None = None


class Scorer(Protocol):
    def __call__(self, output: dict, golden: Any) -> ScorerResult: ...


class JudgeModel(Protocol):
    def __call__(self, rubric: str) -> float: ...


@dataclass
class JudgeDimension:
    name: str
    score: float
    max_score: float = 5.0
    rationale: str | None = None

    @property
    def passed(self, threshold: float = 0.6) -> bool:
        return self.score / self.max_score >= threshold


@dataclass
class JudgeReport:
    dimensions: list[JudgeDimension]
    overall_score: float
    max_score: float
    passed: bool
    details: str | None = None
    cost: float | None = None


@dataclass
class StageTrace:
    stage: str
    model: str | None
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    output_length_chars: int | None = None
    validation_flags: dict[str, bool] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model": self.model,
            "latency_seconds": self.latency_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "output_length_chars": self.output_length_chars,
            "validation_flags": dict(self.validation_flags),
            "error": self.error,
        }


@dataclass
class EvalReport:
    scorer_results: list[ScorerResult]
    judge_report: JudgeReport | None
    aggregate_pass_rate: float
    total_scorers: int
    passed_scorers: int
    failure_taxonomy: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_rate": self.aggregate_pass_rate,
            "total_scorers": self.total_scorers,
            "passed_scorers": self.passed_scorers,
            "failure_taxonomy": dict(self.failure_taxonomy),
            "judge_report": None,
            "golden_set_version": 1,
            "ran_at": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


__all__ = [
    "EvalReport",
    "JudgeDimension",
    "JudgeModel",
    "JudgeReport",
    "Scorer",
    "ScorerResult",
    "StageTrace",
]
