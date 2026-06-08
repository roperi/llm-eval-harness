from __future__ import annotations

import re

from llm_eval_harness.contracts import Scorer, ScorerResult
from llm_eval_harness.golden import SummaryGoldenRecord


def _get_summary_text(output: dict) -> str:
    summary_text = ""
    if isinstance(output, dict):
        summary_text = str(output.get("summary_text") or output.get("summary") or "")
    return summary_text


def required_fields_scorer(output: dict, golden: SummaryGoldenRecord) -> ScorerResult:
    summary_text = _get_summary_text(output)
    missing: list[str] = []
    for field in golden.expected_fields_present:
        if not summary_text.strip():
            missing.append(field)
    return ScorerResult(
        name="required_fields",
        passed=len(missing) == 0,
        failures=missing if missing else None,
        failure_category="missing_field" if missing else None,
    )


HALLUCINATED_NAME_PATTERNS: list[tuple[str, str]] = [
    (r"\bJohn\s+Smithh\b", "John Smith"),
    (r"\bJon\s+Smith\b", "John Smith"),
    (r"\bJhon\s+Smith\b", "John Smith"),
    (r"\bJanne\s+Doe\b", "Jane Doe"),
    (r"\bJain\s+Doe\b", "Jane Doe"),
]


def hallucinated_name_scorer(output: dict, golden: SummaryGoldenRecord) -> ScorerResult:
    summary_text = _get_summary_text(output)
    failures: list[str] = []
    for pattern, canonical in HALLUCINATED_NAME_PATTERNS:
        if re.search(pattern, summary_text, re.IGNORECASE):
            failures.append(f"Found misspelling matching '{pattern}'; expected '{canonical}'")

    known_hosts = set(golden.known_hosts or [])
    if known_hosts:
        found_names = set(re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", summary_text))
        unexpected = found_names - known_hosts
        for name in unexpected:
            if name not in (golden.expected_spellings or {}):
                failures.append(f"Potential hallucinated name '{name}' not in known_hosts")

    return ScorerResult(
        name="hallucinated_name",
        passed=len(failures) == 0,
        failures=failures if failures else None,
        failure_category="hallucinated_name" if failures else None,
    )


def quote_attribution_scorer(output: dict, golden: SummaryGoldenRecord) -> ScorerResult:
    summary_text = _get_summary_text(output)
    quote_pattern = re.compile(r'"([^"]+)"')
    quotes = quote_pattern.findall(summary_text)
    failures: list[str] = []
    for quote in quotes:
        idx = summary_text.index(f'"{quote}"')
        before = summary_text[:idx]
        lines_before = before.strip().split("\n")
        context = lines_before[-1] if lines_before else ""
        if not re.search(r"\b(said|says|added|replied|asked|told|noted)\b", context, re.IGNORECASE):
            failures.append(f'Unattributed quote: "{quote}"')
    return ScorerResult(
        name="quote_attribution",
        passed=len(failures) == 0,
        failures=failures if failures else None,
        failure_category="unattributed_quote" if failures else None,
    )


def coverage_scorer(output: dict, golden: SummaryGoldenRecord) -> ScorerResult:
    summary_text = _get_summary_text(output)
    failures: list[str] = []
    for term in golden.min_coverage_terms or []:
        if term.lower() not in summary_text.lower():
            failures.append(f"Missing required term: '{term}'")
    return ScorerResult(
        name="coverage",
        passed=len(failures) == 0,
        failures=failures if failures else None,
        failure_category="low_coverage" if failures else None,
    )


def build_summary_scorers() -> list[Scorer]:
    return [
        required_fields_scorer,
        hallucinated_name_scorer,
        quote_attribution_scorer,
        coverage_scorer,
    ]


__all__ = [
    "build_summary_scorers",
    "coverage_scorer",
    "hallucinated_name_scorer",
    "quote_attribution_scorer",
    "required_fields_scorer",
]
