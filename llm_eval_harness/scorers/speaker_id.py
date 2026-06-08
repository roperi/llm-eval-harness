from __future__ import annotations

from typing import Any

from llm_eval_harness.contracts import Scorer, ScorerResult
from llm_eval_harness.golden import SpeakerIdentityGoldenRecord


def _get_mappings(output: dict) -> list[dict[str, Any]]:
    metadata = output.get("speaker_identity_metadata") or output.get("metadata") or {}
    if isinstance(metadata, dict):
        return list(metadata.get("mappings") or [])
    return []


def _get_resolved_label(mapping: dict) -> str:
    return str(mapping.get("resolved_label") or mapping.get("speaker_name") or "")


def contract_validator(output: dict, golden: SpeakerIdentityGoldenRecord) -> ScorerResult:
    mappings = _get_mappings(output)
    unresolved: list[str] = []
    for m in mappings:
        label = _get_resolved_label(m)
        status = m.get("status", "")
        if not label or status == "unresolved":
            unresolved.append(str(m.get("channel") or m.get("speaker") or "?"))
    return ScorerResult(
        name="contract_validator",
        passed=len(unresolved) <= golden.expected_unresolved_count_max,
        failures=unresolved if unresolved else None,
        failure_category="unresolved_speaker" if unresolved else None,
    )


def host_label_correctness(output: dict, golden: SpeakerIdentityGoldenRecord) -> ScorerResult:
    mappings = _get_mappings(output)
    expected_hosts = set(golden.expected_host_labels)
    resolved_hosts: set[str] = set()
    for m in mappings:
        if m.get("status") == "resolved":
            resolved_hosts.add(str(m.get("channel") or m.get("speaker") or ""))

    failures: list[str] = []
    for host in expected_hosts:
        if host not in resolved_hosts:
            failures.append(f"Expected host channel '{host}' not resolved as host")
    return ScorerResult(
        name="host_label_correctness",
        passed=len(failures) == 0,
        failures=failures if failures else None,
        failure_category="host_mismatch" if failures else None,
    )


def duplicate_label_detection(output: dict, golden: SpeakerIdentityGoldenRecord) -> ScorerResult:
    mappings = _get_mappings(output)
    label_to_channels: dict[str, list[str]] = {}
    for m in mappings:
        label = _get_resolved_label(m)
        channel = str(m.get("channel") or m.get("speaker") or "")
        if label:
            label_to_channels.setdefault(label, []).append(channel)

    failures: list[str] = []
    for label, channels in label_to_channels.items():
        if len(channels) > 1:
            common_prefixes = ["Unknown Speaker", "Unresolved", "Speaker_", "SPEAKER_"]
            is_placeholder = any(label.startswith(p) for p in common_prefixes)
            if not is_placeholder:
                failures.append(f"Label '{label}' assigned to multiple channels: {channels}")
    return ScorerResult(
        name="duplicate_label_detection",
        passed=len(failures) == 0,
        failures=failures if failures else None,
        failure_category="duplicate_label" if failures else None,
    )


def build_speaker_id_scorers() -> list[Scorer]:
    return [
        contract_validator,
        host_label_correctness,
        duplicate_label_detection,
    ]


__all__ = [
    "build_speaker_id_scorers",
    "contract_validator",
    "duplicate_label_detection",
    "host_label_correctness",
]
