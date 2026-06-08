from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GoldenSetError(Exception):
    pass


SUPPORTED_VERSIONS = {1}


@dataclass
class SummaryGoldenRecord:
    episode_date: str
    title: str
    transcript_path: str
    expected_fields_present: list[str]
    expected_spellings: dict[str, str] | None = None
    min_coverage_terms: list[str] | None = None
    forbidden_terms: list[str] | None = None
    known_hosts: list[str] | None = None


@dataclass
class SpeakerIdentityGoldenRecord:
    episode_date: str
    diarized_turns: list[dict[str, Any]]
    expected_host_labels: list[str]
    expected_unresolved_count_max: int = 0
    expected_known_speakers: dict[str, str] | None = None
    known_hosts: list[str] | None = None


@dataclass
class GoldenSet:
    version: int
    summary_records: list[SummaryGoldenRecord]
    speaker_identity_records: list[SpeakerIdentityGoldenRecord]


def load_golden_set(path: str) -> GoldenSet:
    path_obj = Path(path)
    if not path_obj.exists():
        raise GoldenSetError(f"Golden set not found: {path}")
    if not path_obj.is_file():
        raise GoldenSetError(f"Golden set path is not a file: {path}")

    try:
        raw = json.loads(path_obj.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GoldenSetError(f"Invalid JSON in golden set: {exc}") from exc

    if not isinstance(raw, dict):
        raise GoldenSetError("Golden set must be a JSON object")
    version = raw.get("version")
    if not isinstance(version, int):
        raise GoldenSetError("Golden set must have an integer 'version' field")
    if version not in SUPPORTED_VERSIONS:
        raise GoldenSetError(
            f"Unsupported golden set version {version}; supported: {SUPPORTED_VERSIONS}"
        )

    summary_records = []
    for entry in raw.get("summary_episodes", []):
        if not isinstance(entry, dict):
            raise GoldenSetError("Each summary episode entry must be a JSON object")
        if not entry.get("episode_date"):
            raise GoldenSetError("Each summary episode entry must have a non-empty 'episode_date'")
        summary_records.append(
            SummaryGoldenRecord(
                episode_date=str(entry["episode_date"]),
                title=str(entry.get("title", "")),
                transcript_path=str(entry.get("transcript_path", "")),
                expected_fields_present=list(entry.get("expected_fields_present", [])),
                expected_spellings=entry.get("expected_spellings"),
                min_coverage_terms=entry.get("min_coverage_terms"),
                forbidden_terms=entry.get("forbidden_terms"),
                known_hosts=entry.get("known_hosts"),
            )
        )

    speaker_records = []
    for entry in raw.get("speaker_identity_episodes", []):
        if not isinstance(entry, dict):
            raise GoldenSetError("Each speaker identity episode entry must be a JSON object")
        if not entry.get("episode_date"):
            raise GoldenSetError(
                "Each speaker identity episode entry must have a non-empty 'episode_date'"
            )
        speaker_records.append(
            SpeakerIdentityGoldenRecord(
                episode_date=str(entry["episode_date"]),
                diarized_turns=list(entry.get("diarized_turns", [])),
                expected_host_labels=list(entry.get("expected_host_labels", [])),
                expected_unresolved_count_max=int(entry.get("expected_unresolved_count_max", 0)),
                expected_known_speakers=entry.get("expected_known_speakers"),
                known_hosts=entry.get("known_hosts"),
            )
        )

    return GoldenSet(
        version=version,
        summary_records=summary_records,
        speaker_identity_records=speaker_records,
    )


__all__ = [
    "GoldenSet",
    "GoldenSetError",
    "SpeakerIdentityGoldenRecord",
    "SummaryGoldenRecord",
    "load_golden_set",
]
