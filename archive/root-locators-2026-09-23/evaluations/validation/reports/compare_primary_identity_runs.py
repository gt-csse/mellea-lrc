"""Compare saved primary root-identity runs without provider or model calls.

Example:
    uv run python -m evaluations.validation.reports.compare_primary_identity_runs \
      --annotations ../mellea-lrc-datasets/primary/documents \
      --before ../mellea-lrc-datasets/run-artifacts/33-primary-third-party-body-reviewed-v1/root_body_corroboration_resolution \
      --after data/run-artifacts/35-primary-body-retry-main-v1/root_body_corroboration_resolution \
      --output evaluations/validation/reports/primary-v33-v35-comparison.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from evaluations.validation.score_root_identity import _gold_roots, _root_outcomes, score_root_identity

_TRANSIENT_STATUS = re.compile(r"(?<!\d)(?:429|500|502|503|504)(?!\d)")
_TIMEOUT = re.compile(r"timed? out|timeout", re.IGNORECASE)


def _root_records(artifacts: Path) -> dict[tuple[str, str, int, int], dict]:
    records = {}
    for path in sorted(artifacts.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_path = payload.get("source_metadata", {}).get("path")
        document = Path(source_path).stem if source_path else path.stem
        for citation in payload.get("citations", []):
            if citation.get("root_id") != citation.get("citation_id"):
                continue
            source = citation.get("fields", {})
            span = source.get("locator_span")
            kind = source.get("kind")
            if not isinstance(span, dict) or not isinstance(kind, str):
                continue
            key = (document, kind, int(span["start"]), int(span["end"]))
            if key in records:
                raise ValueError(f"Duplicate serialized root: {key}")
            records[key] = citation
    return records


def _transient_kind(error: object, status: object = None) -> str | None:
    if status is not None and str(status) in {"429", "500", "502", "503", "504"}:
        return str(status)
    if error is None:
        return None
    message = str(error)
    match = _TRANSIENT_STATUS.search(message)
    if match:
        return match.group()
    return "timeout" if _TIMEOUT.search(message) else None


def _provider_failures(records: dict[tuple[str, str, int, int], dict]) -> dict:
    search = Counter()
    fetch = Counter()
    fetch_terminal = Counter()
    search_roots = set()
    fetch_roots = set()
    any_roots = set()
    search_429_roots = set()
    fetch_429_roots = set()
    for key, citation in records.items():
        for event in citation.get("trace", []):
            details = event.get("details", {})
            if event.get("stage") == "root_body_corroboration_search":
                validation = details.get("validation", {})
                attempts = validation.get("search_attempts", [])
                kinds = [
                    kind
                    for attempt in attempts
                    if (kind := _transient_kind(attempt.get("error"))) is not None
                ]
                if not kinds:
                    fallback = _transient_kind(validation.get("error"))
                    if fallback:
                        kinds.append(fallback)
                for kind in kinds:
                    search[kind] += 1
                    search_roots.add(key)
                    any_roots.add(key)
                    if kind == "429":
                        search_429_roots.add(key)
            elif event.get("stage") == "root_body_corroboration_resolution":
                for item in details.get("fetches", []):
                    kind = _transient_kind(item.get("error"), item.get("http_status"))
                    if kind:
                        fetch[kind] += 1
                        if item.get("outcome") == "failed":
                            fetch_terminal[kind] += 1
                        fetch_roots.add(key)
                        any_roots.add(key)
                        if kind == "429":
                            fetch_429_roots.add(key)
    return {
        "search_error_events": dict(sorted(search.items())),
        "fetch_error_attempts": dict(sorted(fetch.items())),
        "fetch_terminal_failures": dict(sorted(fetch_terminal.items())),
        "affected_roots": {
            "search": len(search_roots),
            "fetch": len(fetch_roots),
            "either": len(any_roots),
        },
        "affected_roots_429": {
            "search": len(search_429_roots),
            "fetch": len(fetch_429_roots),
            "either": len(search_429_roots | fetch_429_roots),
        },
        "scope": "All serialized roots, including roots without an identity gold label.",
    }


def _summary(score: dict) -> dict:
    all_root = score["scores"]["all_root"]
    admission = all_root["positive_admission"]
    return {
        "labeled_roots": all_root["labeled_roots"],
        "reached_roots": all_root["stage_coverage"]["reached"],
        "gold_correct_roots": admission["gold"],
        "true_admissions": admission["tp"],
        "false_admissions": admission["fp"],
        "missed_correct_roots": admission["fn"],
        "recall": admission["recall"],
        "precision": admission["precision"],
        "f1": admission["f1"],
    }


def compare_runs(annotations: Path, before: Path, after: Path) -> dict:
    before_score = score_root_identity(annotations=annotations, artifacts=before)
    after_score = score_root_identity(annotations=annotations, artifacts=after)
    before_outcomes, before_docs = _root_outcomes(before)
    after_outcomes, after_docs = _root_outcomes(after)
    if before_docs != after_docs:
        raise ValueError("Artifact runs cover different source documents")
    gold = _gold_roots(annotations, document_names=before_docs)
    before_records = _root_records(before)
    after_records = _root_records(after)
    before_summary, after_summary = _summary(before_score), _summary(after_score)
    changes = []
    for key in sorted(set(before_outcomes) | set(after_outcomes)):
        before_outcome = before_outcomes.get(key, "not_reached")
        after_outcome = after_outcomes.get(key, "not_reached")
        if before_outcome == after_outcome:
            continue
        document, kind, start, end = key
        citation = after_records.get(key) or before_records.get(key) or {}
        changes.append(
            {
                "document": document,
                "kind": kind,
                "locator_span": {"start": start, "end": end},
                "locator_text": citation.get("fields", {}).get("matched_text"),
                "gold_label": gold.get(key),
                "before": before_outcome,
                "after": after_outcome,
            }
        )
    return {
        "artifact_type": "primary_root_identity_saved_run_comparison",
        "annotations": str(annotations.resolve()),
        "before_artifacts": str(before.resolve()),
        "after_artifacts": str(after.resolve()),
        "document_count": len(before_docs),
        "interpretation": "Observed saved-run difference only; repeated model judgments and provider availability may both change outcomes. Neither run imposed a retrospective evidence date.",
        "score": {
            "before": before_summary,
            "after": after_summary,
            "delta": {
                name: after_summary[name] - before_summary[name]
                for name in (
                    "true_admissions",
                    "false_admissions",
                    "missed_correct_roots",
                    "recall",
                    "precision",
                    "f1",
                )
            },
        },
        "provider_transient_failures": {
            "before": _provider_failures(before_records),
            "after": _provider_failures(after_records),
            "definition": "429, 500, 502, 503, 504, or timeout recorded in root body search/fetch traces; repeated trace entries remain separate attempts.",
        },
        "changed_root_outcomes": changes,
        "changed_annotated_root_count": sum(row["gold_label"] is not None for row in changes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.annotations, args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
