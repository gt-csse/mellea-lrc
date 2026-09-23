"""Rebuild the primary v33 root-identity false-negative audit without API calls.

Join annotation files by the artifact's recorded source path. Two serialized
artifact filenames omit a trailing hyphen that remains in the annotation and
source filenames, so joining by artifact stem would silently omit gold roots.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _gold_key(row: dict) -> tuple[str, int, int]:
    span = row["locator"]
    return row["kind"], span["start"], span["end"]


def _artifact_key(citation: dict) -> tuple[str, int, int] | None:
    if citation.get("citation_id") != citation.get("root_id"):
        return None
    source = citation.get("fields", {})
    span = source.get("locator_span")
    if not isinstance(span, dict):
        return None
    return source["kind"], span["start"], span["end"]


def _stage_summaries(trace: list[dict]) -> dict[str, dict]:
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for event in trace:
        by_stage[event["stage"]].append(event)
    result = {}
    for stage, events in sorted(by_stage.items()):
        counts = Counter(event["outcome"] for event in events)
        candidate_counts = [
            validation["candidate_count"]
            for event in events
            if isinstance((validation := event.get("details", {}).get("validation")), dict)
            and isinstance(validation.get("candidate_count"), int)
        ]
        result[stage] = {
            "outcomes": dict(sorted(counts.items())),
            "last_outcome": events[-1]["outcome"],
            "last_message": events[-1].get("message"),
            "largest_candidate_count": max(candidate_counts) if candidate_counts else None,
        }
    return result


def _provider_status(trace: list[dict]) -> dict:
    search_events = [event for event in trace if event["stage"] == "root_body_corroboration_search"]
    resolution_events = [event for event in trace if event["stage"] == "root_body_corroboration_resolution"]
    search_errors = sorted(
        {
            str(error)
            for event in search_events
            if (error := event.get("details", {}).get("validation", {}).get("error"))
        }
    )
    fetches = [fetch for event in resolution_events for fetch in event.get("details", {}).get("fetches", [])]
    fetch_errors = sorted({str(fetch["error"]) for fetch in fetches if fetch.get("error")})
    return {
        "body_search_429": any("429" in error for error in search_errors),
        "body_fetch_429": any(
            fetch.get("http_status") == 429 or "429" in str(fetch.get("error", "")) for fetch in fetches
        ),
        "body_search_errors": search_errors,
        "body_fetch_errors": fetch_errors,
        "body_fetch_outcomes": dict(sorted(Counter(fetch.get("outcome") for fetch in fetches).items())),
    }


def build_report(annotations: Path, artifacts: Path) -> dict:
    annotations = annotations.resolve()
    artifacts = artifacts.resolve()
    misses = []
    gold_count: Counter[str] = Counter()
    outcomes: Counter[tuple[str, str]] = Counter()
    document_count = 0
    for artifact_path in sorted(artifacts.glob("*.json")):
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        source_path = Path(payload["source_metadata"]["path"])
        annotation_path = annotations / f"{source_path.stem}.jsonl"
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        document_count += 1
        gold = {}
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("unit") != "citation" or row.get("is_root") is not True:
                continue
            label = row.get("validation", {}).get("identity", {}).get("label")
            if label not in {"CORRECT_IDENTITY", "WRONG_IDENTITY"}:
                continue
            key = _gold_key(row)
            if key in gold:
                raise ValueError(f"Duplicate annotated root in {annotation_path}: {key}")
            gold[key] = row
            gold_count[label] += 1
        predicted = {}
        for citation in payload["citations"]:
            key = _artifact_key(citation)
            if key is not None:
                if key in predicted:
                    raise ValueError(f"Duplicate artifact root in {artifact_path}: {key}")
                predicted[key] = citation
        for key, row in sorted(gold.items(), key=lambda item: item[0][1:]):
            label = row["validation"]["identity"]["label"]
            citation = predicted.get(key)
            outcome = citation["judgements"]["identity"]["outcome"] if citation else "not_reached"
            outcomes[(label, outcome)] += 1
            if label != "CORRECT_IDENTITY" or outcome == "resolved":
                continue
            trace = citation.get("trace", []) if citation else []
            misses.append(
                {
                    "annotation_id": row["id"],
                    "document": annotation_path.stem,
                    "annotation_path": str(annotation_path),
                    "artifact_path": str(artifact_path),
                    "source_path": str(source_path),
                    "citation_kind": row["kind"],
                    "root_kind": row["identifier"]["kind"],
                    "source_span": row.get("cited_as"),
                    "locator": row["locator"],
                    "final_outcome": outcome,
                    "stage_outcomes": _stage_summaries(trace),
                    "provider_status": _provider_status(trace),
                }
            )
    misses.sort(key=lambda row: (row["document"], row["locator"]["start"], row["annotation_id"]))
    return {
        "artifact_type": "primary_root_identity_recall_audit",
        "artifact_version": "33-primary-third-party-body-reviewed-v1",
        "annotations": str(annotations),
        "artifacts": str(artifacts),
        "join": "artifact source_metadata.path stem to annotation filename, then citation kind and locator span",
        "document_count": document_count,
        "gold_roots": dict(sorted(gold_count.items())),
        "outcomes_by_gold": [
            {"gold_label": label, "outcome": outcome, "count": count}
            for (label, outcome), count in sorted(outcomes.items())
        ],
        "missed_correct_roots": len(misses),
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.annotations, args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.output}: {report['missed_correct_roots']} missed correct roots")


if __name__ == "__main__":
    main()
