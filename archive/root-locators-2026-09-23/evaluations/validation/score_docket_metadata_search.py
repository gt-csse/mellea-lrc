"""Score provider metadata-discovery recall for annotated docket roots.

This is deliberately a retrieval metric, not identity precision. A metadata
search is useful when its persisted candidate list contains the independently
annotated docket record. Later field checks decide whether the filing's name,
court, date, and docket fields make that candidate the right identity.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

Provider = Literal["courtlistener", "govinfo"]

_STAGES: dict[Provider, str] = {
    "courtlistener": "courtlistener_docket_search",
    "govinfo": "govinfo_docket_search",
}
_IDS: dict[Provider, str] = {
    "courtlistener": "docket_id",
    "govinfo": "govinfo_package_id",
}


def score_docket_metadata_search(
    *,
    annotations: Path,
    artifacts: Path,
    provider: Provider,
) -> dict[str, object]:
    """Score whether provider candidates contain annotated independent records."""
    gold = _gold_targets(annotations, provider=provider)
    rows: list[dict[str, object]] = []
    for path in sorted(artifacts.glob("*.json")):
        rows.extend(_artifact_rows(path, provider=provider))

    predicted = {(row["document"], row["start"], row["end"]): row for row in rows}
    gold_keys = set(gold)
    reached = gold_keys & set(predicted)
    target_hits = {
        key for key in reached if gold[key] in {str(value) for value in predicted[key]["candidate_ids"]}
    }
    missed = sorted(reached - target_hits)
    not_reached = sorted(gold_keys - reached)
    attempt_kinds = Counter(
        attempt["kind"] for row in rows for attempt in row["attempts"] if isinstance(attempt.get("kind"), str)
    )
    outcomes = Counter(str(row["outcome"]) for row in rows)
    source_name_status = Counter(str(row["term_plan_outcome"]) for row in rows)

    return {
        "artifact_type": "docket_metadata_search_score",
        "provider": provider,
        "stage": _STAGES[provider],
        "annotations": str(annotations),
        "artifacts": str(artifacts),
        "metric_definition": (
            "recall of annotated provider docket/package records among roots that reached the "
            "metadata-discovery stage; no identity decision or candidate precision is implied"
        ),
        "target_coverage": {
            "annotated_provider_targets": len(gold_keys),
            "reached": len(reached),
            "not_reached": len(not_reached),
        },
        "candidate_retrieval": {
            "targets_retrieved": len(target_hits),
            "targets_missed": len(missed),
            "recall": _ratio(len(target_hits), len(reached)),
        },
        "stage_roots": len(rows),
        "stage_outcomes": dict(sorted(outcomes.items())),
        "term_plan_outcomes": dict(sorted(source_name_status.items())),
        "attempt_kinds": dict(sorted(attempt_kinds.items())),
        "missed_targets": [
            {
                "document": key[0],
                "locator_span": {"start": key[1], "end": key[2]},
                "target_id": gold[key],
                "case_name_terms": predicted[key]["terms"],
                "attempts": predicted[key]["attempts"],
                "candidate_ids": predicted[key]["candidate_ids"],
            }
            for key in missed
        ],
        "not_reached_targets": [
            {
                "document": key[0],
                "locator_span": {"start": key[1], "end": key[2]},
                "target_id": gold[key],
            }
            for key in not_reached
        ],
        "occurrences": sorted(rows, key=lambda row: (str(row["document"]), int(row["start"]))),
    }


def _gold_targets(annotations: Path, *, provider: Provider) -> dict[tuple[str, int, int], str]:
    targets: dict[tuple[str, int, int], str] = {}
    for path in sorted(annotations.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("unit") != "citation" or row.get("kind") != "DocketCitation" or not row.get("is_root"):
                continue
            locator = row.get("locator")
            identity = row.get("validation", {}).get("identity", {})
            evidence = identity.get("evidence", []) if isinstance(identity, dict) else []
            if not isinstance(locator, dict) or not isinstance(evidence, list):
                continue
            ids = {
                str(source["id"])
                for item in evidence
                if isinstance(item, dict)
                and isinstance((source := item.get("source")), dict)
                and source.get("kind") == "docket"
                and _provider_url(source.get("external_url")) == provider
                and source.get("id") is not None
            }
            if len(ids) > 1:
                raise ValueError(f"{path.stem}:{locator}: multiple {provider} docket targets")
            if not ids:
                continue
            key = (path.stem, int(locator["start"]), int(locator["end"]))
            if key in targets:
                raise ValueError(f"Duplicate docket target: {key}")
            targets[key] = ids.pop()
    return targets


def _provider_url(value: object) -> Provider | None:
    if not isinstance(value, str):
        return None
    if "courtlistener.com" in value:
        return "courtlistener"
    if "govinfo.gov" in value:
        return "govinfo"
    return None


def _artifact_rows(path: Path, *, provider: Provider) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for citation in payload.get("citations", []):
        if not isinstance(citation, dict) or citation.get("root_id") != citation.get("citation_id"):
            continue
        source = citation.get("fields")
        if not isinstance(source, dict) or source.get("kind") != "DocketCitation":
            continue
        locator = source.get("locator_span")
        if not isinstance(locator, dict):
            continue
        search = _search_node(citation, stage=_STAGES[provider])
        if search is None:
            continue
        candidates = search.get("candidates", [])
        if not isinstance(candidates, list):
            raise ValueError(f"{path}: search candidates must be a list")
        candidate_ids = tuple(
            str(candidate[_IDS[provider]])
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get(_IDS[provider]) is not None
        )
        term = _term_node(citation, provider=provider)
        attempts = search.get("attempts", [])
        if not isinstance(attempts, list):
            raise ValueError(f"{path}: search attempts must be a list")
        rows.append(
            {
                "document": path.stem,
                "start": int(locator["start"]),
                "end": int(locator["end"]),
                "outcome": search.get("outcome"),
                "candidate_ids": candidate_ids,
                "candidate_count": search.get("candidate_count"),
                "terms": term["details"].get("terms") if term is not None else [],
                "term_plan_outcome": term.get("outcome") if term is not None else "missing",
                "attempts": _attempt_summaries(attempts, provider=provider),
            }
        )
    return rows


def _search_node(citation: dict[str, Any], *, stage: str) -> dict[str, Any] | None:
    matches = []
    for node in citation.get("trace", []):
        if not isinstance(node, dict) or node.get("stage") != stage:
            continue
        details = node.get("details")
        payload = details.get("validation") if isinstance(details, dict) else None
        if isinstance(payload, dict):
            matches.append(payload)
    if len(matches) > 1:
        raise ValueError(f"Multiple search nodes at stage {stage}")
    return matches[0] if matches else None


def _term_node(citation: dict[str, Any], *, provider: Provider) -> dict[str, Any] | None:
    # GovInfo correctly reuses the CourtListener source read rather than paying
    # for a second model call.
    stage = "courtlistener_docket_search"
    suffix = f":{stage}:case_name_terms"
    matches = [
        node
        for node in citation.get("trace", [])
        if isinstance(node, dict)
        and isinstance(node.get("node_id"), str)
        and node["node_id"].endswith(suffix)
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple term plans at stage {stage}")
    return matches[0] if matches else None


def _attempt_summaries(attempts: list[object], *, provider: Provider) -> list[dict[str, object]]:
    identifier = _IDS[provider]
    return [
        {
            "kind": attempt.get("kind"),
            "query": attempt.get("query"),
            "status": attempt.get("status"),
            "candidate_count": attempt.get("candidate_count"),
            "candidate_ids": [
                str(candidate[identifier])
                for candidate in attempt.get("candidates", [])
                if isinstance(candidate, dict) and candidate.get(identifier) is not None
            ],
            "error": attempt.get("error"),
        }
        for attempt in attempts
        if isinstance(attempt, dict)
    ]


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--provider", choices=("courtlistener", "govinfo"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_docket_metadata_search(
        annotations=args.annotations,
        artifacts=args.artifacts,
        provider=args.provider,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
