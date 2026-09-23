"""Score provider metadata-candidate retrieval for exact-miss reporter roots.

This evaluator measures a narrow, useful question: when exact reporter lookup
missed and a *correct-identity* annotation directly identifies a provider
docket/package, did the corresponding metadata-discovery stage retain that
record? It is not an identity metric. A returned docket/package can support a
later evaluation but does not prove that it is the reported decision.

Records attached to a ``WRONG_IDENTITY`` annotation are intentionally outside
this metric. They are the independent evidence that disproves the filing's
stated citation, not a search target the filing's stated case name should
recover. Likewise, an annotation source whose kind is ``opinion`` is a
full-text corroborating citation, not a package-metadata target. Both belong
to later candidate-review/corroboration scores.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

Provider = Literal["courtlistener", "govinfo"]

_STAGE: dict[Provider, str] = {
    "courtlistener": "courtlistener_full_reporter_metadata_search",
    "govinfo": "govinfo_full_reporter_metadata_search",
}
_NODE_TYPE: dict[Provider, str] = {
    "courtlistener": "FullReporterSearchNode",
    "govinfo": "GovInfoFullReporterSearchNode",
}
_CANDIDATE_ID: dict[Provider, str] = {
    "courtlistener": "docket_id",
    "govinfo": "govinfo_package_id",
}


def score_full_reporter_metadata_search(
    *,
    annotations: Path,
    artifacts: Path,
    provider: Provider,
) -> dict[str, object]:
    """Score annotated provider-record recovery for exact reporter lookup misses."""
    gold = _gold_metadata_targets(annotations, provider=provider)
    rows = _search_rows(artifacts, provider=provider)
    by_key = {row["key"]: row for row in rows}
    reached = set(gold) & set(by_key)
    retrieved = {key for key in reached if set(gold[key]["target_ids"]) & set(by_key[key]["candidate_ids"])}
    queryable = {key for key in reached if by_key[key]["attempts"]}
    queryable_retrieved = retrieved & queryable
    missed = sorted(reached - retrieved)
    not_reached = sorted(set(gold) - reached)
    exact_miss_rows = sum(1 for row in rows if row["exact_lookup_outcome"] == "not_found")

    return {
        "artifact_type": "full_reporter_metadata_search_score",
        "provider": provider,
        "stage": _STAGE[provider],
        "annotations": str(annotations),
        "artifacts": str(artifacts),
        "metric_definition": (
            "recall of directly annotated provider docket/package metadata records among exact-reporter-lookup "
            "misses that reached the provider metadata-discovery stage; wrong-identity evidence and opinion-body "
            "corroboration records are excluded; this does not admit or reject identity"
        ),
        "annotation_scope": _annotation_scope(annotations, provider=provider),
        "stage_coverage": {
            "exact_miss_roots_with_annotated_provider_metadata": len(gold),
            "reached": len(reached),
            "not_reached": len(not_reached),
            "all_stage_roots": len(rows),
            "exact_miss_roots_at_stage": exact_miss_rows,
        },
        "candidate_retrieval": {
            "targets_retrieved": len(retrieved),
            "targets_missed": len(missed),
            "recall": _ratio(len(retrieved), len(reached)),
            "queryable_targets": len(queryable),
            "queryable_targets_retrieved": len(queryable_retrieved),
            "queryable_recall": _ratio(len(queryable_retrieved), len(queryable)),
        },
        "stage_outcomes": dict(sorted(Counter(str(row["outcome"]) for row in rows).items())),
        "term_plan_outcomes": dict(sorted(Counter(str(row["term_plan_outcome"]) for row in rows).items())),
        "attempt_kinds": dict(
            sorted(
                Counter(
                    str(attempt["kind"])
                    for row in rows
                    for attempt in row["attempts"]
                    if isinstance(attempt.get("kind"), str)
                ).items()
            )
        ),
        "missed_targets": [_miss_row(key, gold[key], by_key[key]) for key in missed],
        "not_reached_targets": [_not_reached_row(key, gold[key]) for key in not_reached],
        "occurrences": [
            {
                **row,
                "key": {"document": row["key"][0], "start": row["key"][1], "end": row["key"][2]},
            }
            for row in sorted(rows, key=lambda row: row["key"])
        ],
    }


def _gold_metadata_targets(
    annotations: Path, *, provider: Provider
) -> dict[tuple[str, int, int], dict[str, object]]:
    targets: dict[tuple[str, int, int], dict[str, object]] = {}
    for path in _annotation_files(annotations):
        document = _annotation_document_name(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("unit") == "header":
                header_document = row.get("document")
                if isinstance(header_document, str):
                    document = Path(header_document).stem
                continue
            if (
                row.get("unit") != "citation"
                or row.get("kind") != "FullCaseCitation"
                or row.get("is_root") is not True
            ):
                continue
            locator = row.get("locator")
            identity = row.get("validation", {}).get("identity", {})
            if not isinstance(locator, dict) or not isinstance(identity, dict):
                continue
            identity_label = identity.get("label")
            # A source record attached to a wrong-identity annotation commonly
            # identifies the filing's own docket or another independent record
            # used to disprove the citation. Treating it as the cited case's
            # expected retrieval target would make this discovery score false.
            if identity_label != "CORRECT_IDENTITY":
                continue
            target_ids = _provider_targets(identity.get("evidence"), provider=provider)
            if not target_ids:
                continue
            if not document:
                msg = f"{path}: citation appears before an annotation header"
                raise ValueError(msg)
            key = (document, int(locator["start"]), int(locator["end"]))
            if key in targets:
                msg = f"Duplicate annotated full-reporter metadata target: {key}"
                raise ValueError(msg)
            targets[key] = {
                "target_ids": tuple(sorted(target_ids)),
                "identity_label": identity_label,
                "locator_quote": locator.get("quote"),
            }
    return targets


def _annotation_scope(annotations: Path, *, provider: Provider) -> dict[str, int]:
    """Expose annotation records deliberately excluded from this narrow score."""
    scope: Counter[str] = Counter()
    for path in _annotation_files(annotations):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if (
                row.get("unit") != "citation"
                or row.get("kind") != "FullCaseCitation"
                or row.get("is_root") is not True
            ):
                continue
            identity = row.get("validation", {}).get("identity", {})
            if not isinstance(identity, dict):
                continue
            evidence = identity.get("evidence")
            if not isinstance(evidence, list):
                continue
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                source = item.get("source")
                if not isinstance(source, dict) or _source_provider(source.get("external_url")) != provider:
                    continue
                if identity.get("label") != "CORRECT_IDENTITY":
                    scope["wrong_identity_provider_records_excluded"] += 1
                elif item.get("shows") != "independent_record" or source.get("kind") != "docket":
                    scope["opinion_or_nonmetadata_records_excluded"] += 1
                else:
                    scope["direct_metadata_records"] += 1
    return dict(sorted(scope.items()))


def _annotation_files(annotations: Path) -> tuple[Path, ...]:
    if annotations.is_file():
        return (annotations,)
    return tuple(sorted(annotations.glob("*.jsonl")))


def _annotation_document_name(path: Path) -> str:
    if path.name == "primary.jsonl":
        # Combined primary annotations use citation ids such as 001-o01 rather
        # than one source file per document. The caller replaces this marker
        # from each citation id below.
        return ""
    return path.stem


def _provider_targets(evidence: object, *, provider: Provider) -> set[str]:
    if not isinstance(evidence, list):
        return set()
    targets: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        if not isinstance(source, dict):
            continue
        external_url = source.get("external_url")
        source_id = source.get("id")
        kind = source.get("kind")
        # A docket source is direct metadata evidence. An opinion source
        # identifies text that cites the case and is scored by the later
        # full-text corroboration evaluator instead.
        if (
            item.get("shows") != "independent_record"
            or kind != "docket"
            or not isinstance(external_url, str)
            or not isinstance(source_id, str)
        ):
            continue
        if provider == "courtlistener":
            if "courtlistener.com/docket/" in external_url:
                targets.add(source_id)
        elif "govinfo.gov/app/details/USCOURTS-" in external_url:
            targets.add(source_id)
    return targets


def _source_provider(external_url: object) -> Provider | None:
    if not isinstance(external_url, str):
        return None
    if "courtlistener.com" in external_url:
        return "courtlistener"
    if "govinfo.gov" in external_url:
        return "govinfo"
    return None


def _search_rows(artifacts: Path, *, provider: Provider) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(artifacts.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        document = path.stem
        for citation in payload.get("citations", []):
            if not isinstance(citation, dict) or citation.get("root_id") != citation.get("citation_id"):
                continue
            source = citation.get("fields")
            if not isinstance(source, dict) or source.get("kind") != "FullCaseCitation":
                continue
            locator = source.get("locator_span")
            if not isinstance(locator, dict):
                continue
            node = _search_node(citation, provider=provider)
            if node is None:
                continue
            attempts = node.get("attempts", [])
            candidates = node.get("candidates", [])
            if not isinstance(attempts, list) or not isinstance(candidates, list):
                raise ValueError(f"{path}: malformed metadata-search node")
            term = _term_node(citation, provider=provider)
            rows.append(
                {
                    "key": (document, int(locator["start"]), int(locator["end"])),
                    "reporter_locator": node.get("reporter_locator"),
                    "outcome": node.get("outcome"),
                    "candidate_count": node.get("candidate_count"),
                    "candidate_ids": tuple(
                        str(candidate[_CANDIDATE_ID[provider]])
                        for candidate in candidates
                        if isinstance(candidate, dict) and candidate.get(_CANDIDATE_ID[provider]) is not None
                    ),
                    "attempts": _attempt_summaries(attempts, provider=provider),
                    "terms": term["details"].get("terms") if term is not None else [],
                    "term_plan_outcome": term.get("outcome") if term is not None else "missing",
                    "exact_lookup_outcome": _exact_lookup_outcome(citation),
                }
            )
    return rows


def _search_node(citation: dict[str, Any], *, provider: Provider) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for node in citation.get("trace", []):
        if not isinstance(node, dict) or node.get("stage") != _STAGE[provider]:
            continue
        details = node.get("details")
        payload = details.get("validation") if isinstance(details, dict) else None
        if isinstance(payload, dict) and details.get("validation_node_type") == _NODE_TYPE[provider]:
            matches.append(payload)
    if len(matches) > 1:
        msg = f"Multiple {_NODE_TYPE[provider]} records in one citation trace"
        raise ValueError(msg)
    return matches[0] if matches else None


def _term_node(citation: dict[str, Any], *, provider: Provider) -> dict[str, Any] | None:
    preferred_stage = _STAGE[provider]
    fallback_stage = _STAGE["courtlistener"]
    for stage in (preferred_stage, fallback_stage):
        suffix = f":{stage}:case_name_terms"
        matches = [
            node
            for node in citation.get("trace", [])
            if isinstance(node, dict)
            and isinstance(node.get("node_id"), str)
            and node["node_id"].endswith(suffix)
        ]
        if len(matches) > 1:
            msg = f"Multiple term plans at {stage}"
            raise ValueError(msg)
        if matches:
            return matches[0]
    return None


def _exact_lookup_outcome(citation: dict[str, Any]) -> str | None:
    for node in citation.get("trace", []):
        if not isinstance(node, dict):
            continue
        details = node.get("details")
        validation = details.get("validation") if isinstance(details, dict) else None
        if (
            isinstance(validation, dict)
            and details.get("validation_node_type") == "ExactLocatorLookupNode"
            and isinstance(validation.get("outcome"), str)
        ):
            return str(validation["outcome"])
    return None


def _attempt_summaries(attempts: list[object], *, provider: Provider) -> list[dict[str, object]]:
    identifier = _CANDIDATE_ID[provider]
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


def _miss_row(
    key: tuple[str, int, int], gold: dict[str, object], row: dict[str, object]
) -> dict[str, object]:
    return {
        "document": key[0],
        "locator_span": {"start": key[1], "end": key[2]},
        **gold,
        "case_name_terms": row["terms"],
        "attempts": row["attempts"],
        "candidate_ids": row["candidate_ids"],
    }


def _not_reached_row(key: tuple[str, int, int], gold: dict[str, object]) -> dict[str, object]:
    return {"document": key[0], "locator_span": {"start": key[1], "end": key[2]}, **gold}


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--provider", choices=("courtlistener", "govinfo"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_full_reporter_metadata_search(
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
