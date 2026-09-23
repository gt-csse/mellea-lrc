"""Build and score a development-only, provenance-grounded route sidecar.

Only annotation evidence that names a provider record and identifies a matching
retrieval route receives an expected stage. Other rows stay unknown. Scoring
requires both retrieval of that target record and a stage decision matching
the annotated identity label; merely invoking a stage is not a hit.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = {"CORRECT_IDENTITY", "WRONG_IDENTITY"}


def _reporter_path_matches(row: dict[str, Any], evidence: dict[str, Any]) -> bool:
    identifier = row.get("identifier", {})
    if not isinstance(identifier, dict) or identifier.get("kind") != "reporter":
        return False
    path = evidence.get("span", {}).get("text", "").strip("/")
    reporter = re.sub(r"[^a-z0-9]+", "-", str(identifier.get("reporter", "")).lower()).strip("-")
    expected = (str(identifier.get("volume", "")), reporter, f"{identifier.get('page', '')}.json")
    return tuple(path.split("/")[-3:]) == expected


def _expectations_for(row: dict[str, Any]) -> tuple[list[dict[str, str]], str | None, str | None]:
    identity = row["validation"]["identity"]
    evidence = identity.get("evidence", [])
    # A direct reporter route requires the actual CourtListener citation-lookup
    # record and a case-at-locator assertion. Opinion/ruling evidence alone is
    # deliberately insufficient: it may simply be the case being cited.
    reporter_hits = [
        e
        for e in evidence
        if e.get("shows") == "case_at_locator"
        and e.get("source", {}).get("kind") == "cluster"
        and "/sources/courtlistener/citation-lookup/" in f"/{e.get('span', {}).get('text', '').lstrip('/')}"
        and _reporter_path_matches(row, e)
        and e.get("source", {}).get("id") is not None
    ]
    retrievals: dict[tuple[str, str, str, str], dict[str, str]] = {}
    if row.get("kind") == "FullCaseCitation":
        for e in reporter_hits:
            record_id, source_path = str(e["source"]["id"]), str(e["span"]["text"])
            retrieval = {
                "provider": "courtlistener",
                "record_kind": "cluster",
                "record_id": record_id,
                "source_path": source_path,
                "source_field": str(e["span"].get("path", "")),
                "stage": "full_reporter_locator_exact_lookup",
            }
            retrievals[(retrieval["provider"], retrieval["record_kind"], record_id, source_path)] = retrieval

    # Docket retrieval labels require a docket root and an explicit record path
    # from CourtListener or GovInfo. Generic docket corroboration for a reporter
    # citation is never relabeled as reporter metadata search.
    if row.get("kind") == "DocketCitation":
        direct = [
            e
            for e in evidence
            if e.get("shows") in {"case_at_locator", "independent_record"}
            and e.get("source", {}).get("kind") == "docket"
            and e.get("source", {}).get("id") is not None
        ]
        if direct:
            e = next(
                (
                    e
                    for e in direct
                    if e.get("span", {})
                    .get("text", "")
                    .endswith(f"/dockets/{e.get('source', {}).get('id')}.json")
                ),
                None,
            )
            if e is not None:
                record_id, source_path = str(e["source"]["id"]), str(e["span"]["text"])
                retrieval = {
                    "provider": "courtlistener",
                    "record_kind": "docket",
                    "record_id": record_id,
                    "source_path": source_path,
                    "source_field": str(e["span"].get("path", "")),
                    "stage": "docket_root_identity",
                }
                retrievals[(retrieval["provider"], retrieval["record_kind"], record_id, source_path)] = (
                    retrieval
                )
            e = next(
                (
                    e
                    for e in direct
                    if "/sources/govinfo/" in f"/{e.get('span', {}).get('text', '').lstrip('/')}"
                    and e.get("span", {})
                    .get("text", "")
                    .endswith(f"/{e.get('source', {}).get('id')}.summary.json")
                ),
                None,
            )
            if e is not None:
                record_id, source_path = str(e["source"]["id"]), str(e["span"]["text"])
                retrieval = {
                    "provider": "govinfo",
                    "record_kind": "package",
                    "record_id": record_id,
                    "source_path": source_path,
                    "source_field": str(e["span"].get("path", "")),
                    "stage": "docket_root_identity",
                }
                retrievals[(retrieval["provider"], retrieval["record_kind"], record_id, source_path)] = (
                    retrieval
                )

    if retrievals:
        identity_basis = identity.get("basis")
        enough = _decision_evidence_matches_targets(identity, list(retrievals.values()))
        if (
            row.get("kind") == "FullCaseCitation"
            and any(r["record_kind"] == "cluster" for r in retrievals.values())
            and identity_basis in {"record_at_locator_agrees", "record_at_locator_disagrees"}
            and enough
        ):
            decision_stage = "full_reporter_locator_identity_resolution"
            decision_reason = "The annotation directly adjudicates the exact reporter record at its locator."
        elif (
            row.get("kind") == "DocketCitation"
            and identity_basis in {"docket_key_names_case", "record_at_locator_disagrees"}
            and enough
        ):
            decision_stage = "docket_root_identity"
            decision_reason = (
                "The docket-root basis directly names or contradicts the case at the cited docket."
            )
        else:
            decision_stage = None
            decision_reason = "Source establishes the retrieval target, but its annotation basis does not establish that this stage can reach the gold identity verdict."
        return list(retrievals.values()), decision_stage, decision_reason
    if any(e.get("source", {}).get("kind") in {"opinion", "ruling"} for e in evidence):
        why = "Opinion/ruling evidence may be the cited authority and does not establish a body-corroboration route."
    elif any(e.get("source", {}).get("kind") == "docket" for e in evidence):
        why = "Docket evidence lacks a route-specific target/provider match for this root kind or validation basis."
    else:
        why = "Annotation provenance does not establish a unique provider record and pipeline route."
    return [], None, why


def _decision_evidence_matches_targets(identity: dict[str, Any], targets: list[dict[str, str]]) -> bool:
    """Require field evidence to point to cited target records for the verdict."""
    evidence = identity.get("evidence", [])
    fields = identity.get("fields", {})
    supported_disagreement = False
    for name, field in fields.items():
        label = field.get("label") if isinstance(field, dict) else None
        if label in {None, "not_stated", "unknown"}:
            continue
        indexes = field.get("evidence", []) if isinstance(field, dict) else []
        supported = []
        for index in indexes:
            if not isinstance(index, int) or index < 0 or index >= len(evidence):
                continue
            item = evidence[index]
            source = item.get("source", {})
            evidence_path = item.get("span", {}).get("text", "")
            same_target = any(
                str(source.get("id")) == target["record_id"]
                and (
                    (
                        target["record_kind"] == "cluster"
                        and source.get("kind") == "cluster"
                        and evidence_path == target["source_path"]
                    )
                    or (
                        target["record_kind"] == "docket"
                        and source.get("kind") == "docket"
                        and evidence_path == target["source_path"]
                    )
                    or (
                        target["record_kind"] == "package"
                        and source.get("kind") == "docket"
                        and evidence_path == target["source_path"]
                    )
                )
                for target in targets
            )
            if item.get("shows") in {"case_at_locator", "independent_record"} and same_target:
                supported.append(item)
        if label == "disagrees" and supported:
            supported_disagreement = True
        if identity.get("label") == "CORRECT_IDENTITY" and label == "agrees" and not supported:
            return False
    if identity.get("label") == "WRONG_IDENTITY":
        return supported_disagreement
    return True


def build_expectations(annotations: Path) -> list[dict[str, Any]]:
    if not annotations.is_dir():
        raise ValueError(f"Annotation directory does not exist: {annotations}")
    rows: list[dict[str, Any]] = []
    for path in sorted(annotations.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            source = json.loads(line)
            if source.get("unit") != "citation" or source.get("is_root") is not True:
                continue
            identity = source.get("validation", {}).get("identity", {})
            if identity.get("label") not in LABELS or not isinstance(source.get("locator"), dict):
                continue
            retrievals, decision_stage, reason = _expectations_for(source)
            rows.append(
                {
                    "document": path.stem,
                    "kind": source["kind"],
                    "locator_span": {
                        "start": int(source["locator"]["start"]),
                        "end": int(source["locator"]["end"]),
                    },
                    "annotation_id": source["id"],
                    "identity_label": identity["label"],
                    "expected_retrievals": retrievals,
                    "expected_identity_decision_stage": decision_stage,
                    "decision_stage_reason": reason if decision_stage else None,
                    "identity_decision_unknown_reason": reason if retrievals and not decision_stage else None,
                    "unknown_reason": None if retrievals else reason,
                    "annotation_provenance": identity,
                }
            )
    rows.sort(key=lambda r: (r["document"], r["locator_span"]["start"], r["kind"]))
    keys = [(r["document"], r["kind"], r["locator_span"]["start"], r["locator_span"]["end"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate annotated root key")
    return rows


def _candidate_ids(validation: dict[str, Any], record_kind: str) -> set[str]:
    """Read IDs only from returned cluster/docket/package candidate objects."""
    candidates: list[Any] = []
    if record_kind == "cluster":
        cluster = validation.get("cluster")
        if isinstance(cluster, dict):
            candidates.append(cluster)
        listed = validation.get("candidate_clusters", [])
        if isinstance(listed, list):
            candidates.extend(listed)
    else:
        # Search validation may contain raw provider attempts; shortlist nodes
        # carry normalized candidate rows. Ignore prompts, query strings,
        # messages, and every field outside these explicit candidate lists.
        for attempt in validation.get("attempts", []) if isinstance(validation.get("attempts"), list) else []:
            if isinstance(attempt, dict):
                candidates.extend(
                    attempt.get("candidates", []) if isinstance(attempt.get("candidates"), list) else []
                )
        candidates.extend(
            validation.get("candidates", []) if isinstance(validation.get("candidates"), list) else []
        )
    ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if record_kind == "cluster":
            values = [candidate.get("cluster_id"), candidate.get("clusterId")]
        elif record_kind == "docket":
            values = [candidate.get("docket_id"), candidate.get("docketId")]
        else:
            values = [
                candidate.get("packageId"),
                candidate.get("package_id"),
                candidate.get("govinfo_package_id"),
            ]
        ids.update(str(value) for value in values if value is not None)
    return ids


def _artifact_roots(artifacts: Path) -> tuple[dict[tuple[str, str, int, int], dict[str, Any]], set[str]]:
    if not artifacts.is_dir():
        raise ValueError(f"Artifact directory does not exist: {artifacts}")
    roots: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    docs: set[str] = set()
    for path in sorted(artifacts.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("source_metadata", {})
        document = Path(meta.get("path", path.stem)).stem
        if document in docs:
            raise ValueError(f"Multiple artifacts map to document {document!r}")
        docs.add(document)
        for citation in payload.get("citations", []):
            if citation.get("root_id") != citation.get("citation_id"):
                continue
            fields = citation.get("fields", {})
            span, kind = fields.get("locator_span"), fields.get("kind")
            if not isinstance(span, dict) or not isinstance(kind, str):
                continue
            key = (document, kind, int(span["start"]), int(span["end"]))
            roots[key] = citation
    return roots, docs


def _target_retrieved(citation: dict[str, Any], target: dict[str, str]) -> bool:
    actual_stage = target["stage"]
    for trace in citation.get("trace", []):
        if trace.get("stage") != actual_stage:
            continue
        if target["provider"] in {"courtlistener", "govinfo"} and target["record_kind"] != "cluster":
            # Docket stage trace batches both providers. Attribute a candidate
            # only to the provider named by the query node, not by coincidental
            # equality with a candidate from the other provider.
            node_id = trace.get("node_id", "").lower()
            if target["provider"] not in node_id or ":query:" not in node_id or ":shortlist" in node_id:
                continue
        validation = trace.get("details", {}).get("validation", {})
        if target["record_id"] in _candidate_ids(validation, target["record_kind"]):
            return True
    return False


def _first_identity_decision(citation: dict[str, Any]) -> dict[str, str] | None:
    for trace in citation.get("trace", []):
        node_id = trace.get("node_id", "")
        outcome = trace.get("outcome")
        if outcome in {"resolved", "no_match"} and node_id.endswith(
            (":locator_identity_resolution", ":identity_resolution")
        ):
            return {"stage": str(trace.get("stage", "")), "outcome": str(outcome), "node_id": str(node_id)}
    return None


def _decision_matches_expected_stage(expected: str, actual: str) -> bool:
    if expected == "full_reporter_locator_identity_resolution":
        return actual in {
            "full_reporter_locator_unique_identity",
            "full_reporter_locator_ambiguity_resolution",
        }
    return actual == expected


def evaluate_routes(expectations: list[dict[str, Any]], artifacts: Path) -> dict[str, Any]:
    roots, docs = _artifact_roots(artifacts)
    counts: Counter[str] = Counter()
    results: list[dict[str, Any]] = []
    for row in expectations:
        key = (row["document"], row["kind"], row["locator_span"]["start"], row["locator_span"]["end"])
        citation = roots.get(key)
        retrievals = row["expected_retrievals"]
        if not retrievals:
            counts["unknown_needs_review"] += 1
        retrieval_results = []
        for target in retrievals:
            counts[f"retrieval:{target['provider']}:{target['stage']}"] += 1
            retrieved = _target_retrieved(citation, target) if citation else False
            retrieval_results.append({"target_record": target, "retrieved": retrieved})
        expected_stage = row["expected_identity_decision_stage"]
        decision = _first_identity_decision(citation) if citation else None
        expected_decision = "resolved" if row["identity_label"] == "CORRECT_IDENTITY" else "no_match"
        stage_family_match = bool(
            expected_stage
            and decision
            and _decision_matches_expected_stage(expected_stage, decision["stage"])
        )
        decision_correct = bool(stage_family_match and decision["outcome"] == expected_decision)
        targets_retrieved = bool(retrievals) and all(r["retrieved"] for r in retrieval_results)
        results.append(
            {
                "document": row["document"],
                "kind": row["kind"],
                "locator_span": row["locator_span"],
                "annotation_id": row["annotation_id"],
                "expected_retrievals": retrieval_results,
                "all_expected_targets_retrieved": targets_retrieved,
                "expected_identity_decision_stage": expected_stage,
                "expected_decision": expected_decision if expected_stage else None,
                "first_identity_decision": decision,
                "decision_stage_matches_expectation": stage_family_match,
                "expected_stage_gold_judgment_correct": decision_correct,
                "target_and_decision_success": targets_retrieved and decision_correct,
            }
        )
    assessable_decisions = [r for r in results if r["expected_identity_decision_stage"]]
    by_route: dict[str, dict[str, int]] = {}
    route_names = sorted(
        {
            target["provider"] + ":" + target["stage"]
            for row in expectations
            for target in row["expected_retrievals"]
        }
    )
    for route in route_names:
        matches = [
            retrieval_result
            for row in results
            for retrieval_result in row["expected_retrievals"]
            if retrieval_result["target_record"]["provider"]
            + ":"
            + retrieval_result["target_record"]["stage"]
            == route
        ]
        by_route[route] = {
            "expected_target_records": len(matches),
            "retrieved": sum(m["retrieved"] for m in matches),
        }
    by_kind: dict[str, dict[str, int]] = {}
    for kind in sorted({r["kind"] for r in results}):
        subset = [r for r in results if r["kind"] == kind]
        by_kind[kind] = {
            "labeled_roots": sum(1 for row in expectations if row["kind"] == kind),
            "roots_with_expected_retrieval": sum(bool(r["expected_retrievals"]) for r in subset),
            "expected_target_records": sum(len(r["expected_retrievals"]) for r in subset),
            "target_records_retrieved": sum(m["retrieved"] for r in subset for m in r["expected_retrievals"]),
            "roots_with_expected_identity_decision_stage": sum(
                bool(r["expected_identity_decision_stage"]) for r in subset
            ),
            "roots_with_retrieval_but_no_expected_decision": sum(
                bool(r["expected_retrievals"]) and not r["expected_identity_decision_stage"] for r in subset
            ),
            "first_decision_in_expected_stage_family": sum(
                r["decision_stage_matches_expectation"] for r in subset
            ),
            "expected_stage_gold_judgment_correct": sum(
                r["expected_stage_gold_judgment_correct"] for r in subset
            ),
            "target_and_decision_success": sum(r["target_and_decision_success"] for r in subset),
        }
    return {
        "artifact_type": "dev_root_route_findability",
        "stage_counts": dict(sorted(counts.items())),
        "labeled_roots": len(expectations),
        "unknown_needs_review": counts["unknown_needs_review"],
        "roots_with_expected_retrieval": sum(bool(r["expected_retrievals"]) for r in expectations),
        "expected_target_records": sum(len(r["expected_retrievals"]) for r in expectations),
        "target_records_retrieved": sum(m["retrieved"] for r in results for m in r["expected_retrievals"]),
        "roots_with_expected_identity_decision_stage": len(assessable_decisions),
        "roots_with_retrieval_but_no_expected_decision": sum(
            bool(r["expected_retrievals"]) and not r["expected_identity_decision_stage"] for r in expectations
        ),
        "first_decision_in_expected_stage_family": sum(
            r["decision_stage_matches_expectation"] for r in assessable_decisions
        ),
        "expected_stage_gold_judgment_correct": sum(
            r["expected_stage_gold_judgment_correct"] for r in assessable_decisions
        ),
        "target_and_decision_success": sum(r["target_and_decision_success"] for r in assessable_decisions),
        "by_route": by_route,
        "by_kind": by_kind,
        "missing_documents": sorted({r["document"] for r in expectations} - docs),
        "roots": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifacts", type=Path, help="Optional saved checkpoint for offline target-and-decision scoring"
    )
    parser.add_argument("--report", type=Path, help="Write the full offline route score as JSON")
    args = parser.parse_args()
    rows = build_expectations(args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "roots_with_expected_retrieval": sum(bool(r["expected_retrievals"]) for r in rows),
                "roots_with_expected_identity_decision_stage": sum(
                    bool(r["expected_identity_decision_stage"]) for r in rows
                ),
                "unknown_needs_review": sum(not r["expected_retrievals"] for r in rows),
            },
            sort_keys=True,
        )
    )
    if args.artifacts:
        report = evaluate_routes(rows, args.artifacts)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "stage_counts",
                        "labeled_roots",
                        "roots_with_expected_retrieval",
                        "expected_target_records",
                        "target_records_retrieved",
                        "roots_with_expected_identity_decision_stage",
                        "roots_with_retrieval_but_no_expected_decision",
                        "first_decision_in_expected_stage_family",
                        "expected_stage_gold_judgment_correct",
                        "target_and_decision_success",
                        "missing_documents",
                        "by_route",
                        "by_kind",
                    )
                },
                sort_keys=True,
            )
        )
    elif args.report:
        parser.error("--report requires --artifacts")


if __name__ == "__main__":
    main()
