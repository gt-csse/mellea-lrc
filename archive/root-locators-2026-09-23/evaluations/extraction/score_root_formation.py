"""Score the filing-internal root graph without replaying any stage.

The evaluator compares complete reporter and docket locator occurrences by
their source spans. It reports root-locator detection separately from the edge
that attaches every locator occurrence to its filing-stated root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluations.extraction.locator_eval_common import score_sets

_ROOT_KINDS = frozenset({"FullCaseCitation", "DocketCitation"})
_LocatorKey = tuple[str, str, int, int]
_RootEdge = tuple[_LocatorKey, _LocatorKey]


def score_root_formation(*, annotations: Path, artifacts: Path) -> dict[str, object]:
    """Score formed roots and locator-to-root edges from persisted documents."""
    predicted_edges, document_names = _predicted_edges(artifacts)
    gold_edges = _gold_edges(annotations, document_names=document_names)
    gold_roots = {root for _occurrence, root in gold_edges}
    predicted_roots = {root for _occurrence, root in predicted_edges}
    return {
        "artifact_type": "root_formation_score",
        "annotations": str(annotations),
        "artifacts": str(artifacts),
        "root_locators": score_sets(gold_roots, predicted_roots),
        "locator_attribution": score_sets(gold_edges, predicted_edges),
    }


def _gold_edges(annotations: Path, *, document_names: set[str]) -> set[_RootEdge]:
    """Read the annotation graph using root locator spans rather than row ids."""
    edges: set[_RootEdge] = set()
    for path in sorted(annotations.glob("*.jsonl")):
        if path.stem not in document_names:
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        locator_by_id = {
            str(row["id"]): _annotation_key(path.stem, row) for row in rows if _is_root_locator(row)
        }
        for row in rows:
            if not _is_root_locator(row):
                continue
            occurrence = _annotation_key(path.stem, row)
            root_id = row.get("root_id")
            root = locator_by_id.get(str(root_id))
            if root is None:
                msg = f"{path}: root locator {row['id']!r} points to a non-locator root {root_id!r}"
                raise ValueError(msg)
            edges.add((occurrence, root))
    return edges


def _predicted_edges(artifacts: Path) -> tuple[set[_RootEdge], set[str]]:
    """Read the graph directly from serialized citation records."""
    edges: set[_RootEdge] = set()
    documents: dict[str, Path] = {}
    for path in sorted(artifacts.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_metadata = payload.get("source_metadata")
        source_path = source_metadata.get("path") if isinstance(source_metadata, dict) else None
        document = Path(source_path).stem if isinstance(source_path, str) and source_path else path.stem
        if document in documents:
            raise ValueError(
                f"Multiple artifacts map to source document {document!r}: {documents[document]} and {path}"
            )
        documents[document] = path
        citations = payload.get("citations")
        if not isinstance(citations, list):
            msg = f"{path}: serialized document has no citation list"
            raise ValueError(msg)
        locator_by_id = {
            str(citation["citation_id"]): _prediction_key(document, citation)
            for citation in citations
            if _is_predicted_root_locator(citation)
        }
        for citation in citations:
            if not _is_predicted_root_locator(citation):
                continue
            occurrence = _prediction_key(document, citation)
            root_id = citation.get("root_id")
            root = locator_by_id.get(str(root_id))
            if root is None:
                msg = f"{path}: locator {citation.get('citation_id')!r} has unknown root {root_id!r}"
                raise ValueError(msg)
            edges.add((occurrence, root))
    return edges, set(documents)


def _is_root_locator(row: dict[str, Any]) -> bool:
    return (
        row.get("unit") == "citation"
        and row.get("kind") in _ROOT_KINDS
        and isinstance(row.get("locator"), dict)
    )


def _annotation_key(document: str, row: dict[str, Any]) -> _LocatorKey:
    locator = row["locator"]
    return document, str(row["kind"]), int(locator["start"]), int(locator["end"])


def _is_predicted_root_locator(citation: dict[str, Any]) -> bool:
    source = citation.get("fields")
    return (
        isinstance(source, dict)
        and source.get("kind") in _ROOT_KINDS
        and isinstance(source.get("locator_span"), dict)
    )


def _prediction_key(document: str, citation: dict[str, Any]) -> _LocatorKey:
    source = citation["fields"]
    locator = source["locator_span"]
    return document, str(source["kind"]), int(locator["start"]), int(locator["end"])


def main() -> None:
    """Write or print a reusable root-formation score artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_root_formation(annotations=args.annotations, artifacts=args.artifacts)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
