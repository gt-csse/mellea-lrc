"""Checkpoint the no-audit docket reader and site-hunting chain.

This is an occurrence-level evaluation runner.  It writes each completed
document immediately, so a stopped model run resumes without repeating calls.
``occurrences.jsonl`` is rebuilt from those document artifacts after every
checkpoint.  It gives an annotator one row for every known gold, admitted but
unannotated locator, and declined candidate, with a pointer to the serialized
document and, where applicable, its review node.

The chain is deliberately limited to locator admission::

    federal CM/ECF reader -> docket site hunting

The independent docket/court audit is disabled.  Courts, dates, and case names
may still be read after an admission because that is the normal root-stage
operation, but they neither admit nor reject a locator in this run.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mellea_lrc.core.citations import DocketCitation
from mellea_lrc.extraction import grow_roots, stable
from mellea_lrc.extraction.adjudication import hunt_docket_locators, suspected_dockets
from mellea_lrc.extraction.types import Document
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.serialization import deserialize_document, serialize_document

SITE_STAGE = "docket_site_hunting"
DATASETS = (
    "primary",
    "hallucination-set-1",
    "hallucination-set-2",
    "reliable-high-profile",
    "reliable-low-profile",
)

SpanKey = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class Corpus:
    """The text and optional citation annotation for one named dataset."""

    name: str
    texts: Path
    annotations: Path


def _corpus(data: Path, name: str) -> Corpus:
    """Locate one of the supported datasets without duplicating its text."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}")
    root = data / name
    return Corpus(
        name=name,
        texts=root / ("documents_txt" if name == "primary" else "filings_txt"),
        annotations=root / "documents",
    )


def _atomic_json(path: Path, value: object) -> None:
    """Write JSON atomically so an interrupted run keeps its last checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the compact occurrence report atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _span(value: dict[str, Any]) -> tuple[int, int]:
    """Read a serialized half-open span."""
    return int(value["start"]), int(value["end"])


def _annotation_gold(corpus: Corpus) -> tuple[set[SpanKey], set[SpanKey], dict[str, dict[str, Any]]]:
    """Read docket gold, noncase evidence spans, and document headers."""
    gold: set[SpanKey] = set()
    noncase: set[SpanKey] = set()
    headers: dict[str, dict[str, Any]] = {}
    if not corpus.annotations.exists():
        return gold, noncase, headers
    for path in sorted(corpus.annotations.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or rows[0].get("unit") != "header":
            raise ValueError(f"{path}: expected a header row")
        header = rows[0]
        document = str(header["document"])
        headers[document] = header
        for row in rows[1:]:
            if row.get("unit") != "citation" or row.get("kind") != "DocketCitation":
                continue
            locator = row.get("locator")
            if not isinstance(locator, dict):
                raise ValueError(f"{path}: docket citation without a locator span")
            start, end = _span(locator)
            gold.add((document, start, end))
        for row in rows[1:]:
            if row.get("unit") != "noncase_citation":
                continue
            for field in ("span", "cited_as"):
                span = row.get(field)
                if isinstance(span, dict) and {"start", "end"} <= span.keys():
                    start, end = _span(span)
                    noncase.add((document, start, end))
    return gold, noncase, headers


def _source_metadata(path: Path, data: Path) -> dict[str, Any]:
    """Record the exact text a span indexes, without copying it into the run."""
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path.relative_to(data)),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "length": len(text),
    }


def _docket_spans(document: Document) -> set[tuple[int, int]]:
    """Return active docket locator spans from an in-memory document."""
    return {
        (record.locator_span.start, record.locator_span.end)
        for record in document.citations
        if isinstance(record.stated, DocketCitation) and not record.withdrawn
    }


def _serialized_docket_spans(document: dict[str, Any]) -> set[tuple[int, int]]:
    """Return active docket locator spans from a saved document artifact."""
    result: set[tuple[int, int]] = set()
    for record in document["citations"]:
        citation = record.get("stated", record["source"])
        if citation["citation_type"] == "DocketCitation" and not record.get("withdrawn_by"):
            result.add(_span(citation["locator_span"]))
    return result


def _accepted_site_nodes(document: dict[str, Any]) -> dict[tuple[int, int], str]:
    """Map each admitted site-hunting locator to its trace-node id."""
    nodes: dict[tuple[int, int], str] = {}
    for record in document["citations"]:
        citation = record.get("stated", record["source"])
        if citation["citation_type"] != "DocketCitation" or record.get("withdrawn_by"):
            continue
        for node in record.get("trace", []):
            if node.get("stage") == SITE_STAGE and node.get("outcome") == "accepted":
                nodes[_span(citation["locator_span"])] = str(node["node_id"])
    return nodes


def _declined_site_nodes(document: dict[str, Any]) -> dict[tuple[int, int], str]:
    """Map each declined candidate to its document-node id."""
    nodes: dict[tuple[int, int], str] = {}
    for node in document.get("nodes", []):
        if node.get("stage") != SITE_STAGE or node.get("outcome") != "declined":
            continue
        candidate = node.get("details", {}).get("candidate", {})
        span = candidate.get("span")
        if isinstance(span, dict):
            nodes[_span(span)] = str(node["node_id"])
    return nodes


def _score(predicted: set[SpanKey], gold: set[SpanKey]) -> dict[str, int | float | None]:
    """Return exact occurrence metrics, or state plainly that gold is absent."""
    true_positive = len(predicted & gold)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    if not gold:
        return {
            "gold": 0,
            "predicted": len(predicted),
            "tp": 0,
            "fp": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold)
    return {
        "gold": len(gold),
        "predicted": len(predicted),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _overlapping(span: tuple[int, int], others: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return spans that share at least one character with ``span``."""
    start, end = span
    return sorted(other for other in others if max(start, other[0]) < min(end, other[1]))


def _commit() -> str | None:
    """Record the source revision that produced a run, if available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() or None


def _source_worktree_dirty() -> bool | None:
    """Say whether the producing checkout had uncommitted source changes."""
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, check=False, text=True)
    if result.returncode:
        return None
    return bool(result.stdout.strip())


def _document_artifact_path(run_root: Path, document: str) -> Path:
    """Give each source document one independently checkpointed artifact."""
    return run_root / "documents" / f"{Path(document).stem}.json"


def _read_saved(path: Path) -> dict[str, Any] | None:
    """Load a complete checkpoint, rejecting a partial or obsolete shape."""
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or "final" not in value or "initial" not in value:
        raise ValueError(f"{path}: not a docket-site-hunting schema-1 checkpoint")
    deserialize_document(value["final"])
    return value


def _occurrences(
    *,
    saved: list[dict[str, Any]],
    gold: set[SpanKey],
    noncase: set[SpanKey],
    annotations_present: bool,
    run_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify every gold, prediction, and reviewed candidate by occurrence."""
    rows: list[dict[str, Any]] = []
    normal: set[SpanKey] = set()
    admitted: set[SpanKey] = set()
    final: set[SpanKey] = set()
    candidates: set[SpanKey] = set()
    declined: set[SpanKey] = set()

    for saved_document in saved:
        name = str(saved_document["document"])
        artifact = _document_artifact_path(run_root, name)
        relative_artifact = str(artifact.relative_to(run_root))
        initial = saved_document["initial"]
        initial_dockets = {_span(span) for span in initial["docket_spans"]}
        candidate_spans = {_span(site["span"]) for site in initial["site_candidates"]}
        serialized = saved_document["final"]
        final_spans = _serialized_docket_spans(serialized)
        accepted_nodes = _accepted_site_nodes(serialized)
        declined_nodes = _declined_site_nodes(serialized)

        normal |= {(name, start, end) for start, end in initial_dockets}
        # ``hunt_docket_locators`` reschedules candidates after every admission.
        # A later pass can therefore inspect a span absent from the initial
        # proposal set; declined review nodes are the durable record of those
        # later candidates and must count here too.
        candidate_spans |= set(accepted_nodes) | set(declined_nodes)
        candidates |= {(name, start, end) for start, end in candidate_spans}
        final |= {(name, start, end) for start, end in final_spans}
        admitted |= {(name, start, end) for start, end in accepted_nodes}
        declined |= {(name, start, end) for start, end in declined_nodes}

        universe = initial_dockets | candidate_spans | final_spans
        gold_spans = {(start, end) for doc, start, end in gold if doc == name}
        noncase_spans = {(start, end) for doc, start, end in noncase if doc == name}
        universe |= gold_spans
        for start, end in sorted(universe):
            key = (name, start, end)
            node_id: str | None = None
            related_gold = _overlapping((start, end), gold_spans - {(start, end)})
            related_predictions = _overlapping((start, end), final_spans - {(start, end)})
            related_noncase = _overlapping((start, end), noncase_spans)
            if key in gold:
                if (start, end) in initial_dockets:
                    outcome = "gold_deterministic"
                elif (start, end) in accepted_nodes:
                    outcome = "gold_site_admitted"
                    node_id = accepted_nodes[(start, end)]
                elif (start, end) in candidate_spans:
                    outcome = "gold_site_declined"
                    node_id = declined_nodes.get((start, end))
                elif related_predictions:
                    outcome = "gold_locator_span_mismatch"
                else:
                    outcome = "gold_not_proposed"
            elif related_gold:
                outcome = "locator_span_mismatch"
            elif related_noncase:
                outcome = "annotated_noncase_overlap"
            elif (start, end) in initial_dockets:
                outcome = "unannotated_deterministic"
            elif (start, end) in accepted_nodes:
                outcome = "unannotated_site_admitted"
                node_id = accepted_nodes[(start, end)]
            else:
                outcome = "candidate_declined"
                node_id = declined_nodes.get((start, end))
            row: dict[str, Any] = {
                "document": name,
                "span": {"start": start, "end": end},
                "kind": "docket_locator_occurrence",
                "annotated_gold": key in gold,
                "annotation_status": "available" if annotations_present else "not_yet_annotated",
                "outcome": outcome,
                "artifact": {"document": relative_artifact},
            }
            if node_id is not None:
                row["artifact"]["node_id"] = node_id
            if related_gold:
                row["overlapping_annotated_locator_spans"] = [
                    {"start": other_start, "end": other_end} for other_start, other_end in related_gold
                ]
            if related_predictions:
                row["overlapping_predicted_locator_spans"] = [
                    {"start": other_start, "end": other_end} for other_start, other_end in related_predictions
                ]
            if related_noncase:
                row["overlapping_noncase_annotation_spans"] = [
                    {"start": other_start, "end": other_end} for other_start, other_end in related_noncase
                ]
            rows.append(row)

    rows.sort(key=lambda row: (str(row["document"]), row["span"]["start"], row["span"]["end"]))
    review_queue = [
        row
        for row in rows
        if row["outcome"] in {"unannotated_deterministic", "unannotated_site_admitted"}
    ]
    metrics = {
        "deterministic": _score(normal, gold),
        "site_hunting_admissions": _score(admitted, gold - normal),
        "combined_no_audit": _score(final, gold),
        "counts": {
            "site_candidates": len(candidates),
            "site_admitted": len(admitted),
            "site_declined": len(declined),
            "final_docket_locators": len(final),
            "locator_span_mismatches": sum(
                row["outcome"] == "gold_locator_span_mismatch" for row in rows
            ),
            "annotated_noncase_overlaps": sum(row["outcome"] == "annotated_noncase_overlap" for row in rows),
            "unannotated_admissions_for_review": len(review_queue),
        },
    }
    return rows, metrics


def _write_reports(
    *,
    run_root: Path,
    corpus: Corpus,
    saved: list[dict[str, Any]],
    gold: set[SpanKey],
    noncase: set[SpanKey],
    annotations_present: bool,
    started_at: str,
) -> dict[str, Any]:
    """Refresh compact reports from all completed document checkpoints."""
    rows, metrics = _occurrences(
        saved=saved,
        gold=gold,
        noncase=noncase,
        annotations_present=annotations_present,
        run_root=run_root,
    )
    _atomic_jsonl(run_root / "occurrences.jsonl", rows)
    _atomic_jsonl(
        run_root / "annotation-review.jsonl",
        [
            row
            for row in rows
            if row["outcome"] in {"unannotated_deterministic", "unannotated_site_admitted"}
        ],
    )
    manifest = {
        "schema_version": 1,
        "dataset": corpus.name,
        "created_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "source_commit": _commit(),
        "source_worktree_dirty": _source_worktree_dirty(),
        "chain": ["federal_cmecf_reader", "docket_site_hunting"],
        "docket_audit": False,
        "site_hunting": "iterative: each admission refines the document before the next review",
        "documents_completed": len(saved),
        "annotations_present": annotations_present,
        "metrics": metrics,
        "occurrence_report": "occurrences.jsonl",
        "annotation_review": "annotation-review.jsonl",
        "document_artifacts": "documents/",
    }
    _atomic_json(run_root / "manifest.json", manifest)
    return manifest


async def run(*, data: Path, dataset: str, output: Path, resume: bool) -> dict[str, Any]:
    """Evaluate one dataset and checkpoint each document as soon as it finishes."""
    load_dotenv(".env")
    corpus = _corpus(data, dataset)
    gold, noncase, headers = _annotation_gold(corpus)
    paths = [path for path in sorted(corpus.texts.glob("*.txt")) if " (before clean)" not in path.name]
    if not paths:
        raise ValueError(f"{corpus.texts}: no active filing text")
    names = {path.name for path in paths}
    if headers and set(headers) != names:
        raise ValueError(f"{dataset}: annotation and text filenames differ")

    run_root = output / dataset
    run_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    saved: list[dict[str, Any]] = []
    missing: list[Path] = []
    for path in paths:
        checkpoint = _read_saved(_document_artifact_path(run_root, path.name)) if resume else None
        if checkpoint is None:
            missing.append(path)
        else:
            expected = _source_metadata(path, data)
            if checkpoint.get("text") != expected:
                raise ValueError(f"{path}: source text changed since its saved checkpoint")
            saved.append(checkpoint)
    annotations_present = bool(headers)
    manifest = _write_reports(
        run_root=run_root,
        corpus=corpus,
        saved=saved,
        gold=gold,
        noncase=noncase,
        annotations_present=annotations_present,
        started_at=started_at,
    )
    if not missing:
        print(f"{dataset}: resumed {len(saved)} completed documents", flush=True)
        return manifest

    rules = replace(stable(), docket_auditor=None)
    for index, path in enumerate(missing, start=1):
        # An OpenRouter connection is external state, not part of an extraction
        # document. Flash can leave a connection idle after a long sequence of
        # reviews, so do not share a session within this evaluator.  Omitting a
        # session makes each review create its own bounded provider session;
        # the Document remains the only state carried from one move to the next.
        source = _source_metadata(path, data)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            initial = grow_roots(preprocess(path.read_text(encoding="utf-8")), rules=rules)
        sites = suspected_dockets(initial)
        final = await hunt_docket_locators(initial, rules=rules)
        checkpoint = {
            "schema_version": 1,
            "document": path.name,
            "text": source,
            "initial": {
                "docket_spans": [
                    {"start": start, "end": end} for start, end in sorted(_docket_spans(initial))
                ],
                "site_candidates": [
                    {
                        "span": {"start": site.locator_span.start, "end": site.locator_span.end},
                        "generator": "suspected_dockets",
                    }
                    for site in sites
                ],
            },
            "final": serialize_document(final),
        }
        _atomic_json(_document_artifact_path(run_root, path.name), checkpoint)
        saved.append(checkpoint)
        manifest = _write_reports(
            run_root=run_root,
            corpus=corpus,
            saved=saved,
            gold=gold,
            noncase=noncase,
            annotations_present=annotations_present,
            started_at=started_at,
        )
        counts = manifest["metrics"]["counts"]
        print(
            f"[{index}/{len(missing)}] {dataset} {path.name[:36]:<36} "
            f"candidates={len(sites):2} admitted-total={counts['site_admitted']:3}",
            flush=True,
        )
    return manifest


def main() -> None:
    """Parse CLI arguments and run the model-backed evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/run-artifacts/1-roots/docket-site-hunting"),
        help="Directory containing one checkpointed run directory per dataset.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoints and rerun every document.")
    arguments = parser.parse_args()
    manifest = asyncio.run(
        run(data=arguments.data, dataset=arguments.dataset, output=arguments.output, resume=not arguments.no_resume)
    )
    print(json.dumps(manifest["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
