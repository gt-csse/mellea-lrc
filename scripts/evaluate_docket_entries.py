"""Replay saved docket hunting, persist the entry stage, and score its readings.

This does not call a model. Replaying saved site decisions gives earlier rule
entries their new `docket_entries` node instead of retaining a legacy creation-
node reading from an older checkpoint.

Run from the repository root::

    uv run python -m scripts.evaluate_docket_entries
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mellea_lrc.extraction import find_docket_locators, find_full_reporter_locators, resolve_docket_entries
from mellea_lrc.extraction.docket_hunting import STAGE as HUNT_STAGE
from mellea_lrc.model import Document, FullDocketCitation, Span
from mellea_lrc.preprocessing.document_index import is_within
from scripts.run_docket_site_hunt import (
    _atomic_write,
    _checkpoint,
    _source_document,
    _validate_checkpoint,
    select_documents,
)

SETS = (
    "primary",
    "hallucination-set-1",
    "hallucination-set-2",
    "reliable-high-profile",
    "reliable-low-profile",
)


def _replay(source: Document, saved: Document) -> Document:
    """Retain every hunting verdict and trace while rebuilding current fields."""
    # Eyecite occasionally prints harmless overlap diagnostics while reading.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = find_docket_locators(find_full_reporter_locators(source))
    saved_by_id = {citation.id: citation for citation in saved.citations}
    for review in saved.site_reviews:
        if review.citation_id is not None:
            prior = saved_by_id[review.citation_id]
            if not isinstance(prior, FullDocketCitation):
                raise ValueError("A docket site review points to a non-docket citation")
            locator = prior.locator[-1]
            document = document.add_citation(
                FullDocketCitation.from_locator(
                    citation_id=prior.id,
                    stage=HUNT_STAGE,
                    source=document.text,
                    span=locator.span,
                    number_span=locator.number_span,
                )
            )
        document = document.add_site_review(review)
    document = document.complete(HUNT_STAGE)
    old_sites = [(citation.id, citation.locator_span) for citation in saved.citations]
    new_sites = [(citation.id, citation.locator_span) for citation in document.citations]
    if (
        document.stage_runs != saved.stage_runs
        or document.site_reviews != saved.site_reviews
        or new_sites != old_sites
    ):
        raise ValueError("Saved hunting decisions do not replay to the same locator sites")
    return document


def _span(raw: dict[str, Any]) -> Span:
    return Span(start=int(raw["start"]), end=int(raw["end"]))


def _span_json(span: Span) -> dict[str, int]:
    return {"start": span.start, "end": span.end}


def _gold(path: Path, filename: str, digest: str, index_spans: tuple[Span, ...]) -> dict[Span, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or rows[0].get("unit") != "header":
        raise ValueError(f"Missing annotation header: {path}")
    header = rows[0]
    if header.get("document") != filename or header.get("text", {}).get("sha256") != digest:
        raise ValueError(f"Annotation source differs from manifest: {path}")
    result: dict[Span, dict] = {}
    for row in rows[1:]:
        if row.get("unit") != "citation" or row.get("kind") != "DocketCitation":
            continue
        locator = _span(row["locator"])
        if not is_within(locator, index_spans):
            if locator in result:
                raise ValueError(f"Duplicate annotated docket locator: {path} {locator}")
            result[locator] = row
    return result


def _score(document: Document, gold: dict[Span, dict]) -> tuple[Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter(documents=1)
    details: list[dict[str, Any]] = []
    counts["gold_entries"] = sum(isinstance(row.get("docket_entry"), dict) for row in gold.values())
    for citation in document.citations:
        if not isinstance(citation, FullDocketCitation) or not citation.docket_entry:
            continue
        entry = citation.docket_entry[-1]
        counts["predicted_entries"] += 1
        if entry.normalizable:
            counts["normalizable"] += 1
        row = gold.get(citation.locator_span)
        gold_entry = row.get("docket_entry") if row else None
        if isinstance(gold_entry, dict):
            outcome = "wrong_span"
            if entry.span == _span(gold_entry):
                counts["exact_gold_spans"] += 1
                outcome = "wrong_number"
                if entry.normalizable and entry.get_normalized() == str(gold_entry["number"]):
                    counts["correct_normalization"] += 1
                    outcome = "correct"
        elif row and isinstance(row.get("cited_as"), dict):
            cited_as = _span(row["cited_as"])
            outcome = (
                "unlabeled_in_citation"
                if cited_as.start <= entry.span.start and entry.span.end <= cited_as.end
                else "unlabeled_outside_citation"
            )
        else:
            outcome = "unmatched_locator"
        counts[outcome] += 1
        details.append(
            {
                "citation_id": citation.id,
                "locator_span": _span_json(citation.locator_span),
                "entry_span": _span_json(entry.span),
                "quote": entry.quote,
                "normalized": entry.get_normalized() if entry.normalizable else None,
                "outcome": outcome,
                "gold_id": row.get("id") if row else None,
            }
        )
    return counts, details


def evaluate(data_root: Path, input_dir: Path, output_dir: Path, sets: tuple[str, ...]) -> dict:
    by_set: dict[str, Counter[str]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    for name, filename, metadata in select_documents(data_root, sets):
        source = _source_document(data_root, name, filename, metadata)
        saved_path = _checkpoint(input_dir, name, filename)
        saved = Document.model_validate_json(saved_path.read_text(encoding="utf-8"))
        _validate_checkpoint(saved, source, saved_path)
        before = _replay(source, saved)
        document = resolve_docket_entries(before)
        if document.get_stage(HUNT_STAGE) != before:
            raise ValueError("Docket-entry stage did not preserve its prior checkpoint")
        restored = Document.model_validate_json(document.model_dump_json())
        if restored != document or restored.get_stage(HUNT_STAGE) != before:
            raise ValueError("Docket-entry checkpoint does not round-trip")
        _atomic_write(_checkpoint(output_dir, name, filename), document.model_dump_json(indent=2) + "\n")

        annotation = data_root / name / "documents" / f"{Path(filename).stem}.jsonl"
        gold = _gold(annotation, filename, metadata["sha256"], source.index_spans)
        counts, rows = _score(document, gold)
        by_set.setdefault(name, Counter()).update(counts)
        details[f"{name}/{filename}"] = rows
    totals: Counter[str] = Counter()
    for counts in by_set.values():
        totals.update(counts)
    summary = {
        "stage": "docket_entries",
        "basis": "Saved docket-site decisions replayed without model calls; exact source spans outside index masks",
        "sets": {name: dict(counts) for name, counts in by_set.items()},
        "totals": dict(totals),
    }
    _atomic_write(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    _atomic_write(output_dir / "occurrences.json", json.dumps(details, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--input-dir", type=Path, default=Path("local/docket-site-hunt-entry-mask"))
    parser.add_argument("--output-dir", type=Path, default=Path("local/docket-entry-stage"))
    parser.add_argument("--set", dest="sets", action="append", choices=SETS)
    args = parser.parse_args()
    summary = evaluate(args.data_root, args.input_dir, args.output_dir, tuple(args.sets or SETS))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
