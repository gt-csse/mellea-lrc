"""Replay saved docket-site decisions into a separate docket-entry checkpoint.

This artifact producer makes no model call and reads no annotation. It is only
needed for older hunting runs, where entries were attached on creation nodes.
The new ``docket_entries`` stage receives those readings without changing site
decisions. Score its saved Documents separately with
``python -m evaluations.score_stages --stage docket_entries``.

Run from the repository root::

    uv run python -m scripts.replay_docket_entries
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import Counter
from pathlib import Path

from mellea_lrc.extraction import find_docket_locators, find_full_reporter_locators, resolve_docket_entries
from mellea_lrc.extraction.docket_entries import STAGE as ENTRY_STAGE
from mellea_lrc.extraction.docket_hunting import STAGE as HUNT_STAGE
from mellea_lrc.model import Document, FullDocketCitation
from scripts.run_docket_site_hunt import (
    SETS,
    _atomic_write,
    _checkpoint,
    _source_document,
    _validate_checkpoint,
    select_documents,
)

DEFAULT_SETS = tuple(name for name in SETS if name != "final-held-out")


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


def replay(data_root: Path, input_dir: Path, output_dir: Path, sets: tuple[str, ...]) -> dict:
    counts: Counter[str] = Counter()
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
        counts[name] += 1
    summary = {"stage": ENTRY_STAGE, "documents": sum(counts.values()), "sets": dict(counts)}
    _atomic_write(output_dir / "run.json", json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--input-dir", type=Path, default=Path("local/docket-site-hunt-entry-mask"))
    parser.add_argument("--output-dir", type=Path, default=Path("local/docket-entry-stage"))
    parser.add_argument("--set", dest="sets", action="append", choices=SETS)
    args = parser.parse_args()
    summary = replay(args.data_root, args.input_dir, args.output_dir, tuple(args.sets or DEFAULT_SETS))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
