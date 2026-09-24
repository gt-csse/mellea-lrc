"""Score docket rule spans and unreviewed site proposals against five annotated sets.

The rule reader and proposal generator see only each filing's text and the
``index_spans`` in its ``documents.json`` entry. Annotation rows are loaded
after proposals have been generated and are used only to count exact span
matches. This script makes no model or provider calls.

Run from the repository root::

    uv run python scripts/score_docket_site_proposals.py
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from mellea_lrc.extraction import find_docket_locators, find_full_reporter_locators
from mellea_lrc.extraction.docket_hunting import suspected_dockets
from mellea_lrc.model import Document, FullDocketCitation, Span
from mellea_lrc.model.preprocessed_document import PreprocessingBackend, PreprocessingMetadata
from mellea_lrc.model.source import SourceFormat, SourceMetadata
from mellea_lrc.preprocessing.document_index import is_within

SETS = (
    "primary",
    "hallucination-set-1",
    "hallucination-set-2",
    "reliable-high-profile",
    "reliable-low-profile",
)
COUNT_FIELDS = (
    "documents",
    "eligible_gold_docket_locators",
    "rule_found",
    "exact_proposed_among_rule_misses",
    "remaining_misses",
    "total_proposals",
)


def _span(raw: dict[str, Any]) -> Span:
    return Span(start=int(raw["start"]), end=int(raw["end"]))


def _gold_spans(path: Path, *, filename: str, digest: str, index_spans: tuple[Span, ...]) -> set[Span]:
    """Read annotated docket locator spans only after predictions are fixed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty annotation file")
    header = json.loads(lines[0])
    if header.get("unit") != "header" or header.get("document") != filename:
        raise ValueError(f"{path}: annotation header does not match {filename}")
    if header.get("text", {}).get("sha256") != digest:
        raise ValueError(f"{path}: annotation text hash differs from documents.json")

    gold: set[Span] = set()
    for line in lines[1:]:
        row = json.loads(line)
        if row.get("unit") != "citation" or row.get("kind") != "DocketCitation":
            continue
        locator = row.get("locator")
        if not isinstance(locator, dict):
            raise ValueError(f"{path}: docket citation lacks a locator span")
        span = _span(locator)
        if not is_within(span, index_spans):
            gold.add(span)
    return gold


def score_set(data_root: Path, name: str) -> dict[str, int]:
    """Count exact half-open locator spans for one dataset."""
    root = data_root / name
    manifest = json.loads((root / "documents.json").read_text(encoding="utf-8"))["documents"]
    text_dir = root / ("documents_txt" if name == "primary" else "filings_txt")
    counts = dict.fromkeys(COUNT_FIELDS, 0)

    for filename, metadata in sorted(manifest.items()):
        source = (text_dir / filename).read_text(encoding="utf-8")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest != metadata["sha256"] or len(source) != metadata["length"]:
            raise ValueError(f"{text_dir / filename}: text differs from documents.json")
        index_spans = tuple(_span(raw) for raw in metadata.get("index_spans", ()))

        # No annotation data is available to either extraction function.
        document = Document(
            source_metadata=SourceMetadata(path=str(text_dir / filename), format=SourceFormat.TEXT),
            text=source,
            preprocessing_metadata=PreprocessingMetadata(
                backend=PreprocessingBackend(metadata.get("backend", "plain_text")),
                backend_version=metadata.get("backend_version"),
            ),
            index_spans=index_spans,
        )
        # Eyecite can print benign overlap diagnostics for otherwise valid input.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = find_docket_locators(find_full_reporter_locators(document))
        rule = {
            citation.locator_span
            for citation in document.citations
            if isinstance(citation, FullDocketCitation)
        }
        proposals = {candidate.locator_span for candidate in suspected_dockets(document)}

        gold = _gold_spans(
            root / "documents" / f"{Path(filename).stem}.jsonl",
            filename=filename,
            digest=digest,
            index_spans=index_spans,
        )
        missed_by_rule = gold - rule
        counts["documents"] += 1
        counts["eligible_gold_docket_locators"] += len(gold)
        counts["rule_found"] += len(gold & rule)
        counts["exact_proposed_among_rule_misses"] += len(missed_by_rule & proposals)
        counts["remaining_misses"] += len(missed_by_rule - proposals)
        counts["total_proposals"] += len(proposals)
    return counts


def score(data_root: Path, names: tuple[str, ...] = SETS) -> dict[str, object]:
    """Return per-set counts and an aggregate with the same field definitions."""
    sets = {name: score_set(data_root, name) for name in names}
    totals = {field: sum(counts[field] for counts in sets.values()) for field in COUNT_FIELDS}
    return {
        "metric": (
            "Exact annotated DocketCitation locator spans outside "
            "documents.json index_spans; proposals are unreviewed candidates"
        ),
        "sets": sets,
        "totals": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    args = parser.parse_args()
    result = json.dumps(score(args.data_root), indent=2) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
