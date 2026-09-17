"""Evaluate admitted docket spans separately from raw locator discovery."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluations.extraction.locator_eval_common import (
    DEFAULT_DATA,
    GrownAnnotation,
    grow_annotated_corpus,
    locator_span,
    score_sets,
)
from mellea_lrc.core.citations import DocketCitation
from mellea_lrc.extraction.reading.docket_audit import MADE_BY


def eval_docket_audit(corpus: Sequence[GrownAnnotation]) -> dict[str, Any]:
    """Score retained docket occurrences and report the audit's decisions."""
    gold: set[tuple[str, int, int]] = set()
    admitted: set[tuple[str, int, int]] = set()
    reasons: Counter[str] = Counter()
    candidates = withdrawn = 0
    for sample in corpus:
        for row in sample.citation_rows:
            if row.get("unit") == "citation" and row.get("kind") == "DocketCitation":
                span = locator_span(row)
                if span is not None:
                    gold.add((sample.document_name, *span))
        for record in sample.document.citations:
            if not isinstance(record.source, DocketCitation):
                continue
            candidates += 1
            withdrawn += record.withdrawn
            if not record.withdrawn:
                admitted.add((sample.document_name, record.locator_span.start, record.locator_span.end))
            decision = next((node for node in reversed(record.trace) if node.made_by == MADE_BY), None)
            reasons[str(decision.details["reason"]) if decision else "not_audited"] += 1
    return {
        "dataset": "annotation-v4.0 corpus",
        "documents": len(corpus),
        "docket_audit": {
            "candidates": candidates,
            "accepted": candidates - withdrawn,
            "withdrawn": withdrawn,
            "reasons": dict(sorted(reasons.items())),
            "accepted_spans": score_sets(gold, admitted),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_DATA / "annotation-v4.0" / "documents")
    parser.add_argument("--texts-root", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(json.dumps(eval_docket_audit(grow_annotated_corpus(args.annotations, args.texts_root)), indent=2))


if __name__ == "__main__":
    main()
