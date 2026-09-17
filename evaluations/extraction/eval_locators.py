"""Evaluate exact locator occurrence spans on annotation-v4.0."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluations.extraction.locator_eval_common import (
    CASE_LOCATOR_KINDS,
    DEFAULT_DATA,
    GrownAnnotation,
    grow_annotated_corpus,
    locator_span,
    score_sets,
)


def eval_locators(corpus: Sequence[GrownAnnotation]) -> dict[str, Any]:
    """Score every complete locator span, independent of root assignment.

    An occurrence is (document, start, end). Repeated identifiers at different
    positions count separately. Neither ``is_root`` nor any context field is
    required for a match.
    """
    gold: set[tuple[str, int, int]] = set()
    gold_by_kind: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    predicted: set[tuple[str, int, int]] = set()
    predicted_by_kind: dict[str, set[tuple[str, int, int]]] = defaultdict(set)

    for sample in corpus:
        name = sample.document_name
        for row in sample.citation_rows:
            if row.get("unit") != "citation" or row.get("kind") not in CASE_LOCATOR_KINDS:
                continue
            span = locator_span(row)
            if span is None:
                continue
            occurrence = (name, *span)
            gold.add(occurrence)
            gold_by_kind[str(row["kind"])].add(occurrence)

        for locator in sample.document.locators:
            occurrence = (name, locator.span.start, locator.span.end)
            predicted.add(occurrence)
            predicted_by_kind[locator.kind.value].add(occurrence)

    return {
        "dataset": "annotation-v4.0 corpus",
        "documents": len(corpus),
        "locator_spans": score_sets(gold, predicted),
        "locator_spans_by_kind": {
            kind: score_sets(gold_by_kind[kind], predicted_by_kind[kind])
            for kind in sorted(gold_by_kind.keys() | predicted_by_kind.keys())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_DATA / "annotation-v4.0" / "documents",
    )
    parser.add_argument("--texts-root", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(json.dumps(eval_locators(grow_annotated_corpus(args.annotations, args.texts_root)), indent=2))


if __name__ == "__main__":
    main()
