"""Evaluate reported reporter locators and audit-admitted docket locators."""

from __future__ import annotations

import argparse
import json
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
from mellea_lrc.core.citations import DocketCitation, FullCaseCitation


def eval_locators(corpus: Sequence[GrownAnnotation]) -> dict[str, Any]:
    """Score the two complete locator kinds at their meaningful stage.

    Reporter locators are scored as read. Docket locators are scored only after
    the docket audit: a raw candidate that the audit withdraws is neither a
    docket prediction nor a false positive in this metric. Colocation has its
    own evaluator because it is a pre-audit structural relation.
    """
    reporter_gold: set[tuple[str, int, int]] = set()
    docket_gold: set[tuple[str, int, int]] = set()
    reporter_predicted: set[tuple[str, int, int]] = set()
    docket_predicted: set[tuple[str, int, int]] = set()

    for sample in corpus:
        name = sample.document_name
        for row in sample.citation_rows:
            if row.get("unit") != "citation":
                continue
            span = locator_span(row)
            if span is None:
                continue
            occurrence = (name, *span)
            if row.get("kind") == "FullCaseCitation":
                reporter_gold.add(occurrence)
            elif row.get("kind") == "DocketCitation":
                docket_gold.add(occurrence)

        for record in sample.document.citations:
            occurrence = (name, record.locator_span.start, record.locator_span.end)
            if isinstance(record.source, FullCaseCitation):
                reporter_predicted.add(occurrence)
            elif isinstance(record.source, DocketCitation) and not record.withdrawn:
                docket_predicted.add(occurrence)

    return {
        "dataset": "annotation-v4.0 corpus",
        "documents": len(corpus),
        "reporter_locators": score_sets(reporter_gold, reporter_predicted),
        "docket_locators": score_sets(docket_gold, docket_predicted),
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
