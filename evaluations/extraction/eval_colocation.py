"""Evaluate exact colocation membership on annotation-v4.0."""

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


def eval_colocation(corpus: Sequence[GrownAnnotation]) -> dict[str, Any]:
    """Score exact member spans for colocation groups only."""
    gold_groups: set[tuple[str, tuple[tuple[int, int], ...]]] = set()
    predicted_groups: set[tuple[str, tuple[tuple[int, int], ...]]] = set()

    for sample in corpus:
        name = sample.document_name
        gold_members: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in sample.citation_rows:
            if row.get("unit") != "citation" or row.get("kind") not in CASE_LOCATOR_KINDS:
                continue
            colocation_id = row.get("colocation_id")
            span = locator_span(row)
            if isinstance(colocation_id, str) and span is not None:
                gold_members[colocation_id].append(span)
        for members in gold_members.values():
            if len(members) > 1:
                gold_groups.add((name, tuple(sorted(members))))

        records_by_id = {record.citation_id: record for record in sample.document.citations}
        for group in sample.document.colocations:
            member_spans = tuple(
                sorted(
                    (
                        records_by_id[citation_id].locator_span.start,
                        records_by_id[citation_id].locator_span.end,
                    )
                    for citation_id in group
                    if citation_id in records_by_id
                )
            )
            if len(member_spans) > 1:
                predicted_groups.add((name, member_spans))

    return {
        "dataset": "annotation-v4.0 corpus",
        "documents": len(corpus),
        "colocation": score_sets(gold_groups, predicted_groups),
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
    print(json.dumps(eval_colocation(grow_annotated_corpus(args.annotations, args.texts_root)), indent=2))


if __name__ == "__main__":
    main()
