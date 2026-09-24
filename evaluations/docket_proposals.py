"""Diagnose unreviewed docket-site proposals from a saved rule checkpoint.

Proposals are not an admitted stage product, so this diagnostic is separate
from ``score_stages``. It reads a saved Document, recovers ``docket_locators``,
and tests whether proposed sites cover that stage's remaining gold misses.
It makes no model call and changes no citation.

Run from the repository root::

    uv run python -m evaluations.docket_proposals --run-dir local/run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluations.annotations import SETS, annotated_documents, span
from mellea_lrc.extraction._site_hunting.candidates import suspected_dockets
from mellea_lrc.extraction.docket_locator import STAGE as DOCKET_LOCATORS_STAGE
from mellea_lrc.model import FullDocketCitation

COUNT_FIELDS = (
    "documents",
    "eligible_gold_docket_locators",
    "rule_found",
    "exact_proposed_among_rule_misses",
    "remaining_misses",
    "total_proposals",
)


def score_set(data_root: Path, run_dir: Path, name: str) -> dict[str, int]:
    """Count exact half-open spans, excluding the document's index masks."""
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    for _, _, saved, rows in annotated_documents(data_root, run_dir, (name,)):
        document = saved.get_stage(DOCKET_LOCATORS_STAGE)
        rule = {
            citation.locator_span
            for citation in document.citations
            if isinstance(citation, FullDocketCitation)
        }
        proposals = {candidate.locator_span for candidate in suspected_dockets(document)}
        gold = {span(row["locator"]) for row in rows if row["kind"] == "DocketCitation"}
        missed_by_rule = gold - rule
        counts["documents"] += 1
        counts["eligible_gold_docket_locators"] += len(gold)
        counts["rule_found"] += len(gold & rule)
        counts["exact_proposed_among_rule_misses"] += len(missed_by_rule & proposals)
        counts["remaining_misses"] += len(missed_by_rule - proposals)
        counts["total_proposals"] += len(proposals)
    return counts


def score(data_root: Path, run_dir: Path, names: tuple[str, ...] = SETS) -> dict[str, object]:
    sets = {name: score_set(data_root, run_dir, name) for name in names}
    totals = {field: sum(counts[field] for counts in sets.values()) for field in COUNT_FIELDS}
    return {
        "diagnostic": "Unreviewed proposals for gold docket locators missed at docket_locators",
        "sets": sets,
        "totals": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--set", dest="sets", action="append", choices=SETS)
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    args = parser.parse_args()
    result = json.dumps(score(args.data_root, args.run_dir, tuple(args.sets or SETS)), indent=2) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
