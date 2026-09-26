"""Render saved per-stage score summaries as Markdown without rescoring.

Run from the repository root::

    python -m evaluations.render_stage_report \
        --summary-dir local/evaluations/root-stage-eval \
        --output local/evaluations/extraction-stages.md

Each stage's ``summary.json`` is produced by ``evaluations.score_stages``.
The renderer only formats those saved counts; it does not read filings, call a
model, or select representative examples.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evaluations.annotations import SETS
from evaluations.score_stages import ELIGIBILITY, FIELD_STAGES, RELATIONSHIP_STAGES, STAGES
from mellea_lrc.extraction.case_names import STAGE as CASE_NAMES_STAGE
from mellea_lrc.extraction.colocations import STAGE as COLOCATIONS_STAGE
from mellea_lrc.extraction.courts import STAGE as COURTS_STAGE
from mellea_lrc.extraction.dates import STAGE as DATES_STAGE
from mellea_lrc.extraction.docket_entries import STAGE as DOCKET_ENTRIES_STAGE
from mellea_lrc.extraction.docket_locator import STAGE as DOCKET_LOCATORS_STAGE
from mellea_lrc.extraction.docket_site_hunting import STAGE as DOCKET_HUNT_STAGE
from mellea_lrc.extraction.full_reporter_locator import STAGE as FULL_REPORTER_LOCATORS_STAGE
from mellea_lrc.extraction.pin_cites import STAGE as PIN_CITES_STAGE
from mellea_lrc.extraction.roots import STAGE as ROOTS_STAGE
from mellea_lrc.extraction.short_reporter_locator import STAGE as SHORT_REPORTER_CITATIONS_STAGE

STAGE_NAMES = {
    FULL_REPORTER_LOCATORS_STAGE: "Full reporter locators (rule)",
    DOCKET_LOCATORS_STAGE: "Docket locators (rule)",
    DOCKET_HUNT_STAGE: "Docket locators (site hunting)",
    DOCKET_ENTRIES_STAGE: "Docket entries",
    COLOCATIONS_STAGE: "Colocation",
    CASE_NAMES_STAGE: "Case names",
    COURTS_STAGE: "Courts",
    DATES_STAGE: "Dates",
    PIN_CITES_STAGE: "Pin cites",
    ROOTS_STAGE: "Root formation",
    SHORT_REPORTER_CITATIONS_STAGE: "Short reporter citations",
}
REPORT_ORDER = tuple(STAGE_NAMES)
SCORED_PREDICTION_COUNTS = (
    "exact_spans",
    "wrong_span",
    "missing_span",
    "duplicate_prediction",
    "unmatched_locator",
    "outside_stage_scope",
)


def _count(summary: Mapping[str, Any], name: str) -> int:
    return int(summary.get(name, 0))


def _fraction(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "—"
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


def _field_row(label: str, summary: Mapping[str, Any]) -> str:
    exact = _count(summary, "exact_spans")
    scored = sum(_count(summary, name) for name in SCORED_PREDICTION_COUNTS)
    normalized = _count(summary, "normalization_correct")
    cells = (
        label,
        str(_count(summary, "eligible_gold")),
        str(_count(summary, "predicted")),
        _fraction(exact, scored),
        _fraction(exact, _count(summary, "gold_with_span")),
        _fraction(normalized, _count(summary, "normalization_checked")),
        _fraction(normalized, _count(summary, "gold_with_normalization")),
    )
    return "| " + " | ".join(cells) + " |"


def _group_row(label: str, summary: Mapping[str, Any]) -> str:
    cells = (
        label,
        f"{_count(summary, 'eligible_locators')}/{_count(summary, 'gold_locators')}",
        _fraction(_count(summary, "exact_groups"), _count(summary, "gold_groups")),
    )
    return "| " + " | ".join(cells) + " |"


def _colocation_row(label: str, summary: Mapping[str, Any]) -> str:
    cells = (
        label,
        f"{_count(summary, 'eligible_locators')}/{_count(summary, 'gold_locators')}",
        _fraction(_count(summary, "exact_groups"), _count(summary, "gold_groups")),
        _fraction(_count(summary, "correct_links"), _count(summary, "predicted_links")),
        _fraction(_count(summary, "correct_links"), _count(summary, "gold_links")),
    )
    return "| " + " | ".join(cells) + " |"


def render_stage_report(results: Mapping[str, Mapping[str, Any]], *, source_label: str) -> str:
    """Format the completed extraction stages in a fixed display order."""
    if set(STAGE_NAMES) != set(STAGES):
        raise ValueError("Stage display names must cover every scored stage")
    if not results or set(results) - set(STAGES):
        raise ValueError("Expected one or more known extraction stages")
    available_order = tuple(stage for stage in REPORT_ORDER if stage in results)
    for stage, result in results.items():
        if result.get("stage") != stage:
            raise ValueError(f"Summary stage mismatch for {stage}")
    set_names = tuple(name for name in SETS if name in results[available_order[0]]["sets"])
    expected_sets = set(set_names)
    for stage in available_order:
        if set(results[stage]["sets"]) != expected_sets:
            raise ValueError(f"Set coverage differs in {stage}")

    documents = _count(results[available_order[0]]["totals"], "documents")
    if documents == 0 or any(
        _count(results[stage]["totals"], "documents") != documents for stage in available_order
    ):
        raise ValueError("Every stage must score the same nonempty document set")
    lines = [
        "# Incremental extraction stage evaluation",
        "",
        "<!-- Generated by evaluations.render_stage_report from saved score summaries. -->",
        "",
        f"**Score input:** `{source_label}`. **Documents:** {documents}. "
        "This formatter does not rescore documents or call a model.",
        "",
        "The saved summaries were scored from stage checkpoints by "
        "`evaluations.score_stages`. Total rows pool occurrences across sets "
        "rather than averaging set percentages. Refresh those summaries after "
        "changing annotations or saved Documents; rendering alone cannot update scores.",
        "",
        "Each score concerns only readings or relationships written by that stage. "
        "A downstream field is eligible only when its parent locator existed before the stage; "
        "these are conditional incremental scores, not end-to-end citation or identity scores.",
        "",
        "For fields, span precision uses scored predictions and span recall uses eligible "
        "annotated spans. Predictions for unlabeled fields are excluded from precision. "
        "Normalization accuracy uses matched source evidence with independent normalized gold; "
        "normalization recall uses all eligible normalized gold. A dash means there is no "
        "denominator. Inferred courts may have no source span. Case names and short reporter "
        "citations have no independent normalized-gold target.",
        "",
        "Colocation pair scores count unordered pairs of full locators placed in the "
        "same colocation group. Root formation reports exact groups only. Root/leaf "
        "linkage scoring belongs after `grow_leaves`.",
        "",
        "## Totals at a glance",
        "",
        "| Field stage | Eligible gold | Predictions | Span P | Span R | Norm accuracy | Norm recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in available_order:
        if stage in FIELD_STAGES:
            lines.append(_field_row(STAGE_NAMES[stage], results[stage]["totals"]))
    lines.extend(
        (
            "",
            "| Group stage | Eligible full locators | Exact groups / gold groups |",
            "| --- | ---: | ---: |",
        )
    )
    for stage in available_order:
        if stage in RELATIONSHIP_STAGES:
            lines.append(_group_row(STAGE_NAMES[stage], results[stage]["totals"]))
    if COLOCATIONS_STAGE in results:
        colocation_totals = results[COLOCATIONS_STAGE]["totals"]
        lines.extend(
            (
                "",
                "Colocation pairs: "
                f"{_fraction(_count(colocation_totals, 'correct_links'), _count(colocation_totals, 'predicted_links'))} "
                "precision; "
                f"{_fraction(_count(colocation_totals, 'correct_links'), _count(colocation_totals, 'gold_links'))} "
                "recall. Root groups include singletons.",
            )
        )

    lines.extend(
        (
            "",
            "## Diagnostic counts",
            "",
            "A wrong-span prediction also leaves its gold field missed, so these columns "
            "are not disjoint. Normalization failures cover all unnormalizable predictions; "
            "normalization mismatches require matched evidence with normalized gold.",
            "",
            "| Stage | Missed gold | Wrong span | Normalization failures | "
            "Normalization mismatches | Unlabeled field | Unmatched locator |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for stage in available_order:
        if stage not in FIELD_STAGES:
            continue
        total = results[stage]["totals"]
        values = (
            STAGE_NAMES[stage],
            *(
                str(_count(total, key))
                for key in (
                    "missed_gold",
                    "wrong_span",
                    "normalization_errors",
                    "normalization_wrong",
                    "unlabeled_field",
                    "unmatched_locator",
                )
            ),
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(("", "## Results by stage and set", ""))
    for stage in available_order:
        result = results[stage]
        lines.extend((f"### {STAGE_NAMES[stage]}", "", f"Eligibility: {ELIGIBILITY[stage]}.", ""))
        if stage in FIELD_STAGES:
            lines.extend(
                (
                    "| Set (documents) | Eligible gold | Predictions | Span P | Span R | "
                    "Norm accuracy | Norm recall |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                )
            )
            row = _field_row
        elif stage == COLOCATIONS_STAGE:
            lines.extend(
                (
                    "| Set (documents) | Eligible full locators | Exact groups / gold groups | "
                    "Correct colocated pairs / predicted pairs | "
                    "Correct colocated pairs / gold pairs |",
                    "| --- | ---: | ---: | ---: | ---: |",
                )
            )
            row = _colocation_row
        else:
            lines.extend(
                (
                    "| Set (documents) | Eligible full locators | Exact groups / gold groups |",
                    "| --- | ---: | ---: |",
                )
            )
            row = _group_row
        for name in set_names:
            summary = result["sets"][name]
            lines.append(row(f"{name} ({_count(summary, 'documents')})", summary))
        lines.extend((row(f"**Total** ({documents})", result["totals"]), ""))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write Markdown here; otherwise print to stdout")
    args = parser.parse_args()
    results = {}
    for stage in STAGES:
        path = args.summary_dir / stage / "summary.json"
        results[stage] = json.loads(path.read_text(encoding="utf-8"))
    report = render_stage_report(results, source_label=str(args.summary_dir))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
