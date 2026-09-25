"""Render a saved identity-stage score summary without rerunning predictions.

Run from the repository root::

    uv run python -m evaluations.render_identity_report \
        --summary local/evaluations/reporter-exact/summary.json \
        --output local/evaluations/reporter-exact/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluations.annotations import SETS

FIELDS = (
    ("case_name", "Case name"),
    ("court", "Court"),
    ("date", "Date"),
)


def _count(row: dict[str, Any], key: str) -> int:
    return int(row.get(key, 0))


def _fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})" if denominator else "—"


def _row(name: str, row: dict[str, Any]) -> str:
    cells = (
        name,
        str(_count(row, "documents")),
        str(_count(row, "gold_reporter_roots")),
        str(_count(row, "predicted_reporter_roots")),
        str(_count(row, "decided_total")),
        _fraction(_count(row, "correct_decisions"), _count(row, "scored_decisions")),
        _fraction(_count(row, "correct_decisions"), _count(row, "gold_reporter_roots")),
        _fraction(_count(row, "correct_admissions"), _count(row, "scored_admissions")),
        _fraction(_count(row, "correct_admissions"), _count(row, "gold_correct_reporter_roots")),
    )
    return "| " + " | ".join(cells) + " |"


def _field_row(name: str, label: str, field: dict[str, Any]) -> str:
    cells = (
        name,
        label,
        str(field["gold_stated"]),
        str(field["unique_gold"]),
        str(field["eligible_gold"]),
        _fraction(field["correct_predictions"], field["decided"]),
        _fraction(field["decided"], field["eligible_gold"]),
        _fraction(field["correct_gold"], field["gold_stated"]),
    )
    return "| " + " | ".join(cells) + " |"


def _field_gap_row(name: str, label: str, field: dict[str, Any]) -> str:
    no_unique = field["gold_stated"] - field["unique_gold"]
    if no_unique < 0:
        raise ValueError("Field's unique lookup count exceeds its stated gold count")
    cells = (
        name,
        label,
        str(no_unique),
        str(field["missing_reading"]),
        str(field["misaligned_reading"]),
        str(field["incorrect_predictions"]),
        str(field["undetermined"]),
        str(field["omitted_gold"]),
        str(field["unscored_judgments"]),
    )
    return "| " + " | ".join(cells) + " |"


def _confusion_row(label: str, field: dict[str, Any]) -> str:
    confusion = field["confusion"]
    agrees = confusion["agrees"]
    disagrees = confusion["disagrees"]
    cells = (
        label,
        str(agrees["match"]),
        str(agrees["mismatch"]),
        str(disagrees["match"]),
        str(disagrees["mismatch"]),
        str(agrees["undetermined"]),
        str(disagrees["undetermined"]),
    )
    return "| " + " | ".join(cells) + " |"


def render_identity_report(result: dict[str, Any], *, source_label: str) -> str:
    """Format the saved counts and show the denominator of each metric."""
    if not result.get("stage") or not result.get("sets"):
        raise ValueError("Expected a nonempty identity-stage score summary")
    if set(result["sets"]) - set(SETS):
        raise ValueError("Identity-stage summary includes an unknown set")
    total = result["totals"]
    if _count(total, "documents") != sum(_count(row, "documents") for row in result["sets"].values()):
        raise ValueError("Identity-stage document totals do not agree")
    run = result.get("prediction_run")
    run_lines = (
        [
            f"Prediction run: `{run['root_rules']}` extraction rules; "
            f"docket site hunting {'on' if run['hunt_dockets'] else 'off'}; "
            f"linked docket court retrieval {'on' if run['court_docket_fetch'] else 'off'}.",
            "",
        ]
        if isinstance(run, dict)
        else []
    )
    lines = [
        f"# Identity at `{result['stage']}`",
        "",
        "<!-- Generated from saved counts by evaluations.render_identity_report. -->",
        "",
        f"**Score input:** `{source_label}`. This report makes no provider or model calls.",
        "",
        *run_lines,
        "A decided root is a CORRECT_IDENTITY or WRONG_IDENTITY verdict; deferred roots abstain. "
        "Decision precision divides correct decided roots by scored decided roots. "
        "Decision recall uses every labeled gold reporter root represented by an unmasked full citation, "
        "including roots missed by extraction. "
        "Admission precision and recall count only correct-identity admissions, with all gold-correct "
        "reporter roots as the recall denominator. Unlabeled identity sets show a dash, not a zero score.",
        "",
        "| Set | Documents | Gold reporter roots | Predicted roots | Decided | Decision precision | Decision recall | Admission precision | Admission recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(_row(name, result["sets"][name]) for name in SETS if name in result["sets"])
    lines.extend((_row("Total", total), ""))
    lines.extend(
        (
            "Decision recall across all annotated root types: "
            f"{_fraction(_count(total, 'correct_decisions'), _count(total, 'gold_all_roots'))}.",
            "",
            "Deferred reporter roots: "
            f"{_count(total, 'deferred_total')} "
            f"(review {_count(total, 'deferred_to_review')}, "
            f"ambiguity {_count(total, 'deferred_to_ambiguity')}, "
            f"search {_count(total, 'deferred_to_search')}).",
            "",
            "## Field judgments",
            "",
            "Gold field labels belong to the annotated root occurrence. A predicted root represented "
            "by a later citation is unscored for fields, even when its identity can be scored. "
            "A field comparison is scored only when its reading matches the annotated quote, span, "
            "and available normalized value; a court inferred from its reporter must have the "
            "annotated court ID. Misaligned readings are extraction or normalization errors. "
            "Gold marked `not_stated` is excluded from "
            "field accuracy and recall. An undetermined judgment is an abstention.",
            "",
            "Annotated representative roots available for field scoring: "
            f"{_count(total, 'field_gold_roots')} of {_count(total, 'gold_reporter_roots')} "
            "identity-labeled reporter roots.",
            "",
            "Comparison accuracy is correct determinate judgments divided by scored determinate "
            "judgments. Judgment coverage is determinate judgments divided by aligned readings "
            "at unique exact lookups. Correct / stated gold shows this stage's field recall across "
            "the available unmasked annotated root occurrences with that field stated, including "
            "those without a unique exact lookup.",
            "",
            "| Set | Field | Stated gold | Unique exact | Aligned reading | Comparison accuracy | Judgment coverage | Correct / stated gold |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for name in SETS:
        if name in result["sets"]:
            lines.extend(
                _field_row(name, label, result["sets"][name]["fields"][field]) for field, label in FIELDS
            )
    lines.extend(_field_row("Total", label, total["fields"][field]) for field, label in FIELDS)
    lines.extend(
        (
            "",
            "Gold fields marked `not_stated`: "
            + ", ".join(
                f"{total['fields'][field]['gold_not_stated']} {label.lower()}" for field, label in FIELDS
            )
            + ". These are not counted as agreeing or disagreeing fields.",
            "",
            "A field can lack a unique exact lookup, have no aligned reading, or receive no "
            "determinate judgment. Unscored outputs include judgments on later representative "
            "occurrences, misaligned readings, and unmatched or unlabeled occurrences; they do "
            "not enter comparison accuracy.",
            "",
            "| Set | Field | No unique exact | Missing reading | Misaligned reading | Wrong judgment | Undetermined | Omitted judgment | Unscored outputs |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for name in SETS:
        if name in result["sets"]:
            lines.extend(
                _field_gap_row(name, label, result["sets"][name]["fields"][field]) for field, label in FIELDS
            )
    lines.extend(_field_gap_row("Total", label, total["fields"][field]) for field, label in FIELDS)
    lines.extend(
        (
            "",
            "Of the unscored outputs, judgments on later representative occurrences account for "
            + ", ".join(
                f"{total['fields'][field]['changed_occurrence_judgments']} {label.lower()}"
                for field, label in FIELDS
            )
            + ".",
            "",
            "### Comparison directions across scored sets",
            "",
            "`Agrees → mismatch` rejects an agreeing field; `Disagrees → match` accepts a "
            "disagreeing field. These counts use only aligned readings of the annotated root occurrence.",
            "",
            "| Field | Agrees → match | Agrees → mismatch | Disagrees → match | Disagrees → mismatch | Agrees → undetermined | Disagrees → undetermined |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    lines.extend(_confusion_row(label, total["fields"][field]) for field, label in FIELDS)
    lines.extend(
        (
            "",
            "The reporter-locator membership gate found the queried citation in "
            f"{_count(total, 'locator_membership_match')} unique lookup records, found a different "
            f"listed citation in {_count(total, 'locator_membership_mismatch')}, and could not parse "
            f"the listing in {_count(total, 'locator_membership_unknown')}. This is a saved-response "
            "diagnostic; the annotations do not label the provider's citation list independently.",
            "",
            "The occurrence file beside this summary lists each prediction and each gold root that "
            "was not correctly decided, plus field judgments, readings, returned record values, "
            "locator membership, and gaps, with exact locator spans.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text(encoding="utf-8"))
    report = render_identity_report(result, source_label=str(args.summary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
