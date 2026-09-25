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
            f"docket site hunting {'on' if run['hunt_dockets'] else 'off'}.",
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
            "The occurrence file beside this summary lists each prediction and each gold root that "
            "was not correctly decided, including its exact locator span.",
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
