"""Evaluate the three reporter-root field judgments at each completed stage.

This reads saved Documents and annotations. It never reruns extraction, a
lookup, or a model review. Each stage scores only judgments it wrote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluations.annotations import SETS, annotated_documents
from evaluations.score_identity import evaluate as score_unique_lookup
from evaluations.score_reporter_lookup_ambiguous import evaluate as score_ambiguous_lookup
from evaluations.score_reporter_lookup_ambiguous_llm import evaluate as score_ambiguous_review
from evaluations.score_reporter_lookup_unique_llm import evaluate as score_unique_review
from evaluations.score_stages import STAGES as EXTRACTION_STAGES
from mellea_lrc.validation.stage_names import (
    REPORTER_ROOT_LOOKUP,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
    REPORTER_ROOT_LOOKUP_UNIQUE_LLM,
)

FIELDS = ("case_name", "court", "date")
EVALUATORS = {
    REPORTER_ROOT_LOOKUP: score_unique_lookup,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS: score_ambiguous_lookup,
    REPORTER_ROOT_LOOKUP_UNIQUE_LLM: score_unique_review,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM: score_ambiguous_review,
}


def _stage_order(data_root: Path, run_dir: Path, sets: tuple[str, ...]) -> tuple[str, ...]:
    order: tuple[str, ...] | None = None
    for name, filename, document, _ in annotated_documents(data_root, run_dir, sets):
        if order is None:
            order = document.stage_runs
        elif document.stage_runs != order:
            raise ValueError(f"{name}/{filename}: completed stages differ from the rest of the run")
    if not order:
        raise ValueError("No completed stages in the selected run")
    unsupported = set(order) - set(EXTRACTION_STAGES) - set(EVALUATORS)
    if unsupported:
        raise ValueError(f"Completed stages have no evaluator: {', '.join(sorted(unsupported))}")
    selected = tuple(stage for stage in order if stage in EVALUATORS)
    if not selected:
        raise ValueError("Run has no completed reporter field-judgment stage")
    return selected


def _field_scores(summary: dict[str, Any], stage: str) -> dict[str, dict[str, float | None]]:
    fields = summary["fields"]
    return {
        field: {
            "precision": fields[field]["precision"],
            "recall": fields[field]["conditional_recall" if stage == REPORTER_ROOT_LOOKUP else "recall"],
        }
        for field in FIELDS
    }


def evaluate(data_root: Path, run_dir: Path, sets: tuple[str, ...] = ("primary",)) -> dict[str, Any]:
    """Score only case-name, court, and date judgments newly written per stage."""
    selected = tuple(dict.fromkeys(sets))
    if not selected or any(name not in SETS for name in selected):
        raise ValueError("Select one or more known annotated sets")
    order = _stage_order(data_root, run_dir, selected)
    stages: dict[str, Any] = {}
    occurrences: dict[str, Any] = {}
    for stage in order:
        scored = EVALUATORS[stage](data_root, run_dir, selected)
        if set(scored["sets"]) != set(selected):
            raise ValueError(f"{stage}: scored sets differ from the requested sets")
        occurrences[stage] = scored.pop("occurrences")
        stages[stage] = {
            "sets": {name: _field_scores(scored["sets"][name], stage) for name in selected},
            "totals": _field_scores(scored["totals"], stage),
        }
    return {
        "sets": list(selected),
        "stage_order": list(order),
        "stages": stages,
        "occurrences": occurrences,
    }


def render_report(result: dict[str, Any], *, source_label: str) -> str:
    """Render only the requested precision and recall for the three fields."""
    lines = [
        "# Incremental reporter field judgments",
        "",
        f"<!-- Generated from {source_label} by evaluations.evaluate_run. -->",
        "",
        "Each row scores judgments written by that stage. Precision divides correct decided "
        "judgments by decided judgments; recall divides correctly judged annotated roots by "
        "annotated roots eligible for that stage. An unresolved judgment counts as a recall miss.",
        "",
        "| Stage | Set | Field | Precision | Recall |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for stage in result["stage_order"]:
        for name in result["sets"]:
            for field in FIELDS:
                score = result["stages"][stage]["sets"][name][field]
                precision = "—" if score["precision"] is None else f"{score['precision']:.1%}"
                recall = "—" if score["recall"] is None else f"{score['recall']:.1%}"
                lines.append(f"| {stage} | {name} | {field.replace('_', ' ')} | {precision} | {recall} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--set", dest="sets", action="append", choices=SETS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir / "evaluation"
    result = evaluate(args.data_root, args.run_dir, tuple(args.sets or ("primary",)))
    occurrences = result.pop("occurrences")
    summary_path = output_dir / "summary.json"
    report = render_report(result, source_label=str(summary_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "occurrences.json").write_text(json.dumps(occurrences, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(output_dir / "report.md")


if __name__ == "__main__":
    main()
