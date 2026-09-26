"""Score every completed stage in one saved run and write one cumulative report.

Run from the repository root::

    uv run python -m evaluations.evaluate_run --run-dir local/reporter-exact-court-primary

The default set is ``primary`` and the default output directory is
``<run-dir>/evaluation``. This command reads saved Documents and annotations;
it does not repeat extraction, contact a provider, or call a model. Select
other sets explicitly with repeated ``--set`` options.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evaluations.annotations import SETS, annotated_documents
from evaluations.docket_proposals import score as score_docket_proposals
from evaluations.render_identity_report import render_identity_report
from evaluations.render_reporter_lookup_ambiguous import render_report as render_ambiguity_report
from evaluations.render_reporter_lookup_unique_llm import render_report as render_unique_llm_report
from evaluations.render_stage_report import render_stage_report
from evaluations.score_identity import evaluate as score_identity
from evaluations.score_reporter_lookup_ambiguous import evaluate as score_ambiguity
from evaluations.score_reporter_lookup_unique_llm import evaluate as score_unique_llm
from evaluations.score_stages import STAGES as EXTRACTION_STAGES
from evaluations.score_stages import evaluate as score_extraction_stage
from mellea_lrc.extraction.docket_locator import STAGE as DOCKET_LOCATORS_STAGE
from mellea_lrc.validation.reporter_root_lookup import STAGE as REPORTER_IDENTITY_STAGE
from mellea_lrc.validation.reporter_root_lookup_ambiguous import STAGE as REPORTER_AMBIGUITY_STAGE
from mellea_lrc.validation.reporter_root_lookup_unique_llm import STAGE as REPORTER_UNIQUE_LLM_STAGE

SCORED_STAGES = frozenset(
    (*EXTRACTION_STAGES, REPORTER_IDENTITY_STAGE, REPORTER_AMBIGUITY_STAGE, REPORTER_UNIQUE_LLM_STAGE)
)


def _stage_order(data_root: Path, run_dir: Path, sets: tuple[str, ...]) -> tuple[str, ...]:
    """Require every selected Document to have the same completed stage chain."""
    order: tuple[str, ...] | None = None
    found = False
    for name, filename, document, _ in annotated_documents(data_root, run_dir, sets):
        found = True
        if order is None:
            order = document.stage_runs
        elif document.stage_runs != order:
            raise ValueError(f"{name}/{filename}: completed stages differ from the rest of the run")
    if not found or not order:
        raise ValueError("No completed stages in the selected run")
    unsupported = set(order) - SCORED_STAGES
    if unsupported:
        raise ValueError(f"Completed stages have no evaluator: {', '.join(sorted(unsupported))}")
    return order


def evaluate(data_root: Path, run_dir: Path, sets: tuple[str, ...] = ("primary",)) -> dict[str, Any]:
    """Return cumulative scores and occurrence details from saved checkpoints."""
    selected = tuple(dict.fromkeys(sets))
    if not selected or any(name not in SETS for name in selected):
        raise ValueError("Select one or more known annotated sets")
    order = _stage_order(data_root, run_dir, selected)
    summaries: dict[str, dict[str, Any]] = {}
    occurrences: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for stage in order:
        if stage == REPORTER_IDENTITY_STAGE:
            result = score_identity(data_root, run_dir, selected)
        elif stage == REPORTER_AMBIGUITY_STAGE:
            result = score_ambiguity(data_root, run_dir, selected)
        elif stage == REPORTER_UNIQUE_LLM_STAGE:
            result = score_unique_llm(data_root, run_dir, selected)
        else:
            result = score_extraction_stage(data_root, run_dir, stage, selected)
        occurrences[stage] = result.pop("occurrences")
        if set(result["sets"]) != set(selected):
            raise ValueError(f"{stage}: scored sets differ from the requested sets")
        summaries[stage] = result
    diagnostics = (
        {"docket_site_proposals": score_docket_proposals(data_root, run_dir, selected)}
        if DOCKET_LOCATORS_STAGE in order
        else {}
    )
    return {
        "sets": list(selected),
        "stage_order": list(order),
        "stages": summaries,
        "diagnostics": diagnostics,
        "occurrences": occurrences,
    }


def _demote_headings(markdown: str) -> str:
    return re.sub(r"^(#+) ", lambda match: f"#{match.group(1)} ", markdown, flags=re.MULTILINE)


def render_report(result: dict[str, Any], *, source_label: str) -> str:
    """Compose stage reports, including identity field judgments, from scores."""
    stages = result["stages"]
    order = result["stage_order"]
    if set(order) != set(stages) or not order:
        raise ValueError("Cumulative summary must contain every completed stage once")
    extraction = {stage: stages[stage] for stage in order if stage in EXTRACTION_STAGES}
    lines = [
        "# Cumulative stage evaluation",
        "",
        "<!-- Generated by evaluations.evaluate_run from saved Document checkpoints. -->",
        "",
        f"**Sets:** {', '.join(result['sets'])}. **Score input:** `{source_label}`.",
        "",
        "Each completed stage is scored independently from its saved checkpoint. The "
        "reporter sections include case-name, court, and date field judgments; "
        "extraction field scores use different denominators.",
        "",
    ]
    if extraction:
        lines.extend(
            (
                _demote_headings(render_stage_report(extraction, source_label=source_label)).rstrip(),
                "",
            )
        )
    if REPORTER_IDENTITY_STAGE in stages:
        identity = stages[REPORTER_IDENTITY_STAGE]
        if set(identity["sets"]) != set(result["sets"]):
            raise ValueError("Identity set coverage differs from the cumulative report")
        lines.extend(
            (
                _demote_headings(render_identity_report(identity, source_label=source_label)).rstrip(),
                "",
            )
        )
    if REPORTER_AMBIGUITY_STAGE in stages:
        ambiguity = stages[REPORTER_AMBIGUITY_STAGE]
        if set(ambiguity["sets"]) != set(result["sets"]):
            raise ValueError("Ambiguity set coverage differs from the cumulative report")
        lines.extend(
            (
                _demote_headings(render_ambiguity_report(ambiguity, source_label=source_label)).rstrip(),
                "",
            )
        )
    if REPORTER_UNIQUE_LLM_STAGE in stages:
        unique_llm = stages[REPORTER_UNIQUE_LLM_STAGE]
        if set(unique_llm["sets"]) != set(result["sets"]):
            raise ValueError("Unique-reporter model set coverage differs from the cumulative report")
        lines.extend(
            (
                _demote_headings(render_unique_llm_report(unique_llm, source_label=source_label)).rstrip(),
                "",
            )
        )
    proposals = result.get("diagnostics", {}).get("docket_site_proposals")
    if proposals is not None:
        if set(proposals["sets"]) != set(result["sets"]):
            raise ValueError("Docket proposal set coverage differs from the cumulative report")
        lines.extend(
            (
                "## Docket site proposal diagnostic",
                "",
                "These proposals have not been reviewed or admitted as locators. The counts "
                "measure how often the candidate generator covers docket locators missed by "
                "the rule stage.",
                "",
                "| Set | Gold docket locators | Found by rule | Rule misses proposed | "
                "Remaining gold misses | Total proposals |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for name in result["sets"]:
            row = proposals["sets"][name]
            lines.append(
                f"| {name} | {row['eligible_gold_docket_locators']} | {row['rule_found']} | "
                f"{row['exact_proposed_among_rule_misses']} | {row['remaining_misses']} | "
                f"{row['total_proposals']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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
