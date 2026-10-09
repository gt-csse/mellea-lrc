"""Score a diagnostic pinpoint cohort offline, without claiming a workflow run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluations.annotations import align_citation_annotations, annotations_by_site
from evaluations.validate_pincite import (
    FULL_OPINION_SUBSTAGE,
    PAGE_SUPPORT_SUBSTAGE,
    _entries,
    _review_verdict,
    _score_found_page_locations,
    _score_judgments,
    _score_page_locations,
)
from mellea_lrc.model import Document
from mellea_lrc.model.citations.reporter_pinpoint import ReporterCitationSupportReview

_SUBSTAGES = {"cited_pages": PAGE_SUPPORT_SUBSTAGE, "full_opinion": FULL_OPINION_SUBSTAGE}


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _score_target(document: Document, case: dict, review: ReporterCitationSupportReview) -> dict[str, Any]:
    substage = _SUBSTAGES[case["scope"]]
    target = next(citation for citation in document.citations if citation.id == case["citation_id"])
    row = align_citation_annotations([target], annotations_by_site(document))[0][1]
    if row is None or row["id"] != case["annotation_id"]:
        raise ValueError(f"Target does not align with {case['annotation_id']}")
    if review.scope.value != case["scope"] or review not in _entries(
        target, "reporter_support_reviews", substage
    ):
        raise ValueError(f"Target review does not belong to {substage}")
    # This transient scoring view keeps native roots and attachment histories.
    # Only the target's chosen review is exposed as a prediction. Never persist
    # this view: the saved Document remains the genuine pending native result.
    view = document.model_copy(
        update={
            "citations": tuple(
                citation.model_copy(
                    update={"reporter_support_reviews": (review,) if citation.id == target.id else ()}
                )
                for citation in document.citations
            )
        }
    )
    judgments, unscored = _score_judgments(view, substage)
    content = judgments["total"]
    page = _score_page_locations(view, substage)["total"]
    found = _score_found_page_locations(view, substage)["total"]
    decision = review.decision
    attempts = len(review.ivr.attempts) if review.ivr is not None else 0
    finding = row.get("validation", {}).get("pincite", {})
    return {
        "valid": decision is not None,
        "failure_reason": review.failure_reason,
        "model": review.ivr.model if review.ivr is not None else None,
        "attempts": attempts,
        "repair_attempts": max(0, attempts - 1),
        "result": decision.result.value if decision is not None else None,
        "content_verdict": _review_verdict(review).value,
        "correct_page": decision.correct_page if decision is not None else None,
        "found_pages": [target.model_dump(mode="json") for target in decision.found_pages]
        if decision is not None
        else [],
        "gold": {
            "content": finding.get("label"),
            "correct_page": finding.get("correct_page"),
            "found_pages": finding.get("found_pages", []),
        },
        "content": {
            "correct": content.correct,
            "predicted": content.predicted,
            "undetermined": content.undetermined,
            "unscored_definitive": unscored,
        },
        "page": {
            "correct": page.correct,
            "predicted": page.predicted,
            "unknown": page.unlocated,
            "unscored": page.unscored,
        },
        "found": {
            "correct": found.correct,
            "predicted": found.predicted,
            "unknown": found.unlocated,
            "unscored": found.unscored,
        },
    }


def _aggregate(records: list[dict[str, Any] | None]) -> dict[str, Any]:
    available = [record for record in records if record is not None]
    seconds = [record["seconds"] for record in available if isinstance(record.get("seconds"), (int, float))]
    result: dict[str, Any] = {
        "selected": len(records),
        "scored": len(available),
        "pending_or_missing_native": len(records) - len(available),
        "valid": sum(record["valid"] for record in available),
        "failed": sum(not record["valid"] for record in available),
        "repaired_reviews": sum(record["repair_attempts"] > 0 for record in available),
        "repair_attempts": sum(record["repair_attempts"] for record in available),
        "models": dict(Counter(record["model"] or "unspecified" for record in available)),
        "latency": {
            "observed": len(seconds),
            "total_seconds": round(sum(seconds), 2) if seconds else None,
            "mean_seconds": round(sum(seconds) / len(seconds), 2) if seconds else None,
        },
    }
    for category in ("content", "page", "found"):
        counts: Counter[str] = Counter()
        for record in available:
            counts.update(record[category])
        result[category] = dict(counts)
        result[category]["reference_agreement" if category == "found" else "precision"] = (
            counts["correct"] / counts["predicted"] if counts["predicted"] else None
        )
    return result


def _fraction(score: dict[str, Any], category: str) -> str:
    values = score[category]
    return f"{values.get('correct', 0)}/{values.get('predicted', 0)}" if values.get("predicted") else "—"


def _page(value: bool | None) -> str:
    return "unknown" if value is None else str(value).lower()


def _readme(summary: dict[str, Any]) -> str:
    profile = summary["comparison_profile"]
    label = profile["name"]
    lines = [
        f"# Scoped {label} pinpoint comparison",
        "",
        f"This selected diagnostic cohort compares saved baseline reviews with {label} ({profile['model']}) reviews of the same retained citation evidence. "
        "The cases emphasize disagreements and difficult source conditions; these figures are not an unbiased corpus estimate. "
        "Only the target citation is scored in each native Document. Full-document gold denominators and corpus recall are omitted.",
        "",
        "The production defaults are unchanged. This renderer runs offline and makes no model or retrieval calls. "
        + ("The NRP probe uses no paid-model calls. " if label.startswith("nrp_") else "")
        + "Baselines retain their actual saved model and earlier prompt. New reviews use the current prompt, "
        "so differences cannot be attributed to model choice alone.",
        "",
        "Content and page placement are scored independently with the existing evaluation functions. "
        "A cited-pages negative remains undetermined for content; full-opinion contradiction or not_found is definitive at the review boundary. "
        "These are review-boundary metrics, not the later final judgment. Found-page consistency uses positive native reference ranges "
        "and known wrong written targets; an unlisted page is not assumed false.",
        "",
        "Native Documents contain the target production support review and grounded evidence appended to the preceding checkpoint. "
        "They do not claim completion of the review substage, semantic stage, or workflow. Failures and IVR repair attempts are retained.",
        "",
        "Fractions below are correct/predicted. Unknown and unscored page assessments are separate in summary.json.",
        "",
        "| Scope | Review | Valid | Repaired reviews / extra attempts | Content | Content unknown | Page | Page unknown | Found consistency | Mean seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scope, comparisons in summary["by_scope"].items():
        for name, score in comparisons.items():
            name = label if name == "current" else name
            latency = score["latency"]["mean_seconds"]
            lines.append(
                f"| {scope} | {name} | {score['valid']}/{score['selected']} | "
                f"{score['repaired_reviews']}/{score['repair_attempts']} | {_fraction(score, 'content')} | "
                f"{score['content'].get('undetermined', 0)} | {_fraction(score, 'page')} | "
                f"{score['page'].get('unknown', 0)} | {_fraction(score, 'found')} | "
                f"{latency if latency is not None else '—'} |"
            )
    lines.extend(
        [
            "",
            f"| Citation | Scope | Gold content / page | Baseline content | {label} content | Baseline page → {label} page | {label} valid / attempts |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["cases"]:
        case, baseline, current = row["case"], row["baseline"], row["current"]
        finding = baseline["gold"]
        baseline_content = baseline["content_verdict"]
        current_content = current["content_verdict"] if current is not None else row["status"]
        page = _page(current["correct_page"]) if current is not None else "pending"
        valid = (
            f"{str(current['valid']).lower()} / {current['attempts']}" if current is not None else "pending"
        )
        filename = f"{case['annotation_id']}__{case['scope']}.json"
        lines.append(
            f"| [{case['annotation_id']}](outcomes/{filename}) | {case['scope']} | "
            f"{finding['content']} / {_page(finding['correct_page'])} | {baseline_content} | "
            f"{current_content} | {_page(baseline['correct_page'])} → {page} | {valid} |"
        )
    lines.append("")
    failures = [
        (row["case"]["annotation_id"], row["case"]["scope"], row["failure_reason"])
        for row in summary["cases"]
        if row["failure_reason"]
    ]
    if failures:
        lines.extend(["## Recorded failures", ""])
        lines.extend(f"- {identifier} ({scope}): {reason}" for identifier, scope, reason in failures)
        lines.append("")
    return "\n".join(lines)


def render_comparison(destination: Path) -> dict[str, Any]:
    """Render saved target reviews using native content and page gold semantics."""
    profile = json.loads((destination / "profile.json").read_text())
    originals: dict[str, Document] = {}
    rows = []
    for input_path in sorted((destination / "inputs").glob("*.json")):
        saved_input = json.loads(input_path.read_text())
        case = saved_input["case"]
        source = case["document"]
        if source not in originals:
            originals[source] = Document.model_validate_json(Path(source).read_text())
        original = originals[source]
        baseline = ReporterCitationSupportReview.model_validate(saved_input["baseline"])
        row = {
            "case": case,
            "baseline": _score_target(original, case, baseline),
            "current": None,
            "status": "pending",
            "failure_reason": None,
        }
        outcome_path = destination / "outcomes" / input_path.name
        native_path = destination / "documents" / input_path.name
        if outcome_path.is_file():
            outcome = json.loads(outcome_path.read_text())
            if outcome["case"] != case:
                raise ValueError(f"Outcome case differs from input: {input_path.name}")
            row["failure_reason"] = outcome.get("failure_reason")
            if native_path.is_file():
                native = Document.model_validate_json(native_path.read_text())
                target = next(citation for citation in native.citations if citation.id == case["citation_id"])
                reviews = _entries(target, "reporter_support_reviews", _SUBSTAGES[case["scope"]])
                if not reviews:
                    raise ValueError(f"Native target review is absent: {input_path.name}")
                row["current"] = _score_target(native, case, reviews[-1])
                row["current"]["seconds"] = outcome.get("seconds")
                row["failure_reason"] = reviews[-1].failure_reason or row["failure_reason"]
                row["status"] = "accepted" if reviews[-1].decision is not None else "failed"
                row["seconds"] = outcome.get("seconds")
            else:
                row["status"] = "missing_native_document"
                row["failure_reason"] = row["failure_reason"] or "Saved outcome has no native Document"
        rows.append(row)
    if not rows:
        raise ValueError(f"No saved comparison inputs: {destination / 'inputs'}")
    summary = {
        "scope": "Selected diagnostic target reviews; precision and unknown counts only, no corpus recall",
        "comparison_profile": {"name": profile["name"], "model": profile["model"]},
        "by_scope": {
            scope: {
                name: _aggregate([row[name] for row in rows if row["case"]["scope"] == scope])
                for name in ("baseline", "current")
            }
            for scope in _SUBSTAGES
        },
        "cases": rows,
    }
    _save(destination / "summary.json", summary)
    (destination / "README.md").write_text(_readme(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", type=Path, required=True)
    args = parser.parse_args()
    render_comparison(args.render)


if __name__ == "__main__":
    main()
