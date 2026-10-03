"""Compare named profiles on one saved full-opinion review, without retrieval."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from mellea_lrc.llm.profiles import NRP_KIMI, NRP_QWEN, OPENROUTER_LUNA, LlmProfile
from mellea_lrc.model.citations.reporter_pinpoint import OpinionReviewScope
from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_pinpoint_review.reviewer import (
    IvrReporterPinpointReviewer,
    ReporterPinpointReviewContext,
)

PROFILES = {profile.name: profile for profile in (NRP_QWEN, NRP_KIMI, OPENROUTER_LUNA)}


def render_comparison(destination: Path) -> None:
    """Render saved outcomes without repeating a model request."""
    results = json.loads((destination / "summary.json").read_text())
    context = json.loads((destination / "context.json").read_text())["context"]
    lines = [
        "# Full-opinion model comparison",
        "",
        f"Citation: `{context['citation_id']}`. Shared source prefix: {len(context['prefix']):,} characters.",
        "",
        "Every profile receives the same complete saved opinion, occurrence context, "
        "schema, and grounding checks. No annotation labels enter the prompt. "
        "Acceptance means schema and grounding validation passed; it does not establish judgment accuracy.",
        "",
        "| Profile | Accepted | Attempts | Seconds | Content judgment | Correct page | Found pages |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    failures = []
    for result in results:
        decision, run = result["decision"], result["run"]
        pages = (
            ", ".join(
                str(page["first"]) if page["first"] == page["last"] else f"{page['first']}–{page['last']}"
                for page in decision["found_pages"]
            )
            if decision
            else "—"
        )
        name = result["profile"]["name"]
        lines.append(
            f"| [{name}]({name}.json) | {'yes' if decision else 'no'} | "
            f"{len(run['attempts']) if run else '—'} | {result['seconds']} | "
            f"{decision['result'] if decision else '—'} | "
            f"{str(decision['correct_page']).lower() if decision else '—'} | {pages or '[]'} |"
        )
        if result["failure"]:
            failures.append(f"{name}: {result['failure']}")
    if failures:
        lines.extend(["", *failures])
    (destination / "README.md").write_text("\n".join(lines).rstrip() + "\n")


async def compare(document_path: Path, citation_id: str, profile_names: list[str]) -> Path:
    document = Document.model_validate_json(await asyncio.to_thread(document_path.read_text))
    input_path = await asyncio.to_thread(document_path.resolve)
    citation = next(item for item in document.citations if item.id == citation_id)
    context = ReporterPinpointReviewContext.from_document(document, citation, OpinionReviewScope.FULL_OPINION)
    destination = Path(__file__).parent / "comparisons" / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    destination.mkdir(parents=True)
    (destination / "context.json").write_text(
        json.dumps(
            {"input_document": str(input_path), "context": asdict(context)},
            indent=2,
            default=lambda value: value.model_dump(mode="json"),
        )
        + "\n"
    )
    print(destination, flush=True)

    async def probe(profile: LlmProfile) -> dict:
        started = monotonic()
        result = {"profile": asdict(profile), "citation_id": citation_id}
        try:
            reviewer = IvrReporterPinpointReviewer.from_profile(profile)
            outcome = await reviewer(context)
            result.update(
                decision=outcome.decision.model_dump(mode="json") if outcome.decision else None,
                run=outcome.run.model_dump(mode="json") if outcome.run else None,
                failure=outcome.failure_reason,
            )
        except Exception as error:
            result.update(decision=None, run=None, failure=f"{type(error).__name__}: {error}")
        result["seconds"] = round(monotonic() - started, 2)
        (destination / f"{profile.name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({key: result[key] for key in ("profile", "seconds", "decision", "failure")}),
            flush=True,
        )
        return result

    results = await asyncio.gather(*(probe(PROFILES[name]) for name in profile_names))
    (destination / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    render_comparison(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", required=True, type=Path)
    parser.add_argument("--citation-id", required=True)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    args = parser.parse_args()
    if len(args.profiles) != len(set(args.profiles)):
        parser.error("Profiles must be distinct")
    asyncio.run(compare(args.document, args.citation_id, args.profiles))


if __name__ == "__main__":
    main()
