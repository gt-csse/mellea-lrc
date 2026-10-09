"""Test one named model profile on saved pinpoint reviews without retrieval.

Inputs are recovered before each review, not from its final judgment. Historical
outputs are references, not controls for a model-only comparison when prompts
have changed. Gold annotations enter the separate scorer only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from pydantic import BaseModel, TypeAdapter

from mellea_lrc.llm.profiles import LlmProfile
from mellea_lrc.model.citations.reporter_pinpoint import OpinionReviewScope
from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_pinpoint_review import review_citation
from mellea_lrc.validation.reporter_pinpoint_review.reviewer import (
    IvrReporterPinpointReviewer,
    ReporterPinpointReviewContext,
    ReporterPinpointReviewOutcome,
)

BEFORE = {
    "cited_pages": "validate_pincite.citation_preparation.evidence",
    "full_opinion": "validate_pincite.support_review.page_review",
}
SUBSTAGES = {
    "cited_pages": "validate_pincite.support_review.page_review",
    "full_opinion": "validate_pincite.support_review.full_opinion_review",
}


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=lambda item: item.model_dump(mode="json") if isinstance(item, BaseModel) else str(item),
        )
        + "\n"
    )
    temporary.replace(path)


async def compare(cohort_path: Path, profile_name: str, destination: Path | None) -> Path:
    profile = LlmProfile.from_env(profile_name)
    secret = profile.resolve().api_key
    cohort = json.loads(await asyncio.to_thread(cohort_path.read_text))
    destination = destination or (
        Path(__file__).parent / "results" / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    )
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in (("cohort.json", cohort), ("profile.json", asdict(profile))):
        path = destination / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Cannot resume with a different {name}")
        save(path, value)

    originals: dict[str, Document] = {}
    checkpoints: dict[tuple[str, str], Document] = {}
    inputs = []
    identifiers = set()
    for case in cohort["cases"]:
        name = f"{case['annotation_id']}__{case['scope']}"
        if name in identifiers:
            raise ValueError(f"Duplicate review: {name}")
        identifiers.add(name)
        path, scope = case["document"], case["scope"]
        if path not in originals:
            originals[path] = Document.model_validate_json(await asyncio.to_thread(Path(path).read_text))
        original = originals[path]
        key = (path, scope)
        if key not in checkpoints:
            checkpoints[key] = original.get_substage(BEFORE[scope])
        document = checkpoints[key]
        citation = next(item for item in document.citations if item.id == case["citation_id"])
        context = ReporterPinpointReviewContext.from_document(document, citation, OpinionReviewScope(scope))
        original_citation = next(item for item in original.citations if item.id == citation.id)
        baseline = next(
            entry
            for entry in reversed(original_citation.reporter_support_reviews)
            if entry.scope.value == scope and entry.evidence_index == context.evidence_index
        )
        saved_input = {"case": case, "context": asdict(context), "baseline": baseline}
        input_path = destination / "inputs" / f"{name}.json"
        # Freeze the production context before any requests, without annotation gold.
        encoded = json.loads(json.dumps(saved_input, default=lambda item: item.model_dump(mode="json")))
        if input_path.exists() and json.loads(input_path.read_text()) != encoded:
            raise ValueError(f"Current input differs from the saved review: {name}")
        save(input_path, encoded)
        inputs.append((name, case, document, citation, context))
    print(f"Saved {len(inputs)} review inputs: {destination}", flush=True)

    semaphore = asyncio.Semaphore(3)

    async def probe(name, case, document, citation, context) -> None:
        path = destination / "outcomes" / f"{name}.json"
        native_path = destination / "documents" / f"{name}.json"
        if path.exists() and native_path.exists():
            return
        async with semaphore:
            started = monotonic()
            if path.exists():
                result = json.loads(path.read_text())
                outcome = TypeAdapter(ReporterPinpointReviewOutcome).validate_python(
                    {key: result[key] for key in ("decision", "run", "failure_reason")}
                )
            else:
                try:
                    outcome = await IvrReporterPinpointReviewer.from_profile(profile)(context)
                except Exception as error:
                    outcome = ReporterPinpointReviewOutcome(
                        None, failure_reason=f"{type(error).__name__}: {error}".replace(secret, "[redacted]")
                    )
                result = {
                    "case": case,
                    "profile": asdict(profile),
                    "decision": outcome.decision,
                    "run": outcome.run,
                    "failure_reason": outcome.failure_reason,
                    "seconds": round(monotonic() - started, 2),
                }
                save(path, result)

            async def saved_reviewer(received):
                if received != context:
                    raise ValueError("Offline materialization received a different review context")
                return outcome

            reviewed = await review_citation(
                document,
                citation,
                substage=SUBSTAGES[case["scope"]],
                scope=OpinionReviewScope(case["scope"]),
                reviewer=saved_reviewer,
            )
            serialized = reviewed.model_dump_json(indent=2)
            restored = Document.model_validate_json(serialized)
            if restored != reviewed:
                raise ValueError(f"Native document roundtrip differs: {name}")
            native_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = native_path.with_suffix(".json.tmp")
            temporary.write_text(serialized + "\n")
            temporary.replace(native_path)
            print(
                json.dumps(
                    {
                        "review": name,
                        "seconds": result["seconds"],
                        "result": outcome.decision.result.value if outcome.decision else None,
                        "attempts": len(outcome.run.attempts) if outcome.run else 0,
                        "failure": outcome.failure_reason,
                    }
                ),
                flush=True,
            )

    await asyncio.gather(*(probe(*item) for item in inputs))
    from experiments.pinpoint.score_saved_reviews import render_comparison

    render_comparison(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--destination", type=Path, help="Reuse a saved experiment without repeating calls")
    args = parser.parse_args()
    asyncio.run(compare(args.cohort, args.profile, args.destination))


if __name__ == "__main__":
    main()
