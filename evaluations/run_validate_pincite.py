"""Append pinpoint-validation stages to saved cumulative Documents."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from mellea_lrc.api import (
    Document,
    index_reporter_root_opinion_pages,
    judge_reporter_citation_pinpoints,
    prepare_reporter_citation_pinpoint_evidence,
    read_reporter_citation_propositions,
    reporter_root_opinion_retrieval,
    resolve_reporter_citation_pages,
    review_reporter_citation_full_opinions,
    review_reporter_citation_opinions,
    review_reporter_citation_pinpoint_pages,
)
from mellea_lrc.providers.courtlistener import CourtListenerClient
from mellea_lrc.validation.reporter_citation_full_opinion_review import (
    MODEL_PROFILE as FULL_REVIEW_PROFILE,
)
from mellea_lrc.validation.reporter_citation_full_opinion_review import STAGE as FULL_REVIEW_STAGE
from mellea_lrc.validation.reporter_citation_opinion_review import MODEL_PROFILE as REVIEW_PROFILE
from mellea_lrc.validation.reporter_citation_opinion_review import STAGE as REVIEW_STAGE
from mellea_lrc.validation.reporter_citation_page_resolution import STAGE as RESOLUTION_STAGE
from mellea_lrc.validation.reporter_citation_pinpoint_evidence import STAGE as EVIDENCE_STAGE
from mellea_lrc.validation.reporter_citation_pinpoint_judgment import STAGE as JUDGMENT_STAGE
from mellea_lrc.validation.reporter_citation_pinpoint_page_review import (
    MODEL_PROFILE as PAGE_REVIEW_PROFILE,
)
from mellea_lrc.validation.reporter_citation_pinpoint_page_review import STAGE as PAGE_REVIEW_STAGE
from mellea_lrc.validation.reporter_citation_propositions import MODEL_PROFILE as PROPOSITION_PROFILE
from mellea_lrc.validation.reporter_citation_propositions import STAGE as PROPOSITION_STAGE
from mellea_lrc.validation.reporter_root_opinion_page_index import STAGE as INDEX_STAGE
from mellea_lrc.validation.reporter_root_opinion_retrieval import STAGE as RETRIEVAL_STAGE

_RESULTS_ROOT = Path(__file__).resolve().parent / "results"
_STAGES = (
    RETRIEVAL_STAGE,
    INDEX_STAGE,
    RESOLUTION_STAGE,
    REVIEW_STAGE,
    PROPOSITION_STAGE,
    EVIDENCE_STAGE,
    PAGE_REVIEW_STAGE,
    FULL_REVIEW_STAGE,
    JUDGMENT_STAGE,
)
_MODEL_PROFILES = {
    REVIEW_STAGE: REVIEW_PROFILE,
    PROPOSITION_STAGE: PROPOSITION_PROFILE,
    PAGE_REVIEW_STAGE: PAGE_REVIEW_PROFILE,
    FULL_REVIEW_STAGE: FULL_REVIEW_PROFILE,
}


def _write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


async def run(
    input_documents: Path | None,
    *,
    resume_run: Path | None = None,
    stop_after: str = JUDGMENT_STAGE,
    workers: int = 3,
) -> Path:
    """Persist one cumulative Document per filing, including each completed stage."""
    if isinstance(workers, bool) or workers < 1:
        raise ValueError("Pinpoint workers must be a positive integer")
    if stop_after not in _STAGES:
        raise ValueError(f"Unknown pinpoint stop stage: {stop_after}")
    if resume_run is None:
        if input_documents is None:
            raise ValueError("Specify saved input Documents")
        parent = json.loads((input_documents.parent / "run.json").read_text())
        if parent["status"] != "complete":
            raise ValueError("Pinpoint input must be a complete run")
        run_dir = _RESULTS_ROOT / parent["set"] / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        record = {
            "set": parent["set"],
            "workflow": "validate_pincite",
            "status": "running",
            "filings": parent["filings"],
            "input_documents": str(input_documents),
            "stop_after": stop_after,
            "workers": workers,
            "model_profiles": {stage: asdict(profile) for stage, profile in _MODEL_PROFILES.items()},
            "input_sha256": {
                name: hashlib.sha256((input_documents / f"{name}.json").read_bytes()).hexdigest()
                for name in parent["filings"]
            },
        }
        run_dir.mkdir(parents=True)
        (run_dir / "documents").mkdir()
    else:
        run_dir = resume_run
        record = json.loads((run_dir / "run.json").read_text())
        if record["workflow"] != "validate_pincite":
            raise ValueError("Resume is not a validate_pincite run")
        if record["model_profiles"] != {stage: asdict(profile) for stage, profile in _MODEL_PROFILES.items()}:
            raise ValueError(
                "Pinpoint model profiles changed; restore the run's saved settings before resuming"
            )
        input_documents = Path(record["input_documents"])
        stop_after = record["stop_after"]
        workers = record["workers"]
        parent = json.loads((input_documents.parent / "run.json").read_text())
        if parent["status"] != "complete" or (parent["set"], parent["filings"]) != (
            record["set"],
            record["filings"],
        ):
            raise ValueError("Pinpoint input run changed")
    enabled = _STAGES[: _STAGES.index(stop_after) + 1]
    record["status"] = "running"
    record.pop("error", None)
    _write(run_dir / "run.json", json.dumps(record, indent=2) + "\n")
    try:
        with CourtListenerClient() as client:
            semaphore = asyncio.Semaphore(workers)

            async def process(index: int, name: str) -> None:
                async with semaphore:
                    source = input_documents / f"{name}.json"
                    if hashlib.sha256(source.read_bytes()).hexdigest() != record["input_sha256"][name]:
                        raise ValueError(f"Pinpoint input changed: {name}")
                    original = Document.model_validate_json(source.read_text())
                    prior = tuple(stage for stage in original.stage_runs if stage in _STAGES)
                    if prior != enabled[: len(prior)]:
                        raise ValueError(f"Unexpected pinpoint input stages: {name}")
                    stages = enabled[len(prior) :]
                    artifact = run_dir / "documents" / f"{name}.json"
                    document = (
                        Document.model_validate_json(artifact.read_text()) if artifact.exists() else original
                    )
                    finished = document.stage_runs[len(original.stage_runs) :]
                    if finished != stages[: len(finished)]:
                        raise ValueError(f"Unexpected pinpoint checkpoint: {name}")
                    if document.get_stage(original.stage_runs[-1]) != original:
                        raise ValueError(f"Saved pinpoint input differs: {name}")
                    for stage in stages[len(finished) :]:
                        if stage == RETRIEVAL_STAGE:
                            document = reporter_root_opinion_retrieval(document, client=client)
                        elif stage == INDEX_STAGE:
                            document = index_reporter_root_opinion_pages(document)
                        elif stage == RESOLUTION_STAGE:
                            document = resolve_reporter_citation_pages(document)
                        elif stage == REVIEW_STAGE:
                            document = await review_reporter_citation_opinions(document)
                        elif stage == PROPOSITION_STAGE:
                            document = await read_reporter_citation_propositions(document)
                        elif stage == EVIDENCE_STAGE:
                            document = prepare_reporter_citation_pinpoint_evidence(document)
                        elif stage == PAGE_REVIEW_STAGE:
                            document = await review_reporter_citation_pinpoint_pages(document)
                        elif stage == FULL_REVIEW_STAGE:
                            document = await review_reporter_citation_full_opinions(document)
                        elif stage == JUDGMENT_STAGE:
                            document = judge_reporter_citation_pinpoints(document)
                        if document.stage_runs[-1] != stage:
                            raise ValueError(f"Pinpoint stage did not complete: {stage}")
                        _write(artifact, document.model_dump_json(indent=2) + "\n")
                        print(f"{index}/{len(record['filings'])} {name}: {stage}", flush=True)
                    if not artifact.exists():
                        _write(artifact, document.model_dump_json(indent=2) + "\n")
                    print(f"{index}/{len(record['filings'])} {name}: complete", flush=True)

            tasks = [
                asyncio.create_task(process(index, name)) for index, name in enumerate(record["filings"], 1)
            ]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        _write(run_dir / "run.json", json.dumps(record, indent=2) + "\n")
        raise
    record["status"] = "complete"
    _write(run_dir / "run.json", json.dumps(record, indent=2) + "\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-documents", type=Path)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--stop-after", choices=_STAGES, default=JUDGMENT_STAGE)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if bool(args.input_documents) == bool(args.resume_run):
        parser.error("Choose input Documents or resume a pinpoint run")
    print(
        asyncio.run(
            run(
                args.input_documents.resolve() if args.input_documents else None,
                resume_run=args.resume_run.resolve() if args.resume_run else None,
                stop_after=args.stop_after,
                workers=args.workers,
            )
        )
    )


if __name__ == "__main__":
    main()
