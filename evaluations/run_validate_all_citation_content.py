"""Append citation-content validation stages to saved cumulative Documents."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from evaluations import complete_stage_boundary
from evaluations.sources import validate_source
from mellea_lrc.api import (
    Document,
    index_reporter_root_opinion_pages,
    prepare_reporter_citation_evidence,
    read_reporter_citation_propositions,
    resolve_reporter_citation_pages,
    retrieve_reporter_root_opinions,
    review_reporter_citation_full_opinions_llm,
    review_reporter_citation_pages_llm,
    select_reporter_citation_opinions,
    validate_all_citation_content,
)
from mellea_lrc.llm.profiles import load_profile
from mellea_lrc.providers.courtlistener import CourtListenerClient
from mellea_lrc.validation.index_reporter_root_opinion_pages import SUBSTAGE as INDEX_SUBSTAGE
from mellea_lrc.validation.judge_reporter_citation_content import SUBSTAGE as JUDGMENT_SUBSTAGE
from mellea_lrc.validation.prepare_reporter_citation_evidence import SUBSTAGE as EVIDENCE_SUBSTAGE
from mellea_lrc.validation.read_reporter_citation_propositions import SUBSTAGE as PROPOSITION_SUBSTAGE
from mellea_lrc.validation.resolve_reporter_citation_pages import SUBSTAGE as RESOLUTION_SUBSTAGE
from mellea_lrc.validation.retrieve_reporter_root_opinions import SUBSTAGE as RETRIEVAL_SUBSTAGE
from mellea_lrc.validation.review_reporter_citation_full_opinions_llm import SUBSTAGE as FULL_REVIEW_SUBSTAGE
from mellea_lrc.validation.review_reporter_citation_pages_llm import SUBSTAGE as PAGE_REVIEW_SUBSTAGE
from mellea_lrc.validation.select_reporter_citation_opinions import SUBSTAGE as REVIEW_SUBSTAGE
from mellea_lrc.workflows import validate_workflow_checkpoint

_RESULTS_ROOT = Path(__file__).resolve().parent / "results"
_SUBSTAGES = (
    RETRIEVAL_SUBSTAGE,
    INDEX_SUBSTAGE,
    RESOLUTION_SUBSTAGE,
    REVIEW_SUBSTAGE,
    PROPOSITION_SUBSTAGE,
    EVIDENCE_SUBSTAGE,
    PAGE_REVIEW_SUBSTAGE,
    FULL_REVIEW_SUBSTAGE,
    JUDGMENT_SUBSTAGE,
)
_MODEL_SUBSTAGES = (REVIEW_SUBSTAGE, PROPOSITION_SUBSTAGE, PAGE_REVIEW_SUBSTAGE, FULL_REVIEW_SUBSTAGE)


def _resolved_model_profiles() -> dict[str, dict[str, object]]:
    """Capture the configured profile descriptors without resolving credentials."""
    return {substage: asdict(load_profile(substage)) for substage in _MODEL_SUBSTAGES}


def _write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


async def run(
    input_documents: Path | None,
    *,
    resume_run: Path | None = None,
    input_substage: str | None = None,
    stop_after: str = JUDGMENT_SUBSTAGE,
    workers: int = 3,
) -> Path:
    """Persist one cumulative Document per filing, including each completed substage."""
    if isinstance(workers, bool) or workers < 1:
        raise ValueError("Citation content workers must be a positive integer")
    if stop_after not in _SUBSTAGES:
        raise ValueError(f"Unknown citation content stop substage: {stop_after}")
    if input_substage is not None and input_substage not in _SUBSTAGES:
        raise ValueError(f"Unknown citation content input substage: {input_substage}")
    model_profiles = _resolved_model_profiles()
    if resume_run is None:
        if input_documents is None:
            raise ValueError("Specify saved input Documents")
        parent = json.loads((input_documents.parent / "run.json").read_text())
        if parent["status"] != "complete":
            raise ValueError("Citation content input must be a complete run")
        run_dir = _RESULTS_ROOT / parent["set"] / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        record = {
            "set": parent["set"],
            "workflow": "validate_all_citation_content",
            "status": "running",
            "filings": parent["filings"],
            "input_documents": str(input_documents),
            "input_substage": input_substage,
            "stop_after": stop_after,
            "workers": workers,
            "model_profiles": model_profiles,
            "input_sha256": {
                name: hashlib.sha256((input_documents / f"{name}.json").read_bytes()).hexdigest()
                for name in parent["filings"]
            },
        }
    else:
        run_dir = resume_run
        record = json.loads((run_dir / "run.json").read_text())
        if record["workflow"] != "validate_all_citation_content":
            raise ValueError("Resume is not a validate_all_citation_content run")
        if record["model_profiles"] != model_profiles:
            raise ValueError(
                "Citation content model profiles changed; restore the run's saved settings before resuming"
            )
        if input_substage is not None and input_substage != record.get("input_substage"):
            raise ValueError("Resume must use the saved input substage")
        input_substage = record.get("input_substage")
        if input_substage is not None and input_substage not in _SUBSTAGES:
            raise ValueError(f"Unknown citation content input substage: {input_substage}")
        input_documents = Path(record["input_documents"])
        stop_after = record["stop_after"]
        workers = record["workers"]
        parent = json.loads((input_documents.parent / "run.json").read_text())
        if parent["status"] != "complete" or (parent["set"], parent["filings"]) != (
            record["set"],
            record["filings"],
        ):
            raise ValueError("Citation content input run changed")
    if input_substage is not None and _SUBSTAGES.index(input_substage) >= _SUBSTAGES.index(stop_after):
        raise ValueError("Stop substage must follow the input substage")
    enabled = _SUBSTAGES[: _SUBSTAGES.index(stop_after) + 1]
    inputs: dict[str, Document] = {}
    saved_documents: dict[str, Document] = {}
    for name in record["filings"]:
        source = input_documents / f"{name}.json"
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["input_sha256"][name]:
            raise ValueError(f"Citation content input changed: {name}")
        original = Document.model_validate_json(raw)
        validate_source(original)
        if input_substage is not None:
            original = original.get_substage(input_substage)
        inputs[name] = original
        artifact = run_dir / "documents" / f"{name}.json"
        if artifact.exists():
            saved = Document.model_validate_json(artifact.read_bytes())
            validate_source(saved, source=original)
            saved_documents[name] = saved
    if resume_run is None:
        run_dir.mkdir(parents=True)
        (run_dir / "documents").mkdir()
    record["status"] = "running"
    record.pop("error", None)
    _write(run_dir / "run.json", json.dumps(record, indent=2) + "\n")
    try:
        with CourtListenerClient() as client:
            semaphore = asyncio.Semaphore(workers)

            async def process(index: int, name: str) -> None:
                async with semaphore:
                    original = inputs[name]
                    prior = tuple(substage for substage in original.substage_runs if substage in _SUBSTAGES)
                    if prior != enabled[: len(prior)]:
                        raise ValueError(f"Unexpected citation content input stages: {name}")
                    validate_workflow_checkpoint(original, "validate_all_citation_content")
                    stages = enabled[len(prior) :]
                    artifact = run_dir / "documents" / f"{name}.json"
                    document = saved_documents.get(name, original)
                    finished = document.substage_runs[len(original.substage_runs) :]
                    if finished != stages[: len(finished)]:
                        raise ValueError(f"Unexpected citation content checkpoint: {name}")
                    last = original.runs[-1]
                    recover = document.get_stage if last.kind == "stage" else document.get_substage
                    if recover(last.name) != original:
                        raise ValueError(f"Saved citation content input differs: {name}")
                    validate_workflow_checkpoint(document, "validate_all_citation_content")
                    document = complete_stage_boundary(document, _SUBSTAGES)
                    full_workflow = stop_after == JUDGMENT_SUBSTAGE

                    def save_checkpoint(checkpoint: Document) -> None:
                        completed = checkpoint.substage_runs[len(original.substage_runs) :]
                        if completed != stages[: len(completed)]:
                            raise ValueError(f"Unexpected citation content checkpoint: {name}")
                        substage = checkpoint.substage_runs[-1]
                        if (full_workflow or substage in _MODEL_SUBSTAGES) and (
                            _resolved_model_profiles() != model_profiles
                        ):
                            raise ValueError(
                                f"Citation content model profiles changed during execution: {substage}"
                            )
                        _write(artifact, checkpoint.model_dump_json(indent=2) + "\n")
                        if checkpoint.runs[-1].kind == "substage":
                            print(f"{index}/{len(record['filings'])} {name}: {substage}", flush=True)

                    if full_workflow:
                        if _resolved_model_profiles() != model_profiles:
                            raise ValueError("Citation content model profiles changed during execution")
                        document = await validate_all_citation_content(
                            document, client=client, checkpoint=save_checkpoint
                        )
                    else:
                        for substage in stages[len(finished) :]:
                            if substage in _MODEL_SUBSTAGES and _resolved_model_profiles() != model_profiles:
                                raise ValueError(
                                    f"Citation content model profiles changed during execution: {substage}"
                                )
                            if substage == RETRIEVAL_SUBSTAGE:
                                document = retrieve_reporter_root_opinions(document, client=client)
                            elif substage == INDEX_SUBSTAGE:
                                document = index_reporter_root_opinion_pages(document)
                            elif substage == RESOLUTION_SUBSTAGE:
                                document = resolve_reporter_citation_pages(document)
                            elif substage == REVIEW_SUBSTAGE:
                                document = await select_reporter_citation_opinions(document)
                            elif substage == PROPOSITION_SUBSTAGE:
                                document = await read_reporter_citation_propositions(document)
                            elif substage == EVIDENCE_SUBSTAGE:
                                document = prepare_reporter_citation_evidence(document)
                            elif substage == PAGE_REVIEW_SUBSTAGE:
                                document = await review_reporter_citation_pages_llm(document)
                            elif substage == FULL_REVIEW_SUBSTAGE:
                                document = await review_reporter_citation_full_opinions_llm(document)
                            if document.substage_runs[-1] != substage:
                                raise ValueError(f"Citation content substage did not complete: {substage}")
                            document = complete_stage_boundary(document, _SUBSTAGES)
                            save_checkpoint(document)
                    if document.substage_runs[len(original.substage_runs) :] != stages:
                        raise ValueError(
                            f"Citation content run did not complete every enabled substage: {name}"
                        )
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
    parser.add_argument("--input-substage", choices=_SUBSTAGES)
    parser.add_argument("--stop-after", choices=_SUBSTAGES, default=JUDGMENT_SUBSTAGE)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if bool(args.input_documents) == bool(args.resume_run):
        parser.error("Choose input Documents or resume a citation content run")
    print(
        asyncio.run(
            run(
                args.input_documents.resolve() if args.input_documents else None,
                resume_run=args.resume_run.resolve() if args.resume_run else None,
                input_substage=args.input_substage,
                stop_after=args.stop_after,
                workers=args.workers,
            )
        )
    )


if __name__ == "__main__":
    main()
