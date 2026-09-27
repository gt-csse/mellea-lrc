"""Run the primary extraction and validation workflows into one timestamped artifact."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from mellea_lrc.api import (
    Document,
    docket_root_lookup,
    docket_root_lookup_review,
    govinfo_docket_lookup,
    govinfo_docket_lookup_review,
    grow_roots,
    reporter_root_lookup_ambiguous_llm,
    reporter_root_lookup_unique_llm,
    review_docket_root_equivalence,
    validate_roots,
)
from mellea_lrc.model import FullDocketCitation

_SET = "primary"
_DATA_ROOT = Path(__file__).resolve().parents[2] / "mellea-lrc-datasets"
_RESULTS_ROOT = Path(__file__).resolve().parent / "results" / _SET
_ROOT_STAGE = "10_roots"
_ROOT_STAGES = (
    "1_full_reporter_locators",
    "2_docket_locators",
    "3_docket_locator_site_hunting",
    "4_docket_entries",
    "5_colocations",
    "6_case_names",
    "7_courts",
    "8_dates",
    "9_pin_cites",
    _ROOT_STAGE,
)
_RUN_STAGES = (
    *_ROOT_STAGES,
    "11_docket_root_equivalence_review",
    "12_reporter_root_lookup",
    "13_reporter_root_lookup_ambiguous",
    "14_reporter_root_lookup_unique_llm",
    "15_reporter_root_lookup_ambiguous_llm",
    "16_docket_root_lookup",
    "17_docket_root_lookup_review",
    "18_govinfo_docket_lookup",
    "19_govinfo_docket_lookup_review",
)
_REPORTER_REVIEW_INPUT_STAGE = "13_reporter_root_lookup_ambiguous"
_REPORTER_REVIEW_INPUT_STAGES = _RUN_STAGES[: _RUN_STAGES.index(_REPORTER_REVIEW_INPUT_STAGE) + 1]
_DOCKET_LOOKUP_STAGE = "16_docket_root_lookup"
_DOCKET_REVIEW_INPUT_STAGE = "17_docket_root_lookup_review"
_DOCKET_REVIEW_INPUT_STAGES = _RUN_STAGES[: _RUN_STAGES.index(_DOCKET_REVIEW_INPUT_STAGE) + 1]


def _write_json(path: Path, value: object) -> None:
    """Replace a saved artifact only after its complete JSON has been written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _check_source(document: Document, source: Document) -> None:
    """Reject a stale checkpoint before it can trigger any provider calls."""
    if (
        document.source_path is None
        or source.source_path is None
        or Path(document.source_path).resolve() != Path(source.source_path).resolve()
        or document.source_metadata.format != source.source_metadata.format
    ):
        raise ValueError(f"Saved Document source differs from current source: {source.source_path}")
    for field in ("text", "preprocessing_metadata", "index_spans"):
        if getattr(document, field) != getattr(source, field):
            raise ValueError(f"Saved Document {field} differs from current source: {source.source_path}")


def _load_document(path: Path, source: Document) -> Document:
    document = Document.model_validate_json(path.read_text(encoding="utf-8"))
    _check_source(document, source)
    return document


def _has_transient_docket_lookup_failure(document: Document) -> bool:
    """A saved provider failure leaves a docket lookup run incomplete."""
    for root in document.roots:
        if not isinstance(root, FullDocketCitation):
            continue
        for lookup in (root.docket_lookup, root.govinfo_docket_lookup):
            if lookup is None:
                continue
            for failure in (lookup.failure, *(attempt.failure for attempt in lookup.attempts)):
                if failure is not None and (
                    failure.failure_type == "transport_error"
                    or failure.upstream_status_code == 429
                    or (failure.upstream_status_code is not None and failure.upstream_status_code >= 500)
                ):
                    return True
    return False


def _reuse_docket_lookup(document: Document, saved: Document) -> Document:
    """Replay the unchanged docket roots' saved lookup nodes after reporter review."""
    checkpoint = saved.get_stage(_DOCKET_LOOKUP_STAGE)
    for root in checkpoint.roots:
        if isinstance(root, FullDocketCitation) and root.nodes[-1].stage == _DOCKET_LOOKUP_STAGE:
            document = document.replace_citation(root)
    return document.complete(_DOCKET_LOOKUP_STAGE)


async def _run(
    data_root: Path,
    results_root: Path,
    from_roots_documents: Path | None,
    resume_run: Path | None = None,
    from_reporter_review_documents: Path | None = None,
    reuse_docket_lookups: bool = False,
    from_docket_review_documents: Path | None = None,
) -> Path:
    if (
        sum(
            item is not None
            for item in (from_roots_documents, from_reporter_review_documents, from_docket_review_documents)
        )
        > 1
    ):
        raise ValueError("Choose one saved Document checkpoint")
    if reuse_docket_lookups and from_reporter_review_documents is None and resume_run is None:
        raise ValueError("Reusing docket lookups requires reporter-review Documents")
    if resume_run is None:
        filenames = sorted(
            json.loads((data_root / _SET / "documents.json").read_text(encoding="utf-8"))["documents"]
        )
        if not filenames:
            raise ValueError(f"No documents in {_SET}")
        started_at = datetime.now(UTC)
        run_dir = results_root / started_at.strftime("%Y-%m-%dT%H-%M-%SZ")
        run_dir.mkdir(parents=True, exist_ok=False)
        documents_dir = run_dir / "documents"
        documents_dir.mkdir()
        run_record = {
            "set": _SET,
            "started_at": started_at.isoformat(),
            "status": "running",
            "filings": filenames,
            "data_root": str(data_root),
            "from_roots_documents": str(from_roots_documents) if from_roots_documents else None,
            "from_reporter_review_documents": (
                str(from_reporter_review_documents) if from_reporter_review_documents else None
            ),
            "from_docket_review_documents": (
                str(from_docket_review_documents) if from_docket_review_documents else None
            ),
            "reuse_docket_lookups": reuse_docket_lookups,
            "source_sha256": {
                filename: hashlib.sha256(
                    (data_root / _SET / "documents_txt" / filename).read_bytes()
                ).hexdigest()
                for filename in filenames
            },
        }
        record_path = run_dir / "run.json"
        _write_json(record_path, run_record)
    else:
        run_dir = resume_run
        record_path = run_dir / "run.json"
        run_record = json.loads(record_path.read_text(encoding="utf-8"))
        if run_record.get("set") != _SET:
            raise ValueError(f"Run is not for {_SET}: {run_dir}")
        data_root = Path(run_record.get("data_root") or _DATA_ROOT)
        saved_roots = run_record.get("from_roots_documents")
        from_roots_documents = Path(saved_roots) if saved_roots else None
        saved_reporter_review = run_record.get("from_reporter_review_documents")
        from_reporter_review_documents = Path(saved_reporter_review) if saved_reporter_review else None
        saved_docket_review = run_record.get("from_docket_review_documents")
        from_docket_review_documents = Path(saved_docket_review) if saved_docket_review else None
        reuse_docket_lookups = bool(run_record.get("reuse_docket_lookups", False))
        if (
            sum(
                item is not None
                for item in (
                    from_roots_documents,
                    from_reporter_review_documents,
                    from_docket_review_documents,
                )
            )
            > 1
        ):
            raise ValueError("Saved run has more than one Document checkpoint")
        if reuse_docket_lookups and from_reporter_review_documents is None:
            raise ValueError("Saved run cannot reuse docket lookups without reporter-review Documents")
        filenames = run_record["filings"]
        current_filenames = sorted(
            json.loads((data_root / _SET / "documents.json").read_text(encoding="utf-8"))["documents"]
        )
        if filenames != current_filenames:
            raise ValueError("Run filing manifest differs from the current dataset")
        recorded_hashes = run_record.get("source_sha256")
        if recorded_hashes is not None:
            current_hashes = {
                filename: hashlib.sha256(
                    (data_root / _SET / "documents_txt" / filename).read_bytes()
                ).hexdigest()
                for filename in filenames
            }
            if recorded_hashes != current_hashes:
                raise ValueError("Run source content differs from the original run")
        documents_dir = run_dir / "documents"
        if not documents_dir.is_dir():
            raise ValueError(f"Run has no Documents directory: {run_dir}")
        run_record["data_root"] = str(data_root)
        run_record["status"] = "running"
        run_record.pop("completed_at", None)
        _write_json(record_path, run_record)

    try:
        # Check every saved checkpoint before any provider-backed stage starts.
        sources = {
            filename: Document.from_source(data_root / _SET / "documents_txt" / filename)
            for filename in filenames
        }
        completed: set[str] = set()
        for filename in filenames:
            source = sources[filename]
            artifact = documents_dir / f"{filename}.json"
            if artifact.exists():
                saved_document = _load_document(artifact, source)
                if saved_document.stage_runs == _RUN_STAGES and not _has_transient_docket_lookup_failure(
                    saved_document
                ):
                    completed.add(filename)
            if filename not in completed and from_roots_documents is not None:
                saved = from_roots_documents / f"{filename}.json"
                roots = _load_document(saved, source).get_stage(_ROOT_STAGE)
                if roots.stage_runs != _ROOT_STAGES:
                    raise ValueError(f"Saved roots are incomplete for {filename}")
            if filename not in completed and from_reporter_review_documents is not None:
                saved = from_reporter_review_documents / f"{filename}.json"
                saved_document = _load_document(saved, source)
                ready = saved_document.get_stage(_REPORTER_REVIEW_INPUT_STAGE)
                if ready.stage_runs != _REPORTER_REVIEW_INPUT_STAGES:
                    raise ValueError(f"Saved reporter-review input is incomplete for {filename}")
                if reuse_docket_lookups:
                    saved_document.get_stage(_DOCKET_LOOKUP_STAGE)
            if filename not in completed and from_docket_review_documents is not None:
                saved = from_docket_review_documents / f"{filename}.json"
                ready = _load_document(saved, source).get_stage(_DOCKET_REVIEW_INPUT_STAGE)
                if ready.stage_runs != _DOCKET_REVIEW_INPUT_STAGES:
                    raise ValueError(f"Saved docket-review input is incomplete for {filename}")

        for index, filename in enumerate(filenames, start=1):
            source = sources[filename]
            artifact = documents_dir / f"{filename}.json"
            if filename in completed:
                print(f"{index}/{len(filenames)} {filename} (saved)", flush=True)
                continue
            if artifact.exists():
                print(f"{index}/{len(filenames)} {filename} (incomplete; rerunning)", flush=True)
            if from_docket_review_documents is not None:
                saved = from_docket_review_documents / f"{filename}.json"
                document = _load_document(saved, source).get_stage(_DOCKET_REVIEW_INPUT_STAGE)
                document = govinfo_docket_lookup(document)
                document = await govinfo_docket_lookup_review(document)
            elif from_reporter_review_documents is not None:
                saved = from_reporter_review_documents / f"{filename}.json"
                saved_document = _load_document(saved, source)
                document = saved_document.get_stage(_REPORTER_REVIEW_INPUT_STAGE)
                document = await reporter_root_lookup_unique_llm(document)
                document = await reporter_root_lookup_ambiguous_llm(document)
                document = (
                    _reuse_docket_lookup(document, saved_document)
                    if reuse_docket_lookups
                    else docket_root_lookup(document)
                )
                document = await docket_root_lookup_review(document)
                document = govinfo_docket_lookup(document)
                document = await govinfo_docket_lookup_review(document)
            elif from_roots_documents is None:
                document = await grow_roots(
                    source,
                    hunt_dockets=True,
                    review_docket_roots=True,
                )
            else:
                saved = from_roots_documents / f"{filename}.json"
                document = _load_document(saved, source).get_stage(_ROOT_STAGE)
                document = await review_docket_root_equivalence(document)
            if from_docket_review_documents is None and from_reporter_review_documents is None:
                document = await validate_roots(document)
            if document.stage_runs != _RUN_STAGES:
                raise ValueError(f"Run did not complete every stage for {filename}")
            _write_json(artifact, document.model_dump(mode="json"))
            print(f"{index}/{len(filenames)} {filename}", flush=True)
        incomplete = [
            filename
            for filename in filenames
            if _has_transient_docket_lookup_failure(
                _load_document(documents_dir / f"{filename}.json", sources[filename])
            )
        ]
        if incomplete:
            raise RuntimeError(
                "Docket search ended with transient provider failures in "
                + ", ".join(incomplete)
                + "; resume this run after the provider recovers"
            )
    except BaseException:
        run_record["status"] = "failed"
        _write_json(record_path, run_record)
        raise

    run_record["status"] = "complete"
    run_record["completed_at"] = datetime.now(UTC).isoformat()
    _write_json(record_path, run_record)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--from-roots-documents",
        type=Path,
        help="Resume saved Documents at their 10_roots checkpoint before the docket-root review",
    )
    parser.add_argument(
        "--from-reporter-review-documents",
        type=Path,
        help="Resume saved Documents at stage 13 before the reporter LLM reviews",
    )
    parser.add_argument(
        "--from-docket-review-documents",
        type=Path,
        help="Resume saved Documents at stage 17 before GovInfo docket lookup and review",
    )
    parser.add_argument(
        "--reuse-docket-lookups",
        action="store_true",
        help="Reuse saved stage-16 docket lookups while rerunning reporter and docket LLM reviews",
    )
    parser.add_argument(
        "--resume-run", type=Path, help="Continue an interrupted run in its existing timestamp directory"
    )
    args = parser.parse_args()
    if (
        sum(
            item is not None
            for item in (
                args.from_roots_documents,
                args.from_reporter_review_documents,
                args.from_docket_review_documents,
            )
        )
        > 1
    ):
        parser.error("Choose one saved Document checkpoint")
    if args.reuse_docket_lookups and not args.from_reporter_review_documents:
        parser.error("--reuse-docket-lookups requires --from-reporter-review-documents")
    if args.resume_run and any(
        value is not None
        for value in (
            args.data_root,
            args.results_root,
            args.from_roots_documents,
            args.from_reporter_review_documents,
            args.from_docket_review_documents,
        )
    ):
        parser.error("--resume-run reuses its run.json inputs; do not pass other input paths")
    run_dir = asyncio.run(
        _run(
            (args.data_root or _DATA_ROOT).resolve(),
            (args.results_root or _RESULTS_ROOT).resolve(),
            args.from_roots_documents.resolve() if args.from_roots_documents else None,
            args.resume_run.resolve() if args.resume_run else None,
            args.from_reporter_review_documents.resolve() if args.from_reporter_review_documents else None,
            args.reuse_docket_lookups,
            args.from_docket_review_documents.resolve() if args.from_docket_review_documents else None,
        )
    )
    print(run_dir)


if __name__ == "__main__":
    main()
