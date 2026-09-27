"""Run the primary extraction and validation workflows into one timestamped artifact."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from mellea_lrc.api import Document, grow_roots, review_docket_root_equivalence, validate_roots
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
    "docket_root_lookup",
    "docket_root_lookup_review",
    "reporter_root_lookup",
    "reporter_root_lookup_ambiguous",
    "reporter_root_lookup_unique_llm",
    "reporter_root_lookup_ambiguous_llm",
)


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
    """A saved 429/5xx search is incomplete evidence, even if stages finished."""
    for root in document.roots:
        if not isinstance(root, FullDocketCitation) or root.docket_lookup is None:
            continue
        for attempt in root.docket_lookup.attempts:
            failure = attempt.failure
            if (
                failure is not None
                and failure.upstream_status_code is not None
                and (failure.upstream_status_code == 429 or failure.upstream_status_code >= 500)
            ):
                return True
    return False


async def _run(
    data_root: Path,
    results_root: Path,
    from_roots_documents: Path | None,
    resume_run: Path | None = None,
) -> Path:
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

        for index, filename in enumerate(filenames, start=1):
            source = sources[filename]
            artifact = documents_dir / f"{filename}.json"
            if filename in completed:
                print(f"{index}/{len(filenames)} {filename} (saved)", flush=True)
                continue
            if artifact.exists():
                print(f"{index}/{len(filenames)} {filename} (incomplete; rerunning)", flush=True)
            if from_roots_documents is None:
                document = await grow_roots(
                    source,
                    hunt_dockets=True,
                    review_docket_roots=True,
                )
            else:
                saved = from_roots_documents / f"{filename}.json"
                document = _load_document(saved, source).get_stage(_ROOT_STAGE)
                document = await review_docket_root_equivalence(document)
            document = await validate_roots(document)
            if document.stage_runs != _RUN_STAGES:
                raise ValueError(f"Run did not complete every stage for {filename}")
            _write_json(artifact, document.model_dump(mode="json"))
            print(f"{index}/{len(filenames)} {filename}", flush=True)
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
        "--resume-run", type=Path, help="Continue an interrupted run in its existing timestamp directory"
    )
    args = parser.parse_args()
    if args.resume_run and any(
        value is not None for value in (args.data_root, args.results_root, args.from_roots_documents)
    ):
        parser.error("--resume-run reuses its run.json inputs; do not pass other input paths")
    run_dir = asyncio.run(
        _run(
            (args.data_root or _DATA_ROOT).resolve(),
            (args.results_root or _RESULTS_ROOT).resolve(),
            args.from_roots_documents.resolve() if args.from_roots_documents else None,
            args.resume_run.resolve() if args.resume_run else None,
        )
    )
    print(run_dir)


if __name__ == "__main__":
    main()
