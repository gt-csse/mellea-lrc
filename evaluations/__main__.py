"""Run the primary extraction and validation workflows into one timestamped artifact."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from datetime import UTC, date, datetime
from pathlib import Path

from mellea_lrc.api import (
    Document,
    corroborate_root_locator_bodies,
    courtlistener_opinion_locator_body_search,
    courtlistener_recap_locator_body_search,
    docket_root_lookup,
    docket_root_lookup_review,
    govinfo_docket_lookup,
    govinfo_docket_lookup_review,
    govinfo_opinion_locator_body_search,
    grow_roots,
    reporter_root_lookup_ambiguous_llm,
    reporter_root_lookup_unique_llm,
    review_docket_root_equivalence,
    review_locator_body_evidence,
    validate_roots,
)
from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerConfig
from mellea_lrc.model import FullDocketCitation
from mellea_lrc.validation.body_search.courtlistener_opinion import STAGE as _COURTLISTENER_OPINION_STAGE
from mellea_lrc.validation.body_search.courtlistener_recap import STAGE as _COURTLISTENER_RECAP_STAGE
from mellea_lrc.validation.body_search.govinfo import STAGE as _GOVINFO_OPINION_STAGE
from mellea_lrc.validation.locator_body_review import STAGE as _LOCATOR_BODY_REVIEW_STAGE

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
    _COURTLISTENER_OPINION_STAGE,
    _COURTLISTENER_RECAP_STAGE,
    _GOVINFO_OPINION_STAGE,
    _LOCATOR_BODY_REVIEW_STAGE,
)
_BODY_SEARCH_STAGES = (
    _COURTLISTENER_OPINION_STAGE,
    _COURTLISTENER_RECAP_STAGE,
    _GOVINFO_OPINION_STAGE,
)
_REPORTER_REVIEW_INPUT_STAGE = "13_reporter_root_lookup_ambiguous"
_REPORTER_REVIEW_INPUT_STAGES = _RUN_STAGES[: _RUN_STAGES.index(_REPORTER_REVIEW_INPUT_STAGE) + 1]
_DOCKET_LOOKUP_STAGE = "16_docket_root_lookup"
_DOCKET_REVIEW_INPUT_STAGE = "17_docket_root_lookup_review"
_DOCKET_REVIEW_INPUT_STAGES = _RUN_STAGES[: _RUN_STAGES.index(_DOCKET_REVIEW_INPUT_STAGE) + 1]
_VALIDATION_INPUT_STAGE = "19_govinfo_docket_lookup_review"
_VALIDATION_INPUT_STAGES = _RUN_STAGES[: _RUN_STAGES.index(_VALIDATION_INPUT_STAGE) + 1]
_FILING_DATES_FILE = "filing_dates.json"


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


def _load_primary_filing_dates(
    data_root: Path, filenames: list[str]
) -> tuple[dict[str, object], dict[str, date], str]:
    """Read a complete, sourced filing-date manifest before provider work."""
    path = data_root / _SET / _FILING_DATES_FILE
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("filings"), dict):
        raise ValueError(f"Filing-date manifest needs a filings object: {path}")
    entries = manifest["filings"]
    expected = set(filenames)
    actual = set(entries)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Filing-date manifest coverage differs: missing={missing}, extra={extra}")
    filing_dates: dict[str, date] = {}
    for filename in filenames:
        entry = entries[filename]
        if not isinstance(entry, dict):
            raise ValueError(f"Filing-date entry must be an object: {filename}")
        value = entry.get("date")
        if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
            raise ValueError(f"Filing date must be YYYY-MM-DD: {filename}")
        try:
            filing_dates[filename] = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"Invalid filing date for {filename}: {value}") from error
        provenance = entry.get("provenance")
        if not isinstance(provenance, dict) or any(
            not isinstance(provenance.get(field), str) or not provenance[field].strip()
            for field in ("source_pdf", "basis", "evidence")
        ):
            raise ValueError(f"Filing date needs source_pdf, basis, and evidence: {filename}")
        page = provenance.get("page")
        if type(page) is not int or page < 1:
            raise ValueError(f"Filing date provenance needs a positive page: {filename}")
    return manifest, filing_dates, digest


def _check_body_search_cutoffs(document: Document, cutoff: date | None) -> None:
    """Never reuse body evidence gathered under a different retrospective date."""
    for citation in document.full_locators:
        for search in citation.body_searches:
            if search.retrospective_date != cutoff:
                raise ValueError(
                    f"Saved body search cutoff differs for {document.source_path}: "
                    f"{search.retrospective_date} != {cutoff}"
                )


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


def _transient_body_failure_stage(document: Document) -> str | None:
    """Find the first provider stage whose saved search or fetch needs retry."""
    failed_stages: set[str] = set()
    for root in document.roots:
        node_stages = {node.id: node.stage for node in root.nodes}
        for search in root.body_searches:
            for failure in (*search.failures, *(attempt.failure for attempt in search.attempts)):
                if failure is not None and (
                    failure.failure_type == "transport_error"
                    or failure.status_code == 429
                    or (failure.status_code is not None and failure.status_code >= 500)
                ):
                    failed_stages.add(node_stages[search.node_id])
    return next((stage for stage in _BODY_SEARCH_STAGES if stage in failed_stages), None)


def _has_transient_body_search_failure(document: Document) -> bool:
    """A transient body search or fetch cannot be treated as a negative result."""
    return _transient_body_failure_stage(document) is not None


async def _retry_body_stages(
    document: Document,
    first_stage: str,
    retrospective_date: date | None,
    courtlistener_client: CourtListenerClient | None = None,
) -> Document:
    """Run the failed provider, later providers, and the cross-provider review."""
    client_kwargs = {"client": courtlistener_client} if courtlistener_client is not None else {}
    if first_stage == _COURTLISTENER_OPINION_STAGE:
        document = courtlistener_opinion_locator_body_search(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
    if first_stage in (_COURTLISTENER_OPINION_STAGE, _COURTLISTENER_RECAP_STAGE):
        document = courtlistener_recap_locator_body_search(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
    document = govinfo_opinion_locator_body_search(document, retrospective_date=retrospective_date)
    return await review_locator_body_evidence(document)


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
    from_validation_documents: Path | None = None,
    retrospective_date: date | None = None,
    courtlistener_pool: str | None = None,
    primary_filing_dates: bool = False,
) -> Path:
    if courtlistener_pool not in (None, "reserved", "proxy"):
        raise ValueError(f"Unsupported CourtListener pool: {courtlistener_pool}")
    if primary_filing_dates and retrospective_date is not None:
        raise ValueError("Choose either primary filing dates or one retrospective date")
    if resume_run is not None and primary_filing_dates:
        raise ValueError("Resume uses the filing-date mode saved in run.json")
    if (
        sum(
            item is not None
            for item in (
                from_roots_documents,
                from_reporter_review_documents,
                from_docket_review_documents,
                from_validation_documents,
            )
        )
        > 1
    ):
        raise ValueError("Choose one saved Document checkpoint")
    if reuse_docket_lookups and from_reporter_review_documents is None and resume_run is None:
        raise ValueError("Reusing docket lookups requires reporter-review Documents")
    filing_dates: dict[str, date] = {}
    if resume_run is None:
        filenames = sorted(
            json.loads((data_root / _SET / "documents.json").read_text(encoding="utf-8"))["documents"]
        )
        if not filenames:
            raise ValueError(f"No documents in {_SET}")
        filing_date_manifest: dict[str, object] | None = None
        filing_date_manifest_sha256: str | None = None
        if primary_filing_dates:
            filing_date_manifest, filing_dates, filing_date_manifest_sha256 = _load_primary_filing_dates(
                data_root, filenames
            )
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
            "from_validation_documents": (
                str(from_validation_documents) if from_validation_documents else None
            ),
            "retrospective_date": retrospective_date.isoformat() if retrospective_date else None,
            "primary_filing_dates": primary_filing_dates,
            "filing_date_manifest": filing_date_manifest,
            "filing_date_manifest_sha256": filing_date_manifest_sha256,
            "filing_dates": {name: value.isoformat() for name, value in filing_dates.items()},
            "courtlistener_pool": courtlistener_pool,
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
        saved_validation = run_record.get("from_validation_documents")
        from_validation_documents = Path(saved_validation) if saved_validation else None
        saved_retrospective_date = run_record.get("retrospective_date")
        retrospective_date = (
            date.fromisoformat(saved_retrospective_date) if saved_retrospective_date else None
        )
        primary_filing_dates = bool(run_record.get("primary_filing_dates", False))
        if primary_filing_dates and retrospective_date is not None:
            raise ValueError("Saved run has conflicting retrospective date settings")
        saved_pool = run_record.get("courtlistener_pool")
        if saved_pool not in (None, "reserved", "proxy"):
            raise ValueError(f"Unsupported saved CourtListener pool: {saved_pool}")
        if courtlistener_pool is not None and courtlistener_pool != saved_pool:
            history = run_record.setdefault("courtlistener_pool_history", [saved_pool or "proxy"])
            history.append(courtlistener_pool)
        courtlistener_pool = courtlistener_pool or saved_pool
        if courtlistener_pool is not None:
            run_record["courtlistener_pool"] = courtlistener_pool
        reuse_docket_lookups = bool(run_record.get("reuse_docket_lookups", False))
        if (
            sum(
                item is not None
                for item in (
                    from_roots_documents,
                    from_reporter_review_documents,
                    from_docket_review_documents,
                    from_validation_documents,
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
        if primary_filing_dates:
            manifest, filing_dates, digest = _load_primary_filing_dates(data_root, filenames)
            if (
                run_record.get("filing_date_manifest") != manifest
                or run_record.get("filing_date_manifest_sha256") != digest
                or run_record.get("filing_dates")
                != {name: value.isoformat() for name, value in filing_dates.items()}
            ):
                raise ValueError("Filing-date manifest differs from the original run")
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
        run_record.pop("transient_failures", None)
        _write_json(record_path, run_record)

    courtlistener_client: CourtListenerClient | None = None
    try:
        # Check every saved checkpoint before any provider-backed stage starts.
        sources = {
            filename: Document.from_source(data_root / _SET / "documents_txt" / filename)
            for filename in filenames
        }
        completed: set[str] = set()
        retry_body: dict[str, tuple[Document, str]] = {}
        for filename in filenames:
            source = sources[filename]
            cutoff = filing_dates.get(filename, retrospective_date)
            artifact = documents_dir / f"{filename}.json"
            if artifact.exists():
                saved_document = _load_document(artifact, source)
                _check_body_search_cutoffs(saved_document, cutoff)
                if saved_document.stage_runs == _RUN_STAGES:
                    if not _has_transient_docket_lookup_failure(saved_document):
                        if failed_stage := _transient_body_failure_stage(saved_document):
                            previous_stage = _RUN_STAGES[_RUN_STAGES.index(failed_stage) - 1]
                            retry_body[filename] = (
                                saved_document.get_stage(previous_stage),
                                failed_stage,
                            )
                        else:
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
            if filename not in completed and from_validation_documents is not None:
                saved = from_validation_documents / f"{filename}.json"
                ready = _load_document(saved, source).get_stage(_VALIDATION_INPUT_STAGE)
                if ready.stage_runs != _VALIDATION_INPUT_STAGES:
                    raise ValueError(f"Saved validation input is incomplete for {filename}")

        if courtlistener_pool == "reserved" and len(completed) != len(filenames):
            base_url = CourtListenerConfig.from_env().base_url
            token = os.getenv("COURTLISTENER_API_TOKEN_RESERVED", "").strip()
            if not token:
                raise ValueError("COURTLISTENER_API_TOKEN_RESERVED must be configured for the reserved pool")
            courtlistener_client = CourtListenerClient(
                CourtListenerConfig(base_url=base_url, pool="reserved", token=token)
            )
        elif courtlistener_pool == "proxy" and len(completed) != len(filenames):
            courtlistener_client = CourtListenerClient(CourtListenerConfig.from_env())

        for index, filename in enumerate(filenames, start=1):
            source = sources[filename]
            cutoff = filing_dates.get(filename, retrospective_date)
            artifact = documents_dir / f"{filename}.json"
            if filename in completed:
                print(f"{index}/{len(filenames)} {filename} (saved)", flush=True)
                continue
            if artifact.exists():
                print(f"{index}/{len(filenames)} {filename} (incomplete; rerunning)", flush=True)
            if filename in retry_body:
                document, failed_stage = retry_body[filename]
            elif from_validation_documents is not None:
                saved = from_validation_documents / f"{filename}.json"
                document = _load_document(saved, source).get_stage(_VALIDATION_INPUT_STAGE)
            elif from_docket_review_documents is not None:
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
            if filename in retry_body:
                document = await _retry_body_stages(document, failed_stage, cutoff, courtlistener_client)
            elif (
                from_validation_documents is not None
                or from_docket_review_documents is not None
                or from_reporter_review_documents is not None
            ):
                client_kwargs = (
                    {"courtlistener_client": courtlistener_client} if courtlistener_client is not None else {}
                )
                document = await corroborate_root_locator_bodies(
                    document, retrospective_date=cutoff, **client_kwargs
                )
            else:
                client_kwargs = (
                    {"courtlistener_client": courtlistener_client} if courtlistener_client is not None else {}
                )
                document = await validate_roots(document, retrospective_date=cutoff, **client_kwargs)
            if document.stage_runs != _RUN_STAGES:
                raise ValueError(f"Run did not complete every stage for {filename}")
            _check_body_search_cutoffs(document, cutoff)
            _write_json(artifact, document.model_dump(mode="json"))
            print(f"{index}/{len(filenames)} {filename}", flush=True)
        transient_failures = {"docket": [], "body": []}
        for filename in filenames:
            saved_document = _load_document(documents_dir / f"{filename}.json", sources[filename])
            if _has_transient_docket_lookup_failure(saved_document):
                transient_failures["docket"].append(filename)
            if _has_transient_body_search_failure(saved_document):
                transient_failures["body"].append(filename)
        if any(transient_failures.values()):
            run_record["transient_failures"] = transient_failures
            raise RuntimeError(
                "Search ended with transient provider failures in "
                + ", ".join(sorted({*transient_failures["docket"], *transient_failures["body"]}))
                + "; resume this run after the provider recovers"
            )
    except BaseException:
        run_record["status"] = "failed"
        _write_json(record_path, run_record)
        raise
    finally:
        if courtlistener_client is not None:
            courtlistener_client.close()

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
        "--from-validation-documents",
        type=Path,
        help="Resume saved Documents at stage 19 before locator-first body search and review",
    )
    parser.add_argument(
        "--retrospective-date",
        type=date.fromisoformat,
        help="Use only body evidence issued on or before this ISO date",
    )
    parser.add_argument(
        "--primary-filing-dates",
        action="store_true",
        help="Use each primary filing's sourced date from primary/filing_dates.json as its body-evidence cutoff",
    )
    parser.add_argument(
        "--reuse-docket-lookups",
        action="store_true",
        help="Reuse saved stage-16 docket lookups while rerunning reporter and docket LLM reviews",
    )
    parser.add_argument(
        "--resume-run", type=Path, help="Continue an interrupted run in its existing timestamp directory"
    )
    parser.add_argument(
        "--courtlistener-pool",
        choices=("reserved", "proxy"),
        help="Select the reserved token or the proxy's rotating tokens for body searches",
    )
    args = parser.parse_args()
    if (
        sum(
            item is not None
            for item in (
                args.from_roots_documents,
                args.from_reporter_review_documents,
                args.from_docket_review_documents,
                args.from_validation_documents,
            )
        )
        > 1
    ):
        parser.error("Choose one saved Document checkpoint")
    if args.reuse_docket_lookups and not args.from_reporter_review_documents:
        parser.error("--reuse-docket-lookups requires --from-reporter-review-documents")
    if args.primary_filing_dates and args.retrospective_date is not None:
        parser.error("Choose either --primary-filing-dates or --retrospective-date")
    if args.resume_run and (
        args.primary_filing_dates
        or any(
            value is not None
            for value in (
                args.data_root,
                args.results_root,
                args.from_roots_documents,
                args.from_reporter_review_documents,
                args.from_docket_review_documents,
                args.from_validation_documents,
                args.retrospective_date,
            )
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
            args.from_validation_documents.resolve() if args.from_validation_documents else None,
            args.retrospective_date,
            args.courtlistener_pool,
            args.primary_filing_dates,
        )
    )
    print(run_dir)


if __name__ == "__main__":
    main()
