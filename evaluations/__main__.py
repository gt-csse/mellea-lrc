"""Run the primary extraction and validation workflows into one timestamped artifact."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path
from time import monotonic

from evaluations import complete_stage_boundary
from mellea_lrc.api import (
    Document,
    docket_root_lookup_courtlistener_llm_review,
    docket_root_lookup_courtlistener_retrieval,
    docket_root_lookup_govinfo_llm_review,
    docket_root_lookup_govinfo_retrieval,
    grow_leaves,
    grow_roots,
    intended_case_courtlistener_opinion_retrieval,
    intended_case_courtlistener_recap_retrieval,
    intended_case_govinfo_opinion_retrieval,
    intended_case_llm_selection,
    locator_body_courtlistener_opinion_retrieval,
    locator_body_courtlistener_recap_retrieval,
    locator_body_govinfo_opinion_retrieval,
    locator_body_llm_judgment,
    reporter_root_lookup_ambiguous_llm_judgment,
    reporter_root_lookup_unique_llm_judgment,
    validate_pincite,
    validate_roots,
)
from mellea_lrc.configuration import read_env, required_setting
from mellea_lrc.llm.profiles import load_profile
from mellea_lrc.model import FullDocketCitation
from mellea_lrc.model.execution import get_workflow
from mellea_lrc.providers.courtlistener import CourtListenerClient, CourtListenerConfig
from mellea_lrc.validation.body_search.intended_case_courtlistener_opinion_retrieval import (
    SUBSTAGE as _COURTLISTENER_OPINION_FIELD_SUBSTAGE,
)
from mellea_lrc.validation.body_search.intended_case_courtlistener_recap_retrieval import (
    SUBSTAGE as _COURTLISTENER_RECAP_FIELD_SUBSTAGE,
)
from mellea_lrc.validation.body_search.intended_case_govinfo_opinion_retrieval import (
    SUBSTAGE as _GOVINFO_OPINION_FIELD_SUBSTAGE,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_opinion_retrieval import (
    SUBSTAGE as _COURTLISTENER_OPINION_SUBSTAGE,
)
from mellea_lrc.validation.body_search.locator_body_courtlistener_recap_retrieval import (
    SUBSTAGE as _COURTLISTENER_RECAP_SUBSTAGE,
)
from mellea_lrc.validation.body_search.locator_body_govinfo_opinion_retrieval import (
    SUBSTAGE as _GOVINFO_OPINION_SUBSTAGE,
)
from mellea_lrc.validation.fields_aggregated_identity import SUBSTAGE as _FIELD_IDENTITY_SUBSTAGE
from mellea_lrc.validation.intended_case_llm_selection import SUBSTAGE as _INTENDED_CASE_REVIEW_SUBSTAGE
from mellea_lrc.validation.locator_body_llm_judgment import SUBSTAGE as _LOCATOR_BODY_REVIEW_SUBSTAGE
from mellea_lrc.workflows import _require_body_search_cutoff, _require_workflow_prefix

_SET = "primary"
_DATA_ROOT = Path(__file__).resolve().parents[2] / "mellea-lrc-datasets"
_RESULTS_ROOT = Path(__file__).resolve().parent / "results" / _SET
_ROOT_SUBSTAGE = "grow_roots.root_formation.rule"
_ROOT_SUBSTAGES = (
    "grow_roots.locator_discovery.full_reporter_locators",
    "grow_roots.locator_discovery.docket_locators",
    "grow_roots.locator_discovery.docket_hunting",
    "grow_roots.field_reading.docket_entries",
    "grow_roots.field_reading.colocations",
    "grow_roots.field_reading.case_names",
    "grow_roots.field_reading.courts",
    "grow_roots.field_reading.dates",
    "grow_roots.field_reading.pin_cites",
    _ROOT_SUBSTAGE,
)
_RUN_SUBSTAGES = (
    *_ROOT_SUBSTAGES,
    "grow_roots.root_formation.docket_llm_reassignment",
    "validate_roots.reporter_lookup.cluster_retrieval",
    "validate_roots.reporter_lookup.docket_retrieval",
    "validate_roots.reporter_lookup.unique_rule_judgment",
    "validate_roots.reporter_lookup.ambiguous_rule_judgment",
    "validate_roots.reporter_lookup.unique_llm_judgment",
    "validate_roots.reporter_lookup.ambiguous_llm_judgment",
    "validate_roots.docket_lookup.courtlistener_retrieval",
    "validate_roots.docket_lookup.courtlistener_review",
    "validate_roots.docket_lookup.govinfo_retrieval",
    "validate_roots.docket_lookup.govinfo_review",
    _FIELD_IDENTITY_SUBSTAGE,
    _COURTLISTENER_OPINION_SUBSTAGE,
    _COURTLISTENER_RECAP_SUBSTAGE,
    _GOVINFO_OPINION_SUBSTAGE,
    _LOCATOR_BODY_REVIEW_SUBSTAGE,
)
_BODY_SEARCH_SUBSTAGES = (
    _COURTLISTENER_OPINION_SUBSTAGE,
    _COURTLISTENER_RECAP_SUBSTAGE,
    _GOVINFO_OPINION_SUBSTAGE,
)
_FIELD_SUBSTAGES = (
    _COURTLISTENER_OPINION_FIELD_SUBSTAGE,
    _COURTLISTENER_RECAP_FIELD_SUBSTAGE,
    _GOVINFO_OPINION_FIELD_SUBSTAGE,
    _INTENDED_CASE_REVIEW_SUBSTAGE,
)
_FIELD_RUN_SUBSTAGES = (*_RUN_SUBSTAGES, *_FIELD_SUBSTAGES)
_REPORTER_REVIEW_INPUT_SUBSTAGE = "validate_roots.reporter_lookup.ambiguous_rule_judgment"
_REPORTER_REVIEW_INPUT_SUBSTAGES = _RUN_SUBSTAGES[: _RUN_SUBSTAGES.index(_REPORTER_REVIEW_INPUT_SUBSTAGE) + 1]
_DOCKET_LOOKUP_SUBSTAGE = "validate_roots.docket_lookup.courtlistener_retrieval"
_DOCKET_REVIEW_INPUT_SUBSTAGE = "validate_roots.docket_lookup.courtlistener_review"
_DOCKET_REVIEW_INPUT_SUBSTAGES = _RUN_SUBSTAGES[: _RUN_SUBSTAGES.index(_DOCKET_REVIEW_INPUT_SUBSTAGE) + 1]
_VALIDATION_INPUT_SUBSTAGE = "validate_roots.docket_lookup.govinfo_review"
_VALIDATION_INPUT_SUBSTAGES = _RUN_SUBSTAGES[: _RUN_SUBSTAGES.index(_VALIDATION_INPUT_SUBSTAGE) + 1]
_REPORTER_TO_GOVINFO_SUBSTAGES = _RUN_SUBSTAGES[
    _RUN_SUBSTAGES.index("validate_roots.reporter_lookup.cluster_retrieval") : _RUN_SUBSTAGES.index(
        _VALIDATION_INPUT_SUBSTAGE
    )
    + 1
]
_BODY_CHECKPOINT_SUBSTAGES = (*_BODY_SEARCH_SUBSTAGES, _LOCATOR_BODY_REVIEW_SUBSTAGE)
_VALIDATION_CHECKPOINT_SUBSTAGES = (
    *_REPORTER_TO_GOVINFO_SUBSTAGES,
    _FIELD_IDENTITY_SUBSTAGE,
    *_BODY_CHECKPOINT_SUBSTAGES,
)
_E2E_WORKFLOWS = ("grow_roots", "validate_roots", "grow_leaves", "validate_pincite")
_E2E_SUBSTAGES = (
    *_RUN_SUBSTAGES,
    *(
        substage.name
        for workflow in ("grow_leaves", "validate_pincite")
        for stage in get_workflow(workflow).stages
        for substage in stage.substages
    ),
)


def _configured_model_profiles() -> dict[str, dict[str, object]]:
    """Snapshot configured E2E packages without storing credential values."""
    values = read_env()
    return {
        substage: asdict(load_profile(substage, environ=values))
        for substage in _E2E_SUBSTAGES
        if values.get("MELLEA_LRC_SUBSTAGE_" + substage.upper().replace(".", "_") + "_PROFILE")
    }


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()


def _annotation_sha256(data_root: Path, filenames: list[str]) -> dict[str, str]:
    return {
        filename: hashlib.sha256(
            (data_root / _SET / "documents" / f"{Path(filename).stem}.jsonl").read_bytes()
        ).hexdigest()
        for filename in filenames
    }


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


def _annotation_case_cutoffs(data_root: Path, filenames: list[str]) -> tuple[dict[str, date], dict[str, str]]:
    """Read case cutoffs from annotation headers, the dataset's source of truth."""

    def parse_iso(value: object, path: Path) -> date:
        if not isinstance(value, str):
            raise ValueError(f"Annotation header has an invalid filing or cutoff date: {path}")
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"Annotation header has an invalid filing or cutoff date: {path}") from error
        if parsed.isoformat() != value:
            raise ValueError(f"Annotation header has an invalid filing or cutoff date: {path}")
        return parsed

    filing_dates: dict[str, date] = {}
    cutoff_dates: dict[str, date] = {}
    source_documents: dict[str, str] = {}
    header_hashes: dict[str, str] = {}
    for filename in filenames:
        path = data_root / _SET / "documents" / f"{Path(filename).stem}.jsonl"
        with path.open("rb") as rows:
            raw = rows.readline()
        header_hashes[filename] = hashlib.sha256(raw).hexdigest()
        header = json.loads(raw)
        if header.get("unit") != "header" or header.get("document") != filename:
            raise ValueError(f"Annotation header does not identify {filename}: {path}")
        filing = header.get("filing")
        cutoff = header.get("case_cutoff")
        if not isinstance(filing, dict) or not isinstance(cutoff, dict):
            raise ValueError(f"Annotation header needs filing and case_cutoff: {path}")
        filing_dates[filename] = parse_iso(filing.get("date"), path)
        cutoff_dates[filename] = parse_iso(cutoff.get("date"), path)
        provenance = filing.get("provenance")
        if (
            not isinstance(provenance, dict)
            or any(
                not isinstance(provenance.get(field), str) or not provenance[field].strip()
                for field in ("source_pdf", "basis", "evidence")
            )
            or type(provenance.get("page")) is not int
            or provenance["page"] < 1
        ):
            raise ValueError(f"Annotation filing date needs sourced PDF evidence: {path}")
        source_document = cutoff.get("source_document")
        if not isinstance(source_document, str) or not source_document:
            raise ValueError(f"Annotation case cutoff needs a source_document: {path}")
        source_documents[filename] = source_document
    for filename in filenames:
        source = source_documents[filename]
        group = [name for name in filenames if source_documents[name] == source]
        if source not in group:
            raise ValueError(f"Case cutoff source is not a sampled filing: {filename}: {source}")
        earliest = min(filing_dates[name] for name in group)
        if filing_dates[source] != earliest or any(cutoff_dates[name] != earliest for name in group):
            raise ValueError(f"Case cutoff is not the earliest sampled filing: {filename}")
    return cutoff_dates, header_hashes


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


def _transient_docket_failure_substage(document: Document) -> str | None:
    """Locate the earliest failed docket provider so E2E resumes its boundary."""
    failed: set[str] = set()
    for root in document.roots:
        if not isinstance(root, FullDocketCitation):
            continue
        node_substages = {node.id: node.substage for node in root.nodes}
        for lookup in (root.docket_lookup, root.govinfo_docket_lookup):
            if lookup is None:
                continue
            for failure in (lookup.failure, *(attempt.failure for attempt in lookup.attempts)):
                if failure is not None and (
                    failure.failure_type == "transport_error"
                    or failure.upstream_status_code == 429
                    or (failure.upstream_status_code is not None and failure.upstream_status_code >= 500)
                ):
                    failed.add(node_substages[lookup.node_id])
    return next((substage for substage in _RUN_SUBSTAGES if substage in failed), None)


def _transient_body_failure_substage(document: Document) -> str | None:
    """Find the first provider substage whose saved search or fetch needs retry."""
    failed_substages: set[str] = set()
    for root in document.roots:
        node_stages = {node.id: node.substage for node in root.nodes}
        for search in root.body_searches:
            for failure in (*search.failures, *(attempt.failure for attempt in search.attempts)):
                if failure is not None and (
                    failure.failure_type == "transport_error"
                    or failure.status_code == 429
                    or (failure.status_code is not None and failure.status_code >= 500)
                ):
                    failed_substages.add(node_stages[search.node_id])
    return next((substage for substage in _BODY_SEARCH_SUBSTAGES if substage in failed_substages), None)


def _has_transient_body_search_failure(document: Document) -> bool:
    """A transient body search or fetch cannot be treated as a negative result."""
    return _transient_body_failure_substage(document) is not None


def _transient_field_failure_substage(document: Document) -> str | None:
    """Find the first case-name provider whose saved search or fetch needs retry."""
    failed_substages: set[str] = set()
    for root in document.roots:
        node_stages = {node.id: node.substage for node in root.nodes}
        for search in root.field_body_searches:
            for failure in (*search.failures, *(attempt.failure for attempt in search.attempts)):
                if failure is not None and (
                    failure.failure_type == "transport_error"
                    or failure.status_code == 429
                    or (failure.status_code is not None and failure.status_code >= 500)
                ):
                    failed_substages.add(node_stages[search.node_id])
    return next((substage for substage in _FIELD_SUBSTAGES[:-1] if substage in failed_substages), None)


async def _retry_body_substages(
    document: Document,
    first_substage: str,
    retrospective_date: date | None,
    courtlistener_client: CourtListenerClient | None = None,
    checkpoint: Callable[[Document], None] | None = None,
) -> Document:
    """Run the failed provider, later providers, and the cross-provider review."""
    client_kwargs = {"client": courtlistener_client} if courtlistener_client is not None else {}
    if first_substage == _COURTLISTENER_OPINION_SUBSTAGE:
        document = locator_body_courtlistener_opinion_retrieval(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
        if checkpoint is not None:
            checkpoint(document)
    if first_substage in (_COURTLISTENER_OPINION_SUBSTAGE, _COURTLISTENER_RECAP_SUBSTAGE):
        document = locator_body_courtlistener_recap_retrieval(
            document, retrospective_date=retrospective_date, **client_kwargs
        )
        if checkpoint is not None:
            checkpoint(document)
    document = locator_body_govinfo_opinion_retrieval(document, retrospective_date=retrospective_date)
    if checkpoint is not None:
        checkpoint(document)
    document = await locator_body_llm_judgment(document)
    if checkpoint is not None:
        checkpoint(document)
    return document


def _reuse_docket_lookup(document: Document, saved: Document) -> Document:
    """Replay the unchanged docket roots' saved lookup nodes after reporter review."""
    checkpoint = saved.get_substage(_DOCKET_LOOKUP_SUBSTAGE)
    for root in checkpoint.roots:
        if isinstance(root, FullDocketCitation) and root.nodes[-1].substage == _DOCKET_LOOKUP_SUBSTAGE:
            document = document.replace_citation(root)
    return document.complete_substage(_DOCKET_LOOKUP_SUBSTAGE)


async def _continue_field_substages(
    document: Document,
    *,
    filename: str,
    run_dir: Path,
    retrospective_date: date | None,
    courtlistener_client: CourtListenerClient | None,
) -> Document:
    """Append each remaining substage and atomically save its cumulative Document."""
    for substage in _FIELD_SUBSTAGES:
        if substage in document.substage_runs:
            continue
        if document.substage_runs != _FIELD_RUN_SUBSTAGES[: _FIELD_RUN_SUBSTAGES.index(substage)]:
            raise ValueError(f"Field discovery cannot skip a substage for {filename}")
        if substage == _COURTLISTENER_OPINION_FIELD_SUBSTAGE:
            kwargs = {"client": courtlistener_client} if courtlistener_client is not None else {}
            document = intended_case_courtlistener_opinion_retrieval(
                document, retrospective_date=retrospective_date, **kwargs
            )
        elif substage == _COURTLISTENER_RECAP_FIELD_SUBSTAGE:
            kwargs = {"client": courtlistener_client} if courtlistener_client is not None else {}
            document = intended_case_courtlistener_recap_retrieval(
                document, retrospective_date=retrospective_date, **kwargs
            )
        elif substage == _GOVINFO_OPINION_FIELD_SUBSTAGE:
            document = intended_case_govinfo_opinion_retrieval(
                document, retrospective_date=retrospective_date
            )
        else:
            document = await intended_case_llm_selection(document)
        if document.substage_runs != _FIELD_RUN_SUBSTAGES[: _FIELD_RUN_SUBSTAGES.index(substage) + 1]:
            raise ValueError(f"Field discovery did not complete {substage} for {filename}")
        document = complete_stage_boundary(document, _FIELD_RUN_SUBSTAGES)
        _require_body_search_cutoff(document, retrospective_date)
        _write_json(run_dir / "documents" / f"{filename}.json", document.model_dump(mode="json"))
        if _transient_field_failure_substage(document) == substage:
            raise RuntimeError(
                f"Case-name body search at {substage} had a transient provider failure for {filename}; "
                "resume this run after the provider recovers"
            )
    return document


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
    annotation_case_cutoffs: bool = False,
    from_locator_review_documents: Path | None = None,
    from_checkpoint_documents: Path | None = None,
    checkpoint_substage: str | None = None,
    end_to_end: bool = False,
    workers: int | None = None,
) -> Path:
    if workers is not None and (type(workers) is not int or workers < 1):
        raise ValueError("End-to-end workers must be a positive integer")
    if workers is not None and not end_to_end and resume_run is None:
        raise ValueError("--workers requires end-to-end mode")
    if end_to_end and (
        reuse_docket_lookups
        or any(
            directory is not None
            for directory in (
                from_roots_documents,
                from_reporter_review_documents,
                from_docket_review_documents,
                from_validation_documents,
                from_locator_review_documents,
                from_checkpoint_documents,
            )
        )
    ):
        raise ValueError("End-to-end mode starts from primary sources; saved input modes cannot be combined")
    if end_to_end and courtlistener_pool == "reserved":
        raise ValueError("End-to-end mode uses the configured CourtListener proxy without a personal token")
    if courtlistener_pool not in (None, "reserved", "proxy"):
        raise ValueError(f"Unsupported CourtListener pool: {courtlistener_pool}")
    if annotation_case_cutoffs and retrospective_date is not None:
        raise ValueError("Choose either annotation case cutoffs or one retrospective date")
    if resume_run is not None and annotation_case_cutoffs:
        raise ValueError("Resume uses the cutoff mode saved in run.json")
    if (from_checkpoint_documents is None) != (checkpoint_substage is None):
        raise ValueError("Choose both checkpoint Documents and a checkpoint substage")
    if checkpoint_substage is not None and checkpoint_substage not in _VALIDATION_CHECKPOINT_SUBSTAGES:
        raise ValueError(f"Unsupported validation checkpoint: {checkpoint_substage}")
    if (
        sum(
            item is not None
            for item in (
                from_roots_documents,
                from_reporter_review_documents,
                from_docket_review_documents,
                from_validation_documents,
                from_locator_review_documents,
                from_checkpoint_documents,
            )
        )
        > 1
    ):
        raise ValueError("Choose one saved Document checkpoint")
    if reuse_docket_lookups and from_reporter_review_documents is None and resume_run is None:
        raise ValueError("Reusing docket lookups requires reporter-review Documents")
    case_cutoffs: dict[str, date] = {}
    model_profiles = _configured_model_profiles() if end_to_end else None
    if resume_run is None:
        filenames = sorted(
            json.loads((data_root / _SET / "documents.json").read_text(encoding="utf-8"))["documents"]
        )
        if not filenames:
            raise ValueError(f"No documents in {_SET}")
        annotation_hashes = _annotation_sha256(data_root, filenames) if end_to_end else None
        if end_to_end:
            workers = 3 if workers is None else workers
        annotation_header_sha256: dict[str, str] = {}
        if annotation_case_cutoffs:
            case_cutoffs, annotation_header_sha256 = _annotation_case_cutoffs(data_root, filenames)
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
            "from_locator_review_documents": (
                str(from_locator_review_documents) if from_locator_review_documents else None
            ),
            "from_checkpoint_documents": (
                str(from_checkpoint_documents) if from_checkpoint_documents else None
            ),
            "checkpoint_substage": checkpoint_substage,
            "retrospective_date": retrospective_date.isoformat() if retrospective_date else None,
            "annotation_case_cutoffs": annotation_case_cutoffs,
            "annotation_header_sha256": annotation_header_sha256,
            "case_cutoffs": {name: value.isoformat() for name, value in case_cutoffs.items()},
            "courtlistener_pool": courtlistener_pool,
            "reuse_docket_lookups": reuse_docket_lookups,
            "source_sha256": {
                filename: hashlib.sha256(
                    (data_root / _SET / "documents_txt" / filename).read_bytes()
                ).hexdigest()
                for filename in filenames
            },
        }
        if end_to_end:
            run_record.update(
                end_to_end=True,
                workflows=list(_E2E_WORKFLOWS),
                model_profiles=model_profiles,
                git_commit=_git_commit(),
                annotation_sha256=annotation_hashes,
                workers=workers,
                substage_seconds={},
            )
        record_path = run_dir / "run.json"
        _write_json(record_path, run_record)
    else:
        run_dir = resume_run
        record_path = run_dir / "run.json"
        run_record = json.loads(record_path.read_text(encoding="utf-8"))
        if end_to_end and not run_record.get("end_to_end"):
            raise ValueError("Resume is not an end-to-end run")
        end_to_end = bool(run_record.get("end_to_end", False))
        if workers is not None and not end_to_end:
            raise ValueError("--workers requires end-to-end mode")
        if end_to_end:
            if workers is not None and workers != run_record["workers"]:
                raise ValueError("Resume must use the saved end-to-end worker count")
            workers = run_record["workers"]
            model_profiles = _configured_model_profiles()
            if run_record["model_profiles"] != model_profiles:
                raise ValueError(
                    "End-to-end model profiles changed; restore the saved .env settings before resuming"
                )
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
        saved_locator_review = run_record.get("from_locator_review_documents")
        from_locator_review_documents = Path(saved_locator_review) if saved_locator_review else None
        saved_checkpoint_documents = run_record.get("from_checkpoint_documents")
        from_checkpoint_documents = Path(saved_checkpoint_documents) if saved_checkpoint_documents else None
        checkpoint_substage = run_record.get("checkpoint_substage")
        if (from_checkpoint_documents is None) != (checkpoint_substage is None):
            raise ValueError("Saved run has an incomplete checkpoint source")
        if checkpoint_substage is not None and checkpoint_substage not in _VALIDATION_CHECKPOINT_SUBSTAGES:
            raise ValueError(f"Unsupported saved validation checkpoint: {checkpoint_substage}")
        saved_retrospective_date = run_record.get("retrospective_date")
        retrospective_date = (
            date.fromisoformat(saved_retrospective_date) if saved_retrospective_date else None
        )
        annotation_case_cutoffs = bool(run_record.get("annotation_case_cutoffs", False))
        if annotation_case_cutoffs and retrospective_date is not None:
            raise ValueError("Saved run has conflicting retrospective date settings")
        saved_pool = run_record.get("courtlistener_pool")
        if saved_pool not in (None, "reserved", "proxy"):
            raise ValueError(f"Unsupported saved CourtListener pool: {saved_pool}")
        if courtlistener_pool is not None and courtlistener_pool != saved_pool:
            history = run_record.setdefault("courtlistener_pool_history", [saved_pool or "proxy"])
            history.append(courtlistener_pool)
        courtlistener_pool = courtlistener_pool or saved_pool
        if end_to_end and courtlistener_pool == "reserved":
            raise ValueError(
                "End-to-end mode uses the configured CourtListener proxy without a personal token"
            )
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
                    from_locator_review_documents,
                    from_checkpoint_documents,
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
            raise ValueError("Run document list differs from the current dataset")
        if end_to_end and run_record["annotation_sha256"] != _annotation_sha256(data_root, filenames):
            raise ValueError("Run annotation content differs from saved provenance")
        if annotation_case_cutoffs:
            case_cutoffs, header_hashes = _annotation_case_cutoffs(data_root, filenames)
            if run_record.get("annotation_header_sha256") != header_hashes or run_record.get(
                "case_cutoffs"
            ) != {name: value.isoformat() for name, value in case_cutoffs.items()}:
                raise ValueError("Annotation case cutoffs differ from the original run")
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
        run_record.pop("error", None)
        _write_json(record_path, run_record)

    courtlistener_client: CourtListenerClient | None = None
    try:
        # Check every saved checkpoint before any provider-backed substage starts.
        sources = {
            filename: Document.from_source(data_root / _SET / "documents_txt" / filename)
            for filename in filenames
        }
        completed: set[str] = set()
        retry_body: dict[str, str] = {}
        documents: dict[str, Document] = {}
        reporter_inputs: dict[str, Document] = {}
        enabled = (
            _E2E_SUBSTAGES
            if end_to_end
            else _FIELD_RUN_SUBSTAGES
            if from_locator_review_documents is not None
            else _RUN_SUBSTAGES
        )
        input_directory, input_substage = next(
            (
                (directory, substage)
                for directory, substage in (
                    (from_roots_documents, _ROOT_SUBSTAGE),
                    (from_reporter_review_documents, _REPORTER_REVIEW_INPUT_SUBSTAGE),
                    (from_docket_review_documents, _DOCKET_REVIEW_INPUT_SUBSTAGE),
                    (from_validation_documents, _VALIDATION_INPUT_SUBSTAGE),
                    (from_locator_review_documents, _LOCATOR_BODY_REVIEW_SUBSTAGE),
                    (from_checkpoint_documents, checkpoint_substage),
                )
                if directory is not None
            ),
            (None, None),
        )
        for filename in filenames:
            source = sources[filename]
            cutoff = case_cutoffs.get(filename, retrospective_date)
            original = source
            if input_directory is not None:
                supplied = _load_document(input_directory / f"{filename}.json", source)
                original = supplied.get_substage(input_substage)
                if original.substage_runs != enabled[: enabled.index(input_substage) + 1]:
                    raise ValueError(f"Saved input checkpoint is incomplete for {filename}")
                _require_body_search_cutoff(original, cutoff)
                if from_locator_review_documents is not None and (
                    _has_transient_docket_lookup_failure(original)
                    or _has_transient_body_search_failure(original)
                ):
                    raise ValueError(f"Saved locator-body review has transient failures for {filename}")
                if reuse_docket_lookups:
                    supplied.get_substage(_DOCKET_LOOKUP_SUBSTAGE)
                    reporter_inputs[filename] = supplied
            artifact = documents_dir / f"{filename}.json"
            document = _load_document(artifact, source) if artifact.exists() else original
            _require_body_search_cutoff(document, cutoff)
            if document.substage_runs != enabled[: len(document.substage_runs)]:
                raise ValueError(f"Saved evaluation checkpoint skips a substage for {filename}")
            if len(document.substage_runs) < len(original.substage_runs) or (
                original.substage_runs and document.get_substage(input_substage) != original
            ):
                raise ValueError(f"Saved evaluation input differs for {filename}")
            _require_workflow_prefix(document, "grow_roots")
            _require_workflow_prefix(document, "validate_roots")
            if end_to_end:
                _require_workflow_prefix(document, "grow_leaves")
                _require_workflow_prefix(document, "validate_pincite")
            marked = complete_stage_boundary(document, enabled)
            if artifact.exists() and marked != document:
                _write_json(artifact, marked.model_dump(mode="json"))
            document = marked
            if from_locator_review_documents is not None:
                if failed_substage := _transient_field_failure_substage(document):
                    previous_substage = enabled[enabled.index(failed_substage) - 1]
                    document = document.get_substage(previous_substage)
                elif artifact.exists() and document.substage_runs == enabled:
                    completed.add(filename)
            elif _has_transient_docket_lookup_failure(document):
                # Replaying the run's input reruns its docket lookup and review.
                if end_to_end:
                    failed_substage = _transient_docket_failure_substage(document)
                    assert failed_substage is not None
                    document = document.get_substage(enabled[enabled.index(failed_substage) - 1])
                else:
                    document = original
            elif failed_substage := _transient_body_failure_substage(document):
                if document.substage_runs == enabled:
                    retry_body[filename] = failed_substage
                previous_substage = enabled[enabled.index(failed_substage) - 1]
                document = document.get_substage(previous_substage)
            elif artifact.exists() and document.substage_runs == enabled:
                completed.add(filename)
            documents[filename] = document

        if courtlistener_pool == "reserved" and len(completed) != len(filenames):
            courtlistener_client = CourtListenerClient(
                replace(
                    CourtListenerConfig.from_env(),
                    pool="reserved",
                    token=required_setting(read_env(), "COURTLISTENER_API_TOKEN_RESERVED"),
                )
            )
        elif (courtlistener_pool == "proxy" or end_to_end) and len(completed) != len(filenames):
            courtlistener_client = CourtListenerClient(CourtListenerConfig.from_env())

        async def process(index: int, filename: str) -> None:
            cutoff = case_cutoffs.get(filename, retrospective_date)
            artifact = documents_dir / f"{filename}.json"
            if filename in completed:
                print(f"{index}/{len(filenames)} {filename} (saved)", flush=True)
                return
            document = documents[filename]
            document = complete_stage_boundary(document, enabled)
            checkpoint_started = monotonic()
            timed_substages = set(document.substage_runs)

            def check_model_profiles() -> None:
                if end_to_end and _configured_model_profiles() != model_profiles:
                    raise ValueError("End-to-end model profiles changed during execution")

            def save_checkpoint(checkpoint: Document) -> None:
                nonlocal checkpoint_started
                check_model_profiles()
                substage = checkpoint.substage_runs[-1]
                if checkpoint.substage_runs != enabled[: enabled.index(substage) + 1]:
                    raise ValueError(f"Stage did not complete in source order for {filename}: {substage}")
                _require_body_search_cutoff(checkpoint, cutoff)
                _write_json(artifact, checkpoint.model_dump(mode="json"))
                if end_to_end and checkpoint.runs[-1].kind == "substage" and substage not in timed_substages:
                    now = monotonic()
                    run_record["substage_seconds"].setdefault(filename, {})[substage] = round(
                        now - checkpoint_started, 4
                    )
                    checkpoint_started = now
                    timed_substages.add(substage)
                    _write_json(record_path, run_record)
                if end_to_end and _transient_docket_failure_substage(checkpoint) == substage:
                    raise RuntimeError(
                        f"Docket search at {substage} had a transient provider failure for {filename}; "
                        "resume this run after the provider recovers"
                    )
                if _transient_body_failure_substage(checkpoint) == substage:
                    raise RuntimeError(
                        f"Locator-body search at {substage} had a transient provider failure for "
                        f"{filename}; resume this run after the provider recovers"
                    )

            if from_locator_review_documents is not None:
                document = await _continue_field_substages(
                    document,
                    filename=filename,
                    run_dir=run_dir,
                    retrospective_date=cutoff,
                    courtlistener_client=courtlistener_client,
                )
            else:
                check_model_profiles()
                if from_reporter_review_documents is not None:
                    for substage, review in (
                        (
                            "validate_roots.reporter_lookup.unique_llm_judgment",
                            reporter_root_lookup_unique_llm_judgment,
                        ),
                        (
                            "validate_roots.reporter_lookup.ambiguous_llm_judgment",
                            reporter_root_lookup_ambiguous_llm_judgment,
                        ),
                    ):
                        if substage not in document.substage_runs:
                            document = await review(document)
                            save_checkpoint(document)
                    if _DOCKET_LOOKUP_SUBSTAGE not in document.substage_runs:
                        document = complete_stage_boundary(document, enabled)
                        document = (
                            _reuse_docket_lookup(document, reporter_inputs[filename])
                            if reuse_docket_lookups
                            else docket_root_lookup_courtlistener_retrieval(document)
                        )
                        save_checkpoint(document)
                    if _DOCKET_REVIEW_INPUT_SUBSTAGE not in document.substage_runs:
                        document = await docket_root_lookup_courtlistener_llm_review(document)
                        save_checkpoint(document)
                if from_docket_review_documents is not None or from_reporter_review_documents is not None:
                    if "validate_roots.docket_lookup.govinfo_retrieval" not in document.substage_runs:
                        document = docket_root_lookup_govinfo_retrieval(document)
                        save_checkpoint(document)
                    if _VALIDATION_INPUT_SUBSTAGE not in document.substage_runs:
                        document = await docket_root_lookup_govinfo_llm_review(document)
                        save_checkpoint(document)
                if "grow_roots.root_formation.docket_llm_reassignment" not in document.substage_runs:
                    document = await grow_roots(
                        document,
                        hunt_dockets=True,
                        review_docket_roots=True,
                        checkpoint=save_checkpoint,
                    )
                    save_checkpoint(document)
                document = complete_stage_boundary(document, enabled)
                if filename in retry_body:
                    document = await _retry_body_substages(
                        document, retry_body[filename], cutoff, courtlistener_client, save_checkpoint
                    )
                else:
                    client_kwargs = (
                        {"courtlistener_client": courtlistener_client}
                        if courtlistener_client is not None
                        else {}
                    )
                    if end_to_end:
                        client_kwargs["search_other_fields"] = False
                    document = await validate_roots(
                        document,
                        retrospective_date=cutoff,
                        checkpoint=save_checkpoint,
                        **client_kwargs,
                    )
                if end_to_end:
                    check_model_profiles()
                    document = await grow_leaves(document, review_leaves=True, checkpoint=save_checkpoint)
                    check_model_profiles()
                    document = await validate_pincite(
                        document,
                        client=courtlistener_client,
                        review_opinions=True,
                        checkpoint=save_checkpoint,
                    )
            if document.substage_runs != enabled:
                raise ValueError(f"Run did not complete every substage for {filename}")
            document = complete_stage_boundary(document, enabled)
            _require_body_search_cutoff(document, cutoff)
            _write_json(artifact, document.model_dump(mode="json"))
            print(f"{index}/{len(filenames)} {filename}", flush=True)

        if end_to_end:
            semaphore = asyncio.Semaphore(workers)

            async def limited_process(index: int, filename: str) -> None:
                async with semaphore:
                    await process(index, filename)

            tasks = [
                asyncio.create_task(limited_process(index, filename))
                for index, filename in enumerate(filenames, start=1)
            ]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        else:
            for index, filename in enumerate(filenames, start=1):
                await process(index, filename)
        transient_failures: dict[str, list[str]] = {"docket": [], "body": []}
        if from_locator_review_documents is not None:
            transient_failures["field"] = []
        for filename in filenames:
            saved_document = _load_document(documents_dir / f"{filename}.json", sources[filename])
            if _has_transient_docket_lookup_failure(saved_document):
                transient_failures["docket"].append(filename)
            if _has_transient_body_search_failure(saved_document):
                transient_failures["body"].append(filename)
            if from_locator_review_documents is not None and _transient_field_failure_substage(
                saved_document
            ):
                transient_failures["field"].append(filename)
        if any(transient_failures.values()):
            run_record["transient_failures"] = transient_failures
            raise RuntimeError(
                "Search ended with transient provider failures in "
                + ", ".join(sorted(set().union(*transient_failures.values())))
                + "; resume this run after the provider recovers"
            )
    except BaseException as error:
        run_record["status"] = "failed"
        if end_to_end:
            run_record["error"] = f"{type(error).__name__}: {error}"
        _write_json(record_path, run_record)
        raise
    finally:
        if courtlistener_client is not None:
            courtlistener_client.close()

    run_record["status"] = "complete"
    run_record["completed_at"] = datetime.now(UTC).isoformat()
    _write_json(record_path, run_record)
    if end_to_end:
        from evaluations.score_run import score_run

        try:
            score_run(run_dir, _E2E_WORKFLOWS)
        except BaseException:
            run_record["status"] = "failed"
            _write_json(record_path, run_record)
            raise
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help="Run all four workflows in order from primary sources and render four reports in one artifact",
    )
    parser.add_argument("--workers", type=int, help="End-to-end document workers (default: 3)")
    parser.add_argument(
        "--from-roots-documents",
        type=Path,
        help="Resume saved Documents at their grow_roots.root_formation.rule checkpoint before the docket-root review",
    )
    parser.add_argument(
        "--from-reporter-review-documents",
        type=Path,
        help="Resume saved Documents at substage 13 before the reporter LLM reviews",
    )
    parser.add_argument(
        "--from-docket-review-documents",
        type=Path,
        help="Resume saved Documents at substage 17 before GovInfo docket lookup and review",
    )
    parser.add_argument(
        "--from-validation-documents",
        type=Path,
        help="Resume saved Documents at substage 19 before locator-first body search and review",
    )
    parser.add_argument(
        "--from-locator-review-documents",
        type=Path,
        help="Continue saved substage-23 Documents through case-name body discovery and review",
    )
    parser.add_argument(
        "--from-checkpoint-documents",
        type=Path,
        help="Replay validation after a chosen saved substage, including retrieval-only stages",
    )
    parser.add_argument(
        "--checkpoint-substage",
        choices=_VALIDATION_CHECKPOINT_SUBSTAGES,
        help="Stage to recover from each saved Document",
    )
    parser.add_argument(
        "--retrospective-date",
        type=date.fromisoformat,
        help="Use only body evidence issued on or before this ISO date",
    )
    parser.add_argument(
        "--annotation-case-cutoffs",
        action="store_true",
        help="Use each case's earliest sampled filing date from its annotation header",
    )
    parser.add_argument(
        "--reuse-docket-lookups",
        action="store_true",
        help="Reuse saved substage-16 docket lookups while rerunning reporter and docket LLM reviews",
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
                args.from_locator_review_documents,
                args.from_checkpoint_documents,
            )
        )
        > 1
    ):
        parser.error("Choose one saved Document checkpoint")
    if args.reuse_docket_lookups and not args.from_reporter_review_documents:
        parser.error("--reuse-docket-lookups requires --from-reporter-review-documents")
    if bool(args.from_checkpoint_documents) != bool(args.checkpoint_substage):
        parser.error("--from-checkpoint-documents requires --checkpoint-substage")
    if args.annotation_case_cutoffs and args.retrospective_date is not None:
        parser.error("Choose either --annotation-case-cutoffs or --retrospective-date")
    if args.resume_run and (
        args.annotation_case_cutoffs
        or any(
            value is not None
            for value in (
                args.data_root,
                args.results_root,
                args.from_roots_documents,
                args.from_reporter_review_documents,
                args.from_docket_review_documents,
                args.from_validation_documents,
                args.from_locator_review_documents,
                args.from_checkpoint_documents,
                args.checkpoint_substage,
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
            args.annotation_case_cutoffs,
            args.from_locator_review_documents.resolve() if args.from_locator_review_documents else None,
            args.from_checkpoint_documents.resolve() if args.from_checkpoint_documents else None,
            args.checkpoint_substage,
            args.end_to_end,
            args.workers,
        )
    )
    print(run_dir)


if __name__ == "__main__":
    main()
