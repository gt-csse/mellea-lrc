"""Persist and resume a document through the four existing workflow APIs.

Execution metadata and completed model reviews live beside the native Document;
they do not change its serialization or its stage recovery semantics.
"""

# Local run setup and checkpoint commits finish synchronously before advancing.
# ruff: noqa: ASYNC240

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.llm.journal import ReviewJournal
from mellea_lrc.llm.profiles import load_profile
from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.body_evidence import BodyEvidenceFailure
from mellea_lrc.model.citations.docket_lookup import DocketLookupFailure
from mellea_lrc.model.document import Document
from mellea_lrc.providers.courtlistener import CourtListenerConfig
from mellea_lrc.providers.govinfo import GovInfoConfig
from mellea_lrc.workflows import (
    grow_leaves,
    grow_roots,
    validate_all_citation_content,
    validate_roots_citation_identity,
)

_MODEL_SUBSTAGES = (
    "grow_roots.discover_root_locators.hunt_docket_locators",
    "grow_roots.form_root_groups.reassign_docket_roots_llm",
    "validate_roots_citation_identity.lookup_reporter_roots.review_unique_llm",
    "validate_roots_citation_identity.lookup_reporter_roots.review_ambiguous_llm",
    "validate_roots_citation_identity.lookup_docket_roots.review_courtlistener_dockets_llm",
    "validate_roots_citation_identity.lookup_docket_roots.review_govinfo_dockets_llm",
    "validate_roots_citation_identity.corroborate_locator_bodies.review_locator_bodies_llm",
    "grow_leaves.grow_short_reporter_leaves.attribute_citations",
    "grow_leaves.grow_reference_leaves.attribute_citations",
    "grow_leaves.grow_id_leaves.attribute_citations",
    "grow_leaves.grow_supra_leaves.review_attributions_llm",
    "grow_leaves.correct_leaf_readings.correct_leaf_fields",
    "validate_all_citation_content.prepare_citation_evidence.select_opinions_llm",
    "validate_all_citation_content.prepare_citation_evidence.read_propositions",
    "validate_all_citation_content.review_citation_content.review_cited_pages_llm",
    "validate_all_citation_content.review_citation_content.review_full_opinions_llm",
)


def _state_dir(output: Path) -> Path:
    return output.with_name(output.name + ".run")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Replace one complete file without exposing a truncated checkpoint."""
    with TemporaryDirectory(dir=path.parent, prefix=f".{path.name}-") as directory:
        temporary = Path(directory) / path.name
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


@contextmanager
def _owner(directory: Path):
    """The operating system releases ownership even after a crashed process."""
    with (directory / "owner.lock").open("a+") as owner:
        try:
            if os.name == "nt":
                import msvcrt

                owner.write("\0")
                owner.flush()
                owner.seek(0)
                msvcrt.locking(owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError) as error:
            raise RuntimeError(f"Run is already active: {directory}") from error
        try:
            yield
        finally:
            if os.name == "nt":
                owner.seek(0)
                msvcrt.locking(owner.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(owner, fcntl.LOCK_UN)


def _configuration(fields: FieldExtractionConfig) -> dict:
    """Record resolved settings, never credential values or a copy of .env."""
    profiles = {name: load_profile(name) for name in _MODEL_SUBSTAGES}
    for profile in profiles.values():
        profile.resolve()  # Fail before preprocessing or provider work.
    courtlistener, govinfo = CourtListenerConfig.from_env(), GovInfoConfig.from_env()
    return {
        "field_extraction": fields.model_dump(mode="json"),
        "model_profiles": {name: asdict(profile) for name, profile in profiles.items()},
        "courtlistener": {
            "base_url": courtlistener.base_url,
            "timeout_seconds": courtlistener.timeout_seconds,
        },
        "govinfo": {"base_url": govinfo.base_url, "timeout_seconds": govinfo.timeout_seconds},
    }


def _implementation_fingerprint() -> dict:
    package = Path(__file__).parent
    return {
        "code": {str(path.relative_to(package)): _sha(path) for path in sorted(package.rglob("*.py"))},
        "dependencies": {
            name: version(name)
            for name in ("mellea", "openai", "pydantic", "eyecite", "reporters-db", "courts-db", "rapidfuzz")
        },
    }


def _check(record: dict) -> None:
    """Keep one run's settings and implementation fixed across its operations."""
    if _configuration(FieldExtractionConfig.from_env()) != record["configuration"]:
        raise ValueError("Execution configuration changed; restore the original .env before resuming")
    if _implementation_fingerprint() != record["implementation"]:
        raise ValueError("Execution implementation changed; resume with the original code and dependencies")


def _document_input(document: Document) -> dict:
    """The preprocessed input stays fixed while citation histories grow."""
    raw = document.model_dump(mode="json")
    return {key: raw[key] for key in ("source_metadata", "text", "preprocessing_metadata", "index_spans")}


def _input_digest(document: Document) -> str:
    raw = json.dumps(_document_input(document), ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _transient_substage(document: Document) -> str | None:
    """A provider outage cannot commit a completed negative search result."""
    failed = set()

    def transient(failure: DocketLookupFailure | BodyEvidenceFailure | None) -> bool:
        if failure is None:
            return False
        status = (
            failure.upstream_status_code if isinstance(failure, DocketLookupFailure) else failure.status_code
        )
        return (
            failure.failure_type == "transport_error"
            or status == 429
            or (status is not None and status >= 500)
        )

    for root in document.roots:
        nodes = {node.id: node.substage for node in root.nodes}
        if isinstance(root, FullDocketCitation):
            for lookup in (root.docket_lookup, root.govinfo_docket_lookup):
                if lookup is not None and any(
                    transient(failure)
                    for failure in (lookup.failure, *(item.failure for item in lookup.attempts))
                ):
                    failed.add(nodes[lookup.node_id])
        for search in (*root.body_searches, *root.field_body_searches):
            if any(
                transient(failure)
                for failure in (*search.failures, *(item.failure for item in search.attempts))
            ):
                failed.add(nodes[search.node_id])
    return next((name for name in document.substage_runs if name in failed), None)


async def _execute(document: Document, output: Path, record: dict, fields: FieldExtractionConfig) -> Document:
    directory = _state_dir(output)
    record.update(status="running", error=None, updated_at=datetime.now(UTC).isoformat())
    _write_json(directory / "run.json", record)

    def save(checkpoint: Document) -> None:
        _check(record)
        if _input_digest(checkpoint) != record["input_sha256"]:
            raise ValueError("Workflow changed the saved document input")
        failed = _transient_substage(checkpoint)
        if failed is not None:
            preserved = directory / "failed-checkpoints"
            preserved.mkdir(exist_ok=True)
            raw = checkpoint.model_dump(mode="json")
            digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
            _write_json(preserved / f"{digest}.json", raw)
            raise RuntimeError(f"Transient provider failure at {failed}; resume after the provider recovers")
        _write_json(output, checkpoint.model_dump(mode="json"))

    try:
        with ReviewJournal(directory / "reviews", document, before_review=lambda: _check(record)):
            for name, workflow, options in (
                (
                    "grow_roots",
                    grow_roots,
                    {"hunt_dockets": True, "review_docket_roots": True, "field_extraction": fields},
                ),
                ("validate_roots_citation_identity", validate_roots_citation_identity, {}),
                ("grow_leaves", grow_leaves, {"field_extraction": fields}),
                ("validate_all_citation_content", validate_all_citation_content, {}),
            ):
                _check(record)
                print(f"Running {name}...", flush=True)
                document = await workflow(document, checkpoint=save, **options)
                save(document)
    except BaseException as error:
        record.update(
            status="interrupted", error=type(error).__name__, updated_at=datetime.now(UTC).isoformat()
        )
        _write_json(directory / "run.json", record)
        raise
    record.update(
        status="complete", completed_document_sha256=_sha(output), updated_at=datetime.now(UTC).isoformat()
    )
    _write_json(directory / "run.json", record)
    print(f"Saved {output}", flush=True)
    return document


async def validate(source: Path, output: Path) -> Document:
    """Start a new run. Existing output or run state is never overwritten."""
    source, output = source.resolve(), output.absolute()
    directory = _state_dir(output)
    if output.exists() or directory.exists():
        raise FileExistsError(f"Run already exists; use resume: {output}")
    if not source.is_file():
        raise FileNotFoundError(source)
    fields = FieldExtractionConfig.from_env()
    fields.validate_runtime()
    configuration, implementation = _configuration(fields), _implementation_fingerprint()
    source_sha256 = _sha(source)
    document = Document.from_source(source)
    if _sha(source) != source_sha256:
        raise ValueError("Original source changed during preprocessing")
    record = {
        "document": str(output),
        "source_path": str(source),
        "source_sha256": source_sha256,
        "input_sha256": _input_digest(document),
        "configuration": configuration,
        "implementation": implementation,
    }
    _check(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    with _owner(directory):
        _write_json(output, document.model_dump(mode="json"))
        _write_json(directory / "run.json", record)
        return await _execute(document, output, record, fields)


async def resume(output: Path) -> Document:
    """Continue the same run using its Document and unchanged execution settings."""
    output = output.absolute()
    directory = _state_dir(output)
    if not output.is_file() or not (directory / "run.json").is_file():
        raise ValueError(f"Resume requires the saved Document and its .run directory: {output}")
    with _owner(directory):
        record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if record["document"] != str(output):
            raise ValueError("Run metadata belongs to a different document path")
        source = Path(record["source_path"])
        if not source.is_file() or _sha(source) != record["source_sha256"]:
            raise ValueError("Original source changed or is unavailable")
        document = Document.model_validate_json(output.read_bytes())
        if _input_digest(document) != record["input_sha256"]:
            raise ValueError("Saved document input changed")
        fields = FieldExtractionConfig.from_env()
        _check(record)
        if record.get("status") == "complete":
            if _sha(output) != record["completed_document_sha256"]:
                raise ValueError("Completed document changed")
            return document
        fields.validate_runtime()
        return await _execute(document, output, record, fields)
