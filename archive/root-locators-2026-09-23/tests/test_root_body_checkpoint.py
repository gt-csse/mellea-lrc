"""Checkpoint retry selection must distinguish exhausted failures from recovered attempts."""

import asyncio
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from mellea_lrc.model.document import Document
from scripts import run_root_body_checkpoint as runner
from scripts.run_root_body_checkpoint import (
    _presearch_prior_checkpoint,
    _prior_checkpoint,
    _resume_run_manifest,
    _retryable_body_failure,
)


def _payload(*, search_error: str | None = None, fetches: list[dict[str, str]] | None = None):
    return {
        "citations": [
            {
                "trace": [
                    {
                        "stage": "root_body_corroboration_resolution",
                        "details": {
                            "validation": {"error": search_error},
                            "fetches": fetches or [],
                        },
                    }
                ]
            }
        ]
    }


def test_retry_failed_body_fetch() -> None:
    assert _retryable_body_failure(
        _payload(fetches=[{"outcome": "failed", "error": "CourtListener request failed with 429"}])
    )


def test_do_not_retry_successful_fetch_after_transient_attempt() -> None:
    assert not _retryable_body_failure(
        _payload(
            fetches=[
                {"outcome": "retry", "error": "CourtListener request failed with 429"},
                {"outcome": "citation_context_found"},
            ]
        )
    )


def test_retry_failed_search_and_timeout_but_not_permanent_error() -> None:
    assert _retryable_body_failure(_payload(search_error="CourtListenerError: 503 unavailable"))
    assert _retryable_body_failure(_payload(search_error="CourtListener request timed out"))
    assert not _retryable_body_failure(_payload(search_error="CourtListener request failed with 404"))


def test_retry_transient_docket_fallback_failure() -> None:
    payload = {
        "citations": [
            {
                "trace": [
                    {
                        "stage": "root_body_corroboration_search",
                        "details": {
                            "validation": {
                                "error": None,
                                "search_attempts": [
                                    {"kind": "raw_locator", "error": None},
                                    {"kind": "parsed_docket", "error": "HTTP 429"},
                                ],
                            }
                        },
                    }
                ]
            }
        ]
    }

    assert _retryable_body_failure(payload)


def test_retry_requires_prior_input_checkpoint(tmp_path) -> None:
    resolution = tmp_path / "old" / "root_body_corroboration_resolution"
    resolution.mkdir(parents=True)
    with pytest.raises(ValueError, match="input-checkpoint manifest"):
        _prior_checkpoint(resolution)

    source = tmp_path / "old-input"
    source.mkdir()
    (resolution.parent / "run-manifest.json").write_text(
        json.dumps({"source_checkpoint": str(source)}), encoding="utf-8"
    )
    assert _prior_checkpoint(resolution) == source


def test_refresh_follows_saved_search_back_to_presearch_document(tmp_path) -> None:
    before_search = tmp_path / "before-search" / "documents"
    before_search.mkdir(parents=True)
    searched = tmp_path / "searched" / "root_body_corroboration_search"
    searched.mkdir(parents=True)
    (searched / "filing.json").write_text(
        json.dumps({"passes": ["root_body_corroboration_search"]}), encoding="utf-8"
    )
    (searched.parent / "run-manifest.json").write_text(
        json.dumps({"source_checkpoint": str(before_search)}), encoding="utf-8"
    )
    resolution = tmp_path / "resolved" / "root_body_corroboration_resolution"
    resolution.mkdir(parents=True)
    (resolution.parent / "run-manifest.json").write_text(
        json.dumps({"source_checkpoint": str(searched)}), encoding="utf-8"
    )

    assert _presearch_prior_checkpoint(resolution) == before_search


def test_resumed_run_keeps_prior_document_provenance(tmp_path) -> None:
    expected = {"source_checkpoint": "old-input", "courtlistener_search_pool": "main"}
    (tmp_path / "run-manifest.json").write_text(
        json.dumps({**expected, "rerun": ["a.json"], "reused": ["b.json"]}), encoding="utf-8"
    )
    assert _resume_run_manifest(tmp_path, expected) == (["a.json"], ["b.json"])
    with pytest.raises(ValueError, match="different courtlistener_search_pool"):
        _resume_run_manifest(tmp_path, {**expected, "courtlistener_search_pool": "reserved"})


def test_explicit_source_provenance_is_attached_before_body_search(monkeypatch, tmp_path) -> None:
    source = tmp_path / "filing.txt"
    source.write_text("Example filing", encoding="utf-8")
    document = Document.from_source(source)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "filing.json").write_text(json.dumps(document.model_dump(mode="json")), encoding="utf-8")
    sidecar = tmp_path / "filings.json"
    sidecar.write_text(json.dumps({"filings": {"filing.txt": {"docket_id": "12345"}}}))
    observed: list[str | None] = []

    async def _search(document, **_kwargs):
        observed.append(document.source_metadata.courtlistener_docket_id)
        return document.evolve(passes=(*document.passes, "root_body_corroboration_search"))

    async def _resolve(document, **_kwargs):
        observed.append(document.source_metadata.courtlistener_docket_id)
        return document.evolve(passes=(*document.passes, "root_body_corroboration_resolution"))

    monkeypatch.setattr(runner, "search_root_body_corroboration", _search)
    monkeypatch.setattr(runner, "resolve_root_body_corroboration", _resolve)
    monkeypatch.setattr(runner, "CourtListenerClient", lambda **_kwargs: object())
    monkeypatch.setattr(runner, "body_text_client", lambda client: client)
    monkeypatch.setattr(runner, "GovInfoClient", object)
    monkeypatch.setattr(runner, "start_mellea_session_from_env", object)

    artifacts = tmp_path / "artifacts"
    asyncio.run(
        runner.run_checkpoint(
            checkpoint,
            artifacts,
            include_search=True,
            source_provenance=sidecar,
        )
    )
    assert observed == ["12345", "12345"]
    persisted = json.loads(
        (artifacts / "root_body_corroboration_resolution" / "filing.json").read_text(encoding="utf-8")
    )
    assert Document.model_validate(persisted).source_metadata.courtlistener_docket_id == "12345"
    manifest = json.loads((artifacts / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_provenance_path"] == str(sidecar.resolve())
    assert manifest["source_provenance_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()


def test_source_provenance_missing_entry_leaves_checkpoint_unchanged(monkeypatch, tmp_path) -> None:
    source = tmp_path / "unlisted.txt"
    source.write_text("Example filing", encoding="utf-8")
    document = Document.from_source(source)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "unlisted.json").write_text(json.dumps(document.model_dump(mode="json")), encoding="utf-8")
    sidecar = tmp_path / "filings.json"
    sidecar.write_text(json.dumps({"filings": {"other.txt": {"docket_id": "12345"}}}))
    observed: list[str | None] = []

    async def _resolve(document, **_kwargs):
        observed.append(document.source_metadata.courtlistener_docket_id)
        return document.evolve(passes=(*document.passes, "root_body_corroboration_resolution"))

    monkeypatch.setattr(runner, "resolve_root_body_corroboration", _resolve)
    monkeypatch.setattr(runner, "CourtListenerClient", lambda **_kwargs: object())
    monkeypatch.setattr(runner, "body_text_client", lambda client: client)
    monkeypatch.setattr(runner, "GovInfoClient", object)
    monkeypatch.setattr(runner, "start_mellea_session_from_env", object)

    asyncio.run(runner.run_checkpoint(checkpoint, tmp_path / "artifacts", source_provenance=sidecar))
    assert observed == [None]


def test_new_source_provenance_prevents_reusing_old_body_result(monkeypatch, tmp_path) -> None:
    source = tmp_path / "filing.txt"
    source.write_text("Example filing", encoding="utf-8")
    document = Document.from_source(source)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "filing.json").write_text(json.dumps(document.model_dump(mode="json")), encoding="utf-8")
    prior_resolution = tmp_path / "old" / "root_body_corroboration_resolution"
    prior_resolution.mkdir(parents=True)
    (prior_resolution / "filing.json").write_text(
        json.dumps(document.model_dump(mode="json")), encoding="utf-8"
    )
    (prior_resolution.parent / "run-manifest.json").write_text(
        json.dumps({"source_checkpoint": str(checkpoint)}), encoding="utf-8"
    )
    sidecar = tmp_path / "filings.json"
    sidecar.write_text(json.dumps({"filings": {"filing.txt": {"docket_id": "12345"}}}))
    observed: list[str | None] = []

    async def _search(document, **_kwargs):
        observed.append(document.source_metadata.courtlistener_docket_id)
        return document.evolve(passes=(*document.passes, "root_body_corroboration_search"))

    async def _resolve(document, **_kwargs):
        return document.evolve(passes=(*document.passes, "root_body_corroboration_resolution"))

    monkeypatch.setattr(runner, "search_root_body_corroboration", _search)
    monkeypatch.setattr(runner, "resolve_root_body_corroboration", _resolve)
    monkeypatch.setattr(runner, "CourtListenerClient", lambda **_kwargs: object())
    monkeypatch.setattr(runner, "body_text_client", lambda client: client)
    monkeypatch.setattr(runner, "GovInfoClient", object)
    monkeypatch.setattr(runner, "start_mellea_session_from_env", object)

    artifacts = tmp_path / "artifacts"
    asyncio.run(
        runner.run_checkpoint(
            checkpoint,
            artifacts,
            include_search=True,
            retry_from=prior_resolution,
            source_provenance=sidecar,
        )
    )
    assert observed == ["12345"]
    manifest = json.loads((artifacts / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["rerun"] == ["filing.json"]
    assert manifest["reused"] == []


def test_same_source_provenance_reuses_clean_body_result(monkeypatch, tmp_path) -> None:
    source = tmp_path / "filing.txt"
    source.write_text("Example filing", encoding="utf-8")
    document = Document.from_source(source)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "filing.json").write_text(json.dumps(document.model_dump(mode="json")), encoding="utf-8")
    sidecar = tmp_path / "filings.json"
    sidecar.write_text(json.dumps({"filings": {"filing.txt": {"docket_id": "12345"}}}))
    prior_resolution = tmp_path / "old" / "root_body_corroboration_resolution"
    prior_resolution.mkdir(parents=True)
    reviewed = runner.attach_source_provenance(document, {"filing.txt": "12345"})
    (prior_resolution / "filing.json").write_text(
        json.dumps(reviewed.model_dump(mode="json")), encoding="utf-8"
    )
    (prior_resolution.parent / "run-manifest.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(checkpoint),
                "source_provenance_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("An unchanged, clean body result should be reused")

    monkeypatch.setattr(runner, "search_root_body_corroboration", unexpected)
    monkeypatch.setattr(runner, "resolve_root_body_corroboration", unexpected)
    monkeypatch.setattr(runner, "CourtListenerClient", lambda **_kwargs: object())
    monkeypatch.setattr(runner, "body_text_client", lambda client: client)
    monkeypatch.setattr(runner, "GovInfoClient", object)
    monkeypatch.setattr(runner, "start_mellea_session_from_env", object)

    artifacts = tmp_path / "artifacts"
    asyncio.run(
        runner.run_checkpoint(
            checkpoint,
            artifacts,
            include_search=True,
            retry_from=prior_resolution,
            source_provenance=sidecar,
        )
    )
    manifest = json.loads((artifacts / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["rerun"] == []
    assert manifest["reused"] == ["filing.json"]
    assert (artifacts / "root_body_corroboration_resolution" / "filing.json").read_text(encoding="utf-8") == (
        prior_resolution / "filing.json"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "script",
    ("run_root_body_checkpoint.py", "run_root_identity_pipeline.py"),
)
def test_direct_cli_imports_without_provider_calls(script: str, tmp_path) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-provenance" in result.stdout
