"""Tests for the executable root pipeline's provider routing."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import date
from types import SimpleNamespace

import pytest

from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerConfig
from mellea_lrc.courtlistener.client import DEFAULT_BASE_URL, body_text_client
from mellea_lrc.model.document import Document
from scripts import run_root_identity_pipeline as pipeline


def test_retrospective_corpus_evaluation_requires_explicit_drafting_dates(tmp_path) -> None:
    with pytest.raises(ValueError, match="drafting-date map"):
        asyncio.run(
            pipeline.run_corpus(
                (tmp_path / "filing.txt",),
                artifacts=tmp_path / "artifacts",
                require_retrospective_dates=True,
            )
        )


def test_corpus_research_allows_no_retrospective_cutoff(monkeypatch, tmp_path) -> None:
    source = tmp_path / "filing.txt"
    source.write_text("Example filing", encoding="utf-8")
    observed = []

    async def _run(path, **kwargs):
        observed.append((path, kwargs["retrospective_date"]))
        return SimpleNamespace(citations=())

    monkeypatch.setattr(pipeline, "CourtListenerClient", object)
    monkeypatch.setattr(pipeline, "GovInfoClient", object)
    monkeypatch.setattr(pipeline, "start_mellea_session_from_env", object)
    monkeypatch.setattr(pipeline, "run", _run)
    completed = asyncio.run(pipeline.run_corpus((source,), artifacts=tmp_path / "artifacts"))

    assert len(completed) == 1
    assert observed == [(source, None)]


def test_body_fetch_uses_selected_proxy_pool_without_direct_credentials() -> None:
    client = CourtListenerClient(
        config=CourtListenerConfig(
            base_url="https://example.modal.run/api/rest/v4/",
            token="example-token",
            pool="reserved",
        )
    )

    body_client = body_text_client(client)

    assert isinstance(body_client, CourtListenerClient)
    assert body_client.config.base_url == client.config.base_url
    assert body_client.config.token is None
    assert body_client.config.pool == "reserved"
    assert "Authorization" not in body_client._headers()
    assert body_client._headers()["x-cl-pool"] == "reserved"


def test_body_fetch_keeps_direct_courtlistener_authentication() -> None:
    client = CourtListenerClient(config=CourtListenerConfig(base_url=DEFAULT_BASE_URL, token="example-token"))

    assert body_text_client(client) is client


def test_body_fetch_keeps_authentication_for_other_custom_hosts() -> None:
    client = CourtListenerClient(
        config=CourtListenerConfig(base_url="https://custom.example/api/rest/v4/", token="example-token")
    )

    assert body_text_client(client) is client


def test_runner_finishes_after_corpus_body_identity_without_open_web(monkeypatch, tmp_path) -> None:
    source = tmp_path / "filing.txt"
    source.write_text("Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024).", encoding="utf-8")
    completed_stages: list[str] = []
    stage_arguments: dict[str, dict[str, object]] = {}
    provider_stages = (
        "grow_roots_document",
        "resolve_docket_root_identities_document",
        "lookup_full_reporter_locators_exact_document",
        "validate_unique_full_reporter_locator_identities_document",
        "review_full_reporter_exact_dates_document",
        "resolve_full_reporter_locator_ambiguities_document",
        "search_courtlistener_full_reporter_roots_document",
        "search_govinfo_full_reporter_roots_document",
        "resolve_full_reporter_search_candidates_document",
        "search_root_body_corroboration_document",
        "resolve_root_body_corroboration_document",
        "review_shared_body_evidence_document",
    )

    def _stage_stub(name):
        async def _run(document, **kwargs):
            completed_stages.append(name)
            stage_arguments[name] = kwargs
            return document

        return _run

    for name in provider_stages:
        monkeypatch.setattr(pipeline, name, _stage_stub(name))

    async def _unexpected_open_web(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Automatic root identity run called open-web search")

    monkeypatch.setattr(pipeline, "search_open_web_roots_document", _unexpected_open_web)
    monkeypatch.setattr(pipeline, "resolve_open_web_root_identities_document", _unexpected_open_web)
    monkeypatch.setattr(pipeline, "body_text_client", lambda client: client)

    asyncio.run(
        pipeline.run(
            source,
            artifacts=tmp_path / "artifacts",
            client=object(),
            govinfo_client=object(),
            session=object(),
            retrospective_date=date(2024, 6, 1),
        )
    )

    assert completed_stages[-3:] == [
        "search_root_body_corroboration_document",
        "resolve_root_body_corroboration_document",
        "review_shared_body_evidence_document",
    ]
    assert all(
        stage_arguments[name]["retrospective_date"] == date(2024, 6, 1)
        for name in provider_stages
        if name not in {"grow_roots_document", "review_shared_body_evidence_document"}
    )


def test_explicit_source_provenance_reaches_first_stage_and_manifest(monkeypatch, tmp_path) -> None:
    source = tmp_path / "filing.txt"
    source.write_text("Example filing", encoding="utf-8")
    sidecar = tmp_path / "filings.json"
    sidecar.write_text(json.dumps({"filings": {source.name: {"docket_id": "12345"}}}), encoding="utf-8")
    observed: list[str | None] = []

    async def _stage(document, **_kwargs):
        observed.append(document.source_metadata.courtlistener_docket_id)
        return document

    monkeypatch.setattr(pipeline, "grow_roots_document", _stage)
    monkeypatch.setattr(pipeline, "resolve_docket_root_identities_document", _stage)
    artifacts = tmp_path / "artifacts"
    result = asyncio.run(
        pipeline.run(
            source,
            artifacts=artifacts,
            client=object(),
            govinfo_client=object(),
            session=object(),
            stop_after_docket_identity=True,
            source_provenance=sidecar,
        )
    )

    assert observed == ["12345", "12345"]
    assert result.source_metadata.courtlistener_docket_id == "12345"
    manifest = json.loads((artifacts / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "source_provenance_path": str(sidecar.resolve()),
        "source_provenance_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    }


def test_unlisted_source_keeps_unknown_provenance(monkeypatch, tmp_path) -> None:
    source = tmp_path / "unlisted.txt"
    source.write_text("Example filing", encoding="utf-8")
    sidecar = tmp_path / "filings.json"
    sidecar.write_text(json.dumps({"filings": {"other.txt": {"docket_id": "12345"}}}))

    async def _stage(document, **_kwargs):
        return document

    monkeypatch.setattr(pipeline, "grow_roots_document", _stage)
    monkeypatch.setattr(pipeline, "resolve_docket_root_identities_document", _stage)
    result = asyncio.run(
        pipeline.run(
            source,
            artifacts=tmp_path / "artifacts",
            client=object(),
            govinfo_client=object(),
            session=object(),
            stop_after_docket_identity=True,
            source_provenance=sidecar,
        )
    )
    assert result.source_metadata.courtlistener_docket_id is None


def _saved_root_checkpoint(source, directory):
    """A native root-stage save with no provider or model calls."""
    document = Document.from_source(source)
    document = document.evolve(passes=(*document.passes, "root_formation"))
    checkpoint = directory / "root_formation" / f"{source.stem}.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps(document.model_dump(mode="json")), encoding="utf-8")
    return checkpoint, document


def test_identity_run_resumes_native_root_checkpoint_without_regrowing(monkeypatch, tmp_path) -> None:
    source = tmp_path / "filing.txt"
    source.write_text("Smith v. Jones, 123 U.S. 456 (1887).", encoding="utf-8")
    checkpoint, saved = _saved_root_checkpoint(source, tmp_path / "roots")
    observed = []

    async def _unexpected_grow(*_args, **_kwargs):
        raise AssertionError("The existing root stage should not run again")

    async def _identity_stage(document, **_kwargs):
        observed.append(document)
        return document

    monkeypatch.setattr(pipeline, "grow_roots_document", _unexpected_grow)
    monkeypatch.setattr(pipeline, "resolve_docket_root_identities_document", _identity_stage)
    artifacts = tmp_path / "identity"
    result = asyncio.run(
        pipeline.run(
            source,
            artifacts=artifacts,
            root_checkpoint=checkpoint,
            client=object(),
            govinfo_client=object(),
            session=object(),
            stop_after_docket_identity=True,
        )
    )

    assert observed == [saved]
    assert result == saved
    retained = artifacts / "root_formation" / "filing.json"
    assert retained.exists()
    assert Document.model_validate(json.loads(retained.read_text(encoding="utf-8"))) == saved


def test_identity_run_adds_source_provenance_to_saved_root_checkpoint(monkeypatch, tmp_path) -> None:
    source = tmp_path / "filing.txt"
    source.write_text("Smith v. Jones, 123 U.S. 456 (1887).", encoding="utf-8")
    checkpoint, _ = _saved_root_checkpoint(source, tmp_path / "roots")
    sidecar = tmp_path / "filings.json"
    sidecar.write_text(json.dumps({"filings": {source.name: {"docket_id": "12345"}}}), encoding="utf-8")
    observed: list[str | None] = []

    async def _identity_stage(document, **_kwargs):
        observed.append(document.source_metadata.courtlistener_docket_id)
        return document

    monkeypatch.setattr(pipeline, "resolve_docket_root_identities_document", _identity_stage)
    artifacts = tmp_path / "identity"
    result = asyncio.run(
        pipeline.run(
            source,
            artifacts=artifacts,
            root_checkpoint=checkpoint,
            client=object(),
            govinfo_client=object(),
            session=object(),
            stop_after_docket_identity=True,
            source_provenance=sidecar,
        )
    )

    assert observed == ["12345"]
    assert result.source_metadata.courtlistener_docket_id == "12345"
    retained = artifacts / "root_formation" / "filing.json"
    assert (
        Document.model_validate(
            json.loads(retained.read_text(encoding="utf-8"))
        ).source_metadata.courtlistener_docket_id
        == "12345"
    )


@pytest.mark.parametrize("invalid_part", ["source_text", "source_path"])
def test_identity_run_rejects_root_checkpoint_for_another_source(monkeypatch, tmp_path, invalid_part) -> None:
    source = tmp_path / "filing.txt"
    source.write_text("Original text", encoding="utf-8")
    checkpoint, _ = _saved_root_checkpoint(source, tmp_path / "roots")
    if invalid_part == "source_text":
        source.write_text("Changed text", encoding="utf-8")
    else:
        other = tmp_path / "other.txt"
        other.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        source = other

    async def _unexpected_identity(*_args, **_kwargs):
        raise AssertionError("Mismatched source must fail before identity requests")

    monkeypatch.setattr(pipeline, "resolve_docket_root_identities_document", _unexpected_identity)
    with pytest.raises(ValueError, match=r"source|checkpoint|text|hash|path"):
        asyncio.run(
            pipeline.run(
                source,
                artifacts=tmp_path / "identity",
                root_checkpoint=checkpoint,
                client=object(),
                govinfo_client=object(),
                session=object(),
                stop_after_docket_identity=True,
            )
        )
