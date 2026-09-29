"""Offline checks for resuming one durable evaluation run."""

from __future__ import annotations

import asyncio
import importlib
import json
from datetime import date
from pathlib import Path

import pytest

from evaluations import __main__ as runner
from mellea_lrc.api import Document, grow_roots
from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.model import (
    DocketLookup,
    DocketLookupAttempt,
    DocketLookupFailure,
    FullDocketCitation,
    Span,
)
from mellea_lrc.model.citations.body_evidence import (
    BodyEvidenceFailure,
    BodySearch,
    BodySearchAttempt,
    BodySource,
)
from mellea_lrc.model.citations.govinfo_lookup import GovInfoDocketLookup, GovInfoLookupAttempt


def _dataset(tmp_path: Path, filenames: tuple[str, ...]) -> Path:
    data_root = tmp_path / "data"
    source_dir = data_root / "primary" / "documents_txt"
    source_dir.mkdir(parents=True)
    for filename in filenames:
        (source_dir / filename).write_text(f"source for {filename}\n", encoding="utf-8")
    (data_root / "primary" / "documents.json").write_text(
        json.dumps({"documents": list(filenames)}), encoding="utf-8"
    )
    return data_root


def _complete(document: Document, stages: tuple[str, ...]) -> Document:
    for stage in stages:
        document = document.complete(stage)
    return document


@pytest.fixture(autouse=True)
def _offline_body_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    async def body(document: Document, *, retrospective_date: date | None = None) -> Document:
        return _complete(document, runner._RUN_STAGES[19:])

    monkeypatch.setattr(runner, "corroborate_root_locator_bodies", body)


def test_resume_skips_valid_documents_and_reuses_timestamp_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filenames = ("001.txt", "002.txt")
    data_root = _dataset(tmp_path, filenames)
    calls: list[str] = []

    async def fake_grow(document: Document, *, hunt_dockets: bool, review_docket_roots: bool) -> Document:
        assert hunt_dockets and review_docket_roots
        calls.append(Path(document.source_path or "").name)
        return _complete(document, runner._RUN_STAGES[:11])

    async def fake_validate(document: Document, *, retrospective_date: date | None = None) -> Document:
        return _complete(document, runner._RUN_STAGES[11:])

    monkeypatch.setattr(runner, "grow_roots", fake_grow)
    monkeypatch.setattr(runner, "validate_roots", fake_validate)
    run_dir = asyncio.run(runner._run(data_root, tmp_path / "results", None))
    assert calls == list(filenames)

    (run_dir / "documents" / "002.txt.json").unlink()
    record_path = run_dir / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["status"] = "failed"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    assert asyncio.run(runner._run(tmp_path / "ignored", tmp_path / "unused", None, run_dir)) == run_dir
    assert calls == [*filenames, "002.txt"]
    assert json.loads(record_path.read_text(encoding="utf-8"))["status"] == "complete"

    (data_root / "primary" / "documents_txt" / "002.txt").write_text("changed source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Run source content differs"):
        asyncio.run(runner._run(tmp_path / "ignored", tmp_path / "unused", None, run_dir))
    assert calls == [*filenames, "002.txt"]


def test_resume_rejects_stale_saved_source_before_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    roots_dir = tmp_path / "roots"
    roots_dir.mkdir()
    source = data_root / "primary" / "documents_txt" / "001.txt"
    roots = _complete(Document.from_source(source), runner._ROOT_STAGES)
    (roots_dir / "001.txt.json").write_text(roots.model_dump_json(), encoding="utf-8")
    source.write_text("changed source\n", encoding="utf-8")

    async def unexpected_review(_document: Document) -> Document:
        pytest.fail("A stale checkpoint must not reach the provider-backed review")

    monkeypatch.setattr(runner, "review_docket_root_equivalence", unexpected_review)
    with pytest.raises(ValueError, match="Saved Document text differs"):
        asyncio.run(runner._run(data_root, tmp_path / "results", roots_dir))


def test_resume_from_reporter_llm_checkpoint_only_runs_later_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    checkpoint_dir = tmp_path / "reporter-review-input"
    checkpoint_dir.mkdir()
    source = data_root / "primary" / "documents_txt" / "001.txt"
    ready = _complete(Document.from_source(source), runner._REPORTER_REVIEW_INPUT_STAGES)
    (checkpoint_dir / "001.txt.json").write_text(ready.model_dump_json(), encoding="utf-8")
    calls: list[str] = []

    async def unique(document: Document) -> Document:
        calls.append("14")
        return document.complete(runner._RUN_STAGES[13])

    async def ambiguous(document: Document) -> Document:
        calls.append("15")
        return document.complete(runner._RUN_STAGES[14])

    def docket(document: Document) -> Document:
        calls.append("16")
        return document.complete(runner._RUN_STAGES[15])

    async def docket_review(document: Document) -> Document:
        calls.append("17")
        return document.complete(runner._RUN_STAGES[16])

    def govinfo(document: Document) -> Document:
        calls.append("18")
        return document.complete(runner._RUN_STAGES[17])

    async def govinfo_review(document: Document) -> Document:
        calls.append("19")
        return document.complete(runner._RUN_STAGES[18])

    monkeypatch.setattr(runner, "reporter_root_lookup_unique_llm", unique)
    monkeypatch.setattr(runner, "reporter_root_lookup_ambiguous_llm", ambiguous)
    monkeypatch.setattr(runner, "docket_root_lookup", docket)
    monkeypatch.setattr(runner, "docket_root_lookup_review", docket_review)
    monkeypatch.setattr(runner, "govinfo_docket_lookup", govinfo)
    monkeypatch.setattr(runner, "govinfo_docket_lookup_review", govinfo_review)
    run_dir = asyncio.run(runner._run(data_root, tmp_path / "results", None, None, checkpoint_dir))

    assert calls == ["14", "15", "16", "17", "18", "19"]
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "complete"
    saved = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text(encoding="utf-8"))
    assert saved.get_stage(runner._REPORTER_REVIEW_INPUT_STAGE) == ready
    assert asyncio.run(runner._run(tmp_path / "ignored", tmp_path / "unused", None, run_dir)) == run_dir
    assert calls == ["14", "15", "16", "17", "18", "19"]


def test_reporter_review_replay_reuses_saved_docket_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    source_path = data_root / "primary" / "documents_txt" / "001.txt"
    source_path.write_text("Acme v. Reed, Case No. 2:31-cv-45821 (S.D.N.Y. 2031).", encoding="utf-8")
    monkeypatch.setattr("mellea_lrc.extraction.docket_site_hunting.suspected_dockets", lambda _: ())
    ready = asyncio.run(grow_roots(Document.from_source(source_path), hunt_dockets=True))
    ready = _complete(ready, runner._RUN_STAGES[10:13])
    root = next(root for root in ready.roots if isinstance(root, FullDocketCitation))
    prior = _complete(ready, runner._RUN_STAGES[13:15])
    recorded = root.record(runner._DOCKET_LOOKUP_STAGE)
    lookup = DocketLookup(
        node_id=recorded.nodes[-1].id,
        attempts=(DocketLookupAttempt(source_type="d", query="docketNumber:(2:31-cv-45821)"),),
    )
    prior = prior.replace_citation(recorded.with_docket_lookup(lookup)).complete(runner._DOCKET_LOOKUP_STAGE)
    checkpoint_dir = tmp_path / "reporter-review-input"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "001.txt.json").write_text(prior.model_dump_json(), encoding="utf-8")

    async def unique(document: Document) -> Document:
        return document.complete(runner._RUN_STAGES[13])

    async def ambiguous(document: Document) -> Document:
        return document.complete(runner._RUN_STAGES[14])

    def unexpected_lookup(_document: Document) -> Document:
        pytest.fail("Saved docket lookup must be reused")

    async def docket_review(document: Document) -> Document:
        return document.complete(runner._RUN_STAGES[16])

    def govinfo(document: Document) -> Document:
        return document.complete(runner._RUN_STAGES[17])

    async def govinfo_review(document: Document) -> Document:
        return document.complete(runner._RUN_STAGES[18])

    monkeypatch.setattr(runner, "reporter_root_lookup_unique_llm", unique)
    monkeypatch.setattr(runner, "reporter_root_lookup_ambiguous_llm", ambiguous)
    monkeypatch.setattr(runner, "docket_root_lookup", unexpected_lookup)
    monkeypatch.setattr(runner, "docket_root_lookup_review", docket_review)
    monkeypatch.setattr(runner, "govinfo_docket_lookup", govinfo)
    monkeypatch.setattr(runner, "govinfo_docket_lookup_review", govinfo_review)
    run_dir = asyncio.run(runner._run(data_root, tmp_path / "results", None, None, checkpoint_dir, True))

    saved = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text(encoding="utf-8"))
    replayed_root = next(root for root in saved.roots if isinstance(root, FullDocketCitation))
    assert replayed_root.docket_lookup == lookup
    assert saved.get_stage(runner._REPORTER_REVIEW_INPUT_STAGE) == ready


def test_resume_from_docket_review_checkpoint_only_runs_govinfo_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    checkpoint_dir = tmp_path / "docket-review-input"
    checkpoint_dir.mkdir()
    source = data_root / "primary" / "documents_txt" / "001.txt"
    ready = _complete(Document.from_source(source), runner._DOCKET_REVIEW_INPUT_STAGES)
    (checkpoint_dir / "001.txt.json").write_text(ready.model_dump_json(), encoding="utf-8")
    calls: list[str] = []

    def govinfo(document: Document) -> Document:
        calls.append("18")
        return document.complete(runner._RUN_STAGES[17])

    async def govinfo_review(document: Document) -> Document:
        calls.append("19")
        return document.complete(runner._RUN_STAGES[18])

    monkeypatch.setattr(runner, "govinfo_docket_lookup", govinfo)
    monkeypatch.setattr(runner, "govinfo_docket_lookup_review", govinfo_review)
    run_dir = asyncio.run(
        runner._run(data_root, tmp_path / "results", None, None, None, False, checkpoint_dir)
    )

    assert calls == ["18", "19"]
    saved = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text(encoding="utf-8"))
    assert saved.stage_runs == runner._RUN_STAGES
    assert saved.get_stage(runner._DOCKET_REVIEW_INPUT_STAGE) == ready


def test_resume_from_validation_checkpoint_runs_only_body_stages_and_reuses_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filenames = ("001.txt", "002.txt")
    data_root = _dataset(tmp_path, filenames)
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    for filename in filenames:
        source = data_root / "primary" / "documents_txt" / filename
        ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_STAGES)
        (checkpoint_dir / f"{filename}.json").write_text(ready.model_dump_json(), encoding="utf-8")
    calls: list[tuple[str, date | None]] = []

    async def body(document: Document, *, retrospective_date: date | None = None) -> Document:
        calls.append((Path(document.source_path or "").name, retrospective_date))
        assert document.stage_runs == runner._VALIDATION_INPUT_STAGES
        return _complete(document, runner._RUN_STAGES[19:])

    monkeypatch.setattr(runner, "corroborate_root_locator_bodies", body)
    cutoff = date(2024, 6, 1)
    run_dir = asyncio.run(
        runner._run(
            data_root,
            tmp_path / "results",
            None,
            from_validation_documents=checkpoint_dir,
            retrospective_date=cutoff,
        )
    )
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["from_validation_documents"] == str(checkpoint_dir)
    assert record["retrospective_date"] == cutoff.isoformat()
    assert calls == [(filename, cutoff) for filename in filenames]
    for filename in filenames:
        saved = Document.model_validate_json((run_dir / "documents" / f"{filename}.json").read_text())
        assert saved.stage_runs == runner._RUN_STAGES
        assert saved.get_stage(runner._VALIDATION_INPUT_STAGE) == Document.model_validate_json(
            (checkpoint_dir / f"{filename}.json").read_text()
        )

    (run_dir / "documents" / "002.txt.json").unlink()
    record["status"] = "failed"
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    assert asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir)) == run_dir
    assert calls == [(filename, cutoff) for filename in filenames] + [("002.txt", cutoff)]


def test_reserved_pool_is_saved_reused_and_closed_without_saving_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filenames = ("001.txt", "002.txt")
    data_root = _dataset(tmp_path, filenames)
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    for filename in filenames:
        source = data_root / "primary" / "documents_txt" / filename
        ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_STAGES)
        (checkpoint_dir / f"{filename}.json").write_text(ready.model_dump_json(), encoding="utf-8")

    monkeypatch.setenv("COURTLISTENER_BASE_URL", "https://proxy.example/api/rest/v4/")
    monkeypatch.setenv("COURTLISTENER_API_TOKEN_RESERVED", "offline-reserved-token")
    calls: list[tuple[str, CourtListenerClient]] = []
    closed: list[CourtListenerClient] = []
    original_close = CourtListenerClient.close

    def close(client: CourtListenerClient) -> None:
        closed.append(client)
        original_close(client)

    async def body(
        document: Document, *, retrospective_date: date | None, courtlistener_client: CourtListenerClient
    ) -> Document:
        assert retrospective_date is None
        calls.append((Path(document.source_path or "").name, courtlistener_client))
        return _complete(document, runner._RUN_STAGES[19:])

    monkeypatch.setattr(CourtListenerClient, "close", close)
    monkeypatch.setattr(runner, "corroborate_root_locator_bodies", body)
    run_dir = asyncio.run(
        runner._run(
            data_root,
            tmp_path / "results",
            None,
            from_validation_documents=checkpoint_dir,
            courtlistener_pool="reserved",
        )
    )

    record_text = (run_dir / "run.json").read_text(encoding="utf-8")
    assert json.loads(record_text)["courtlistener_pool"] == "reserved"
    assert "offline-reserved-token" not in record_text
    assert [filename for filename, _client in calls] == list(filenames)
    assert calls[0][1] is calls[1][1]
    assert closed == [calls[0][1]]
    assert calls[0][1].config.base_url == "https://proxy.example/api/rest/v4/"
    assert calls[0][1].config.pool == "reserved"
    assert calls[0][1].config.token == "offline-reserved-token"

    (run_dir / "documents" / "002.txt.json").unlink()
    assert asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir)) == run_dir
    assert [filename for filename, _client in calls] == [*filenames, "002.txt"]
    assert calls[2][1] is not calls[0][1]
    assert closed == [calls[0][1], calls[2][1]]


def test_resume_existing_run_can_select_and_save_reserved_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    source = data_root / "primary" / "documents_txt" / "001.txt"
    ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_STAGES)
    (checkpoint_dir / "001.txt.json").write_text(ready.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("COURTLISTENER_BASE_URL", "https://proxy.example/api/rest/v4/")
    monkeypatch.setenv("COURTLISTENER_API_TOKEN_RESERVED", "offline-reserved-token")
    clients: list[CourtListenerClient | None] = []

    async def body(
        document: Document,
        *,
        retrospective_date: date | None,
        courtlistener_client: CourtListenerClient | None = None,
    ) -> Document:
        clients.append(courtlistener_client)
        return _complete(document, runner._RUN_STAGES[19:])

    monkeypatch.setattr(runner, "corroborate_root_locator_bodies", body)
    run_dir = asyncio.run(
        runner._run(data_root, tmp_path / "results", None, from_validation_documents=checkpoint_dir)
    )
    record_path = run_dir / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.pop("courtlistener_pool")  # Older runs did not record a pool.
    record_path.write_text(json.dumps(record), encoding="utf-8")
    (run_dir / "documents" / "001.txt.json").unlink()

    assert (
        asyncio.run(
            runner._run(
                tmp_path / "unused",
                tmp_path / "unused",
                None,
                run_dir,
                courtlistener_pool="reserved",
            )
        )
        == run_dir
    )
    assert clients[0] is None
    assert clients[1] is not None
    assert clients[1].config.pool == "reserved"
    record_text = record_path.read_text(encoding="utf-8")
    assert json.loads(record_text)["courtlistener_pool"] == "reserved"
    assert "offline-reserved-token" not in record_text

    (run_dir / "documents" / "001.txt.json").unlink()
    assert asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir)) == run_dir
    assert clients[2] is not None
    assert clients[2].config.pool == "reserved"


def test_resume_can_switch_reserved_run_to_proxy_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    source = data_root / "primary" / "documents_txt" / "001.txt"
    ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_STAGES)
    (checkpoint_dir / "001.txt.json").write_text(ready.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("COURTLISTENER_BASE_URL", "https://proxy.example/api/rest/v4/")
    monkeypatch.setenv("COURTLISTENER_API_TOKEN_RESERVED", "offline-reserved-token")
    clients: list[CourtListenerClient] = []

    async def body(
        document: Document, *, retrospective_date: date | None, courtlistener_client: CourtListenerClient
    ) -> Document:
        clients.append(courtlistener_client)
        return _complete(document, runner._RUN_STAGES[19:])

    monkeypatch.setattr(runner, "corroborate_root_locator_bodies", body)
    run_dir = asyncio.run(
        runner._run(
            data_root,
            tmp_path / "results",
            None,
            from_validation_documents=checkpoint_dir,
            courtlistener_pool="reserved",
        )
    )
    (run_dir / "documents" / "001.txt.json").unlink()
    assert asyncio.run(
        runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir, courtlistener_pool="proxy")
    ) == run_dir
    assert clients[0].config.pool == "reserved"
    assert clients[1].config.pool is None
    assert clients[1].config.token is None
    record_text = (run_dir / "run.json").read_text(encoding="utf-8")
    assert json.loads(record_text)["courtlistener_pool_history"] == ["reserved", "proxy"]
    assert json.loads(record_text)["courtlistener_pool"] == "proxy"
    assert "offline-reserved-token" not in record_text


def test_retry_body_stages_passes_selected_client_to_both_courtlistener_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _complete(Document.from_source("source"), runner._VALIDATION_INPUT_STAGES)
    selected_client = object()
    calls: list[str] = []

    def opinion(document: Document, *, retrospective_date: date | None, client: object) -> Document:
        assert client is selected_client
        calls.append("opinion")
        return document.complete(runner._COURTLISTENER_OPINION_STAGE)

    def recap(document: Document, *, retrospective_date: date | None, client: object) -> Document:
        assert client is selected_client
        calls.append("recap")
        return document.complete(runner._COURTLISTENER_RECAP_STAGE)

    def govinfo(document: Document, *, retrospective_date: date | None) -> Document:
        calls.append("govinfo")
        return document.complete(runner._GOVINFO_OPINION_STAGE)

    async def review(document: Document) -> Document:
        calls.append("review")
        return document.complete(runner._LOCATOR_BODY_REVIEW_STAGE)

    monkeypatch.setattr(runner, "courtlistener_opinion_locator_body_search", opinion)
    monkeypatch.setattr(runner, "courtlistener_recap_locator_body_search", recap)
    monkeypatch.setattr(runner, "govinfo_opinion_locator_body_search", govinfo)
    monkeypatch.setattr(runner, "review_locator_body_evidence", review)
    result = asyncio.run(
        runner._retry_body_stages(document, runner._COURTLISTENER_OPINION_STAGE, None, selected_client)
    )
    assert result.stage_runs == runner._RUN_STAGES
    assert calls == ["opinion", "recap", "govinfo", "review"]


def test_corroboration_workflow_passes_selected_client_to_both_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = importlib.import_module("mellea_lrc.workflows.corroborate_root_locator_bodies")
    document = _complete(Document.from_source("source"), runner._VALIDATION_INPUT_STAGES)
    selected_client = object()
    calls: list[str] = []

    def opinion(document: Document, *, retrospective_date: date | None, client: object) -> Document:
        assert client is selected_client
        calls.append("opinion")
        return document.complete(runner._COURTLISTENER_OPINION_STAGE)

    def recap(document: Document, *, retrospective_date: date | None, client: object) -> Document:
        assert client is selected_client
        calls.append("recap")
        return document.complete(runner._COURTLISTENER_RECAP_STAGE)

    def govinfo(document: Document, *, retrospective_date: date | None) -> Document:
        calls.append("govinfo")
        return document.complete(runner._GOVINFO_OPINION_STAGE)

    async def review(document: Document) -> Document:
        calls.append("review")
        return document.complete(runner._LOCATOR_BODY_REVIEW_STAGE)

    monkeypatch.setattr(workflow, "courtlistener_opinion_locator_body_search", opinion)
    monkeypatch.setattr(workflow, "courtlistener_recap_locator_body_search", recap)
    monkeypatch.setattr(workflow, "govinfo_opinion_locator_body_search", govinfo)
    monkeypatch.setattr(workflow, "review_locator_body_evidence", review)
    result = asyncio.run(
        workflow.corroborate_root_locator_bodies(document, courtlistener_client=selected_client)
    )
    assert result.stage_runs == runner._RUN_STAGES
    assert calls == ["opinion", "recap", "govinfo", "review"]


def test_validation_replay_checks_all_sources_before_body_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filenames = ("001.txt", "002.txt")
    data_root = _dataset(tmp_path, filenames)
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    for filename in filenames:
        source = data_root / "primary" / "documents_txt" / filename
        ready = _complete(Document.from_source(source), runner._VALIDATION_INPUT_STAGES)
        (checkpoint_dir / f"{filename}.json").write_text(ready.model_dump_json(), encoding="utf-8")
    (data_root / "primary" / "documents_txt" / "002.txt").write_text("changed source\n", encoding="utf-8")

    async def unexpected_body(_document: Document, *, retrospective_date: date | None = None) -> Document:
        pytest.fail("A stale validation checkpoint must not reach body providers")

    monkeypatch.setattr(runner, "corroborate_root_locator_bodies", unexpected_body)
    with pytest.raises(ValueError, match="Saved Document text differs"):
        asyncio.run(
            runner._run(data_root, tmp_path / "results", None, from_validation_documents=checkpoint_dir)
        )


@pytest.mark.parametrize(
    ("failed_stage", "failure_type", "status"),
    [
        (runner._COURTLISTENER_OPINION_STAGE, "http_error", 429),
        (runner._COURTLISTENER_RECAP_STAGE, "transport_error", None),
        (runner._GOVINFO_OPINION_STAGE, "http_error", 503),
    ],
)
def test_transient_body_search_failure_replays_from_failed_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
    failure_type: str,
    status: int | None,
) -> None:
    filenames = ("001.txt", "002.txt")
    data_root = _dataset(tmp_path, filenames)
    source = "Acme v. Reed, Case No. 2:31-cv-45821 (S.D.N.Y. 2031)."
    checkpoint_dir = tmp_path / "validation-input"
    checkpoint_dir.mkdir()
    locator = "Case No. 2:31-cv-45821"
    number = "2:31-cv-45821"
    for filename in filenames:
        source_path = data_root / "primary" / "documents_txt" / filename
        source_path.write_text(source, encoding="utf-8")
        ready = Document.from_source(source_path).complete(runner._RUN_STAGES[0])
        root = FullDocketCitation.from_locator(
            citation_id="docket:0",
            stage=runner._RUN_STAGES[1],
            source=source,
            span=Span(source.index(locator), source.index(locator) + len(locator)),
            number_span=Span(source.index(number), source.index(number) + len(number)),
        )
        ready = ready.add_citation(root).complete(runner._RUN_STAGES[1])
        ready = _complete(ready, runner._RUN_STAGES[2:9])
        ready = ready.replace_citation(root.record(runner._ROOT_STAGE).with_root(root.id)).complete(
            runner._ROOT_STAGE
        )
        ready = _complete(ready, runner._RUN_STAGES[10:19])
        (checkpoint_dir / f"{filename}.json").write_text(ready.model_dump_json(), encoding="utf-8")

    cutoff = date(2024, 6, 1)
    calls: list[tuple[str, str]] = []
    stages = (
        (runner._COURTLISTENER_OPINION_STAGE, BodySource.COURTLISTENER_OPINION),
        (runner._COURTLISTENER_RECAP_STAGE, BodySource.COURTLISTENER_RECAP),
        (runner._GOVINFO_OPINION_STAGE, BodySource.GOVINFO_OPINION),
    )

    def fake_provider(stage: str, body_source: BodySource):
        def run(document: Document, *, retrospective_date: date | None = None) -> Document:
            filename = Path(document.source_path or "").name
            assert retrospective_date == cutoff
            assert document.stage_runs == runner._RUN_STAGES[: runner._RUN_STAGES.index(stage)]
            first_call = (filename, stage) not in calls
            calls.append((filename, stage))
            root = document.roots[0].record(stage)
            failure = (
                BodyEvidenceFailure(
                    failure_type=failure_type,
                    message="Provider request failed",
                    status_code=status,
                )
                if filename == "001.txt" and stage == failed_stage and first_call
                else None
            )
            search = BodySearch(
                node_id=root.nodes[-1].id,
                source=body_source,
                retrospective_date=retrospective_date,
                attempts=(BodySearchAttempt(query=number, failure=failure),),
                failures=(failure,) if failure is not None else (),
            )
            return document.replace_citation(root.with_body_search(search)).complete(stage)

        return run

    monkeypatch.setattr(runner, "courtlistener_opinion_locator_body_search", fake_provider(*stages[0]))
    monkeypatch.setattr(runner, "courtlistener_recap_locator_body_search", fake_provider(*stages[1]))
    monkeypatch.setattr(runner, "govinfo_opinion_locator_body_search", fake_provider(*stages[2]))

    async def review(document: Document) -> Document:
        filename = Path(document.source_path or "").name
        assert document.stage_runs == runner._RUN_STAGES[:-1]
        calls.append((filename, runner._LOCATOR_BODY_REVIEW_STAGE))
        return document.complete(runner._LOCATOR_BODY_REVIEW_STAGE)

    monkeypatch.setattr(runner, "review_locator_body_evidence", review)

    async def body(document: Document, *, retrospective_date: date | None = None) -> Document:
        assert document.stage_runs == runner._VALIDATION_INPUT_STAGES
        for stage, _source in stages:
            if stage == runner._COURTLISTENER_OPINION_STAGE:
                document = runner.courtlistener_opinion_locator_body_search(
                    document, retrospective_date=retrospective_date
                )
            elif stage == runner._COURTLISTENER_RECAP_STAGE:
                document = runner.courtlistener_recap_locator_body_search(
                    document, retrospective_date=retrospective_date
                )
            else:
                document = runner.govinfo_opinion_locator_body_search(
                    document, retrospective_date=retrospective_date
                )
        return await runner.review_locator_body_evidence(document)

    monkeypatch.setattr(runner, "corroborate_root_locator_bodies", body)
    with pytest.raises(RuntimeError, match="transient provider failures"):
        asyncio.run(
            runner._run(
                data_root,
                tmp_path / "results",
                None,
                from_validation_documents=checkpoint_dir,
                retrospective_date=cutoff,
            )
        )
    run_dir = next((tmp_path / "results").iterdir())
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["transient_failures"] == {"docket": [], "body": ["001.txt"]}
    initial = Document.model_validate_json(
        (run_dir / "documents" / "001.txt.json").read_text(encoding="utf-8")
    )
    assert runner._transient_body_failure_stage(initial) == failed_stage
    previous_stage = runner._RUN_STAGES[runner._RUN_STAGES.index(failed_stage) - 1]
    first_calls = len(calls)

    assert asyncio.run(runner._run(tmp_path / "unused", tmp_path / "unused", None, run_dir)) == run_dir
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert "transient_failures" not in record
    assert calls[first_calls:] == [
        ("001.txt", stage)
        for stage in (
            *runner._BODY_SEARCH_STAGES[runner._BODY_SEARCH_STAGES.index(failed_stage) :],
            runner._LOCATOR_BODY_REVIEW_STAGE,
        )
    ]
    final = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text(encoding="utf-8"))
    assert final.get_stage(previous_stage) == initial.get_stage(previous_stage)
    assert not runner._has_transient_body_search_failure(final)


@pytest.mark.parametrize(
    ("failure_type", "status"),
    [("http_error", 429), ("transport_error", None)],
)
def test_transient_docket_search_failure_requires_a_rerun(failure_type: str, status: int | None) -> None:
    document = asyncio.run(grow_roots(Document.from_source("Acme v. Reed, Case No. 2:31-cv-45821.")))
    root = document.roots[0]
    assert isinstance(root, FullDocketCitation)
    recorded = root.record("16_docket_root_lookup")
    lookup = DocketLookup(
        node_id=recorded.nodes[-1].id,
        attempts=(
            DocketLookupAttempt(
                source_type="d",
                query="docketNumber:(2:31-cv-45821)",
                failure=DocketLookupFailure(
                    failure_type=failure_type,
                    message="Provider request failed",
                    upstream_status_code=status,
                ),
            ),
        ),
    )
    incomplete = document.replace_citation(recorded.with_docket_lookup(lookup))

    assert runner._has_transient_docket_lookup_failure(incomplete)
    assert not runner._has_transient_docket_lookup_failure(document)


def test_transient_govinfo_search_failure_requires_a_rerun() -> None:
    document = asyncio.run(grow_roots(Document.from_source("Acme v. Reed, Case No. 2:31-cv-45821.")))
    root = document.roots[0]
    assert isinstance(root, FullDocketCitation)
    recorded = root.record("18_govinfo_docket_lookup")
    lookup = GovInfoDocketLookup(
        node_id=recorded.nodes[-1].id,
        attempts=(
            GovInfoLookupAttempt(
                query='collection:uscourts casenumber:("2:31-cv-45821")',
                failure=DocketLookupFailure(
                    failure_type="http_error",
                    message="Provider quota reached",
                    upstream_status_code=429,
                ),
            ),
        ),
    )
    incomplete = document.replace_citation(recorded.with_govinfo_docket_lookup(lookup))
    assert runner._has_transient_docket_lookup_failure(incomplete)


def test_new_run_stays_failed_when_a_saved_search_is_transiently_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))

    async def fake_grow(document: Document, *, hunt_dockets: bool, review_docket_roots: bool) -> Document:
        return _complete(document, runner._RUN_STAGES[:11])

    async def fake_validate(document: Document, *, retrospective_date: date | None = None) -> Document:
        return _complete(document, runner._RUN_STAGES[11:])

    monkeypatch.setattr(runner, "grow_roots", fake_grow)
    monkeypatch.setattr(runner, "validate_roots", fake_validate)
    monkeypatch.setattr(runner, "_has_transient_docket_lookup_failure", lambda _document: True)

    with pytest.raises(RuntimeError, match="transient provider failures"):
        asyncio.run(runner._run(data_root, tmp_path / "results", None))

    run_dir = next((tmp_path / "results").iterdir())
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert (run_dir / "documents" / "001.txt.json").exists()
