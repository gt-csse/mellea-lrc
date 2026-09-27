"""Offline checks for resuming one durable evaluation run."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from evaluations import __main__ as runner
from mellea_lrc.api import Document, grow_roots
from mellea_lrc.model import (
    DocketLookup,
    DocketLookupAttempt,
    DocketLookupFailure,
    FullDocketCitation,
)


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

    async def fake_validate(document: Document) -> Document:
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

    monkeypatch.setattr(runner, "reporter_root_lookup_unique_llm", unique)
    monkeypatch.setattr(runner, "reporter_root_lookup_ambiguous_llm", ambiguous)
    monkeypatch.setattr(runner, "docket_root_lookup", docket)
    monkeypatch.setattr(runner, "docket_root_lookup_review", docket_review)
    run_dir = asyncio.run(runner._run(data_root, tmp_path / "results", None, None, checkpoint_dir))

    assert calls == ["14", "15", "16", "17"]
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "complete"
    saved = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text(encoding="utf-8"))
    assert saved.get_stage(runner._REPORTER_REVIEW_INPUT_STAGE) == ready
    assert asyncio.run(runner._run(tmp_path / "ignored", tmp_path / "unused", None, run_dir)) == run_dir
    assert calls == ["14", "15", "16", "17"]


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

    monkeypatch.setattr(runner, "reporter_root_lookup_unique_llm", unique)
    monkeypatch.setattr(runner, "reporter_root_lookup_ambiguous_llm", ambiguous)
    monkeypatch.setattr(runner, "docket_root_lookup", unexpected_lookup)
    monkeypatch.setattr(runner, "docket_root_lookup_review", docket_review)
    run_dir = asyncio.run(runner._run(data_root, tmp_path / "results", None, None, checkpoint_dir, True))

    saved = Document.model_validate_json((run_dir / "documents" / "001.txt.json").read_text(encoding="utf-8"))
    replayed_root = next(root for root in saved.roots if isinstance(root, FullDocketCitation))
    assert replayed_root.docket_lookup == lookup
    assert saved.get_stage(runner._REPORTER_REVIEW_INPUT_STAGE) == ready


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


def test_new_run_stays_failed_when_a_saved_search_is_transiently_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _dataset(tmp_path, ("001.txt",))

    async def fake_grow(document: Document, *, hunt_dockets: bool, review_docket_roots: bool) -> Document:
        return _complete(document, runner._RUN_STAGES[:11])

    async def fake_validate(document: Document) -> Document:
        return _complete(document, runner._RUN_STAGES[11:])

    monkeypatch.setattr(runner, "grow_roots", fake_grow)
    monkeypatch.setattr(runner, "validate_roots", fake_validate)
    monkeypatch.setattr(runner, "_has_transient_docket_lookup_failure", lambda _document: True)

    with pytest.raises(RuntimeError, match="transient provider failures"):
        asyncio.run(runner._run(data_root, tmp_path / "results", None))

    run_dir = next((tmp_path / "results").iterdir())
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert (run_dir / "documents" / "001.txt.json").exists()
