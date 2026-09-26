"""Prediction runs remain complete and resumable at document boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from evaluations import run_reporter_root_lookup as runner
from evaluations import run_reporter_root_lookup_ambiguous as ambiguity_runner
from evaluations import run_reporter_root_lookup_unique_llm as unique_runner
from mellea_lrc.api import Document, reporter_root_lookup
from mellea_lrc.courtlistener import CourtListenerCitationLookup
from mellea_lrc.model import Span


class FakeLookupClient:
    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        assert (volume, reporter, page) == ("550", "U.S.", "544")
        return CourtListenerCitationLookup.model_validate(
            {
                "citation": "550 U.S. 544",
                "status": 200,
                "clusters": [
                    {
                        "id": 1,
                        "caseNameFull": "Bell Atlantic Corporation v. Twombly",
                        "court_id": "scotus",
                        "dateFiled": "2007-05-21",
                        "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
                    }
                ],
            }
        )


def _corpus(data_root: Path) -> tuple[str, str]:
    name = "primary"
    filename = "sample.txt"
    text = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007). INDEX"
    source = data_root / name / "documents_txt" / filename
    source.parent.mkdir(parents=True)
    source.write_text(text, encoding="utf-8")
    (data_root / name / "documents.json").write_text(
        json.dumps(
            {
                "documents": {
                    filename: {
                        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "length": len(text),
                        "index_spans": [{"start": text.index("INDEX"), "end": len(text)}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return name, filename


def test_provider_failure_preserves_root_checkpoint_and_resume_does_not_reextract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    name, filename = _corpus(data_root)

    def fail_lookup(_document: Document) -> Document:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(runner, "reporter_root_lookup", fail_lookup)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(runner.run_documents(data_root, run_dir, (name,)))

    root_path = run_dir / "root_documents" / name / f"{filename}.json"
    lookup_path = run_dir / "documents" / name / f"{filename}.json"
    roots = Document.model_validate_json(root_path.read_text(encoding="utf-8"))
    assert roots.index_spans == (Span(roots.text.index("INDEX"), len(roots.text)),)
    assert roots.stage_runs[-1] == "roots"
    assert not lookup_path.exists()

    async def reject_reextract(_document: Document, **_options: object) -> Document:
        raise AssertionError("roots were already saved")

    monkeypatch.setattr(runner, "grow_roots", reject_reextract)
    monkeypatch.setattr(
        runner,
        "reporter_root_lookup",
        lambda document: reporter_root_lookup(document, client=FakeLookupClient()),
    )
    counts = asyncio.run(runner.run_documents(data_root, run_dir, (name,)))
    assert counts == {"roots_created": 0, "roots_reused": 1, "lookup_created": 1, "lookup_reused": 0}
    exact = Document.model_validate_json(lookup_path.read_text(encoding="utf-8"))
    assert exact.get_stage("roots") == roots
    assert exact.roots[0].identity_judgments[-1].verdict.value == "correct_identity"

    counts = asyncio.run(runner.run_documents(data_root, run_dir, (name,)))
    assert counts == {"roots_created": 0, "roots_reused": 1, "lookup_created": 0, "lookup_reused": 1}

    ambiguity_dir = tmp_path / "ambiguity-run"
    counts = ambiguity_runner.run_documents(data_root, run_dir, ambiguity_dir, (name,))
    assert counts == {"ambiguity_created": 1, "ambiguity_reused": 0}
    ambiguity = Document.model_validate_json(
        (ambiguity_dir / "documents" / name / f"{filename}.json").read_text(encoding="utf-8")
    )
    assert ambiguity.get_stage("reporter_root_lookup") == exact
    assert ambiguity.stage_runs[-1] == "reporter_root_lookup_ambiguous"
    assert ambiguity_runner.run_documents(data_root, run_dir, ambiguity_dir, (name,)) == {
        "ambiguity_created": 0,
        "ambiguity_reused": 1,
    }


def test_manifest_drift_rejects_saved_run(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    name, filename = _corpus(data_root)
    asyncio.run(runner.run_documents(data_root, run_dir, (name,), through="roots"))

    source = data_root / name / "documents_txt" / filename
    source.write_text(source.read_text(encoding="utf-8") + " changed", encoding="utf-8")
    with pytest.raises(ValueError, match=r"source differs from documents\.json"):
        asyncio.run(runner.run_documents(data_root, run_dir, (name,), through="roots"))


def test_unique_review_runner_preserves_lookup_checkpoint_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    lookup_dir = tmp_path / "lookup"
    review_dir = tmp_path / "review"
    name, filename = _corpus(data_root)
    monkeypatch.setattr(
        runner,
        "reporter_root_lookup",
        lambda document: reporter_root_lookup(document, client=FakeLookupClient()),
    )
    asyncio.run(runner.run_documents(data_root, lookup_dir, (name,)))

    async def fake_review(document: Document) -> Document:
        return document.complete(unique_runner.STAGE)

    monkeypatch.setattr(unique_runner, "reporter_root_lookup_unique_llm", fake_review)
    assert asyncio.run(unique_runner.run_documents(data_root, lookup_dir, review_dir, (name,))) == {
        "review_created": 1,
        "review_reused": 0,
    }
    lookup = Document.model_validate_json(
        (lookup_dir / "documents" / name / f"{filename}.json").read_text(encoding="utf-8")
    )
    reviewed = Document.model_validate_json(
        (review_dir / "documents" / name / f"{filename}.json").read_text(encoding="utf-8")
    )
    assert reviewed.get_stage("reporter_root_lookup") == lookup
    assert reviewed.stage_runs[-1] == unique_runner.STAGE

    async def reject_repeat(_document: Document) -> Document:
        raise AssertionError("Completed review must not run again")

    monkeypatch.setattr(unique_runner, "reporter_root_lookup_unique_llm", reject_repeat)
    assert asyncio.run(unique_runner.run_documents(data_root, lookup_dir, review_dir, (name,))) == {
        "review_created": 0,
        "review_reused": 1,
    }
