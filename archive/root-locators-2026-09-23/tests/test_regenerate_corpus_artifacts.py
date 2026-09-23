"""Tests for resuming corpus identity from native root-stage checkpoints."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import date

import pytest

from mellea_lrc.model.document import Document
from scripts import regenerate_corpus_artifacts as corpus


def test_corpus_runner_forwards_saved_roots_and_records_their_hash(monkeypatch, tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "filing.txt"
    source.write_text("Smith v. Jones, 123 U.S. 456 (1887).", encoding="utf-8")
    roots = tmp_path / "roots"
    checkpoint = roots / "root_formation" / "filing.json"
    checkpoint.parent.mkdir(parents=True)
    root_document = Document.from_source(source).evolve(passes=("root_formation",))
    checkpoint.write_text(json.dumps(root_document.model_dump(mode="json")), encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    inventory = tmp_path / "dates.json"
    inventory.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "set": "primary",
                        "source_txt": str(source),
                        "source_sha256": source_hash,
                        "retrospective_date": "2025-01-02",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    sidecar = tmp_path / "filings.json"
    sidecar.write_text(json.dumps({"filings": {source.name: {"docket_id": "12345"}}}), encoding="utf-8")
    monkeypatch.setitem(corpus.DATASET_SOURCES, "primary", source_dir)
    monkeypatch.setattr(corpus, "CourtListenerClient", object)
    monkeypatch.setattr(corpus, "GovInfoClient", object)
    monkeypatch.setattr(corpus, "start_mellea_session_from_env", object)
    calls = []

    async def _identity_from_checkpoint(path, **kwargs):
        calls.append((path, kwargs))
        return root_document

    monkeypatch.setattr(corpus, "run", _identity_from_checkpoint)
    output = tmp_path / "identity"
    asyncio.run(
        corpus._run(
            argparse.Namespace(
                dataset="primary",
                artifacts=output,
                date_inventory=inventory,
                source_provenance=sidecar,
                root_checkpoints=roots,
                roots_only=False,
                resume=False,
            )
        )
    )

    assert len(calls) == 1
    path, kwargs = calls[0]
    assert path == source.resolve()
    assert kwargs["root_checkpoint"] == checkpoint
    assert kwargs["retrospective_date"] == date(2025, 1, 2)
    assert kwargs["source_provenance"] == sidecar
    manifest = json.loads((output / "corpus-manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["documents"][source.name]["root_checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert manifest["documents"][source.name]["status"] == "complete"
    assert manifest["source_provenance_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()

    sidecar.write_text(json.dumps({"filings": {source.name: {"docket_id": "changed"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Resume configuration changed: source_provenance_sha256"):
        asyncio.run(
            corpus._run(
                argparse.Namespace(
                    dataset="primary",
                    artifacts=output,
                    date_inventory=inventory,
                    source_provenance=sidecar,
                    root_checkpoints=roots,
                    roots_only=False,
                    resume=True,
                )
            )
        )
