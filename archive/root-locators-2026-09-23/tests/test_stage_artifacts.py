"""Tests for opt-in persisted Document stage artifacts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mellea_lrc.api import Document, find_full_reporter_locators
from mellea_lrc.serialization import artifact_directory, serialize


def test_serialize_decorator_writes_a_recoverable_sync_stage_checkpoint(tmp_path: Path) -> None:
    @serialize()
    def locate(document: Document) -> Document:
        return find_full_reporter_locators(document)

    with artifact_directory(tmp_path):
        result = locate(Document.from_plain_text("Example, 123 U.S. 456."))

    # The content digest is implementation detail, so find the one checkpoint
    # rather than baking it into the contract.
    artifacts = list(tmp_path.glob("full_reporter_locator_rule/*.json"))
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert Document.model_validate(payload) == result
    assert artifacts[0].parent == tmp_path / "full_reporter_locator_rule"


def test_serialize_decorator_supports_async_stages(tmp_path: Path) -> None:
    @serialize()
    async def locate(document: Document) -> Document:
        return find_full_reporter_locators(document)

    with artifact_directory(tmp_path):
        result = asyncio.run(locate(Document.from_plain_text("Example, 123 U.S. 456.")))

    artifacts = list(tmp_path.glob("full_reporter_locator_rule/*.json"))
    assert len(artifacts) == 1
    assert Document.model_validate(json.loads(artifacts[0].read_text(encoding="utf-8"))) == result


def test_serialize_decorator_preserves_async_stage_keyword_arguments(tmp_path: Path) -> None:
    @serialize()
    async def locate(document: Document, *, enabled: bool) -> Document:
        assert enabled is True
        return find_full_reporter_locators(document)

    with artifact_directory(tmp_path):
        result = asyncio.run(locate(Document.from_plain_text("Example, 123 U.S. 456."), enabled=True))

    artifacts = list(tmp_path.glob("full_reporter_locator_rule/*.json"))
    assert len(artifacts) == 1
    assert Document.model_validate(json.loads(artifacts[0].read_text(encoding="utf-8"))) == result
