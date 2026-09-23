"""A saved document is sufficient to resume the next public stage."""

import asyncio
from dataclasses import replace

import pytest
from pydantic import BaseModel

from mellea_lrc.api import Document, grow_leaves, grow_roots


def test_native_checkpoint_resumes_leaf_growth_with_complete_operations() -> None:
    source = "Brown v. Board, 347 U.S. 483 (1954). Id. at 495."
    roots = asyncio.run(grow_roots(Document.from_plain_text(source)))

    assert isinstance(roots, BaseModel)
    assert len(roots.citations) == 1
    assert roots.citations[0].has_complete_history

    restored = Document.model_validate_json(roots.model_dump_json())
    assert restored == roots
    leaves = asyncio.run(grow_leaves(restored))
    assert len(leaves.citations) == 2
    assert all(record.has_complete_history for record in leaves.citations)
    assert Document.model_validate(leaves.model_dump(mode="json")) == leaves
    assert len(roots.citations) == 1


def test_document_rejects_citation_without_replayable_creation() -> None:
    roots = asyncio.run(grow_roots(Document.from_plain_text("Brown v. Board, 347 U.S. 483 (1954).")))
    bare_record = replace(roots.citations[0], created_by=None, operations=())

    with pytest.raises(ValueError, match="CREATE-backed operation history"):
        roots.evolve(citations=(bare_record,))


def test_native_dump_rejects_state_changed_without_an_operation() -> None:
    roots = asyncio.run(grow_roots(Document.from_plain_text("Brown v. Board, 347 U.S. 483 (1954).")))
    record = roots.citations[0]
    record.fields = replace(record.fields, page="999")

    with pytest.raises(ValueError, match="state disagrees with operation history"):
        roots.model_dump_json()
