"""A later native Document checkpoint must retain every earlier durable reading."""

import asyncio
from dataclasses import replace

import pytest

from mellea_lrc.api import (
    Document,
    find_full_reporter_locators,
    grow_leaves,
    grow_roots,
    mark_full_reporter_locator_hunting_skipped,
)
from mellea_lrc.model.document import (
    DocumentCreated,
    SourceMetadataUpdated,
    UnreadCaseNamesUpdated,
)
from mellea_lrc.model.operations import judge_citation
from mellea_lrc.model.record import Node, Question, Reads
from mellea_lrc.model.spans import Span
from mellea_lrc.model.stage import isolated_stage
from mellea_lrc.serialization.stage_artifacts import artifact_directory, serialize


def _located_document() -> Document:
    return find_full_reporter_locators(Document.from_plain_text("Brown v. Board, 347 U.S. 483 (1954)."))


def test_evolve_cannot_drop_or_rewrite_prior_citation_history() -> None:
    document = _located_document()
    with pytest.raises(ValueError, match="remove a created citation"):
        document.evolve(citations=())

    record = document.citations[0]
    rewritten = replace(
        record,
        operations=(
            record.operations[0],
            replace(record.operations[1], reason="Rewritten history"),
            *record.operations[2:],
        ),
    )
    with pytest.raises(ValueError, match="rewrite citation"):
        document.evolve(citations=(rewritten,))


def test_evolve_cannot_drop_a_live_operation_added_after_document_creation() -> None:
    document = _located_document()
    earlier = document.snapshot().citations[0]
    judge_citation(
        document.citations[0],
        Node("identity:review", Reads.RECORD, "identity", "test", "reviewed"),
        Question.IDENTITY,
        "resolved",
    )
    with pytest.raises(ValueError, match="rewrite citation"):
        document.evolve(citations=(earlier,))


def test_document_level_updates_replay_from_creation_after_native_reload() -> None:
    first = Document.from_plain_text("Brown v. Board, 347 U.S. 483 (1954).")
    changed_names = first.evolve(unread_case_names=(Span(start=0, end=5),))
    changed_source = changed_names.evolve(
        source_metadata=replace(changed_names.source_metadata, courtlistener_docket_id="123")
    )
    assert [type(event) for event in changed_source.document_events] == [
        DocumentCreated,
        UnreadCaseNamesUpdated,
        SourceMetadataUpdated,
    ]
    assert changed_source.document_events[0].source_metadata == first.source_metadata
    assert changed_source.document_events[1].after == changed_names.unread_case_names
    assert Document.model_validate(changed_source.model_dump(mode="json")) == changed_source

    damaged = changed_source.model_dump(mode="json")
    damaged["document_events"].pop()
    with pytest.raises(ValueError, match="event history"):
        Document.model_validate(damaged)


def test_native_reload_rejects_missing_history_even_with_current_schema() -> None:
    payload = Document.from_plain_text("A case citation.").model_dump(mode="json")
    del payload["document_events"]
    with pytest.raises(ValueError, match="document_events"):
        Document.model_validate(payload)


def test_native_reload_rejects_previous_schema_without_compatibility() -> None:
    payload = Document.from_plain_text("A case citation.").model_dump(mode="json")
    payload["schema_version"] = 20
    with pytest.raises(ValueError, match="schema_version"):
        Document.model_validate(payload)


def test_historical_node_details_cannot_be_rewritten_before_dump() -> None:
    document = mark_full_reporter_locator_hunting_skipped(
        Document.from_plain_text("A case citation."), reason="Disabled"
    )
    document.nodes[0].details["enabled"] = True
    with pytest.raises(ValueError, match="rewrite document nodes"):
        document.model_dump(mode="json")


def test_historical_citation_trace_cannot_be_rewritten_before_dump() -> None:
    document = _located_document()
    document.citations[0].trace[0].details["reader"] = "different"
    with pytest.raises(ValueError, match=r"rewrite citation .* trace"):
        document.model_dump(mode="json")


def test_public_stage_preserves_input_and_rejects_rewritten_node() -> None:
    before = mark_full_reporter_locator_hunting_skipped(
        Document.from_plain_text("A case citation."), reason="Disabled"
    )

    @isolated_stage
    def bad_stage(document: Document) -> Document:
        document.nodes[0].details["enabled"] = True
        return document.evolve(passes=(*document.passes, "bad_stage"))

    with pytest.raises(ValueError, match="rewrite document nodes"):
        bad_stage(before)
    assert before.nodes[0].details == {"enabled": False}


def test_serialize_rejects_non_cumulative_stage_before_writing(tmp_path) -> None:
    @serialize()
    def bad_stage(document: Document) -> Document:
        return Document.from_plain_text(document.text)

    with artifact_directory(tmp_path), pytest.raises(ValueError, match="A stage cannot"):
        bad_stage(_located_document())
    assert not list(tmp_path.rglob("*.json"))


def test_citation_reordering_preserves_history() -> None:
    document = find_full_reporter_locators(Document.from_plain_text("347 U.S. 483; 410 U.S. 113."))
    assert len(document.citations) == 2
    later = document.evolve(citations=tuple(reversed(document.citations)))
    assert {record.citation_id for record in later.citations} == {
        record.citation_id for record in document.citations
    }


def test_real_root_and_leaf_checkpoints_keep_every_earlier_event() -> None:
    roots = asyncio.run(
        grow_roots(Document.from_plain_text("Brown v. Board, 347 U.S. 483 (1954). Id. at 485."))
    )
    leaves = asyncio.run(grow_leaves(Document.model_validate(roots.model_dump(mode="json"))))
    assert len(leaves.citations) > len(roots.citations)
    for earlier in roots.citations:
        later = next(record for record in leaves.citations if record.citation_id == earlier.citation_id)
        assert later.operations[: len(earlier.operations)] == earlier.operations
        assert later.trace[: len(earlier.trace)] == earlier.trace
    assert leaves.passes[: len(roots.passes)] == roots.passes
    assert leaves.document_events[: len(roots.document_events)] == roots.document_events
    assert Document.model_validate(leaves.model_dump(mode="json")) == leaves


def test_a_field_projection_without_an_update_cannot_enter_a_checkpoint() -> None:
    document = _located_document()
    original = document.citations[0]
    with pytest.raises(ValueError, match="state disagrees with operation history"):
        changed = replace(original, fields=replace(original.fields, page="999"))
        document.evolve(citations=(changed,))

    directly_mutated = document.snapshot()
    directly_mutated.citations[0].fields = replace(original.fields, page="999")
    with pytest.raises(ValueError, match="state disagrees with operation history"):
        directly_mutated.model_dump(mode="json")
