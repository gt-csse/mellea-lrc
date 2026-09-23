"""Contracts shared by extraction, validation, and saved document checkpoints."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from mellea_lrc.api import Document, grow_leaves, resolve_case_names, resolve_courts
from mellea_lrc.model.citations import CitationField, CitationKind, FullCaseCitation, Reporter
from mellea_lrc.model.document import Document as ModelDocument
from mellea_lrc.model.operations import (
    assign_root,
    attribute_authority,
    create_citation,
    judge_citation,
    mark_extraction_reviewed,
    observe_citation,
    observe_document,
    resolve_citation,
    update_field,
    update_fields,
    withdraw_citation,
)
from mellea_lrc.model.record import (
    UNJUDGED,
    WITHDRAWN_HEAD_ID,
    Node,
    OperationKind,
    Question,
    Reads,
    Resolution,
)


def _node(node_id: str, reads: Reads = Reads.DOCUMENT) -> Node:
    return Node(node_id, reads, "architecture_test", "test", "read")


def test_document_is_owned_by_the_shared_model() -> None:
    assert Document is ModelDocument


def test_record_and_document_writes_are_exposed_as_operations() -> None:
    record = create_citation("case-1", CitationKind.FULL_CASE, _node("create"))
    for method in (
        "observe",
        "update_field",
        "update_fields",
        "judge",
        "resolve",
        "reattribute",
        "withdraw",
        "mark_extraction_reviewed_by_llm",
    ):
        assert not hasattr(record, method)
    assert not hasattr(Document.from_plain_text("A filing."), "observe")


def _one_record_document() -> Document:
    text = "Brown v. Board, 347 U.S. 483."
    document = Document.from_plain_text(text)
    record = create_citation("case-1", CitationKind.FULL_CASE, _node("create"))
    locator = "347 U.S. 483"
    start = text.index(locator)
    from mellea_lrc.model.spans import Span

    update_fields(
        record,
        _node("read-locator"),
        {
            CitationField.SPAN: Span(start, start + len(locator)),
            CitationField.LOCATOR_SPAN: Span(start, start + len(locator)),
            CitationField.MATCHED_TEXT: locator,
            CitationField.VOLUME: "347",
            CitationField.REPORTER: Reporter(as_written="U.S.", short_name="U.S.", is_scotus=True),
            CitationField.PAGE: "483",
        },
        reason="first reading from the filing",
    )
    record = assign_root(record, record.citation_id, _node("assign-root", Reads.RECORD))
    from dataclasses import replace

    return document.evolve(citations=(record,), passes=("root_formation",))


def test_create_update_and_withdraw_are_separate_evidenced_operations() -> None:
    created = _node("create")
    record = create_citation("case-1", CitationKind.FULL_CASE, created)
    assert record.fields == FullCaseCitation()
    assert [operation.kind for operation in record.operations] == [OperationKind.CREATE]
    assert record.created_by == created.node_id

    first_read = _node("read")
    assert (
        update_field(record, first_read, CitationField.VOLUME, "347", reason="printed in the filing")
        is record
    )
    assert record.operations[-1].kind is OperationKind.FIELD_UPDATE
    assert record.operations[-1].after == "347"
    assert not hasattr(record.operations[-1], "before")
    assert record.operations[-1].node_id == first_read.node_id

    withdrawn = _node("withdraw", Reads.RECORD)
    assert withdraw_citation(record, withdrawn) is record
    assert record.withdrawn
    assert record.root_id == WITHDRAWN_HEAD_ID
    assert record.root_link_node_id == withdrawn.node_id
    assert [node.node_id for node in record.trace] == ["create", "read", "withdraw"]


def test_creation_requires_document_evidence_and_rejects_record_evidence() -> None:
    with pytest.raises(ValueError, match="requires evidence from the document"):
        create_citation("case-1", CitationKind.FULL_CASE, _node("lookup", Reads.RECORD))


def test_stage_neutral_operations_keep_decisions_with_their_evidence() -> None:
    record = create_citation("case-1", CitationKind.FULL_CASE, _node("create"))
    review = _node("review")
    assert mark_extraction_reviewed(record, review) is record
    assert record.extraction_review_node_id == review.node_id

    lookup = _node("lookup", Reads.RECORD)
    assert observe_citation(record, lookup) is record
    resolution = Resolution("123", "Brown v. Board", "1954-05-17", "scotus", lookup.node_id)
    assert resolve_citation(record, lookup, resolution) is record
    assert attribute_authority(record, lookup, "courtlistener:123") is record
    assert judge_citation(record, lookup, Question.IDENTITY, "confirmed") is record
    assert record.found == resolution
    assert record.authority_id == "courtlistener:123"
    assert record.judgement(Question.IDENTITY).node_id == lookup.node_id
    assert [node.node_id for node in record.trace] == ["create", "review", "lookup"]


def test_authority_attribution_requires_record_evidence_and_an_identifier() -> None:
    record = create_citation("case-1", CitationKind.FULL_CASE, _node("create"))
    with pytest.raises(ValueError, match="requires retrieved record evidence"):
        attribute_authority(record, _node("document-reading"), "authority-1")
    with pytest.raises(ValueError, match="nonempty authority ID"):
        attribute_authority(record, _node("lookup", Reads.RECORD), "")
    assert record.authority_id is None


def test_a_returned_document_is_an_independent_in_memory_snapshot() -> None:
    roots = _one_record_document()
    record = roots.citations[0]
    observe_citation(
        record,
        Node(
            "nested-evidence",
            Reads.DOCUMENT,
            "extraction",
            "test",
            "read",
            details={"alternatives": [{"name": "Brown"}]},
        ),
    )
    before = deepcopy(roots.model_dump(mode="json"))

    leaves = asyncio.run(grow_leaves(roots))
    assert roots.model_dump(mode="json") == before
    assert leaves is not roots
    assert leaves.citations[0] is not record

    later_record = leaves.citations[0]
    judge_citation(later_record, _node("later-judgement", Reads.RECORD), Question.IDENTITY, "confirmed")
    assert record.judgement(Question.IDENTITY).outcome == UNJUDGED
    assert all(node.node_id != "later-judgement" for node in record.trace)

    later_details = next(node.details for node in later_record.trace if node.node_id == "nested-evidence")
    later_details["alternatives"][0]["name"] = "Changed later"
    earlier_details = next(node.details for node in record.trace if node.node_id == "nested-evidence")
    assert earlier_details["alternatives"][0]["name"] == "Brown"
    assert roots.model_dump(mode="json") == before


def test_individual_public_stages_keep_each_prior_checkpoint_independent() -> None:
    initial = _one_record_document()
    named = resolve_case_names(initial)
    courted = resolve_courts(named)

    assert initial is not named and named is not courted
    assert initial.citations[0] is not named.citations[0]
    assert named.citations[0] is not courted.citations[0]
    assert "case_name_resolution" not in initial.passes
    assert "case_name_resolution" in named.passes
    assert "court_resolution" not in named.passes
    assert "court_resolution" in courted.passes

    judge_citation(courted.citations[0], _node("later", Reads.RECORD), Question.IDENTITY, "confirmed")
    assert initial.citations[0].judgement(Question.IDENTITY).outcome == UNJUDGED
    assert named.citations[0].judgement(Question.IDENTITY).outcome == UNJUDGED


def test_document_observation_returns_a_new_snapshot() -> None:
    earlier = _one_record_document()
    node = _node("document-finding")
    later = observe_document(earlier, node)
    assert earlier.nodes == ()
    assert later.nodes == (node,)
    assert later is not earlier
    with pytest.raises(ValueError, match="already belongs to a citation"):
        observe_document(earlier, earlier.citations[0].trace[0])


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("findings", "not a list"),
        ("passes", "not a list"),
        ("nodes", "not a list"),
        ("nodes", None),
        ("citations", ["not a citation"]),
    ],
)
def test_current_checkpoint_rejects_malformed_document_collections(key: str, bad_value: object) -> None:
    payload = _one_record_document().model_dump(mode="json")
    payload[key] = bad_value
    with pytest.raises(ValueError, match=key):
        Document.model_validate(payload)


@pytest.mark.parametrize("bad_value", ["not a mapping", {"identity": "not a judgement"}])
def test_current_checkpoint_rejects_malformed_judgements(bad_value: object) -> None:
    payload = _one_record_document().model_dump(mode="json")
    payload["citations"][0]["judgements"] = bad_value
    with pytest.raises(ValueError, match="judgements"):
        Document.model_validate(payload)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [("findings", ["not a finding"]), ("nodes", ["not a node"])],
)
def test_current_checkpoint_rejects_malformed_collection_entries(key: str, bad_value: object) -> None:
    payload = _one_record_document().model_dump(mode="json")
    payload[key] = bad_value
    with pytest.raises(ValueError, match=key):
        Document.model_validate(payload)


def test_current_checkpoint_rejects_malformed_node_dependencies() -> None:
    payload = _one_record_document().model_dump(mode="json")
    payload["citations"][0]["trace"][0]["depends_on"] = "a node id is not a list"
    with pytest.raises(ValueError, match="depends_on"):
        Document.model_validate(payload)


@pytest.mark.parametrize("field", ["trace", "operations"])
def test_current_checkpoint_rejects_null_record_collections(field: str) -> None:
    payload = _one_record_document().model_dump(mode="json")
    payload["citations"][0][field] = None
    with pytest.raises(ValueError, match=field):
        Document.model_validate(payload)


def test_current_checkpoint_rejects_a_missing_mutation_node() -> None:
    payload = _one_record_document().model_dump(mode="json")
    payload["citations"][0]["trace"] = []
    with pytest.raises(ValueError, match=r"creation node|operation"):
        Document.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("root_id", WITHDRAWN_HEAD_ID),
        (
            "found",
            {
                "cluster_id": "123",
                "case_name": None,
                "date_filed": None,
                "court_id": None,
                "node_id": "missing-resolution",
            },
        ),
    ],
)
def test_current_checkpoint_rejects_decisions_inconsistent_with_operations(
    field: str, bad_value: object
) -> None:
    payload = _one_record_document().model_dump(mode="json")
    payload["citations"][0][field] = bad_value
    with pytest.raises(ValueError, match="state disagrees with operation history"):
        Document.model_validate(payload)


def test_current_checkpoint_rejects_a_judgement_inconsistent_with_operations() -> None:
    payload = _one_record_document().model_dump(mode="json")
    payload["citations"][0]["judgements"][Question.IDENTITY.value] = {
        "outcome": "confirmed",
        "node_id": "missing-judgement",
        "message": None,
    }
    with pytest.raises(ValueError, match="state disagrees with operation history"):
        Document.model_validate(payload)


def test_current_checkpoint_rejects_an_old_schema() -> None:
    payload = _one_record_document().model_dump(mode="json")
    payload["schema_version"] -= 1
    with pytest.raises(ValueError, match="schema"):
        Document.model_validate(payload)
