"""Citation creation, field updates, and the evidence behind each change."""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import TypeAdapter

from mellea_lrc.model.case_names import CaseName
from mellea_lrc.model.citations import (
    CanonicalCitation,
    CitationField,
    CitationKind,
    FullCaseCitation,
    empty_citation,
    placed,
)
from mellea_lrc.model.operations import (
    assign_antecedent,
    assign_root,
    attribute_authority,
    create_citation,
    judge_citation,
    mark_extraction_reviewed,
    record_colocation,
    resolve_citation,
    update_field,
    update_fields,
    withdraw_citation,
)
from mellea_lrc.model.record import (
    WITHDRAWN_HEAD_ID,
    CitationOperation,
    CitationRecord,
    FieldUpdate,
    Node,
    OperationKind,
    Question,
    Reads,
    Resolution,
)
from mellea_lrc.model.spans import Span

READ = CaseName(span=Span(start=0, end=6), text="Hassan", defendant="Hassan")
FULLER = CaseName(
    span=Span(start=0, end=23),
    text="United States v. Hassan",
    plaintiff="United States",
    defendant="Hassan",
)


def _record() -> CitationRecord:
    return CitationRecord(
        citation_id="c1",
        fields=placed(
            FullCaseCitation(volume="742", reporter="F.3d", page="104"),
            span=Span(start=0, end=30),
            locator_span=Span(start=9, end=21),
            matched_text="742 F.3d 104",
            case_name=READ,
        ),
    )


def _node(**extra: object) -> Node:
    return Node(
        node_id="case_name:0-6",
        reads=Reads.DOCUMENT,
        stage="case_name",
        made_by="adjudicate_case_name",
        outcome="names_a_citation",
        **extra,
    )


def test_create_citation_only_selects_kind_and_records_its_node() -> None:
    node = _node()
    record = create_citation("c2", CitationKind.FULL_CASE, node)

    assert record.kind is CitationKind.FULL_CASE
    assert record.fields == FullCaseCitation()
    assert record.created_by == node.node_id
    assert record.trace == (node,)
    assert record.field_updates == ()


def test_an_initial_reading_uses_the_same_update_as_a_later_revision() -> None:
    record = create_citation("c2", CitationKind.FULL_CASE, _node())
    locator_node = Node("locator", Reads.DOCUMENT, "extraction", "eyecite", "found")
    update_fields(
        record,
        locator_node,
        {
            CitationField.SPAN: Span(0, 30),
            CitationField.LOCATOR_SPAN: Span(9, 21),
            CitationField.MATCHED_TEXT: "742 F.3d 104",
        },
        reason="read the first locator",
    )
    assert record.fields.matched_text == "742 F.3d 104"
    assert record.field_updates[0].before is None
    assert {item.node_id for item in record.field_updates} == {"locator"}

    update_field(record, _node(), CitationField.CASE_NAME, FULLER, reason="named in the sentence")
    assert record.fields.case_name == FULLER
    assert record.field_updates[-1].before is None


def test_a_field_revision_records_after_and_evidence_pointer() -> None:
    record = _record()
    node = _node()
    update_field(record, node, CitationField.CASE_NAME, FULLER, reason="named in the sentence")

    assert record.fields.case_name == FULLER
    assert [(u.field, u.after) for u in record.field_updates] == [(CitationField.CASE_NAME, FULLER)]
    update = record.operations[-1]
    assert not hasattr(update, "before")
    assert update.reason == "named in the sentence"
    assert update.node_id == node.node_id
    assert next(item for item in record.trace if item.node_id == update.node_id) is node


def test_case_name_property_uses_generic_get_field_for_latest_update() -> None:
    record = create_citation(
        "c2",
        CitationKind.FULL_CASE,
        Node("create-c2", Reads.DOCUMENT, "extraction", "test", "classified"),
    )
    initial = Node("initial-name", Reads.DOCUMENT, "extraction", "test", "read")
    correction = Node("corrected-name", Reads.DOCUMENT, "validation", "test", "corrected")
    update_field(record, initial, CitationField.CASE_NAME, READ, reason="initial case-name reading")
    update_field(record, correction, CitationField.CASE_NAME, FULLER, reason="corrected case-name reading")

    assert record.get_field(CitationField.CASE_NAME) == FULLER
    assert record.case_name == FULLER
    assert record.fields.case_name == FULLER


def test_one_node_can_update_multiple_fields_atomically() -> None:
    record = _record()
    node = _node()
    assert (
        update_fields(
            record,
            node,
            {CitationField.CASE_NAME: FULLER, CitationField.COURT: "ca7"},
            reason="reviewed the citation",
        )
        is record
    )
    assert record.fields.case_name == FULLER
    assert record.fields.court == "ca7"
    assert [(u.field, u.node_id) for u in record.field_updates] == [
        (CitationField.CASE_NAME, node.node_id),
        (CitationField.COURT, node.node_id),
    ]
    assert record.trace == (node,)

    unchanged = record.fields
    prior_updates = record.field_updates
    prior_trace = record.trace
    invalid_node = Node("invalid", Reads.DOCUMENT, "case_name", "review", "invalid")
    with pytest.raises(ValueError, match="has no field"):
        update_fields(
            record,
            invalid_node,
            {CitationField.CASE_NAME: READ, CitationField.PUBLISHER: "x"},
            reason="invalid batch",
        )
    assert record.fields is unchanged
    assert record.field_updates == prior_updates
    assert record.trace == prior_trace


def test_a_node_that_read_a_record_cannot_update_what_the_filing_states() -> None:
    record = _record()
    with pytest.raises(ValueError, match="cannot update filing fields"):
        update_field(
            record,
            Node(
                node_id="lookup",
                reads=Reads.RECORD,
                stage="identity",
                made_by="courtlistener",
                outcome="found",
            ),
            CitationField.CASE_NAME,
            FULLER,
            reason="the archive says so",
        )
    assert record.fields.case_name == READ
    assert record.field_updates == ()


def test_a_field_update_that_changes_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="must change the value"):
        FieldUpdate(field=CitationField.CASE_NAME, before=READ, after=READ, reason="why", node_id="n")


def test_a_root_is_what_extraction_read_and_an_authority_is_what_a_lookup_found() -> None:
    record = _record()
    record = assign_root(record, "c1", Node("root", Reads.RECORD, "root", "rule", "root_created"))
    assert record.authority == "c1"
    attribute_authority(record, Node("authority", Reads.RECORD, "identity", "archive", "found"), "c9")
    assert record.authority == "c9"
    assert record.root_id == "c1"


def test_model_reparse_provenance_points_to_its_review_node() -> None:
    record = _record()
    node = Node("review", Reads.DOCUMENT, "extraction_review", "model", "reviewed")
    mark_extraction_reviewed(record, node)
    assert record.extraction_reviewed_by_llm
    assert record.extraction_review_node_id == node.node_id
    assert node in record.trace


def test_every_citation_mutation_has_an_ordered_replayable_operation() -> None:
    created = Node("create", Reads.DOCUMENT, "extraction", "rule", "classified")
    record = create_citation("c2", CitationKind.FULL_CASE, created)
    reading = Node("reading", Reads.DOCUMENT, "extraction", "rule", "read")
    update_fields(
        record,
        reading,
        {
            CitationField.SPAN: Span(0, 30),
            CitationField.LOCATOR_SPAN: Span(9, 21),
            CitationField.MATCHED_TEXT: "742 F.3d 104",
        },
        reason="first reading",
    )
    # The same reading establishes three fields and the first root link.
    record = assign_root(record, "root-1", reading, resolves_to="root-1")
    assert record.trace == (created, reading)

    rereading = Node("rereading", Reads.DOCUMENT, "extraction", "model", "corrected")
    update_field(record, rereading, CitationField.MATCHED_TEXT, "742 F.3d 105", reason="corrected")
    record = assign_root(record, "root-2", rereading, resolves_to="root-2")
    record = assign_antecedent(record, "root-3", Node("antecedent", Reads.RECORD, "root", "rule", "linked"))
    record = record_colocation(
        record, "group-1", Node("group-1", Reads.RECORD, "colocation", "rule", "grouped")
    )
    record = record_colocation(record, None, Node("ungroup", Reads.RECORD, "colocation", "rule", "ungrouped"))

    first_lookup = Node("lookup-1", Reads.RECORD, "identity", "archive", "candidate")
    first_resolution = Resolution("cluster-1", "Case One", None, None, first_lookup.node_id)
    resolve_citation(record, first_lookup, first_resolution)
    judge_citation(record, first_lookup, Question.IDENTITY, "candidate")
    attribute_authority(record, first_lookup, "authority-1")
    second_lookup = Node("lookup-2", Reads.RECORD, "identity", "archive", "confirmed")
    second_resolution = Resolution("cluster-2", "Case Two", None, None, second_lookup.node_id)
    resolve_citation(record, second_lookup, second_resolution)
    judge_citation(record, second_lookup, Question.IDENTITY, "confirmed")
    attribute_authority(record, second_lookup, "authority-2")

    mark_extraction_reviewed(
        record, Node("review-1", Reads.DOCUMENT, "extraction_review", "model", "reviewed")
    )
    mark_extraction_reviewed(
        record, Node("review-2", Reads.DOCUMENT, "extraction_review", "model", "reviewed")
    )
    withdraw_citation(record, Node("withdraw-1", Reads.RECORD, "identity", "rule", "withdrawn"))
    withdraw_citation(record, Node("withdraw-2", Reads.RECORD, "identity", "rule", "withdrawn"))

    assert [operation.kind for operation in record.operations] == [
        OperationKind.CREATE,
        OperationKind.FIELD_UPDATE,
        OperationKind.FIELD_UPDATE,
        OperationKind.FIELD_UPDATE,
        OperationKind.ROOT_LINK,
        OperationKind.ANTECEDENT_LINK,
        OperationKind.FIELD_UPDATE,
        OperationKind.ROOT_LINK,
        OperationKind.ANTECEDENT_LINK,
        OperationKind.ANTECEDENT_LINK,
        OperationKind.COLOCATION_LINK,
        OperationKind.COLOCATION_LINK,
        OperationKind.RESOLUTION,
        OperationKind.JUDGEMENT,
        OperationKind.AUTHORITY,
        OperationKind.RESOLUTION,
        OperationKind.JUDGEMENT,
        OperationKind.AUTHORITY,
        OperationKind.EXTRACTION_REVIEW,
        OperationKind.EXTRACTION_REVIEW,
        OperationKind.ROOT_LINK,
        OperationKind.ROOT_LINK,
    ]
    assert [
        operation.after for operation in record.operations if operation.kind is OperationKind.ROOT_LINK
    ] == ["root-1", "root-2", WITHDRAWN_HEAD_ID, WITHDRAWN_HEAD_ID]
    assert all(not hasattr(operation, "before") for operation in record.operations)
    assert record.has_complete_history
    assert record.found == second_resolution
    assert record.judgement(Question.IDENTITY).outcome == "confirmed"
    assert record.authority_id == "authority-2"
    assert record.extraction_review_node_id == "review-2"
    assert record.withdrawn
    assert record.root_id == WITHDRAWN_HEAD_ID
    assert record.root_link_node_id == "withdraw-2"
    assert replace(record) == record  # Reconstruction validates every event against the projection.

    adapter = TypeAdapter(CitationRecord)
    payload = adapter.dump_python(record, mode="json")
    assert all("before" not in event for event in payload["operations"])
    assert adapter.validate_python(payload) == record


def test_every_citation_kind_uses_a_native_discriminator() -> None:
    adapter = TypeAdapter(CanonicalCitation)
    for kind in CitationKind:
        citation = empty_citation(kind)
        payload = adapter.dump_python(citation, mode="json")
        assert payload["kind"] == kind.value
        assert type(adapter.validate_python(payload)) is type(citation)


def test_event_payloads_restore_typed_values() -> None:
    record = create_citation(
        "c2", CitationKind.FULL_CASE, Node("create", Reads.DOCUMENT, "extraction", "rule", "classified")
    )
    update_field(
        record,
        Node("read", Reads.DOCUMENT, "name", "model", "found"),
        CitationField.CASE_NAME,
        FULLER,
        reason="case name",
    )
    adapter = TypeAdapter(CitationOperation)
    payload = adapter.dump_python(record.operations[-1], mode="json")
    restored = adapter.validate_python(payload)
    assert restored.after == FULLER
    assert type(restored.after) is CaseName


@pytest.mark.parametrize(
    ("kind", "wrong_reads", "expected"),
    [
        (OperationKind.CREATE, Reads.RECORD, "create requires document evidence"),
        (OperationKind.FIELD_UPDATE, Reads.RECORD, "field_update requires document evidence"),
        (OperationKind.COLOCATION_LINK, Reads.DOCUMENT, "colocation_link requires record evidence"),
        (OperationKind.RESOLUTION, Reads.DOCUMENT, "resolution requires record evidence"),
        (OperationKind.AUTHORITY, Reads.DOCUMENT, "authority requires record evidence"),
        (OperationKind.EXTRACTION_REVIEW, Reads.RECORD, "extraction_review requires document evidence"),
    ],
)
def test_native_reload_rejects_wrong_evidence_source(
    kind: OperationKind, wrong_reads: Reads, expected: str
) -> None:
    record = create_citation(
        "c2",
        CitationKind.FULL_CASE,
        Node("create", Reads.DOCUMENT, "extraction", "rule", "classified"),
    )
    update_field(
        record,
        Node("field", Reads.DOCUMENT, "extraction", "rule", "read"),
        CitationField.MATCHED_TEXT,
        "742 F.3d 104",
        reason="read",
    )
    record = record_colocation(
        record, "group-1", Node("colocation", Reads.RECORD, "structure", "rule", "grouped")
    )
    resolve_citation(
        record,
        Node("resolution", Reads.RECORD, "identity", "archive", "found"),
        Resolution("cluster-1", "Case One", None, None, "resolution"),
    )
    attribute_authority(record, Node("authority", Reads.RECORD, "identity", "rule", "attributed"), "c2")
    mark_extraction_reviewed(record, Node("review", Reads.DOCUMENT, "extraction_review", "model", "reviewed"))

    payload = TypeAdapter(CitationRecord).dump_python(record, mode="json")
    node_id = next(event["node_id"] for event in payload["operations"] if event["kind"] == kind.value)
    next(node for node in payload["trace"] if node["node_id"] == node_id)["reads"] = wrong_reads.value
    with pytest.raises(ValueError, match=expected):
        TypeAdapter(CitationRecord).validate_python(payload)


def test_native_reload_rejects_effect_pointing_to_another_evidence_node() -> None:
    record = create_citation(
        "c2",
        CitationKind.FULL_CASE,
        Node("create", Reads.DOCUMENT, "extraction", "rule", "classified"),
    )
    resolution_node = Node("resolution", Reads.RECORD, "identity", "archive", "found")
    resolve_citation(record, resolution_node, Resolution("cluster-1", "Case One", None, None, "resolution"))
    payload = TypeAdapter(CitationRecord).dump_python(record, mode="json")
    payload["operations"][-1]["after"]["node_id"] = "create"
    payload["found"]["node_id"] = "create"
    with pytest.raises(ValueError, match="resolution names another evidence node"):
        TypeAdapter(CitationRecord).validate_python(payload)


def test_replay_rejects_tampered_event_and_projection() -> None:
    record = create_citation(
        "c2", CitationKind.FULL_CASE, Node("create", Reads.DOCUMENT, "extraction", "rule", "classified")
    )
    first = Node("first", Reads.DOCUMENT, "extraction", "rule", "read")
    second = Node("second", Reads.DOCUMENT, "extraction", "rule", "corrected")
    update_field(record, first, CitationField.MATCHED_TEXT, "742 F.3d 104", reason="first")
    update_field(record, second, CitationField.MATCHED_TEXT, "742 F.3d 105", reason="second")

    operations = list(record.operations)
    operations[-1] = replace(operations[-1], after="742 F.3d 104")
    with pytest.raises(ValueError, match="inconsistent operation history"):
        replace(record, operations=tuple(operations))
    with pytest.raises(ValueError, match="state disagrees with operation history"):
        replace(record, fields=replace(record.fields, matched_text="silently changed"))
