"""Behavioral checks for the public, composable extraction stages."""

import asyncio

import pytest

from mellea_lrc.extraction import (
    find_docket_locators,
    find_full_reporter_locators,
    form_roots,
    grow_roots,
    resolve_case_names,
    resolve_colocations,
    resolve_courts,
    resolve_dates,
    resolve_pin_cites,
    stable,
    start_extraction,
)
from mellea_lrc.model import (
    CitationField,
    Document,
    FullCitation,
    FullCitationKind,
    FullDocketCitation,
    FullReporterCitation,
    latest,
)
from mellea_lrc.preprocessing import preprocess


def _read_span(document: Document, citation_index: int) -> str:
    citation = document.citations[citation_index]
    span = latest(citation.locator_span)
    assert span is not None
    return document.text[span.start : span.end]


def _field_logs(citation: FullCitation) -> dict[str, tuple]:
    return {
        name: getattr(citation, name)
        for name in type(citation).model_fields
        if name not in {"id", "kind", "nodes"}
    }


def test_locator_stages_preserve_exact_occurrences_and_leave_fields_unread() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. Jan. 1, 2024)."
    empty = start_extraction(preprocess(text))

    reporter_only = find_full_reporter_locators(empty)
    assert len(reporter_only.citations) == 1
    assert reporter_only.citations[0].kind == "reporter"
    assert _read_span(reporter_only, 0) == "2024 WL 1234567"
    assert reporter_only.citations[0].case_name == ()
    assert reporter_only.citations[0].court == ()
    assert reporter_only.citations[0].date == ()
    assert reporter_only.colocations == ()

    both = find_docket_locators(reporter_only)
    assert len(both.citations) == 2
    assert [(citation.kind, latest(citation.locator_text)) for citation in both.citations] == [
        ("docket", "No. 1:24-cv-00123"),
        ("reporter", "2024 WL 1234567"),
    ]
    assert _read_span(both, 0) == "No. 1:24-cv-00123"
    assert isinstance(both.citations[0], FullDocketCitation)
    assert isinstance(both.citations[1], FullReporterCitation)
    assert latest(both.citations[0].docket_number) == "1:24-cv-00123"
    assert not hasattr(both.citations[1], "docket_number")
    assert both.colocations == ()
    assert both.citations[1] == reporter_only.citations[0]
    assert len(both.citations[0].nodes) == 2
    assert both.citations[0].nodes[0].stage == "docket_locators"
    assert all(
        update.node_id == both.citations[0].nodes[1].id
        for log in _field_logs(both.citations[0]).values()
        for update in log
    )
    assert "operations" not in Document.model_fields
    assert "nodes" not in Document.model_fields
    assert "updates" not in FullCitation.model_fields


def test_colocation_is_a_separate_grouping_stage() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. 2024)."
    document = start_extraction(preprocess(text))
    document = find_full_reporter_locators(document)
    document = find_docket_locators(document)
    grouped = resolve_colocations(document)

    assert len(grouped.colocations) == 1
    group = grouped.colocations[0]
    assert set(group.citation_ids) == {citation.id for citation in grouped.citations}
    assert all(latest(citation.colocation_id) == group.id for citation in grouped.citations)
    assert all(citation.case_name == () for citation in grouped.citations)
    assert all(citation.court == () for citation in grouped.citations)
    assert all(citation.date == () for citation in grouped.citations)
    assert document.colocations == ()
    reporter_only = find_full_reporter_locators(start_extraction(preprocess(text)))
    grouped_early = resolve_colocations(reporter_only)
    with pytest.raises(ValueError, match="before resolving colocations"):
        find_docket_locators(grouped_early)


def test_field_readers_run_after_colocation_and_keep_adjacent_cases_separate() -> None:
    text = (
        "Brown v. Board of Education, 347 U.S. 483, 495 (1954). "
        "Smith v. Jones, No. 1:24-cv-00123 (D. Ariz. Jan. 1, 2024)."
    )
    document = start_extraction(preprocess(text))
    document = find_full_reporter_locators(document)
    document = find_docket_locators(document)
    document = resolve_colocations(document)
    document = resolve_case_names(document)
    document = resolve_courts(document)
    document = resolve_dates(document)
    document = resolve_pin_cites(document)

    reporter, docket = document.citations
    assert "Brown" in str(latest(reporter.case_name))
    assert "Smith" not in str(latest(reporter.case_name))
    assert "Smith" in str(latest(docket.case_name))
    assert "Brown" not in str(latest(docket.case_name))
    assert "1954" in str(latest(reporter.date))
    assert "2024" in str(latest(docket.date))
    assert latest(reporter.pin_cite) == "495"
    assert latest(reporter.court) is not None
    assert latest(docket.court) is not None
    reporter_name_span = latest(reporter.case_name_span)
    docket_court_span = latest(docket.court_span)
    docket_date_span = latest(docket.date_span)
    reporter_pin_span = latest(reporter.pin_cite_span)
    assert reporter_name_span and docket_court_span and docket_date_span and reporter_pin_span
    assert document.text[reporter_name_span.start : reporter_name_span.end] == latest(reporter.case_name)
    assert document.text[docket_court_span.start : docket_court_span.end] == "D. Ariz."
    assert document.text[docket_date_span.start : docket_date_span.end] == "Jan. 1, 2024"
    assert document.text[reporter_pin_span.start : reporter_pin_span.end] == "495"


def test_repeated_reporter_occurrences_share_a_root_without_losing_spans() -> None:
    text = "See 556 U.S. 662. Later, 556 U.S. 662."
    document = start_extraction(preprocess(text))
    document = find_full_reporter_locators(document)
    document = resolve_colocations(document)
    document = form_roots(document)

    assert len(document.citations) == 2
    first, second = document.citations
    assert first.id != second.id
    assert latest(first.locator_span) != latest(second.locator_span)
    assert latest(first.root_id) == first.id
    assert latest(second.root_id) == first.id
    assert [_read_span(document, index) for index in range(2)] == ["556 U.S. 662"] * 2


def test_courtless_docket_occurrences_are_not_deduplicated_by_number_alone() -> None:
    text = "See Case No. 1:24-cv-00123. Later Case   No.   1:24-cv-00123."
    document = start_extraction(preprocess(text))
    document = find_docket_locators(document)
    document = resolve_colocations(document)
    document = form_roots(document)

    assert len(document.citations) == 2
    assert _read_span(document, 1) == "Case   No.   1:24-cv-00123"
    assert all(latest(citation.court) is None for citation in document.citations)
    assert all(latest(citation.root_id) == citation.id for citation in document.citations)


def test_adjacent_docket_entry_is_read_without_making_it_a_case_number() -> None:
    document = find_docket_locators(Document.from_source("Doc. 10-1, Case No. 1:24-cv-00123."))

    assert len(document.citations) == 1
    citation = document.citations[0]
    assert latest(citation.docket_number) == "1:24-cv-00123"
    assert latest(citation.docket_entry) == "10-1"
    entry_span = latest(citation.docket_entry_span)
    assert entry_span is not None
    assert document.text[entry_span.start : entry_span.end] == "Doc. 10-1"


def test_document_json_roundtrip_recovers_every_field_log() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. 2024)."
    document = asyncio.run(grow_roots(start_extraction(preprocess(text)), rules=stable()))
    restored = Document.model_validate_json(document.model_dump_json())

    assert restored == document
    assert isinstance(restored.citations[0], FullDocketCitation)
    assert isinstance(restored.citations[1], FullReporterCitation)
    assert restored.text == text
    assert all(citation.nodes and latest(citation.locator_text) for citation in restored.citations)
    assert restored.completed_stages
    assert len(restored.citations) == 2
    assert len(restored.colocations) == 1
    assert all(latest(citation.root_id) is not None for citation in restored.citations)

    changed = restored.model_dump(mode="json")
    changed["citations"][0]["locator_text"][-1]["value"] = "a different locator"
    revised = Document.model_validate(changed)
    assert latest(revised.citations[0].locator_text) == "a different locator"
    assert revised.citations[0].locator_text[:-1] == restored.citations[0].locator_text[:-1]


def test_locator_fields_belong_only_to_concrete_full_citations() -> None:
    assert "locator_span" not in FullCitation.model_fields
    assert "locator_text" not in FullCitation.model_fields
    for citation_type in (FullReporterCitation, FullDocketCitation):
        assert "locator_span" in citation_type.model_fields
        assert "locator_text" in citation_type.model_fields


def test_single_citation_field_logs_are_complete_checkpoints() -> None:
    document = Document.from_source("Case No. 1:24-cv-00123.")
    citation_id = "docket:0:23"
    document = document.create_citation("docket_locators", citation_id, FullCitationKind.DOCKET)
    citation = document.citations[0]
    assert len(citation.nodes) == 1
    assert citation.nodes[0].stage == "docket_locators"
    assert all(log == () for log in _field_logs(citation).values())
    assert Document.model_validate_json(document.model_dump_json()) == document

    document = document.update_fields(
        "docket_locators",
        citation_id,
        {
            CitationField.LOCATOR_TEXT: "Case No. 1:24-cv-00123",
            CitationField.DOCKET_NUMBER: "1:24-cv-00123",
            CitationField.COURT: None,
        },
    )
    citation = document.citations[0]
    assert len(citation.nodes) == 2
    assert citation.nodes[0].stage == citation.nodes[1].stage
    assert latest(citation.locator_text) == "Case No. 1:24-cv-00123"
    assert latest(citation.docket_number) == "1:24-cv-00123"
    assert citation.court[0].value is None
    assert latest(citation.court) is None
    assert citation.case_name == ()
    assert {log[-1].node_id for log in (citation.locator_text, citation.docket_number, citation.court)} == {
        citation.nodes[-1].id
    }
    assert Document.model_validate_json(document.model_dump_json()) == document

    changed = document.model_dump(mode="json")
    changed["citations"][0]["nodes"].pop()
    with pytest.raises(ValueError, match="missing or creation node"):
        Document.model_validate(changed)


def test_field_logs_reject_tampered_node_pointers_and_out_of_order_history() -> None:
    document = Document.from_source("Case No. 1:24-cv-00123.")
    document = document.create_citation("locator", "docket:0", FullCitationKind.DOCKET)
    document = document.update_fields("first", "docket:0", {CitationField.COURT: "D. Ariz."})
    document = document.update_fields("second", "docket:0", {CitationField.COURT: "S.D.N.Y."})

    missing = document.model_dump(mode="json")
    missing["citations"][0]["court"][-1]["node_id"] = "missing"
    with pytest.raises(ValueError, match="missing or creation node"):
        Document.model_validate(missing)

    reversed_history = document.model_dump(mode="json")
    reversed_history["citations"][0]["court"][0]["node_id"] = document.citations[0].nodes[-1].id
    reversed_history["citations"][0]["court"][1]["node_id"] = document.citations[0].nodes[-2].id
    with pytest.raises(ValueError, match="out of order"):
        Document.model_validate(reversed_history)


def test_citation_updates_cannot_cross_concrete_locator_types() -> None:
    document = find_docket_locators(Document.from_source("Case No. 1:24-cv-00123."))
    citation = document.citations[0]
    assert isinstance(citation, FullDocketCitation)

    with pytest.raises(ValueError, match="has no reporter field"):
        document.update_fields("review", citation.id, {CitationField.REPORTER: "U.S."})

    assert document.citations[0] == citation


def test_every_checkpoint_and_later_withdrawal_roundtrips_without_erasing_history() -> None:
    document = Document.from_source("Brown v. Board, 347 U.S. 483 (1954).")
    stages = (
        find_full_reporter_locators,
        find_docket_locators,
        resolve_colocations,
        resolve_case_names,
        resolve_courts,
        resolve_dates,
        resolve_pin_cites,
        form_roots,
    )
    for stage in stages:
        previous = {citation.id: citation for citation in document.citations}
        document = stage(document)
        assert Document.model_validate_json(document.model_dump_json()) == document
        for citation in document.citations:
            prior = previous.get(citation.id)
            if prior is None:
                continue
            assert citation.nodes[: len(prior.nodes)] == prior.nodes
            for name, old_log in _field_logs(prior).items():
                assert getattr(citation, name)[: len(old_log)] == old_log

    citation = document.citations[0]
    withdrawn = document.withdraw_citation("review", citation.id)
    withdrawn_citation = withdrawn.citations[0]
    assert latest(withdrawn_citation.root_id) == "__withdrawn__"
    assert withdrawn_citation.nodes[: len(citation.nodes)] == citation.nodes
    assert withdrawn_citation.root_id[:-1] == citation.root_id
    assert withdrawn_citation.root_id[-1].node_id == withdrawn_citation.nodes[-1].id
    for name, old_log in _field_logs(citation).items():
        assert getattr(withdrawn_citation, name)[: len(old_log)] == old_log
    assert Document.model_validate_json(withdrawn.model_dump_json()) == withdrawn

    restored = withdrawn.update_fields("review", citation.id, {CitationField.ROOT_ID: citation.id})
    restored_citation = restored.citations[0]
    assert latest(restored_citation.root_id) == citation.id
    assert restored_citation.root_id[:-1] == withdrawn_citation.root_id
    assert len(restored_citation.nodes) == len(withdrawn_citation.nodes) + 1
    assert Document.model_validate_json(restored.model_dump_json()) == restored

    stale_root = restored.model_dump(mode="json")
    stale_root["citations"][0]["root_id"][-1]["value"] = "missing-citation"
    with pytest.raises(ValueError, match="unknown root"):
        Document.model_validate(stale_root)
