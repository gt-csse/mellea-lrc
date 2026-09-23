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
from mellea_lrc.model import CitationField, Document
from mellea_lrc.preprocessing import preprocess


def _read_span(document: Document, citation_index: int) -> str:
    citation = document.citations[citation_index]
    return document.text[citation.locator_span.start : citation.locator_span.end]


def test_locator_stages_preserve_exact_occurrences_and_leave_fields_unread() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. Jan. 1, 2024)."
    empty = start_extraction(preprocess(text))

    reporter_only = find_full_reporter_locators(empty)
    assert len(reporter_only.citations) == 1
    assert reporter_only.citations[0].kind == "reporter"
    assert _read_span(reporter_only, 0) == "2024 WL 1234567"
    assert reporter_only.citations[0].case_name is None
    assert reporter_only.citations[0].court is None
    assert reporter_only.citations[0].date is None
    assert reporter_only.colocations == ()

    both = find_docket_locators(reporter_only)
    assert len(both.citations) == 2
    assert [(citation.kind, citation.locator_text) for citation in both.citations] == [
        ("docket", "No. 1:24-cv-00123"),
        ("reporter", "2024 WL 1234567"),
    ]
    assert _read_span(both, 0) == "No. 1:24-cv-00123"
    assert [citation.docket_number for citation in both.citations] == ["1:24-cv-00123", None]
    assert both.colocations == ()
    assert both.operations[: len(reporter_only.operations)] == reporter_only.operations


def test_colocation_is_a_separate_grouping_stage() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. 2024)."
    document = start_extraction(preprocess(text))
    document = find_full_reporter_locators(document)
    document = find_docket_locators(document)
    grouped = resolve_colocations(document)

    assert len(grouped.colocations) == 1
    group = grouped.colocations[0]
    assert set(group.citation_ids) == {citation.id for citation in grouped.citations}
    assert all(citation.colocation_id == group.id for citation in grouped.citations)
    assert all(citation.case_name is None for citation in grouped.citations)
    assert all(citation.court is None for citation in grouped.citations)
    assert all(citation.date is None for citation in grouped.citations)
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
    assert "Brown" in str(reporter.case_name)
    assert "Smith" not in str(reporter.case_name)
    assert "Smith" in str(docket.case_name)
    assert "Brown" not in str(docket.case_name)
    assert "1954" in str(reporter.date)
    assert "2024" in str(docket.date)
    assert reporter.pin_cite == "495"
    assert reporter.court is not None
    assert docket.court is not None
    assert document.text[reporter.case_name_span.start : reporter.case_name_span.end] == reporter.case_name
    assert document.text[docket.court_span.start : docket.court_span.end] == "D. Ariz."
    assert document.text[docket.date_span.start : docket.date_span.end] == "Jan. 1, 2024"
    assert document.text[reporter.pin_cite_span.start : reporter.pin_cite_span.end] == "495"


def test_repeated_reporter_occurrences_share_a_root_without_losing_spans() -> None:
    text = "See 556 U.S. 662. Later, 556 U.S. 662."
    document = start_extraction(preprocess(text))
    document = find_full_reporter_locators(document)
    document = resolve_colocations(document)
    document = form_roots(document)

    assert len(document.citations) == 2
    first, second = document.citations
    assert first.id != second.id
    assert first.locator_span != second.locator_span
    assert first.root_id == first.id
    assert second.root_id == first.id
    assert [_read_span(document, index) for index in range(2)] == ["556 U.S. 662"] * 2


def test_courtless_docket_occurrences_are_not_deduplicated_by_number_alone() -> None:
    text = "See Case No. 1:24-cv-00123. Later Case   No.   1:24-cv-00123."
    document = start_extraction(preprocess(text))
    document = find_docket_locators(document)
    document = resolve_colocations(document)
    document = form_roots(document)

    assert len(document.citations) == 2
    assert _read_span(document, 1) == "Case   No.   1:24-cv-00123"
    assert all(citation.court is None for citation in document.citations)
    assert all(citation.root_id == citation.id for citation in document.citations)


def test_adjacent_docket_entry_is_read_without_making_it_a_case_number() -> None:
    document = find_docket_locators(Document.from_source("Doc. 10-1, Case No. 1:24-cv-00123."))

    assert len(document.citations) == 1
    citation = document.citations[0]
    assert citation.docket_number == "1:24-cv-00123"
    assert citation.docket_entry == "10-1"
    assert document.text[citation.docket_entry_span.start : citation.docket_entry_span.end] == "Doc. 10-1"


def test_document_json_roundtrip_recovers_every_stage_operation() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. 2024)."
    document = asyncio.run(grow_roots(start_extraction(preprocess(text)), rules=stable()))
    restored = Document.model_validate_json(document.model_dump_json())

    assert restored == document
    assert restored.text == text
    assert restored.operations
    assert restored.completed_stages
    assert len(restored.citations) == 2
    assert len(restored.colocations) == 1
    assert all(citation.root_id is not None for citation in restored.citations)

    changed = restored.model_dump(mode="json")
    changed["citations"][0]["locator_text"] = "a different locator"
    with pytest.raises(ValueError, match="disagrees with extraction operations"):
        Document.model_validate(changed)


def test_every_checkpoint_and_later_withdrawal_replays_without_erasing_history() -> None:
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
        previous = document.operations
        document = stage(document)
        assert Document.model_validate_json(document.model_dump_json()) == document
        assert document.operations[: len(previous)] == previous

    citation = document.citations[0]
    withdrawn = document.withdraw_citation("review", citation.id)
    assert withdrawn.citations[0].root_id == "__withdrawn__"
    assert withdrawn.operations[: len(document.operations)] == document.operations
    assert Document.model_validate_json(withdrawn.model_dump_json()) == withdrawn

    restored = withdrawn.update_fields("review", citation.id, {CitationField.ROOT_ID: citation.id})
    assert restored.citations[0].root_id == citation.id
    assert len(restored.operations) == len(withdrawn.operations) + 1
    assert Document.model_validate_json(restored.model_dump_json()) == restored
