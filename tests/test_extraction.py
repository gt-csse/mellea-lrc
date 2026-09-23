"""Behavioral checks for extraction and citation-local field histories."""

import pytest

from mellea_lrc.extraction import (
    extract,
    find_docket_locators,
    find_full_reporter_locators,
    form_roots,
    grow_roots,
    resolve_case_names,
    resolve_colocations,
    resolve_courts,
    resolve_dates,
    resolve_pin_cites,
)
from mellea_lrc.model import (
    CitationDate,
    Document,
    FullCitation,
    FullDocketCitation,
    FullReporterCitation,
    Span,
    latest,
)
from mellea_lrc.preprocessing import preprocess


def _read_span(document: Document, citation_index: int) -> str:
    span = document.citations[citation_index].locator_span
    return document.text[span.start : span.end]


def _field_logs(citation: FullCitation) -> dict[str, tuple]:
    return {
        name: getattr(citation, name)
        for name in type(citation).model_fields
        if name not in {"id", "kind", "nodes"}
    }


def _assert_roundtrip(document: Document) -> None:
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_locator_stages_preserve_exact_occurrences_and_create_one_node() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. Jan. 1, 2024)."
    empty = Document.from_preprocessed(preprocess(text))

    reporter_only = find_full_reporter_locators(empty)
    assert len(reporter_only.citations) == 1
    reporter = reporter_only.citations[0]
    assert isinstance(reporter, FullReporterCitation)
    assert _read_span(reporter_only, 0) == "2024 WL 1234567"
    assert reporter.case_name == reporter.court == reporter.date == ()
    assert reporter_only.colocations == ()
    assert len(reporter.nodes) == 1
    assert reporter.nodes[0].stage == "full_reporter_locators"
    assert all(
        update.node_id == reporter.nodes[0].id for log in _field_logs(reporter).values() for update in log
    )

    both = find_docket_locators(reporter_only)
    assert len(both.citations) == 2
    docket, reporter = both.citations
    assert isinstance(docket, FullDocketCitation)
    assert isinstance(reporter, FullReporterCitation)
    assert [(citation.kind, latest(citation.locator_text)) for citation in both.citations] == [
        ("docket", "No. 1:24-cv-00123"),
        ("reporter", "2024 WL 1234567"),
    ]
    assert _read_span(both, 0) == "No. 1:24-cv-00123"
    assert latest(docket.docket_number) == "1:24-cv-00123"
    assert not hasattr(reporter, "docket_number")
    assert both.colocations == ()
    assert reporter == reporter_only.citations[0]
    assert len(docket.nodes) == 1
    assert all(update.node_id == docket.nodes[0].id for log in _field_logs(docket).values() for update in log)
    assert "updates" not in FullCitation.model_fields
    assert "operations" not in Document.model_fields
    _assert_roundtrip(both)


def test_colocation_is_derived_from_citation_assignments() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. 2024)."
    document = find_docket_locators(find_full_reporter_locators(Document.from_source(text)))
    grouped = resolve_colocations(document)

    assert len(grouped.colocations) == 1
    group = grouped.colocations[0]
    assert set(group.citation_ids) == {citation.id for citation in grouped.citations}
    assert all(latest(citation.colocation_id) == group.id for citation in grouped.citations)
    assert all(citation.case_name == citation.court == citation.date == () for citation in grouped.citations)
    assert document.colocations == ()
    assert "colocations" not in Document.model_fields
    _assert_roundtrip(grouped)

    reporter_only = find_full_reporter_locators(Document.from_source(text))
    grouped_early = resolve_colocations(reporter_only)
    with pytest.raises(ValueError, match="before resolving colocations"):
        find_docket_locators(grouped_early)


def test_field_readers_keep_adjacent_cases_and_source_spans_separate() -> None:
    text = (
        "Brown v. Board of Education, 347 U.S. 483, 495 (1954). "
        "Smith v. Jones, No. 1:24-cv-00123 (D. Ariz. Jan. 1, 2024)."
    )
    document = Document.from_source(text)
    for stage in (
        find_full_reporter_locators,
        find_docket_locators,
        resolve_colocations,
        resolve_case_names,
        resolve_courts,
        resolve_dates,
        resolve_pin_cites,
    ):
        document = stage(document)

    reporter, docket = document.citations
    assert "Brown" in str(latest(reporter.case_name))
    assert "Smith" not in str(latest(reporter.case_name))
    assert "Smith" in str(latest(docket.case_name))
    assert "Brown" not in str(latest(docket.case_name))
    assert latest(reporter.date).year == 1954
    assert latest(docket.date).year == 2024
    assert latest(reporter.pin_cite) == "495"
    assert latest(reporter.court) is not None
    assert latest(docket.court) is not None

    name = reporter.case_name[-1]
    court = docket.court[-1]
    date = docket.date[-1]
    pin = reporter.pin_cite[-1]
    assert all(update.span is not None for update in (name, court, date, pin))
    assert text[name.span.start : name.span.end] == name.value
    assert text[court.span.start : court.span.end] == "D. Ariz."
    assert text[date.span.start : date.span.end] == "Jan. 1, 2024"
    assert text[pin.span.start : pin.span.end] == pin.value
    _assert_roundtrip(document)


def test_repeated_reporter_occurrences_share_a_root_without_losing_spans() -> None:
    document = form_roots(
        resolve_colocations(
            find_full_reporter_locators(Document.from_source("See 556 U.S. 662. Later, 556 U.S. 662."))
        )
    )

    assert len(document.citations) == 2
    first, second = document.citations
    assert first.id != second.id
    assert first.locator_span != second.locator_span
    assert latest(first.root_id) == first.id
    assert latest(second.root_id) == first.id
    assert [_read_span(document, index) for index in range(2)] == ["556 U.S. 662"] * 2


def test_courtless_docket_occurrences_are_not_deduplicated_by_number_alone() -> None:
    document = form_roots(
        resolve_colocations(
            find_docket_locators(
                Document.from_source("See Case No. 1:24-cv-00123. Later Case   No.   1:24-cv-00123.")
            )
        )
    )

    assert len(document.citations) == 2
    assert _read_span(document, 1) == "Case   No.   1:24-cv-00123"
    assert all(latest(citation.court) is None for citation in document.citations)
    assert all(latest(citation.root_id) == citation.id for citation in document.citations)


def test_adjacent_docket_entry_has_its_own_evidence_span() -> None:
    document = find_docket_locators(Document.from_source("Doc. 10-1, Case No. 1:24-cv-00123."))

    assert len(document.citations) == 1
    citation = document.citations[0]
    assert isinstance(citation, FullDocketCitation)
    assert latest(citation.docket_number) == "1:24-cv-00123"
    assert latest(citation.docket_entry) == "10-1"
    entry = citation.docket_entry[-1]
    assert entry.span is not None
    assert document.text[entry.span.start : entry.span.end] == "Doc. 10-1"


def test_synchronous_pipeline_and_json_roundtrip() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. 2024)."
    document = grow_roots(Document.from_preprocessed(preprocess(text)))
    _assert_roundtrip(document)

    assert isinstance(document.citations[0], FullDocketCitation)
    assert isinstance(document.citations[1], FullReporterCitation)
    assert document.text == text
    assert all(citation.nodes and latest(citation.locator_text) for citation in document.citations)
    assert document.completed_stages
    assert len(document.citations) == 2
    assert len(document.colocations) == 1
    assert all(latest(citation.root_id) is not None for citation in document.citations)
    assert extract(text) == document


def test_locator_fields_belong_only_to_concrete_full_citations() -> None:
    assert "locator_span" not in FullCitation.model_fields
    assert "locator_text" not in FullCitation.model_fields
    for citation_type in (FullReporterCitation, FullDocketCitation):
        assert "locator_span" not in citation_type.model_fields
        assert "locator_text" in citation_type.model_fields
        assert isinstance(citation_type.locator_span, property)

    document = find_docket_locators(Document.from_source("Case No. 1:24-cv-00123."))
    assert not hasattr(document.citations[0], "reporter")
    invalid = document.model_dump(mode="json")
    invalid["citations"][0]["reporter"] = []
    with pytest.raises(ValueError):
        Document.model_validate(invalid)


def test_explicit_methods_append_a_traceable_history() -> None:
    source = "Smith v. Jones, Case No. 1:24-cv-00123 (D. Ariz. 2024)."
    document = find_docket_locators(Document.from_source(source))
    citation = document.citations[0]
    name = "Smith v. Jones"
    date_text = "2024"
    court_text = "D. Ariz."

    named = citation.record("case_names").with_case_name(source, name, Span(0, len(name)))
    courted = named.record("courts").with_court(
        source, "d-ariz", Span(source.index(court_text), source.index(court_text) + len(court_text))
    )
    dated = courted.record("dates").with_date(
        source, CitationDate(year=2024), Span(source.index(date_text), source.index(date_text) + 4)
    )
    rooted = dated.record("roots").with_root(citation.id)
    document = document.replace_citation(rooted)

    assert [node.stage for node in rooted.nodes] == [
        "docket_locators",
        "case_names",
        "courts",
        "dates",
        "roots",
    ]
    assert named.nodes == rooted.nodes[:2]
    assert courted.nodes == rooted.nodes[:3]
    assert dated.nodes == rooted.nodes[:4]
    assert rooted.case_name[-1].node_id == rooted.nodes[1].id
    assert rooted.court[-1].node_id == rooted.nodes[2].id
    assert rooted.date[-1].node_id == rooted.nodes[3].id
    assert rooted.root_id[-1].node_id == rooted.nodes[4].id
    assert rooted.case_name[-1].span == Span(0, len(name))
    assert latest(rooted.root_id) == citation.id
    _assert_roundtrip(document)

    with pytest.raises(ValueError, match="append-only"):
        document.replace_citation(rooted.model_copy(update={"case_name": ()}))


def test_one_recorded_decision_can_update_two_fields() -> None:
    source = "Smith v. Jones, Case No. 1:24-cv-00123 (D. Ariz. 2024)."
    document = find_docket_locators(Document.from_source(source))
    citation = document.citations[0]
    name = "Smith v. Jones"
    court = "D. Ariz."
    revised = citation.record("review")
    revised = revised.with_case_name(source, name, Span(0, len(name)))
    revised = revised.with_court(
        source, "d-ariz", Span(source.index(court), source.index(court) + len(court))
    )
    document = document.replace_citation(revised)

    assert len(revised.nodes) == len(citation.nodes) + 1
    assert revised.nodes[-1].stage == "review"
    assert revised.case_name[-1].node_id == revised.court[-1].node_id == revised.nodes[-1].id
    assert revised.locator_text == citation.locator_text
    _assert_roundtrip(document)

    year_span = Span(source.index("2024"), source.index("2024") + 4)
    unrecorded = revised.with_date(source, CitationDate(year=2024), year_span)
    with pytest.raises(ValueError, match="new decision node"):
        document.replace_citation(unrecorded)


def test_source_mismatches_are_rejected_at_write_time_and_after_json_loading() -> None:
    source = "Smith v. Jones, Case No. 1:24-cv-00123, at 42 (2024)."
    document = find_docket_locators(Document.from_source(source))
    citation = document.citations[0]
    name_span = Span(0, len("Smith v. Jones"))
    pin_span = Span(source.index("42"), source.index("42") + 2)
    date_span = Span(source.index("2024"), source.index("2024") + 4)

    with pytest.raises(ValueError, match="source span"):
        citation.record("review").with_case_name(source, "Other v. Case", name_span)
    with pytest.raises(ValueError, match="source span"):
        citation.record("review").with_pin_cite(source, "43", pin_span)
    with pytest.raises(ValueError, match="Date year"):
        citation.record("review").with_date(source, CitationDate(year=2023), date_span)
    with pytest.raises(ValueError, match="source span"):
        FullDocketCitation.from_locator(
            citation_id="bad",
            stage="locator",
            source=source,
            span=citation.locator_span,
            docket_number="wrong number",
            docket_number_span=citation.docket_number[-1].span,
        )

    updated = citation.record("case_names").with_case_name(source, "Smith v. Jones", name_span)
    updated = updated.record("pin_cites").with_pin_cite(source, "42", pin_span)
    document = document.replace_citation(updated)
    _assert_roundtrip(document)

    for field in ("locator_text", "case_name", "pin_cite"):
        tampered = document.model_dump(mode="json")
        tampered["citations"][0][field][-1]["value"] = "text absent from source"
        with pytest.raises(ValueError, match="source span"):
            Document.model_validate(tampered)


def test_field_history_rejects_missing_node_and_reversed_order() -> None:
    source = "Case No. 1:24-cv-00123 (D. Ariz. 2024)."
    document = find_docket_locators(Document.from_source(source))
    citation = document.citations[0]
    citation = citation.record("first").with_court(source, "d-ariz")
    citation = citation.record("second").with_court(source, "d-ariz")
    document = document.replace_citation(citation)

    missing = document.model_dump(mode="json")
    missing["citations"][0]["court"][-1]["node_id"] = "missing"
    with pytest.raises(ValueError, match="missing node"):
        Document.model_validate(missing)

    reversed_history = document.model_dump(mode="json")
    reversed_history["citations"][0]["court"][0]["node_id"] = citation.nodes[-1].id
    reversed_history["citations"][0]["court"][1]["node_id"] = citation.nodes[-2].id
    with pytest.raises(ValueError, match="out of order"):
        Document.model_validate(reversed_history)

    same_node = document.model_dump(mode="json")
    same_node["citations"][0]["court"][1]["node_id"] = citation.nodes[-2].id
    with pytest.raises(ValueError, match="out of order"):
        Document.model_validate(same_node)


def test_every_checkpoint_and_withdrawal_restore_preserve_history() -> None:
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
        _assert_roundtrip(document)
        for citation in document.citations:
            prior = previous.get(citation.id)
            if prior is None:
                continue
            assert citation.nodes[: len(prior.nodes)] == prior.nodes
            for name, old_log in _field_logs(prior).items():
                assert getattr(citation, name)[: len(old_log)] == old_log

    citation = document.citations[0]
    withdrawn = document.replace_citation(citation.record("review").withdraw())
    withdrawn_citation = withdrawn.citations[0]
    assert latest(withdrawn_citation.root_id) == "__withdrawn__"
    assert withdrawn_citation.nodes[: len(citation.nodes)] == citation.nodes
    assert withdrawn_citation.root_id[:-1] == citation.root_id
    assert withdrawn_citation.root_id[-1].node_id == withdrawn_citation.nodes[-1].id
    for name, old_log in _field_logs(citation).items():
        assert getattr(withdrawn_citation, name)[: len(old_log)] == old_log
    _assert_roundtrip(withdrawn)

    restored = withdrawn.replace_citation(withdrawn_citation.record("review").with_root(citation.id))
    restored_citation = restored.citations[0]
    assert latest(restored_citation.root_id) == citation.id
    assert restored_citation.root_id[:-1] == withdrawn_citation.root_id
    assert len(restored_citation.nodes) == len(withdrawn_citation.nodes) + 1
    _assert_roundtrip(restored)

    stale_root = restored.model_dump(mode="json")
    stale_root["citations"][0]["root_id"][-1]["value"] = "missing-citation"
    with pytest.raises(ValueError, match="unknown root"):
        Document.model_validate(stale_root)
