"""Citation occurrences retain grounded readings and their local decision history."""

import pytest
from pydantic import ValidationError

from mellea_lrc.model import Span
from mellea_lrc.model.citations import (
    FullCitationKind,
    FullDocketCitation,
    FullReporterCitation,
    Node,
    latest,
)
from mellea_lrc.model.citations.fields import (
    CaseName,
    CaseNameField,
    CaseNameKind,
    CitationDate,
    NotStatedField,
    current_reading,
)

REPORTER_SOURCE = "Alpha v. Beta, 556 U.S. 662, 678 (2009)."
DOCKET_SOURCE = "Alpha v. Beta, Case No. 1:24-cv-00123, ECF No. 128 (D. Md. 2024)."


def quoted_span(source, quote):
    start = source.index(quote)
    return Span(start=start, end=start + len(quote))


def reporter():
    return FullReporterCitation.from_locator(
        citation_id="reporter:0",
        substage="grow_roots.discover.find_reporters",
        source=REPORTER_SOURCE,
        span=quoted_span(REPORTER_SOURCE, "556 U.S. 662"),
    )


def docket():
    return FullDocketCitation.from_locator(
        citation_id="docket:0",
        substage="grow_roots.discover.find_dockets",
        source=DOCKET_SOURCE,
        span=quoted_span(DOCKET_SOURCE, "Case No. 1:24-cv-00123"),
        number_span=quoted_span(DOCKET_SOURCE, "1:24-cv-00123"),
    )


def corrected_reporter():
    initial = (
        reporter()
        .record("grow_roots.read_fields")
        .with_case_name(REPORTER_SOURCE, quoted_span(REPORTER_SOURCE, "Beta"))
    )
    return initial.record("validate_identity.review").with_case_name(
        REPORTER_SOURCE,
        quoted_span(REPORTER_SOURCE, "Alpha v. Beta"),
        normalized=CaseName(kind=CaseNameKind.ADVERSARIAL, plaintiff="Alpha", defendant="Beta"),
    )


@pytest.mark.parametrize(
    "factory,kind", [(reporter, FullCitationKind.REPORTER), (docket, FullCitationKind.DOCKET)]
)
def test_creation_records_a_grounded_locator_and_its_producer(factory, kind):
    citation = factory()
    source = REPORTER_SOURCE if kind is FullCitationKind.REPORTER else DOCKET_SOURCE

    assert citation.kind is kind
    assert len(citation.nodes) == 1
    assert citation.locator[0].node_id == citation.nodes[0].id
    assert citation.site_span == citation.locator_span == citation.locator[0].span
    assert source[citation.locator_span.start : citation.locator_span.end] == citation.locator[0].quote
    assert not citation.case_name
    assert citation.pin_cite is None
    assert citation.get_case_name().kind is CaseNameKind.NOT_STATED
    assert citation.get_pin_cite() is None
    citation.validate_source(source)
    assert type(citation).model_validate_json(citation.model_dump_json()) == citation


def test_one_decision_can_read_several_fields_without_mutating_the_earlier_citation():
    created = reporter()
    read = (
        created.record("grow_roots.read_fields")
        .with_case_name(REPORTER_SOURCE, quoted_span(REPORTER_SOURCE, "Alpha v. Beta"))
        .with_inferred_court("scotus")
        .with_date(REPORTER_SOURCE, quoted_span(REPORTER_SOURCE, "2009"))
        .with_pin_cite(REPORTER_SOURCE, quoted_span(REPORTER_SOURCE, "678"))
    )

    assert len(created.nodes) == 1
    assert created.case_name == ()
    assert created.pin_cite is None
    assert len(read.nodes) == 2
    assert read.get_case_name().as_citation() == "Alpha v. Beta"
    assert latest(read.court).id == "scotus"
    assert read.court[-1].quote is None
    assert latest(read.date) == CitationDate(year=2009)
    assert read.get_pin_cite()[0].first == 678
    assert {log[-1].node_id for log in (read.case_name, read.court, read.date, read.pin_cite)} == {
        read.nodes[-1].id
    }
    assert FullReporterCitation.model_validate_json(read.model_dump_json()) == read


def test_corrected_and_absent_readings_keep_the_complete_history_after_json_loading():
    corrected = corrected_reporter()
    absent = corrected.record("validate_identity.clear_name").with_case_name_not_stated()
    loaded = FullReporterCitation.model_validate_json(absent.model_dump_json())

    assert len(corrected.case_name) == 2
    assert corrected.case_name[0].quote == "Beta"
    assert corrected.case_name[1].quote == "Alpha v. Beta"
    assert corrected.case_name[1].normalized_by == "model"
    assert corrected.get_case_name().as_citation() == "Alpha v. Beta"
    assert loaded == absent
    assert len(loaded.case_name) == 3
    assert loaded.case_name[:2] == corrected.case_name
    assert isinstance(loaded.case_name[-1], NotStatedField)
    assert loaded.get_case_name().kind is CaseNameKind.NOT_STATED
    assert latest(loaded.case_name) is None
    assert current_reading(loaded.case_name) is None
    loaded.validate_source(REPORTER_SOURCE)


@pytest.mark.parametrize("field", ["case_name", "court", "date", "pin_cite"])
def test_explicit_absence_replaces_the_current_reading_without_reviving_an_older_one(field):
    citation = reporter().record("grow_roots.read_fields")
    if field == "case_name":
        citation = citation.with_case_name(REPORTER_SOURCE, quoted_span(REPORTER_SOURCE, "Alpha v. Beta"))
    elif field == "court":
        citation = citation.with_inferred_court("scotus")
    elif field == "date":
        citation = citation.with_date(REPORTER_SOURCE, quoted_span(REPORTER_SOURCE, "2009"))
    else:
        citation = citation.with_pin_cite(REPORTER_SOURCE, quoted_span(REPORTER_SOURCE, "678"))
    previous = getattr(citation, field)[-1]
    cleared = getattr(citation.record("review.clear_field"), f"with_{field}_not_stated")()
    log = getattr(cleared, field)

    assert log[0] == previous
    assert latest(log) is None
    assert current_reading(log) is None
    assert isinstance(log[-1], NotStatedField)
    assert log[-1].quote is None
    assert log[-1].span is None
    with pytest.raises(ValueError):
        log[-1].get_normalized()
    assert FullReporterCitation.model_validate_json(cleared.model_dump_json()) == cleared


def test_a_failed_latest_reading_is_preserved_and_does_not_fall_back_to_an_older_value():
    source = "Alpha v. Beta, 556 U.S. 662 (2009); later printed as February 30, 2024."
    citation = FullReporterCitation.from_locator(
        citation_id="reporter:0",
        substage="discover.reporter",
        source=source,
        span=quoted_span(source, "556 U.S. 662"),
    )
    citation = citation.record("read.date").with_date(source, quoted_span(source, "2009"))
    citation = citation.record("review.date").with_date(source, quoted_span(source, "February 30, 2024"))
    loaded = FullReporterCitation.model_validate_json(citation.model_dump_json())

    assert len(loaded.date) == 2
    assert loaded.date[-1].quote == "February 30, 2024"
    assert not loaded.date[-1].normalizable
    assert current_reading(loaded.date) == loaded.date[-1]
    with pytest.raises(ValueError, match="not normalizable"):
        latest(loaded.date)
    loaded.validate_source(source)


@pytest.mark.parametrize("method", ["with_case_name", "with_date", "with_pin_cite"])
def test_field_changes_require_a_decision_after_creation(method):
    span = quoted_span(
        REPORTER_SOURCE, {"with_case_name": "Beta", "with_date": "2009", "with_pin_cite": "678"}[method]
    )
    with pytest.raises(ValueError):
        getattr(reporter(), method)(REPORTER_SOURCE, span)


def test_a_field_cannot_be_written_twice_under_the_same_decision_node():
    citation = (
        reporter().record("read.name").with_case_name(REPORTER_SOURCE, quoted_span(REPORTER_SOURCE, "Beta"))
    )
    with pytest.raises(ValueError):
        citation.with_case_name(REPORTER_SOURCE, quoted_span(REPORTER_SOURCE, "Alpha v. Beta"))


@pytest.mark.parametrize("node_id", ["missing-node", "another-citation:node:1"])
def test_loading_rejects_a_reading_whose_producer_is_not_owned_by_the_citation(node_id):
    stored = corrected_reporter().model_dump(mode="json")
    stored["case_name"][-1]["node_id"] = node_id
    with pytest.raises(ValidationError):
        FullReporterCitation.model_validate(stored)


def test_loading_rejects_out_of_order_field_history():
    stored = corrected_reporter().model_dump(mode="json")
    stored["case_name"].reverse()
    with pytest.raises(ValidationError):
        FullReporterCitation.model_validate(stored)


def test_loading_rejects_duplicate_entries_at_one_node():
    stored = corrected_reporter().model_dump(mode="json")
    stored["case_name"].append(stored["case_name"][-1])
    with pytest.raises(ValidationError):
        FullReporterCitation.model_validate(stored)


@pytest.mark.parametrize("damage", ["missing_creation", "duplicate_node", "foreign_node"])
def test_loading_rejects_invalid_decision_node_ownership(damage):
    stored = corrected_reporter().model_dump(mode="json")
    if damage == "missing_creation":
        stored["nodes"] = []
    elif damage == "duplicate_node":
        stored["nodes"].append(stored["nodes"][-1])
    else:
        stored["nodes"][-1]["id"] = "another-citation:node:2"
        stored["case_name"][-1]["node_id"] = "another-citation:node:2"
    with pytest.raises(ValidationError):
        FullReporterCitation.model_validate(stored)


def test_loading_rejects_an_ordinary_field_attached_to_the_creation_node():
    citation = reporter()
    stored = citation.model_dump(mode="json")
    stored["case_name"] = [
        CaseNameField.from_source(
            REPORTER_SOURCE, quoted_span(REPORTER_SOURCE, "Beta"), node_id=citation.nodes[0].id
        ).model_dump(mode="json")
    ]
    with pytest.raises(ValidationError):
        FullReporterCitation.model_validate(stored)


@pytest.mark.parametrize("factory", [reporter, docket])
@pytest.mark.parametrize("damage", ["empty", "wrong_creation_node"])
def test_locators_must_be_present_at_the_creation_node(factory, damage):
    citation = factory().record("read.context")
    stored = citation.model_dump(mode="json")
    if damage == "empty":
        stored["locator"] = []
    else:
        stored["locator"][0]["node_id"] = citation.nodes[-1].id
    with pytest.raises(ValidationError):
        type(citation).model_validate(stored)


@pytest.mark.parametrize("factory,wrong_kind", [(reporter, "docket"), (docket, "reporter")])
def test_concrete_citation_types_reject_the_other_locator_kind(factory, wrong_kind):
    citation = factory()
    stored = citation.model_dump(mode="json")
    stored["kind"] = wrong_kind
    with pytest.raises(ValidationError):
        type(citation).model_validate(stored)


def test_source_validation_checks_older_quotes_even_after_a_correction():
    citation = corrected_reporter()
    # The old name is a proper subspan of the corrected name; changing only
    # its saved quote keeps the latest reading and locator untouched.
    stored = citation.model_dump(mode="json")
    old = stored["case_name"][0]
    old["quote"] = "Zeta"
    old["normalized"]["partial"] = "Zeta"
    loaded = FullReporterCitation.model_validate(stored)
    assert loaded.case_name[-1] == citation.case_name[-1]
    with pytest.raises(ValueError, match="does not match its source span"):
        loaded.validate_source(REPORTER_SOURCE)


def test_citation_and_nested_readings_cannot_be_mutated_in_place():
    citation = corrected_reporter()
    with pytest.raises(ValidationError, match="frozen"):
        citation.id = "replacement"
    with pytest.raises(ValidationError, match="frozen"):
        citation.nodes[-1].substage = "rewrite.history"
    with pytest.raises(ValidationError, match="frozen"):
        citation.case_name[-1].quote = "replacement"
    with pytest.raises(TypeError):
        citation.case_name[0] = citation.case_name[-1]


def test_docket_entry_and_number_rereading_retain_the_original_locator_site():
    created = docket()
    read = created.record("read.docket_fields").with_docket_entry(
        DOCKET_SOURCE, quoted_span(DOCKET_SOURCE, "ECF No. 128")
    )
    corrected = read.record("review.docket_number").with_docket_number(
        DOCKET_SOURCE, quoted_span(DOCKET_SOURCE, "24-cv-00123")
    )
    loaded = FullDocketCitation.model_validate_json(corrected.model_dump_json())

    assert loaded == corrected
    assert latest(loaded.docket_entry) == "128"
    assert loaded.locator[0] == created.locator[0]
    assert len(loaded.locator) == 2
    assert latest(loaded.locator).docket_number == "24-cv-00123"
    assert loaded.locator_span == created.locator_span
    loaded.validate_source(DOCKET_SOURCE)


def test_a_docket_rereading_cannot_move_to_a_different_source_site():
    source = DOCKET_SOURCE + " Later, Case No. 2:25-cv-00456."
    citation = docket().record("review.docket_number")
    with pytest.raises(ValueError, match="inside the locator"):
        citation.with_docket_number(source, quoted_span(source, "2:25-cv-00456"))


@pytest.mark.parametrize("substage", ["", " ", "WrongCase", "a.b.c.d", "read fields"])
def test_decision_nodes_require_a_valid_named_producer(substage):
    with pytest.raises(ValidationError):
        Node(id="citation:node:0", substage=substage)


def test_empty_histories_have_no_current_reading():
    assert latest(()) is None
    assert latest(None) is None
    assert current_reading(()) is None
    assert current_reading(None) is None
