"""Invalid readings retain their source evidence and normalization failure."""

import pytest

from mellea_lrc.extraction import find_docket_locators, grow_roots
from mellea_lrc.model import (
    CaseNameField,
    CitationDate,
    CourtField,
    DateField,
    Document,
    FullDocketCitation,
    FullReporterLocator,
    PinCiteField,
    Reporter,
    Span,
    latest,
)
from mellea_lrc.model.citations.fields.court import normalize_court
from mellea_lrc.model.citations.fields.docket import DocketEntryField


@pytest.mark.parametrize(
    ("written", "court_id"),
    [
        ("3d Cir.", "ca3"),
        ("3rd Cir.", "ca3"),
        ("2 nd  Cir.", "ca2"),
        ("D. Md.", "mdd"),
        ("D. Minn.", "mnd"),
        ("D. N. Mar. I.", "nmid"),
    ],
)
def test_court_label_uses_data_derived_token_variants(written: str, court_id: str) -> None:
    source = f"See Case No. 1:24-cv-00123 ({written} 2024)."
    document = grow_roots(Document.from_source(source))
    court = document.citations[0].court[-1]
    assert court.quote == written
    assert court.get_normalized().id == court_id
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_filing_date_verb_is_not_part_of_the_court_quote() -> None:
    source = "See Case No. 1:24-cv-00123 (E.D.N.Y. filed Oct. 14, 2025)."
    document = grow_roots(Document.from_source(source))
    court = document.citations[0].court[-1]
    assert court.quote == "E.D.N.Y."
    assert court.get_normalized().id == "nyed"
    assert document.citations[0].date[-1].get_normalized() == CitationDate(year=2025, month=10, day=14)


def test_court_matching_does_not_confuse_a_circuit_with_its_bankruptcy_panel() -> None:
    assert normalize_court("2 nd  Cir.").id == "ca2"
    assert normalize_court("2 nd  Cir. BAP").id == "bap2"


def test_reporter_locator_keeps_written_variant_and_eyecite_identity() -> None:
    source = "See 347 U. S. 483."
    written = "347 U. S. 483"
    entry = FullReporterLocator.from_source(
        source,
        Span(source.index(written), source.index(written) + len(written)),
        node_id="reporter:node:0",
    )

    assert entry.quote == written
    assert entry.get_normalized().volume == 347
    assert entry.get_normalized().page == "483"
    assert entry.get_normalized().edition == "U.S."
    assert isinstance(entry.get_normalized().reporter, Reporter)
    assert entry.get_normalized().reporter.short_name == "U.S."


@pytest.mark.parametrize(
    ("written", "edition", "page"),
    [
        ("2005  WL  465431", "WL", "465431"),
        ("937\n\nS.W.2d  796", "S.W.2d", "796"),
        ("347 U.\n\nS. 483", "U.S.", "483"),
        ("539  F.  App'x  937", "F. App'x", "937"),
        ("58  N.Y .2d  916", "N.Y.2d", "916"),
        ("777 F. App ' x 516", "F. App'x", "516"),
    ],
)
def test_reporter_discovery_and_normalization_share_the_relaxed_eyecite_reader(
    written: str, edition: str, page: str
) -> None:
    from mellea_lrc.extraction import find_full_reporter_locators

    source = f"See {written}."
    document = find_full_reporter_locators(Document.from_source(source))
    assert len(document.citations) == 1
    entry = document.citations[0].locator[-1]
    assert entry.quote == written
    assert entry.span == Span(4, 4 + len(written))
    assert source[entry.span.start : entry.span.end] == written
    assert entry.get_normalized().edition == edition
    assert entry.get_normalized().page == page
    assert FullReporterLocator.model_validate_json(entry.model_dump_json()) == entry
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_relaxed_reporter_reader_keeps_repeated_source_offsets_distinct() -> None:
    from mellea_lrc.extraction import find_full_reporter_locators

    source = "See 2005  WL  465431. Later, 2005\nWL\n465431."
    document = find_full_reporter_locators(Document.from_source(source))
    assert [citation.locator[-1].quote for citation in document.citations] == [
        "2005  WL  465431",
        "2005\nWL\n465431",
    ]
    assert len({citation.id for citation in document.citations}) == 2
    assert all(
        source[citation.locator_span.start : citation.locator_span.end] == citation.locator[-1].quote
        for citation in document.citations
    )


def test_relaxed_reporter_reader_supplies_context_to_root_formation() -> None:
    source = "Smith v. Jones, 347  U.S.  483 (1954)."
    document = grow_roots(Document.from_source(source))
    citation = document.citations[0]

    assert citation.locator[-1].quote == "347  U.S.  483"
    assert citation.case_name[-1].get_normalized().as_citation() == "Smith v. Jones"
    assert citation.court[-1].get_normalized().id == "scotus"
    assert citation.date[-1].get_normalized().year == 1954
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_ambiguous_eyecite_reporter_keeps_its_failed_reading() -> None:
    # eyecite identifies this as a full citation but cannot choose an edition.
    source = "1 Wash. 2"
    entry = FullReporterLocator.from_source(source, Span(0, len(source)), node_id="reporter:node:0")

    assert entry.quote == source
    assert entry.span == Span(0, len(source))
    assert entry.normalizable is False
    assert "Cannot normalize ambiguous reporter locator" in entry.normalization_error
    assert entry.model_dump(mode="json")["normalized"] is None
    with pytest.raises(ValueError):
        entry.get_normalized()
    assert FullReporterLocator.model_validate_json(entry.model_dump_json()) == entry


def test_docket_number_span_must_be_inside_its_locator() -> None:
    source = "Doc. 10-1, Case No. 1:24-cv-00123."
    locator = "Case No. 1:24-cv-00123"
    with pytest.raises(ValueError, match="inside the locator"):
        FullDocketCitation.from_locator(
            citation_id="docket:0",
            stage="docket_locators",
            source=source,
            span=Span(source.index(locator), source.index(locator) + len(locator)),
            number_span=Span(0, len("Doc. 10-1")),
        )


def test_serialized_docket_number_must_match_its_exact_component_span() -> None:
    document = find_docket_locators(Document.from_source("See Case No. 1:24-cv-00123."))
    saved = document.model_dump(mode="json")
    locator = saved["citations"][0]["locator"][0]

    changed_number = document.model_dump(mode="json")
    changed_number["citations"][0]["locator"][0]["normalized"]["docket_number"] = "00123"
    with pytest.raises(ValueError, match="normalization"):
        Document.model_validate(changed_number)

    locator["number_span"]["start"] += 1
    with pytest.raises(ValueError, match=r"normalization|span"):
        Document.model_validate(saved)


def test_docket_entry_that_cannot_be_normalized_stays_on_citation() -> None:
    source = "Doc. unresolved, Case No. 1:24-cv-00123."
    locator = "Case No. 1:24-cv-00123"
    number = "1:24-cv-00123"
    citation = FullDocketCitation.from_locator(
        citation_id="docket:0",
        stage="docket_locators",
        source=source,
        span=Span(source.index(locator), source.index(locator) + len(locator)),
        number_span=Span(source.index(number), source.index(number) + len(number)),
        docket_entry_span=Span(0, len("Doc. unresolved")),
    )
    entry = citation.docket_entry[-1]
    assert entry.quote == "Doc. unresolved"
    assert entry.span == Span(0, len("Doc. unresolved"))
    assert entry.normalizable is False
    assert "Cannot normalize docket entry" in entry.normalization_error
    with pytest.raises(ValueError):
        entry.get_normalized()
    assert FullDocketCitation.model_validate_json(citation.model_dump_json()) == citation


@pytest.mark.parametrize(
    ("field_type", "source", "error_text"),
    [
        (CaseNameField, "A vs. B", "Adversarial case name"),
        (CourtField, "D. Fiction", "Cannot normalize written court"),
        (DateField, "Feb. 30, 2024", "Invalid citation date"),
        (PinCiteField, "3-2", "Pin cite pages"),
        (DocketEntryField, "garbage 10", "Cannot normalize docket entry"),
    ],
)
def test_unparseable_written_field_preserves_quote_span_and_error(field_type, source, error_text) -> None:
    span = Span(0, len(source))
    field = field_type.from_source(source, span, node_id="field:node:0")

    assert field.quote == source
    assert field.span == span
    assert field.normalizable is False
    assert error_text in field.normalization_error
    saved = field.model_dump(mode="json")
    assert saved["normalizable"] is False
    assert saved["normalized"] is None
    assert saved["normalization_error"] == field.normalization_error
    with pytest.raises(ValueError):
        field.get_normalized()
    with pytest.raises(ValueError):
        field.get_normalized()
    assert field_type.model_validate_json(field.model_dump_json()) == field


def test_failed_reading_remains_loadable_after_normalizer_improves(monkeypatch: pytest.MonkeyPatch) -> None:
    source = "D. Fiction"
    failed = CourtField.from_source(source, Span(0, len(source)), node_id="court:node:0")
    recognized = normalize_court("D. Ariz.")
    monkeypatch.setattr("mellea_lrc.model.citations.fields.court.normalize_court", lambda _quote: recognized)

    restored = CourtField.model_validate_json(failed.model_dump_json())
    assert restored == failed
    assert restored.normalizable is False
    with pytest.raises(ValueError, match="not normalizable"):
        restored.get_normalized()


@pytest.mark.parametrize(
    "source",
    [
        "Alpha v. Beta. Smith v. Jones, 347 U.S. 483 (1954).",
        "In Alpha v. Beta the court cited Smith v. Jones, 347 U.S. 483 (1954).",
        "The court in Smith v. Jones, 347 U.S. 483 (1954).",
        "The court in Oak and Fort Corp. v. Jones, 347 U.S. 483 (1954).",
    ],
)
def test_reporter_name_quote_excludes_prior_names_and_prose(source: str) -> None:
    document = grow_roots(Document.from_source(source))
    name = document.citations[0].case_name[-1]
    assert name.quote == source[name.span.start : name.span.end]
    assert name.get_normalized().defendant == "Jones"
    assert name.get_normalized().plaintiff in {"Smith", "Oak and Fort Corp."}


@pytest.mark.parametrize(
    "components",
    [
        {"year": 2024},
        {"year": 2024, "month": 2},
        {"year": 2024, "month": 2, "day": 29},
    ],
)
def test_citation_date_accepts_valid_precisions(components: dict[str, int]) -> None:
    assert CitationDate(**components).model_dump(exclude_none=True) == components


@pytest.mark.parametrize(
    "components",
    [
        {"year": 0},
        {"year": 10000},
        {"year": 2024, "month": 0},
        {"year": 2024, "month": 13},
        {"year": 2023, "month": 2, "day": 29},
        {"year": 2024, "month": 4, "day": 31},
        {"year": 2024, "month": 2, "day": 0},
        {"year": 2024, "month": 2, "day": 30},
    ],
)
def test_citation_date_rejects_invalid_calendar_components(components: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="Invalid citation date"):
        CitationDate(**components)


def test_citation_date_rejects_day_without_month() -> None:
    with pytest.raises(ValueError, match="day without a month"):
        CitationDate(year=2024, day=1)


@pytest.mark.parametrize(
    "source",
    [
        "See 347 U.S. 483 (D. Fiction 1954).",
        "See Case No. 1:24-cv-00123 (Ct. App. 2024).",
    ],
)
def test_unresolved_written_court_survives_full_pipeline(source: str) -> None:
    document = grow_roots(Document.from_source(source))
    court = document.citations[0].court[-1]
    assert court.quote is not None
    assert source[court.span.start : court.span.end] == court.quote
    assert court.normalizable is False
    assert "Cannot normalize written court" in court.normalization_error
    assert court.model_dump(mode="json")["normalized"] is None
    with pytest.raises(ValueError):
        court.get_normalized()
    restored = Document.model_validate_json(document.model_dump_json())
    assert restored == document
    at_courts = restored.get_stage("courts")
    assert at_courts.citations[0].court[-1] == court


def test_absent_written_court_can_still_be_inferred_from_reporter() -> None:
    document = grow_roots(Document.from_source("See 347 U.S. 483 (1954)."))
    assert latest(document.citations[0].court).id == "scotus"
