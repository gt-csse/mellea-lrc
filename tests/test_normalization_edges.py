"""Invalid citation readings must fail at the point they are normalized."""

import pytest

from mellea_lrc.extraction import find_docket_locators, grow_roots
from mellea_lrc.model import (
    CitationDate,
    Document,
    FullDocketCitation,
    FullReporterLocator,
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
    assert court.normalized.id == court_id
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_filing_date_verb_is_not_part_of_the_court_quote() -> None:
    source = "See Case No. 1:24-cv-00123 (E.D.N.Y. filed Oct. 14, 2025)."
    document = grow_roots(Document.from_source(source))
    court = document.citations[0].court[-1]
    assert court.quote == "E.D.N.Y."
    assert court.normalized.id == "nyed"
    assert document.citations[0].date[-1].normalized == CitationDate(year=2025, month=10, day=14)


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
    assert entry.normalized.volume == 347
    assert entry.normalized.page == "483"
    assert entry.normalized.edition == "U.S."
    assert isinstance(entry.normalized.reporter, Reporter)
    assert entry.normalized.reporter.short_name == "U.S."


def test_ambiguous_eyecite_reporter_raises_on_normalization() -> None:
    # eyecite identifies this as a full citation but cannot choose an edition.
    source = "1 Wash. 2"
    with pytest.raises(ValueError, match="Cannot normalize.*reporter locator"):
        FullReporterLocator.from_source(source, Span(0, len(source)), node_id="reporter:node:0")


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
    with pytest.raises(ValueError, match="Docket number normalization"):
        Document.model_validate(changed_number)

    locator["number_span"]["start"] += 1
    with pytest.raises(ValueError, match="Docket number normalization"):
        Document.model_validate(saved)


def test_docket_entry_that_cannot_be_normalized_raises() -> None:
    source = "Doc. unresolved, Case No. 1:24-cv-00123."
    locator = "Case No. 1:24-cv-00123"
    number = "1:24-cv-00123"
    with pytest.raises(ValueError, match="Cannot normalize docket entry"):
        FullDocketCitation.from_locator(
            citation_id="docket:0",
            stage="docket_locators",
            source=source,
            span=Span(source.index(locator), source.index(locator) + len(locator)),
            number_span=Span(source.index(number), source.index(number) + len(number)),
            docket_entry_span=Span(0, len("Doc. unresolved")),
        )


def test_docket_entry_requires_its_written_label() -> None:
    source = "garbage 10"
    with pytest.raises(ValueError, match="Cannot normalize docket entry"):
        DocketEntryField.from_source(source, Span(0, len(source)), node_id="docket:node:0")


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
    assert name.normalized.defendant == "Jones"
    assert name.normalized.plaintiff in {"Smith", "Oak and Fort Corp."}


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
def test_unresolved_written_court_raises_before_inference(source: str) -> None:
    with pytest.raises(ValueError, match="Cannot normalize written court"):
        grow_roots(Document.from_source(source))


def test_absent_written_court_can_still_be_inferred_from_reporter() -> None:
    document = grow_roots(Document.from_source("See 347 U.S. 483 (1954)."))
    assert latest(document.citations[0].court).id == "scotus"
