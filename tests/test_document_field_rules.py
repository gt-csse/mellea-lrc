"""Document-aware rules retain group boundaries, aliases and local pinpoint spans."""

import asyncio

import pytest

from mellea_lrc.api import (
    Document,
    find_docket_locators,
    find_full_reporter_locators,
    find_short_reporter_citations,
    grow_roots,
    resolve_case_names,
    resolve_colocations,
    resolve_courts,
    resolve_dates,
    resolve_pin_cites,
    resolve_short_reporter_case_names,
    resolve_short_reporter_colocations,
)
from mellea_lrc.config.extraction import ExtractionRules
from mellea_lrc.extraction.resolve_case_names.reader import read_case_name, read_short_case_name
from mellea_lrc.extraction.resolve_pin_cites import read_pin_cite
from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.parsing.case_names import CaseName, CaseNameKind
from mellea_lrc.parsing.span import Span


def full_fields(source: str, *, discovered_docket: str | None = None) -> Document:
    document = find_docket_locators(find_full_reporter_locators(Document.from_source(source)))
    if discovered_docket is not None:
        start = source.index(discovered_docket)
        document = document.add_citation(
            FullDocketCitation.from_locator(
                citation_id=f"hunted:{start}",
                substage="grow_roots.discover_root_locators.hunt_docket_locators",
                source=source,
                span=Span(start, start + len(discovered_docket)),
                number_span=Span(start + 4, start + len(discovered_docket)),
            )
        ).complete_substage("grow_roots.discover_root_locators.hunt_docket_locators")
    document = resolve_colocations(document)
    for stage in (resolve_case_names, resolve_courts, resolve_dates, resolve_pin_cites):
        document = stage(document)
    assert not any("predict_fields" in stage for stage in document.substage_runs)
    return document


def quotes(citation) -> dict[str, str | None]:
    return {
        field: values[-1].quote if (values := getattr(citation, field)) else None
        for field in ("case_name", "court", "date", "pin_cite")
    }


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "Ashcroft v. Iqbal, 556 U.S. 662, 678 (2009).",
            {"case_name": "Ashcroft v. Iqbal", "court": None, "date": "2009", "pin_cite": "678"},
        ),
        (
            "Alpha v. Beta, 123 F.3d 456, 459–60 (2d Cir. Mar. 4, 2000).",
            {"case_name": "Alpha v. Beta", "court": "2d Cir.", "date": "Mar. 4, 2000", "pin_cite": "459–60"},
        ),
        (
            "Smith v. Jones, No. 12-cv-1234, at *5 (S.D.N.Y. Jan. 2, 2020).",
            {"case_name": "Smith v. Jones", "court": "S.D.N.Y.", "date": "Jan. 2, 2020", "pin_cite": "*5"},
        ),
    ],
)
def test_rule_stages_read_exact_source_spans_without_prediction_cache(source, expected):
    document = full_fields(source)
    assert len(document.full_locators) == 1
    assert quotes(document.full_locators[0]) == expected
    assert Document.model_validate_json(document.model_dump_json()) == document


@pytest.mark.parametrize("docket", ["No. 05-cv-4206", "No. 05-4206"])
def test_docket_reporter_group_retains_shared_fields_and_local_pins(docket):
    source = f"Alpha v. Beta, {docket}, 2006 WL 1984362, at *1 (E.D. La. July 13, 2006)."
    document = full_fields(source, discovered_docket=docket if docket == "No. 05-4206" else None)
    assert len(document.colocations) == 1
    assert [quotes(citation) for citation in document.full_locators] == [
        {"case_name": "Alpha v. Beta", "court": "E.D. La.", "date": "July 13, 2006", "pin_cite": None},
        {"case_name": "Alpha v. Beta", "court": "E.D. La.", "date": "July 13, 2006", "pin_cite": "*1"},
    ]


def test_parallel_reporter_group_keeps_each_local_pin():
    document = full_fields("Alpha v. Beta, 123 F.3d 456, 459, 789 F.2d 10, 12 (2d Cir. 2000).")
    assert len(document.colocations) == 1
    assert [quotes(citation) for citation in document.full_locators] == [
        {"case_name": "Alpha v. Beta", "court": "2d Cir.", "date": "2000", "pin_cite": "459"},
        {"case_name": "Alpha v. Beta", "court": "2d Cir.", "date": "2000", "pin_cite": "12"},
    ]


def test_short_reporter_group_uses_names_from_the_document():
    source = (
        "Unknown Entity v. Another Entity, 347 U.S. 483 (1954). "
        "The Court explained Unknown Entity, 347 U.S. at 495, 74 S. Ct. at 690."
    )
    document = asyncio.run(grow_roots(Document.from_source(source)))
    document = find_short_reporter_citations(document)
    document = resolve_short_reporter_colocations(document)
    document = resolve_short_reporter_case_names(document)
    assert [
        (citation.case_name[-1].quote, citation.pin_cite[-1].quote) for citation in document.short_reporters
    ] == [
        ("Unknown Entity", "495"),
        ("Unknown Entity", "690"),
    ]


def test_configured_rule_windows_are_used_by_the_field_stage():
    source = "Alpha v. Beta, 123 F.3d 456 (2d Cir. Jan. 2, 2000)."
    grouped = resolve_colocations(find_full_reporter_locators(Document.from_source(source)))
    narrow = resolve_dates(grouped, ExtractionRules(post_locator_window=8))
    wide = resolve_dates(grouped, ExtractionRules(post_locator_window=120))
    assert narrow.full_locators[0].date == ()
    assert wide.full_locators[0].date[-1].quote == "Jan. 2, 2000"


def test_bounded_rule_helpers_preserve_offsets_and_malformed_pin_tokens():
    assert read_case_name("See Alpha v. Beta, ", 100, None) == Span(104, 117)
    assert read_pin_cite(", 459 n. X (2000)", 0, 16) == Span(2, 10)
    assert read_pin_cite(", 2d Cir.", 0, 9) is None
    assert read_pin_cite(", 12: prose", 0, 11) is None
    assert read_short_case_name("See Alpha, ", 50) == Span(54, 59)
    assert read_short_case_name("The Court explained Alpha, ", 0, ("Alpha",)) == Span(20, 25)


def test_case_name_normalizer_remains_strict():
    assert CaseName.from_quote("In re\nAlpha").kind is CaseNameKind.IN_RE
    assert CaseName.from_quote("Alpha v. Beta").plaintiff == "Alpha"
    with pytest.raises(ValueError):
        CaseName.from_quote("Alpha v.")
