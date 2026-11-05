"""Native locator discovery preserves exact source sites and atomic checkpoints."""

import pytest

from mellea_lrc.extraction import find_docket_locators, find_full_reporter_locators
from mellea_lrc.model import (
    Document,
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
    Rule,
    SourceFormat,
    SourceMetadata,
    Span,
    TableOfAuthoritiesComponent,
)
from mellea_lrc.model.citations import FullDocketCitation, FullReporterCitation

REPORTERS = "grow_roots.discover_root_locators.find_full_reporter_locators"
DOCKETS = "grow_roots.discover_root_locators.find_docket_locators"
DISCOVERY = (find_full_reporter_locators, find_docket_locators)


def document(text, *, index_spans=()):
    return Document.from_preprocessed(
        PreprocessedDocument(
            text=text,
            source_metadata=SourceMetadata(path="brief.pdf", format=SourceFormat.PDF),
            preprocessing_metadata=PreprocessingMetadata(
                backend=PreprocessingBackend.DOCLING,
                backend_version="offline-test",
                rules=(Rule.TABLE_OF_AUTHORITIES,),
            ),
            index_spans=index_spans,
        )
    )


def source_span(text, quote):
    start = text.index(quote)
    return Span(start, start + len(quote))


@pytest.mark.parametrize("steps", [DISCOVERY, tuple(reversed(DISCOVERY))])
def test_discovery_preserves_source_and_orders_both_kinds_without_reading_contextual_fields(steps):
    text = (
        "Préface. Gamma v. Delta, Case No. 1:24-cv-00123-AB, ECF No. 128 (D. Md. 2024). "
        "Alpha v. Beta, 556 U.S. 662, 678 (2009)."
    )
    original = document(text)
    first = steps[0](original)
    result = steps[1](first)

    assert original.citations == original.runs == ()
    assert len(first.citations) == 1
    assert result.text == original.text
    assert result.source_metadata == original.source_metadata
    assert result.preprocessing_metadata == original.preprocessing_metadata
    assert result.source_path == "brief.pdf"
    assert result.index_spans == original.index_spans
    docket, reporter = result.citations
    assert isinstance(docket, FullDocketCitation)
    assert isinstance(reporter, FullReporterCitation)
    assert docket.site_span == source_span(text, "Case No. 1:24-cv-00123-AB")
    assert docket.locator[0].quote == "Case No. 1:24-cv-00123-AB"
    assert docket.locator[0].number_span == source_span(text, "1:24-cv-00123-AB")
    assert docket.locator[0].get_normalized().docket_number == "1:24-cv-00123-AB"
    assert docket.docket_entry == ()
    assert reporter.site_span == source_span(text, "556 U.S. 662")
    assert reporter.locator[0].quote == "556 U.S. 662"
    assert reporter.locator[0].get_normalized().page == "662"
    for citation in result.citations:
        assert citation.case_name == citation.court == citation.date == ()
        assert citation.pin_cite is None
        assert len(citation.nodes) == 1
        assert citation.locator[0].node_id == citation.nodes[0].id
        citation.validate_source(text)
    assert {citation.nodes[0].substage for citation in result.citations} == {REPORTERS, DOCKETS}
    assert Document.model_validate_json(result.model_dump_json()) == result
    assert result.get_substage(first.substage_runs[-1]) == first
    assert result.get_substage(result.substage_runs[-1]) == result


def test_repeated_reporter_and_docket_sites_remain_distinct_occurrences():
    text = "556 U.S. 662; Case No. 24-cv-123. Later: 556 U.S. 662; Case No. 24-cv-123."
    result = find_docket_locators(find_full_reporter_locators(document(text)))

    assert len(result.citations) == len({item.id for item in result.citations}) == 4
    assert [item.locator[0].quote for item in result.citations] == [
        "556 U.S. 662",
        "Case No. 24-cv-123",
        "556 U.S. 662",
        "Case No. 24-cv-123",
    ]
    assert len({item.site_span for item in result.citations}) == 4
    assert [item.site_span.start for item in result.citations] == sorted(
        item.site_span.start for item in result.citations
    )


def test_discovery_retains_table_of_authorities_sites_and_their_source_regions():
    text = "Authorities: 556 U.S. 662.\nArgument: 556 U.S. 662, 678."
    regions = (TableOfAuthoritiesComponent(0, text.index("\n")),)
    result = find_full_reporter_locators(document(text, index_spans=regions))

    assert len(result.citations) == 2
    assert result.index_spans == regions
    assert result.citations[0].site_span.end < regions[0].end
    assert result.citations[1].site_span.start > regions[0].end


@pytest.mark.parametrize("written", ["347U.S.483", "937\n\nS.W.2d  796"])
def test_reporter_discovery_keeps_pdf_whitespace_and_tightly_joined_source_quotes(written):
    text = f"See {written}."
    result = find_full_reporter_locators(document(text))

    assert len(result.citations) == 1
    assert result.citations[0].site_span == Span(4, 4 + len(written))
    assert result.citations[0].locator[0].quote == written
    assert result.citations[0].locator[0].normalizable


def test_ambiguous_reporter_is_retained_as_failed_normalization():
    result = find_full_reporter_locators(document("See 1 Wash. 2."))

    assert len(result.citations) == 1
    field = result.citations[0].locator[0]
    assert field.quote == "1 Wash. 2"
    assert not field.normalizable
    assert "ambiguous reporter" in field.normalization_error
    assert Document.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("discover, name", list(zip(DISCOVERY, (REPORTERS, DOCKETS))))
def test_no_result_discovery_still_commits_a_replayable_completion_and_rejects_rerun(discover, name):
    original = document("This passage has no case citation.")
    result = discover(original)
    restored = Document.model_validate_json(result.model_dump_json())

    assert original.substage_runs == ()
    assert result.citations == ()
    assert restored.substage_runs == (name,)
    assert restored.get_substage(name) == result
    with pytest.raises(ValueError, match="already completed"):
        discover(restored)


def test_no_result_second_substage_does_not_remove_the_first_checkpoint():
    first = find_full_reporter_locators(document("See 556 U.S. 662."))
    result = find_docket_locators(first)
    restored = Document.model_validate_json(result.model_dump_json())

    assert result.citations == first.citations
    assert restored.substage_runs == (REPORTERS, DOCKETS)
    assert restored.get_substage(REPORTERS) == first
    assert restored.get_substage(DOCKETS) == result


@pytest.mark.parametrize("discover, name", list(zip(DISCOVERY, (REPORTERS, DOCKETS))))
def test_discovery_rejects_rerun_with_existing_results(discover, name):
    result = discover(document("See 556 U.S. 662 and Case No. 24-cv-123."))
    with pytest.raises(ValueError, match="already completed"):
        discover(result)
    assert result.substage_runs == (name,)
    assert len(result.citations) == 1


@pytest.mark.parametrize("kind", ["reporter", "docket"])
def test_discovery_skips_an_existing_overlapping_site_and_preserves_its_history(kind):
    text = "See 556 U.S. 662 and Case No. 24-cv-123."
    if kind == "reporter":
        seeded = FullReporterCitation.from_locator(
            citation_id="seeded",
            substage="seed",
            source=text,
            span=source_span(text, "556 U.S. 662"),
        )
        discover = find_full_reporter_locators
    else:
        seeded = FullDocketCitation.from_locator(
            citation_id="seeded",
            substage="seed",
            source=text,
            span=source_span(text, "Case No. 24-cv-123"),
            number_span=source_span(text, "24-cv-123"),
        )
        discover = find_docket_locators
    original = document(text).add_citation(seeded).complete_substage("seed")
    result = discover(original)

    assert result.citations == (seeded,)
    assert len(result.substage_runs) == 2
    assert result.get_substage("seed") == original


@pytest.mark.parametrize(
    "written, number",
    [
        ("No. 23-cr-12", "23-cr-12"),
        ("Case No. 1:24-cv-00123", "1:24-cv-00123"),
        ("Civil Action No. 24-cv-123", "24-cv-123"),
        ("Civ. A. No. 24-cv-123", "24-cv-123"),
        ("Docket No. 2-24-cv-123456", "2-24-cv-123456"),
        ("Case\tNo.\t1:24-cv-00123", "1:24-cv-00123"),
        ("CaseNo.1:24-cv-00123", "1:24-cv-00123"),
        ("cAsE nO. 24-CV-00123-ABC-DE", "24-CV-00123-ABC-DE"),
    ],
)
def test_docket_discovery_recognizes_labelled_cmecf_forms_without_rewriting_the_number(written, number):
    text = f"See {written}; dismissed."
    result = find_docket_locators(document(text))

    assert len(result.citations) == 1
    locator = result.citations[0].locator[0]
    assert locator.quote == written
    assert locator.span == Span(4, 4 + len(written))
    assert locator.number_span == source_span(text, number)
    assert locator.get_normalized().docket_number == number


@pytest.mark.parametrize(
    "text",
    [
        "An unlabelled 1:24-cv-00123 is outside this reader.",
        "No 1:24-cv-00123 lacks the required punctuation.",
        "Case No. 12345 is not a CM/ECF case number.",
        "Docket No. A-12345 is outside this grammar.",
        "Doc. 24-cv-123 and ECF No. 128 are entry references.",
        "Case No. 24-cv-1234567 must not be truncated.",
        "Case No. 24-cv-123-a must not be truncated.",
        "Case No. 24-cv-123-ABCDE-ABCDE-ABCDE must not be truncated.",
        "Case No. 24-cv-123/4 must not be truncated.",
        "Case No. 24-cv-123x must not be truncated.",
    ],
)
def test_docket_discovery_does_not_admit_unlabelled_entries_or_partial_numbers(text):
    result = find_docket_locators(document(text))
    assert result.citations == ()
    assert result.substage_runs == (DOCKETS,)
