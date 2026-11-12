"""Native case-name and docket-entry readings preserve their source and history."""

import pytest

from mellea_lrc.config.extraction import ExtractionRules
from mellea_lrc.extraction.find_docket_locators import find_docket_locators
from mellea_lrc.extraction.find_full_reporter_locators import find_full_reporter_locators
from mellea_lrc.extraction.resolve_case_names import SUBSTAGE as NAMES
from mellea_lrc.extraction.resolve_case_names import resolve_case_names
from mellea_lrc.extraction.resolve_colocations import SUBSTAGE as COLOCATIONS
from mellea_lrc.extraction.resolve_colocations import resolve_colocations
from mellea_lrc.extraction.resolve_docket_entries import SUBSTAGE as ENTRIES
from mellea_lrc.extraction.resolve_docket_entries import resolve_docket_entries
from mellea_lrc.model import Document, PreprocessedDocument, PreprocessingMetadata, SourceMetadata, Span
from mellea_lrc.model.citations.fields.base import NotStatedField


def discovered(text: str) -> Document:
    source = PreprocessedDocument(
        text=text,
        source_metadata=SourceMetadata(path="filing.txt"),
        preprocessing_metadata=PreprocessingMetadata(),
    )
    return find_docket_locators(find_full_reporter_locators(Document.from_preprocessed(source)))


def test_parallel_locators_share_the_exact_case_name_prefix():
    source = "See Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. 2024)."
    grouped = resolve_colocations(discovered(source))
    result = resolve_case_names(grouped)
    assert len(result.colocations) == 1
    for citation in result.citations:
        name = citation.case_name[-1]
        assert name.quote == "Smith v. Jones"
        assert name.span == Span(4, 18)
        assert name.node_id == citation.nodes[-1].id
        assert citation.nodes[-1].substage == NAMES
    restored = Document.model_validate_json(result.model_dump_json())
    assert restored.get_substage(COLOCATIONS) == grouped
    assert restored.get_substage(NAMES) == result


def test_colocated_docket_uses_same_reporter_hint_as_reporter():
    source = "Yardley, Inc. v. Building Co., No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. 2024)."
    grouped = resolve_colocations(discovered(source))
    result = resolve_case_names(grouped)
    assert len(result.colocations) == 1
    assert [citation.case_name[-1].quote for citation in result.citations] == [
        "Yardley, Inc. v. Building Co.",
        "Yardley, Inc. v. Building Co.",
    ]


def test_docket_name_does_not_include_prose_after_a_preceding_citation():
    source = "347 U.S. 483 (1954). The court explained the rule. Alpha v. Beta, Case No. 1:24-cv-00123."
    grouped = resolve_colocations(discovered(source))
    _, docket = resolve_case_names(grouped).citations
    assert docket.case_name[-1].quote == "Alpha v. Beta"


def test_explicit_name_absence_differs_from_unresolved_text():
    grouped = resolve_colocations(discovered("See 347 U.S. 483. The court applied the rule in 556 U.S. 662."))
    first, second = resolve_case_names(grouped).citations
    assert isinstance(first.case_name[-1], NotStatedField)
    assert second.case_name == ()


def test_short_name_window_does_not_treat_cut_off_context_as_absent():
    grouped = resolve_colocations(discovered("Smith v. Jones, 347 U.S. 483."))
    result = resolve_case_names(grouped, ExtractionRules(case_name_window=2))
    assert result.citations == grouped.citations
    assert result.substage_runs[-1] == NAMES


def test_cut_off_party_name_is_not_recorded_as_a_shorter_name():
    source = "Example Global Co. v. Smith, 347 U.S. 483."
    grouped = resolve_colocations(discovered(source))
    result = resolve_case_names(grouped, ExtractionRules(case_name_window=len("al Co. v. Smith, ")))
    assert result.citations[0].case_name == ()


def test_initial_name_reader_preserves_prior_readings():
    grouped = resolve_colocations(discovered("Smith v. Jones, 347 U.S. 483."))
    prior = grouped.citations[0].record("manual_read").with_case_name(grouped.text, Span(9, 14))
    prepared = grouped.replace_citation(prior).complete_substage("manual_read")
    result = resolve_case_names(prepared)
    assert result.citations == prepared.citations
    assert result.substage_runs[-1] == NAMES


@pytest.mark.parametrize(
    ("source", "quote", "number"),
    [
        ("Doc. 10-1, Case No. 1:24-cv-00123.", "Doc. 10-1", "10-1"),
        ("Case No. 1:24-cv-00123, ECF No. 113.", "ECF No. 113", "113"),
        ("Case No. 1:25-bk-11282 (MG) [D.I. 21].", "D.I. 21", "21"),
    ],
)
def test_docket_entry_preserves_its_quote_and_checkpoint_after_colocation(source, quote, number):
    grouped = resolve_colocations(discovered(source))
    result = resolve_docket_entries(grouped)
    (citation,) = result.citations
    (entry,) = citation.docket_entry
    assert entry.quote == source[entry.span.start : entry.span.end] == quote
    assert entry.get_normalized() == number
    assert entry.node_id == citation.nodes[-1].id
    assert citation.nodes[-1].substage == ENTRIES
    restored = Document.model_validate_json(result.model_dump_json())
    assert restored.get_substage(COLOCATIONS) == grouped
    assert restored.get_substage(ENTRIES) == result


@pytest.mark.parametrize(
    "source",
    [
        "Doc. 10, Case No. 1:24-cv-00123, ECF No. 113.",
        "Case No. 1:24-cv-00123, ECF No. 113, Case No. 2:25-cv-00010.",
        "Case No. 1:24-cv-00123, as discussed in ECF No. 113.",
        "Case No. 1:24-cv-00123,\nECF No. 113.",
        "See ECF No. 113 and Doc. 10-1.",
    ],
)
def test_unowned_or_ambiguous_entries_do_not_change_citations(source):
    initial = discovered(source)
    result = resolve_docket_entries(initial)
    assert result.citations == initial.citations
    assert result.substage_runs[-1] == ENTRIES


def test_preexisting_docket_entry_does_not_get_reassigned():
    source = "Case No. 1:24-cv-00123, ECF No. 113, Case No. 2:25-cv-00010."
    initial = discovered(source)
    start = source.index("ECF No. 113")
    prior = initial.citations[0].record("manual_read").with_docket_entry(source, Span(start, start + 11))
    prepared = initial.replace_citation(prior).complete_substage("manual_read")
    result = resolve_docket_entries(prepared)
    assert result.citations == prepared.citations


def test_readers_require_discovery_and_reject_repeat_runs():
    initial = discovered("Smith v. Jones, 347 U.S. 483.")
    with pytest.raises(ValueError, match="Resolve colocations"):
        resolve_case_names(initial)
    with pytest.raises(ValueError, match="Find docket locators"):
        resolve_docket_entries(initial.get_substage(initial.substage_runs[0]))
    read = resolve_docket_entries(resolve_case_names(resolve_colocations(initial)))
    with pytest.raises(ValueError, match="already completed"):
        resolve_case_names(read)
    with pytest.raises(ValueError, match="already completed"):
        resolve_docket_entries(read)
