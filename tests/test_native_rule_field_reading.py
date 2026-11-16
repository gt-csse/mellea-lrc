"""Native court and date reading stays bounded by colocation context."""

from __future__ import annotations

import pytest

from mellea_lrc.extraction.find_docket_locators import find_docket_locators
from mellea_lrc.extraction.find_full_reporter_locators import find_full_reporter_locators
from mellea_lrc.extraction.resolve_colocations import resolve_colocations
from mellea_lrc.extraction.resolve_courts import resolve_courts
from mellea_lrc.extraction.resolve_dates import resolve_dates
from mellea_lrc.model import Document, PreprocessedDocument, PreprocessingMetadata, SourceMetadata
from mellea_lrc.model.citations.fields.base import NotStatedField


def _document(text: str) -> Document:
    return Document.from_preprocessed(
        PreprocessedDocument(
            text=text,
            source_metadata=SourceMetadata(path="filing.txt"),
            preprocessing_metadata=PreprocessingMetadata(),
        )
    )


def _discovered(text: str) -> Document:
    return resolve_colocations(find_docket_locators(find_full_reporter_locators(_document(text))))


def test_group_members_share_one_bounded_parenthetical_for_court_and_date() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. Feb. 2, 2024)."
    grouped = _discovered(text)
    resolved = resolve_dates(resolve_courts(grouped))

    assert len(resolved.colocations) == 1
    for citation in resolved.citations:
        court = citation.court[-1]
        date = citation.date[-1]
        assert court.quote == "D. Ariz."
        assert court.get_normalized().id == "azd"
        assert date.quote == "Feb. 2, 2024"
        assert (date.get_normalized().year, date.get_normalized().month, date.get_normalized().day) == (
            2024,
            2,
            2,
        )
        assert citation.nodes[-2].substage.endswith("resolve_courts")
        assert citation.nodes[-1].substage.endswith("resolve_dates")
    assert resolved.get_substage("grow_roots.read_root_fields.resolve_colocations") == grouped


def test_date_only_parenthetical_infers_supreme_court_and_records_written_date() -> None:
    resolved = resolve_dates(resolve_courts(_discovered("347 U.S. 483 (1954).")))
    citation = resolved.citations[0]

    assert citation.court[-1].quote is None
    assert citation.court[-1].get_normalized().id == "scotus"
    assert citation.date[-1].get_normalized().year == 1954


def test_explicit_court_without_date_records_date_not_stated() -> None:
    resolved = resolve_dates(resolve_courts(_discovered("No. 1:24-cv-00123 (D. Ariz.).")))
    citation = resolved.citations[0]

    assert citation.court[-1].get_normalized().id == "azd"
    assert isinstance(citation.date[-1], NotStatedField)


@pytest.mark.parametrize(
    "text",
    (
        "123 F.3d 456 (explaining events in 2001).",
        "123 F.3d 456\n\nThe statute was enacted (2024).",
        "123 F.3d 456 (D. Mass. 2001, amended 2002).",
    ),
)
def test_prose_truncated_or_ambiguous_context_stays_unread(text: str) -> None:
    resolved = resolve_dates(resolve_courts(_discovered(text)))
    citation = resolved.citations[0]

    assert citation.court == ()
    assert citation.date == ()


def test_citation_shaped_unrecognized_court_is_preserved_as_a_failed_reading() -> None:
    resolved = resolve_dates(resolve_courts(_discovered("123 F.3d 456 (15th Cir. 2001).")))
    citation = resolved.citations[0]

    assert citation.court[-1].quote == "15th Cir."
    assert citation.court[-1].normalizable is False
    assert citation.date[-1].get_normalized().year == 2001


def test_stages_require_colocation_and_preserve_existing_field_history() -> None:
    raw = find_full_reporter_locators(_document("347 U.S. 483 (1954)."))
    with pytest.raises(ValueError, match="Resolve colocations"):
        resolve_courts(raw)
    with pytest.raises(ValueError, match="Resolve colocations"):
        resolve_dates(raw)

    grouped = resolve_colocations(raw)
    first = grouped.citations[0].record("manual.court").with_inferred_court("scotus")
    prepared = grouped.replace_citation(first).complete_substage("manual.court")
    resolved = resolve_courts(prepared)

    assert resolved.citations[0].court == first.court
    assert resolved.citations[0].nodes == first.nodes
