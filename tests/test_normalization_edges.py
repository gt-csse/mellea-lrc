"""Invalid citation readings must fail at the point they are normalized."""

import pytest

from mellea_lrc.extraction import grow_roots
from mellea_lrc.model import CitationDate, Document, FullDocketCitation, Span, latest


@pytest.mark.parametrize(
    ("docket_entry", "include_entry_span"),
    [(None, True), ("10-1", False)],
)
def test_docket_entry_and_span_must_be_supplied_together(
    docket_entry: str | None, include_entry_span: bool
) -> None:
    source = "Doc. 10-1, Case No. 1:24-cv-00123."
    locator = "Case No. 1:24-cv-00123"
    number = "1:24-cv-00123"
    entry_span = Span(0, len("Doc. 10-1")) if include_entry_span else None

    with pytest.raises(ValueError, match="Docket entry and its source span must be supplied together"):
        FullDocketCitation.from_locator(
            citation_id="docket:0",
            stage="docket_locators",
            source=source,
            span=Span(source.index(locator), source.index(locator) + len(locator)),
            docket_number=number,
            docket_number_span=Span(source.index(number), source.index(number) + len(number)),
            docket_entry=docket_entry,
            docket_entry_span=entry_span,
        )


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
        "See Case No. 1:24-cv-00123 (3d Cir. 2024).",
    ],
)
def test_unresolved_written_court_raises_before_inference(source: str) -> None:
    with pytest.raises(ValueError, match="Cannot normalize written court"):
        grow_roots(Document.from_source(source))


def test_absent_written_court_can_still_be_inferred_from_reporter() -> None:
    document = grow_roots(Document.from_source("See 347 U.S. 483 (1954)."))
    assert latest(document.citations[0].court) == "scotus"
