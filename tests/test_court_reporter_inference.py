"""Stage 7 infers a court only when the reporter identifies one court."""

import pytest

from mellea_lrc.api import (
    find_docket_locators,
    find_full_reporter_locators,
    resolve_colocations,
    resolve_courts,
    resolve_dates,
)
from mellea_lrc.model import Document, FullReporterCitation


def _through_courts(locator: str, year: int = 2000) -> Document:
    document = Document.from_source(f"See {locator} ({year}).")
    document = find_full_reporter_locators(document)
    document = find_docket_locators(document)
    document = resolve_colocations(document)
    return resolve_courts(document)


@pytest.mark.parametrize(
    ("locator", "court_id"),
    [
        ("95 L. Ed. 2d 123", "scotus"),
        ("58 N.Y.2d 916", "ny"),
        ("31 N.C. App. 216", "ncctapp"),
        ("12 Cal. App. 5th 123", "calctapp"),
        ("13 How. 363", "scotus"),
    ],
)
def test_unique_reporter_infers_court_at_serialized_stage_7_checkpoint(locator: str, court_id: str) -> None:
    checkpoint = _through_courts(locator)
    assert checkpoint.stage_runs[-1] == "7_courts"
    assert len(checkpoint.citations) == 1
    citation = checkpoint.citations[0]
    assert isinstance(citation, FullReporterCitation)
    assert citation.locator[-1].quote == locator
    assert len(citation.court) == 1

    court = citation.court[-1]
    assert court.quote is None
    assert court.span is None
    assert court.get_normalized().id == court_id
    assert citation.nodes[-1].stage == "7_courts"
    assert court.node_id == citation.nodes[-1].id

    later = resolve_dates(checkpoint)
    restored = Document.model_validate_json(later.model_dump_json())
    assert restored.get_stage("7_courts") == checkpoint


@pytest.mark.parametrize(
    ("locator", "year"),
    [
        ("100 F.3d 123", 2000),
        ("123 P.3d 456", 2000),
        ("99 B.R. 123", 2000),
        ("5 Cranch 137", 1809),
    ],
)
def test_shared_or_ambiguous_reporter_does_not_infer_court(locator: str, year: int) -> None:
    checkpoint = _through_courts(locator, year)
    assert checkpoint.stage_runs[-1] == "7_courts"
    assert len(checkpoint.citations) == 1
    citation = checkpoint.citations[0]
    assert isinstance(citation, FullReporterCitation)
    assert citation.locator[-1].quote == locator
    assert citation.court == ()

    restored = Document.model_validate_json(checkpoint.model_dump_json())
    assert restored.get_stage("7_courts") == checkpoint
