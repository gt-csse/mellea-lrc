"""Full locator pinpoints use local suffixes and replayable decisions."""

import pytest

from mellea_lrc.config.extraction import ExtractionRules
from mellea_lrc.extraction import find_full_reporter_locators, resolve_colocations
from mellea_lrc.extraction.resolve_pin_cites import SUBSTAGE, resolve_pin_cites
from mellea_lrc.model import Document, PreprocessedDocument, PreprocessingMetadata, SourceMetadata
from mellea_lrc.model.citations.fields.base import NotStatedField


def grouped(text: str) -> Document:
    source = PreprocessedDocument(
        text=text,
        source_metadata=SourceMetadata(path="filing.txt"),
        preprocessing_metadata=PreprocessingMetadata(),
    )
    return resolve_colocations(find_full_reporter_locators(Document.from_preprocessed(source)))


def test_parallel_reporters_keep_their_own_pinpoints():
    original = grouped("Alpha v. Beta, 123 F.3d 456, 459–60, 789 F.2d 10, 12 (2d Cir. 2000).")
    resolved = resolve_pin_cites(original)
    assert len(resolved.colocations) == 1
    assert [citation.pin_cite[-1].quote for citation in resolved.citations] == ["459–60", "12"]
    assert [citation.get_pin_cite()[0].last for citation in resolved.citations] == [460, 12]
    restored = Document.model_validate_json(resolved.model_dump_json())
    assert restored == resolved
    assert restored.get_substage(original.substage_runs[-1]) == original
    assert restored.get_substage(SUBSTAGE) == resolved
    with pytest.raises(ValueError, match="already completed"):
        resolve_pin_cites(restored)


def test_pinpoint_absence_and_an_unread_truncated_suffix_are_distinct():
    original = grouped("347 U.S. 483 (1954). Later 556 U.S. 662, 678 (2009).")
    resolved = resolve_pin_cites(original, ExtractionRules(post_locator_window=4))
    assert isinstance(resolved.citations[0].pin_cite[-1], NotStatedField)
    assert resolved.citations[1].pin_cite is None


@pytest.mark.parametrize("written", ["459 n. X", "459 through 460"])
def test_malformed_pinpoint_is_retained_without_a_partial_normalization(written):
    resolved = resolve_pin_cites(grouped(f"123 F.3d 456, {written} (2000)."))
    citation = resolved.citations[0]
    assert citation.pin_cite[-1].quote == written
    assert not citation.pin_cite[-1].normalizable


def test_pinpoints_do_not_include_numbers_from_the_next_paragraph():
    resolved = resolve_pin_cites(grouped("123 F.3d 456, 459,\n\n2. Argument"))
    assert resolved.citations[0].pin_cite[-1].quote == "459"
    resolved = resolve_pin_cites(grouped("123 F.3d 456\n\n2. Argument"))
    assert isinstance(resolved.citations[0].pin_cite[-1], NotStatedField)


def test_clipped_footnote_does_not_turn_into_a_page_only_reading():
    resolved = resolve_pin_cites(
        grouped("123 F.3d 456, 459 n. 7 (2000)."), ExtractionRules(post_locator_window=9)
    )
    assert resolved.citations[0].pin_cite is None
