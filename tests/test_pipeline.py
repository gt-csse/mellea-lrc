"""Tests for the extraction pipeline."""

import pytest

from mellea_lrc.extraction import (
    extract_baseline,
    run_extraction,
)
from mellea_lrc.extraction.augmenters import LLMExtractionAugmenter
from mellea_lrc.preprocessing import preprocess_plain_text_from_string

SAMPLE_TEXT = "Under Norton v. Shelby County, 118 U.S. 425, 442 (1886), an act confers no rights."


def test_run_extraction_without_augmenters_matches_baseline() -> None:
    preprocessed = preprocess_plain_text_from_string(SAMPLE_TEXT)
    baseline = extract_baseline(preprocessed)
    pipelined = run_extraction(preprocessed)

    assert pipelined.text == baseline.text
    assert len(pipelined.citations) == len(baseline.citations)
    assert pipelined.citations[0].citation.kind == baseline.citations[0].citation.kind


class NoopAugmenter:
    """Test augmenter that proves the pipeline remains extensible."""

    def __init__(self) -> None:
        self.called = False

    def augment(self, preprocessed, baseline):  # noqa: ANN001, ANN201
        self.called = True
        assert preprocessed.text == SAMPLE_TEXT
        return baseline


def test_run_extraction_applies_augmenters_in_order() -> None:
    preprocessed = preprocess_plain_text_from_string(SAMPLE_TEXT)
    augmenter = NoopAugmenter()

    result = run_extraction(preprocessed, augmenters=(augmenter,))

    assert augmenter.called
    assert result.citations


def test_llm_augmenter_not_implemented() -> None:
    preprocessed = preprocess_plain_text_from_string(SAMPLE_TEXT)
    baseline = extract_baseline(preprocessed)
    augmenter = LLMExtractionAugmenter()

    with pytest.raises(NotImplementedError, match="LLMExtractionAugmenter is not implemented"):
        augmenter.augment(preprocessed, baseline)
