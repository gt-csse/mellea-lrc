"""Tests for the extraction pipeline."""

from mellea_lrc.extraction import (
    extract_baseline,
    run_extraction,
)
from mellea_lrc.preprocessing import preprocess_plain_text_from_string

SAMPLE_TEXT = "Under Norton v. Shelby County, 118 U.S. 425, 442 (1886), an act confers no rights."


def test_run_extraction_matches_baseline() -> None:
    preprocessed = preprocess_plain_text_from_string(SAMPLE_TEXT)
    baseline = extract_baseline(preprocessed)
    pipelined = run_extraction(preprocessed)

    assert pipelined.text == baseline.text
    assert len(pipelined.citations) == len(baseline.citations)
    assert pipelined.citations[0].citation.kind == baseline.citations[0].citation.kind
