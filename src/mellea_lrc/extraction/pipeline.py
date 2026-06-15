"""Orchestrate baseline extraction and optional augmentation."""

from collections.abc import Sequence

from mellea_lrc.extraction.eyecite import extract_baseline
from mellea_lrc.extraction.protocols import ExtractionAugmenter
from mellea_lrc.extraction.result import DocumentExtraction
from mellea_lrc.preprocessing.document import PreprocessedDocument
from mellea_lrc.preprocessing.plain_text import preprocess_plain_text_from_string


def run_extraction(
    preprocessed: PreprocessedDocument,
    *,
    augmenters: Sequence[ExtractionAugmenter] = (),
) -> DocumentExtraction:
    """Run extraction on a preprocessed document."""
    result = extract_baseline(preprocessed)
    for augmenter in augmenters:
        result = augmenter.augment(preprocessed, result)
    return result


def run_extraction_from_text(
    text: str,
    *,
    source_path: str | None = None,
    augmenters: Sequence[ExtractionAugmenter] = (),
) -> DocumentExtraction:
    """Run the extraction pipeline on raw Layer 2 text."""
    preprocessed = preprocess_plain_text_from_string(text, source_path=source_path)
    return run_extraction(preprocessed, augmenters=augmenters)
