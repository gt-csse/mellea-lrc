"""Protocols for extending the extraction pipeline."""

from typing import Protocol

from mellea_lrc.extraction.result import DocumentExtraction
from mellea_lrc.preprocessing.document import PreprocessedDocument


class ExtractionAugmenter(Protocol):
    """Augment a baseline extraction (e.g. LLM span recovery or field repair)."""

    def augment(
        self,
        preprocessed: PreprocessedDocument,
        baseline: DocumentExtraction,
    ) -> DocumentExtraction:
        """Return an improved extraction built on top of the baseline."""
