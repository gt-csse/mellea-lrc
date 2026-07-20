"""Extract case citations and label them."""

import abc

from mellea_lrc.extraction.types import ExtractedDocument


class BaseExtractor(abc.ABC):
    """Abstract base class for extractors."""

    @abc.abstractmethod
    def extract_citations(self, text: str) -> ExtractedDocument:
        """Identify, retrieve, and classify case law citations."""

    @abc.abstractmethod
    def resolve_citations(self, citations: list) -> list:
        """Group citations with the same reference, e.g., document, bried."""
