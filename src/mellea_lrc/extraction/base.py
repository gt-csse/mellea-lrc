"""Extract case citations and label them."""

import abc


class BaseExtractor(abc.ABC):
    """Abstract base class for extractors."""

    @classmethod
    @abc.abstractmethod
    def get_citations(cls, text: str) -> list:
        """Identify, retrieve, and classify case law citations."""

    @classmethod
    @abc.abstractmethod
    def resolve_citations(cls, citations: list) -> list:
        """Group citations with the same reference, e.g., document, bried."""
