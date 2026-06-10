"""Use Eyecite to extract and label."""

from .base import BaseExtractor


class EyeciteExtractor(BaseExtractor):
    """Extractor that uses Mellea."""

    @classmethod
    def get_citations(cls, text: str) -> list:
        """Identify, retrieve, and classify case law citations.

        Args:
        ----
            text: The document as a plain-text string.

        Returns:
        -------
            A list of citations.

        """
        return super().get_citations(text)

    @classmethod
    def resolve_citations(cls, citations: list) -> list:
        """Group citations with the same reference, e.g., document, bried."""
        return super().resolve_citations(citations)
