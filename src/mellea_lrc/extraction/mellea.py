"""Use Mellea to extract and label."""

from pathlib import Path

import mellea
from mellea.stdlib.components.docs.richdocument import RichDocument

from .base import BaseExtractor


class MelleaExtractor(BaseExtractor):
    """Extractor that uses Mellea."""

    def __init__(self) -> None:
        """Initialize a Mellea session."""
        self._mellea_session = mellea.start_session()

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
        citations = []

        return citations

    @classmethod
    def resolve_citations(cls, citations: list) -> list:
        """Group citations with the same reference, e.g., document, bried."""
        return super().resolve_citations(citations)

    @classmethod
    def extract_structured_text(cls, file_path: Path) -> str:
        """Convert Unstructured file to structred data (e.g., PDF to markdown)."""
        if not file_path.exists() or not file_path.is_file():
            message = f"{file_path} doesn't exists or isn't a file.\n"
            raise Exception(message)
