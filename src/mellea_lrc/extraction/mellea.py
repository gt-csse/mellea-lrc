"""Use Mellea to extract and label."""

from pathlib import Path

import mellea
from mellea.stdlib.sampling.base import RejectionSamplingStrategy
from mellea.backends.model_ids import IBM_GRANITE_4_1_3B

from .base import BaseExtractor
from mellea_lrc.preprocessing import (
    preprocess,
)


class MelleaExtractor(BaseExtractor):
    """Extractor that uses Mellea."""

    _mellea_session = mellea.start_session(backend_name="ollama", model_id=IBM_GRANITE_4_1_3B)

    def __init__(self) -> None:
        """Initialize a Mellea session."""

    @classmethod
    def extract_citations(cls, text: str) -> list:
        """Identify, retrieve, and classify case law citations.

        Args:
        ----
            text: The document as a plain-text string.

        Returns:
        -------
            A list of citations.

        """
        response = cls._mellea_session.instruct(
            "return a list of all case law citations in the document. document: {{text}}",
            user_variables={"text": text},
            requirements=[
                "return a list of case law citations",
                "place each citation on a line",
                "return case citations (i.e., Doe vs. Roe 452 U.S. 4722 (1978)) with original format",
                "only include the case citations",
                "write the names of citions exactly how exactly how they appear in the text.",
                "keep the order of the citations as they appear on the text",
                "include full, short, supra, and id citations",
            ],
            strategy=RejectionSamplingStrategy(loop_budget=4),
        ).value

        return response.split("\n")

    @classmethod
    def resolve_citations(cls, citations: list) -> list:
        """Group citations with the same reference, e.g., document, bried."""
        return super().resolve_citations(citations)

    @classmethod
    def extract_structured_text(cls, file_path: Path | str) -> str:
        """Convert Unstructured file to structred data (e.g., PDF to markdown)."""
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            message = f"{file_path} doesn't exists or isn't a file.\n"
            raise Exception(message)
        document = preprocess(file_path)
        return document.text
