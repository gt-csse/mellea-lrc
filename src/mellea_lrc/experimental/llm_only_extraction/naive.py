"""Use Mellea to extract and label."""

import re
import uuid
from pathlib import Path

from mellea.backends.model_ids import IBM_GRANITE_4_1_3B, ModelIdentifier

from mellea_lrc.core import (
    CanonicalCitation,
    CitationKind,
    FullCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    IdCitation,
    ReferenceCitation,
    ShortCaseCitation,
    SourceFormat,
    SourceMetadata,
    Span,
    SupraCitation,
    UnknownCitation,
)
from mellea_lrc.experimental.llm_only_extraction.base import FoundCitation, MelleaExtractorBase
from mellea_lrc.extraction.types import (
    ExtractedCitation,
    ExtractedDocument,
    ExtractionBackend,
    ExtractionMetadata,
)
from mellea_lrc.preprocessing import (
    PreprocessingBackend,
    PreprocessingMetadata,
    preprocess,
)


class MelleaNaive(MelleaExtractorBase):
    """Extractor that uses Mellea."""

    def __init__(
        self,
        model_id: ModelIdentifier = IBM_GRANITE_4_1_3B,
    ) -> None:
        """Keep the 3B default; session creation and host policy come from the base."""
        super().__init__(model_id)

    def _assemble_canonical_citation(self, kind: str, **kwargs) -> CanonicalCitation:
        """Return an assembled CanonicalCitation class.

        Args:
        ----
            kind: The type of citation (e.g., FullCaseCitation, ShortCaseCitation, etc.)
            kwargs: All of the parameters for the citation (e.g., plaintiff, defendant, etc.)

        Returns:
        -------
            An assembled CanonicalCitation class.

        """
        mapping = {
            CitationKind(citation_type.kind): citation_type
            for citation_type in (
                FullCaseCitation,
                FullJournalCitation,
                FullLawCitation,
                ShortCaseCitation,
                SupraCitation,
                IdCitation,
                UnknownCitation,
                ReferenceCitation,
            )
        }
        kind = CitationKind(kind)
        return mapping[kind](**kwargs)

    def _assemble_extractor_citation(self, text: str, **kwargs) -> ExtractedCitation:
        """Build and return a ExtractedCitation class.

        Args:
        ----
            text: The original text input (i.e., the legal document as plain text).
            kwargs: All of the arguments for assembling a ExtractedCitation (e.g., span, matched text, etc.).

        Returns:
        -------
            An assembled ExtractedCitation class of the citation.

        """
        matched_text = kwargs.get("matched_text", "")
        start_span = kwargs.get("start_span", 0)
        end_span = kwargs.get("end_span", 0)
        citation = kwargs.get("citation", UnknownCitation())
        citation_id = uuid.uuid4().hex
        span = Span(start_span, end_span)
        locator = "123 U.S. 456"
        locator_start = matched_text.index(locator)
        return ExtractedCitation(
            citation_id=citation_id,
            full_span=span,
            matched_text=matched_text,
            citation=citation,
            locator_span=Span(locator_start, locator_start + len(locator)),
        )

    def _load_data(self, document_path: Path) -> str:
        """Return the text content."""
        file_content = ""
        if document_path.exists() and document_path.is_file():
            with document_path.open("r") as file:
                file_content = file.read()
        else:
            msg = f"The document path doesn't exist or isn't a file: {document_path}"
            raise FileNotFoundError(msg)
        return file_content

    def _find_citations(self, text: str) -> list[FoundCitation]:
        """Identify, retrieve case law citations.

        Args:
        ----
            text: The document as a plain-text string.

        Returns:
        -------
            A list of citations.

        """
        citations: list[FoundCitation] = []
        unfound: list = []
        raw_citations = self._naive_strategy(text)
        for raw in raw_citations:
            matched_text = raw.strip()
            if not matched_text:  # Empty
                continue
            span = self._locate_span(text, matched_text)
            if span is None:
                unfound.append(matched_text)
                continue
            citations.append(FoundCitation(matched_text=matched_text, span=span))
        return citations

    @classmethod
    def extract_structured_text(cls, file_path: Path | str) -> str:
        """Convert Unstructured file to structred data (e.g., PDF to markdown)."""
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            message = f"{file_path} doesn't exists or isn't a file.\n"
            raise FileNotFoundError(message)
        document = preprocess(file_path)
        return document.text

    # -------------------- extraction field ---------- #

    def _naive_strategy(self, text: str) -> list[str]:
        """Return a list of case-law citations, one per line."""
        response = self.session.instruct(
            "return a list of all case law citations in the document. document: {{text}}",
            user_variables={"text": text},
            requirements=[
                "return a list of case law citations",
                "place each citation on a line",
                "return case citations (i.e., Doe vs. Roe 452 U.S. 4722 (1978)) with original format",
                "only include the case citations",
                "write the names of citations exactly how exactly how they appear in the text.",
                "keep the order of the citations as they appear on the text",
                "include full, short, supra, and id citations",
            ],
            strategy=None,
        ).value

        return response.splitlines()
