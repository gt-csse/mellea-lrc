"""Use Mellea to extract and label."""

import uuid
import re
from pathlib import Path
from typing import Literal

import mellea
from mellea import MelleaSession
from mellea.backends.model_ids import IBM_GRANITE_4_1_3B
from dotenv import load_dotenv, find_dotenv

from mellea_lrc.extraction.base import BaseExtractor
from mellea_lrc.preprocessing import (
    preprocess,
    PreprocessingBackend,
    PreprocessingMetadata,
)
from mellea_lrc.core import (
    Span,
    CitationKind,
    FullLawCitation,
    ShortCaseCitation,
    SupraCitation,
    IdCitation,
    FullJournalCitation,
    UnknownCitation,
    ReferenceCitation,
    FullCaseCitation,
    CanonicalCitation,
    SourceFormat,
    SourceMetadata,
)

from mellea_lrc.extraction.types import (
    ExtractedDocument,
    ExtractedCitation,
    ExtractionBackend,
    ExtractionMetadata,
)


class MelleaExtractor(BaseExtractor):
    """Extractor that uses Mellea."""

    def __init__(
        self,
        model_id=IBM_GRANITE_4_1_3B,  # noqa: ANN001
    ) -> None:
        """Initialize a Mellea session."""
        self._model_id = model_id
        self._session = None

    @property
    def session(self) -> MelleaSession:
        """Return an initialized Mellea session."""
        if self._session is None:
            load_dotenv(find_dotenv())
            self._session = mellea.start_session(backend_name="ollama", model_id=self._model_id)
        return self._session

    def _locate_span(
        self,
        text: str,
        matched_text: str,
    ) -> Span | None:
        """Return the span of the citation if found, else none."""
        pattern = matched_text.encode("unicode_escape").decode("utf-8")
        match = re.search(pattern, text)
        if match:
            return Span(match.start(), match.end())
        return None

    def _assemble_canonical_citation(self, kind: str, **kwargs) -> CanonicalCitation:
        """Return an assembled CanonicalCitation class.

        Args:
        ----
            kind: The type of citation (e.g., FullCaseCitation, ShortCaseCitation, etc.)
            kwargs: All of the parameters for the citation (e.g., plaintiff, defendant, etc.)

        Returns:
        -------
            An assmbled CanonicalCitation class.

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

    def _assemble_extractor_citation(self, text: str, **kwargs) -> ExtractedCitation:  # noqa: ARG002
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
        start_span = kwargs.get("start_span", -1)
        end_span = kwargs.get("end_span", -1)
        citation = kwargs.get("citation", UnknownCitation())
        citation_id = uuid.uuid4().hex
        span = Span(start_span, end_span)
        return ExtractedCitation(
            citation_id=citation_id, span=span, matched_text=matched_text, citation=citation
        )

    def _load_data(self, document_path: Path) -> str:
        """Return the text content."""
        file_content = ""
        if document_path.exists() and document_path.is_file():
            with document_path.open("r") as file:
                file_content = file.read()
        return file_content

    def extract_citations(self, text: str) -> ExtractedDocument:
        """Identify, retrieve, and classify case law citations.

        Args:
        ----
            text: The document as a plain-text string.

        Returns:
        -------
            A list of citations.

        """
        # Fill in the `SourceMetadata`
        source_metadata = SourceMetadata(format=SourceFormat.PDF)
        # Fill in the `PreprocesingMetadata`
        preprocessing_metadata = PreprocessingMetadata(backend=PreprocessingBackend.DOCLING)
        # Fill in the `Citations`
        citations: list[ExtractedCitation] = []
        unfound: list = []
        raw_citations = self._divide_and_conquer_strategy(text)
        for raw in raw_citations:
            matched_text = raw.strip()
            if not matched_text:  # Empty
                continue
            span = self._locate_span(text, matched_text)
            if span is None:
                unfound.append(matched_text)
            citation = self._assemble_canonical_citation(CitationKind.UNKNOWN)  # TODO: Implement classifier
            arguments = {"citation": citation, "matched_text": matched_text}
            citations.append(self._assemble_extractor_citation(text, **arguments))
        # Fill in the `ExtractedMetadata`
        extraction_metadata = ExtractionMetadata(backend=ExtractionBackend.MELLEA)

        return ExtractedDocument(
            text=text,
            preprocessing_metadata=preprocessing_metadata,
            source_metadata=source_metadata,
            citations=tuple(citations),
            extraction_metadata=extraction_metadata,
        )

    def resolve_citations(self, citations: list) -> list:
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
                "write the names of citions exactly how exactly how they appear in the text.",
                "keep the order of the citations as they appear on the text",
                "include full, short, supra, and id citations",
            ],
            strategy=None,
        ).value

        return response.splitlines()

    def _divide_and_conquer_strategy(self, text: str) -> list[str]:  # noqa: ARG002
        """Breakdown the input and extract the citations."""
        return []

    def _seperate_by_paragraph(self) -> list[str]:
        """Return a list of all paragraphs."""

    @mellea.generative
    def _check_for_citation(self) -> Literal["yes", "no"]:
        """Return `yes` if there is a case-law citation in the paragraph. Otherwise, return `no`."""

    @mellea.generative
    def _return_citation(self) -> str:
        """Return the citation within the paragrah."""

    @mellea.generative
    def _classify_citation(
        self, matched_text: str
    ) -> Literal[
        CitationKind.FULL_CASE,
        CitationKind.SHORT_CASE,
        CitationKind.SUPRA,
        CitationKind.ID,
        CitationKind.UNKNOWN,
    ]:
        """Return the citation type from the given options."""
