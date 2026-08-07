"""Abstract base for the Mellea extraction strategies."""

from __future__ import annotations

import abc
import os
import re
import uuid
from dataclasses import dataclass

import mellea
from dotenv import find_dotenv, load_dotenv
from mellea import MelleaSession
from mellea.backends.model_ids import IBM_GRANITE_4_1_30B, ModelIdentifier
from rapidfuzz import fuzz

from mellea_lrc.core import (
    SourceFormat,
    SourceMetadata,
    Span,
    UnknownCitation,
)
from mellea_lrc.extraction.types import (
    ExtractedCitation,
    ExtractedDocument,
    ExtractionBackend,
    ExtractionMetadata,
)
from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.preprocessing import PreprocessingBackend, PreprocessingMetadata

DEFAULT_MODEL_ID = IBM_GRANITE_4_1_30B
LOCATOR = re.compile(r"\b\d+\s+[A-Z][A-Za-z.]*(?:\s+[A-Z][A-Za-z.]*)*\s+\d+\b")
TRAILING_YEAR = re.compile(r"\s*\(\d{4}\)\s*$")


@dataclass(frozen=True, slots=True)
class FoundCitation:
    """A citation string located in the document, with its character span."""

    matched_text: str
    span: Span


class MelleaExtractorBase(abc.ABC):
    """Shared Mellea extraction pipeline."""

    def __init__(self, model_id: ModelIdentifier = DEFAULT_MODEL_ID) -> None:
        """Store the model identifier; the session is created lazily on first use."""
        self._model_id = model_id
        self._session: MelleaSession | None = None

    @property
    def model_id(self) -> ModelIdentifier:
        """Return the Granite model this extractor talks to."""
        return self._model_id

    @property
    def session(self) -> MelleaSession:
        """Return an initialized Mellea session (Ollama backend), created on demand."""
        if self._session is None:
            load_dotenv(find_dotenv())
            host = os.environ.get("OLLAMA_HOST")
            if not host:
                msg = (
                    "OLLAMA_HOST is not set. Export it (or add it to a .env on the search "
                    "path) so the extractor knows which Ollama server to reach."
                )
                raise RuntimeError(msg)
            self._session = mellea.start_session(
                backend_name="ollama", model_id=self._model_id, base_url=host
            )
        return self._session

    def _locate_span(self, text: str, matched_text: str, start: int = 0) -> Span | None:
        """Return the first exact span of `matched_text` at or after `start`, else None."""
        match = fuzz.partial_ratio_alignment(matched_text, text)
        if match:
            return Span(match.dest_start, match.dest_end)
        return None

    @abc.abstractmethod
    def _find_citations(self, text: str) -> list[FoundCitation]:
        """Locate the case citations in `text`. Implemented by each strategy."""

    def _to_extracted_citation(self, found: FoundCitation) -> ExtractedCitation:
        """Wrap a located citation in an `ExtractedCitation` (classification pending)."""
        return ExtractedCitation(
            citation_id=uuid.uuid4().hex,
            span=found.span,
            locator_span=found.span,
            matched_text=found.matched_text,
            citation=UnknownCitation(),
        )

    def extract_citations(self, text: str) -> ExtractedDocument:
        """Identify and locate case-law citations, returning an `ExtractedDocument`."""
        found = self._find_citations(text)
        citations = tuple(self._to_extracted_citation(item) for item in found)
        return ExtractedDocument(
            text=text,
            source_metadata=SourceMetadata(format=SourceFormat.TEXT),
            preprocessing_metadata=PreprocessingMetadata(backend=PreprocessingBackend.PLAIN_TEXT),
            citations=citations,
            extraction_metadata=ExtractionMetadata(backend=ExtractionBackend.MELLEA),
        )

    def resolve_citations(self, citations: list) -> list:
        """Group citations sharing a reference. Not yet implemented; returns them unchanged."""
        return citations
