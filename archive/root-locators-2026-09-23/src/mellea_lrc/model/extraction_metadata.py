"""Provenance for the citation reading stored on a document."""

from dataclasses import dataclass
from enum import Enum


class Relaxation(str, Enum):
    """How much whitespace damage a citation may carry and still be found."""

    NONE = "none"
    BOUNDED = "bounded"
    FULL = "full"


class ExtractionBackend(str, Enum):
    """Engine that produced the extracted citations."""

    EYECITE = "eyecite"
    MELLEA = "mellea"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class ExtractionMetadata:
    """Provenance for the extraction stage."""

    backend: ExtractionBackend = ExtractionBackend.EYECITE
    backend_version: str | None = None
    relaxation: Relaxation = Relaxation.FULL
    """Which tokenizer read the text."""
