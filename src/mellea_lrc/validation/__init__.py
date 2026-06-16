"""Citation validation against external legal-data services."""

from mellea_lrc.courtlistener.remote import CourtListenerAccessClient, CourtListenerAccessConfig
from mellea_lrc.validation.pipeline import validate_extraction
from mellea_lrc.validation.types import (
    CitationValidation,
    DocumentValidation,
    ValidationStatus,
)

__all__ = [
    "CitationValidation",
    "CourtListenerAccessClient",
    "CourtListenerAccessConfig",
    "DocumentValidation",
    "ValidationStatus",
    "validate_extraction",
]
