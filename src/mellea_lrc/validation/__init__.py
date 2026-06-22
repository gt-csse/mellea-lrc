"""Citation validation against external legal-data services."""

from mellea_lrc.courtlistener.remote import CourtListenerAccessClient, CourtListenerAccessConfig
from mellea_lrc.validation.pipeline import run_validation
from mellea_lrc.validation.types import (
    CitationValidation,
    DocumentValidation,
    ValidationClientMode,
    ValidationStatus,
)

__all__ = [
    "CitationValidation",
    "CourtListenerAccessClient",
    "CourtListenerAccessConfig",
    "DocumentValidation",
    "ValidationClientMode",
    "ValidationStatus",
    "run_validation",
]
