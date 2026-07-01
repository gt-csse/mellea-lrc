"""Citation validation against external legal-data services."""

from mellea_lrc.courtlistener.remote import CourtListenerAccessClient, CourtListenerAccessConfig
from mellea_lrc.validation.pipeline import run_validation
from mellea_lrc.validation.types import (
    AmbiguousCitationValidation,
    CitationValidation,
    CourtResolutionSource,
    CourtResolutionTrace,
    FoundCitationValidation,
    InvalidCitationValidation,
    LookupFailedCitationValidation,
    NotFoundCitationValidation,
    SkippedCitationValidation,
    ThrottledCitationValidation,
    ValidatedDocument,
    ValidationClientMode,
    ValidationMetadata,
    ValidationStatus,
)

__all__ = [
    "AmbiguousCitationValidation",
    "CitationValidation",
    "CourtListenerAccessClient",
    "CourtListenerAccessConfig",
    "CourtResolutionSource",
    "CourtResolutionTrace",
    "FoundCitationValidation",
    "InvalidCitationValidation",
    "LookupFailedCitationValidation",
    "NotFoundCitationValidation",
    "SkippedCitationValidation",
    "ThrottledCitationValidation",
    "ValidatedDocument",
    "ValidationClientMode",
    "ValidationMetadata",
    "ValidationStatus",
    "run_validation",
]
