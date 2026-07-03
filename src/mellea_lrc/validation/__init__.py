"""Citation validation against external legal-data services."""

from mellea_lrc.courtlistener.remote import CourtListenerAccessClient, CourtListenerAccessConfig
from mellea_lrc.validation.pipeline import run_validation
from mellea_lrc.validation.types import (
    AmbiguousCitationValidation,
    CaseNameSearchStatus,
    CaseNameSearchTrace,
    CitationValidation,
    CourtResolutionSource,
    CourtResolutionTrace,
    FoundCitationValidation,
    InvalidCitationValidation,
    LookupFailedCitationValidation,
    NotFoundCitationValidation,
    RetrievedCandidate,
    SkippedCitationValidation,
    ThrottledCitationValidation,
    ValidatedDocument,
    ValidationClientMode,
    ValidationMetadata,
    ValidationStatus,
)

__all__ = [
    "AmbiguousCitationValidation",
    "CaseNameSearchStatus",
    "CaseNameSearchTrace",
    "CitationValidation",
    "CourtListenerAccessClient",
    "CourtListenerAccessConfig",
    "CourtResolutionSource",
    "CourtResolutionTrace",
    "FoundCitationValidation",
    "InvalidCitationValidation",
    "LookupFailedCitationValidation",
    "NotFoundCitationValidation",
    "RetrievedCandidate",
    "SkippedCitationValidation",
    "ThrottledCitationValidation",
    "ValidatedDocument",
    "ValidationClientMode",
    "ValidationMetadata",
    "ValidationStatus",
    "run_validation",
]
