"""Validation result types.

Validation is a deterministic (no-LLM) stage that resolves one citation against
CourtListener. Each outcome is one variant of the ``CitationValidation``
discriminated union. Found validation carries one structured
``CourtResolutionTrace``; ambiguous validation carries one trace per candidate.
Each records how the CL court was resolved (docket GET, cache hit, etc.). Validation only
retrieves data — it never compares the resolved court against the citation
court from extraction; that comparison belongs to the assessment stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.extraction.types import ExtractedDocument

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.types import CourtListenerCitationRecord, ValidationFailureDetail

ValidationClientMode: TypeAlias = Literal["deployed", "sdk", "custom"]
HTTP_OK = 200
EXPECTED_CASE_NAME_PROBES = 2


class ValidationStatus(str, Enum):
    """Canonical validation outcomes for citation existence checks."""

    FOUND = "found"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    INVALID = "invalid"
    THROTTLED = "throttled"
    LOOKUP_FAILED = "lookup_failed"
    SKIPPED = "skipped"


class CourtResolutionSource(str, Enum):
    """How the CourtListener-side court was resolved for a found citation."""

    CLUSTER_PROVIDED = "cluster_provided"
    DOCKET_LOOKUP = "docket_lookup"
    NO_DOCKET_ID = "no_docket_id"
    DOCKET_LOOKUP_FAILED = "docket_lookup_failed"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True, slots=True)
class CourtResolutionTrace:
    """Traced court-resolution work for one found validation.

    Records how the CourtListener-side court slug was obtained (docket GET,
    cluster payload, or no resolution). Validation never compares this against
    the citation court from extraction; that comparison is the assessment
    stage's job. This is the per-citation "validation window" trace that
    mirrors the assessment stage's case-name followup provenance, restricted
    to deterministic (non-LLM) resolution.
    """

    courtlistener_court_id: str | None
    resolved_via: CourtResolutionSource
    docket_id: str | None
    docket_url: str | None
    cached: bool
    error_message: str | None


class CaseNameSearchStatus(str, Enum):
    """Whether/why a case-name search ran for a not-found citation."""

    SEARCHED = "searched"
    PARTIAL = "partial"
    SKIPPED_NO_CASE_NAME = "skipped_no_case_name"
    SKIPPED_PARTIAL_CASE_NAME = "skipped_partial_case_name"
    SEARCH_UNAVAILABLE = "search_unavailable"
    SEARCH_FAILED = "search_failed"
    NOT_ATTEMPTED = "not_attempted"


class CaseNameSearchCorpus(str, Enum):
    """CourtListener corpus searched by one case-name probe."""

    OPINIONS = "o"
    RECAP = "r"


@dataclass(frozen=True, slots=True)
class CaseNameSearchProbe:
    """Independent result from searching one CourtListener corpus."""

    corpus: CaseNameSearchCorpus
    status: CaseNameSearchStatus
    http_status: int | None = None
    case_count: int | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.status is CaseNameSearchStatus.SEARCHED:
            if self.http_status != HTTP_OK:
                msg = "A searched case-name probe requires HTTP 200"
                raise ValueError(msg)
            if self.case_count is None or isinstance(self.case_count, bool) or self.case_count < 0:
                msg = "A searched case-name probe requires a non-negative case_count"
                raise ValueError(msg)
        elif self.case_count is not None:
            msg = "Only a searched case-name probe may carry case_count"
            raise ValueError(msg)
        if self.status not in {
            CaseNameSearchStatus.SEARCHED,
            CaseNameSearchStatus.SEARCH_FAILED,
            CaseNameSearchStatus.SEARCH_UNAVAILABLE,
        }:
            msg = "A probe requires a corpus-level search status"
            raise ValueError(msg)
        if self.status is CaseNameSearchStatus.SEARCH_FAILED and self.http_status == HTTP_OK:
            msg = "A failed case-name probe cannot carry HTTP 200"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CaseNameSearchTrace:
    """Two-corpus case-name search attached to a not-found citation.

    When a reporter lookup 404s but both parties were extracted, we query
    The same engineered query is sent independently to opinions and RECAP.
    Counts remain corpus-scoped: validation does not combine, rank, or interpret
    them, and expresses no opinion about whether a result is the cited case.
    """

    status: CaseNameSearchStatus = CaseNameSearchStatus.NOT_ATTEMPTED
    query: str | None = None
    probes: tuple[CaseNameSearchProbe, ...] = ()

    def __post_init__(self) -> None:
        corpora = tuple(probe.corpus for probe in self.probes)
        if len(corpora) != len(set(corpora)):
            msg = "Case-name trace probe corpora must be unique"
            raise ValueError(msg)
        successes = sum(probe.status is CaseNameSearchStatus.SEARCHED for probe in self.probes)
        if self.status is CaseNameSearchStatus.SEARCHED and successes != EXPECTED_CASE_NAME_PROBES:
            msg = "A searched trace requires two successful corpus probes"
            raise ValueError(msg)
        if self.status is CaseNameSearchStatus.PARTIAL and not (0 < successes < EXPECTED_CASE_NAME_PROBES):
            msg = "A partial trace requires one successful corpus probe"
            raise ValueError(msg)
        if self.status is CaseNameSearchStatus.SEARCH_FAILED and successes:
            msg = "A failed trace cannot contain a successful probe"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ValidationMetadata:
    """Provenance for the validation stage."""

    client_mode: ValidationClientMode
    source: str
    duration_ms: float | None = None


@dataclass(frozen=True, slots=True)
class RetrievedCandidate:
    """One retrieved record with stable identity and candidate-scoped provenance."""

    candidate_id: str
    record: CourtListenerCitationRecord
    court_resolution: CourtResolutionTrace

    def __post_init__(self) -> None:
        if not self.candidate_id:
            msg = "RetrievedCandidate.candidate_id must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class FoundCitationValidation:
    """A locator that retrieved exactly one candidate from CourtListener."""

    status: ClassVar[ValidationStatus] = ValidationStatus.FOUND
    citation_id: str
    locator: str
    source: str
    lookup_status: int
    lookup_cache: str | None
    lookup_key: str | None
    candidate: RetrievedCandidate
    extra_data: ExtraData = field(default_factory=ExtraData)

    @property
    def case_names(self) -> tuple[str, ...]:
        """Return the retrieved candidate's case name when present."""
        return (self.candidate.record.case_name,) if self.candidate.record.case_name else ()


@dataclass(frozen=True, slots=True)
class AmbiguousCitationValidation:
    """A citation that resolves to multiple independently enriched candidates."""

    status: ClassVar[ValidationStatus] = ValidationStatus.AMBIGUOUS
    citation_id: str
    locator: str
    source: str
    lookup_status: int
    lookup_cache: str | None
    lookup_key: str | None
    candidates: tuple[RetrievedCandidate, ...]
    extra_data: ExtraData = field(default_factory=ExtraData)

    def __post_init__(self) -> None:
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            msg = "Ambiguous candidate identifiers must be unique"
            raise ValueError(msg)

    @property
    def case_names(self) -> tuple[str, ...]:
        """Return non-empty case names from retrieved candidates."""
        return tuple(
            candidate.record.case_name for candidate in self.candidates if candidate.record.case_name
        )


@dataclass(frozen=True, slots=True)
class NotFoundCitationValidation:
    """A full citation that CourtListener does not have."""

    status: ClassVar[ValidationStatus] = ValidationStatus.NOT_FOUND
    citation_id: str
    locator: str
    source: str
    lookup_status: int
    lookup_cache: str | None
    lookup_key: str | None
    candidate_search: CaseNameSearchTrace = field(default_factory=CaseNameSearchTrace)
    extra_data: ExtraData = field(default_factory=ExtraData)

    @property
    def case_names(self) -> tuple[str, ...]:
        """Not-found citations have no candidate case names."""
        return ()


@dataclass(frozen=True, slots=True)
class InvalidCitationValidation:
    """A citation missing the volume, reporter, or page required to look it up."""

    status: ClassVar[ValidationStatus] = ValidationStatus.INVALID
    citation_id: str
    source: str

    @property
    def case_names(self) -> tuple[str, ...]:
        """Invalid citations have no candidate case names."""
        return ()


@dataclass(frozen=True, slots=True)
class ThrottledCitationValidation:
    """A lookup that CourtListener rejected with a retryable throttle response."""

    status: ClassVar[ValidationStatus] = ValidationStatus.THROTTLED
    citation_id: str
    locator: str
    source: str
    lookup_status: int
    lookup_cache: str | None
    lookup_key: str | None
    error_message: str | None
    failure_detail: ValidationFailureDetail | None
    extra_data: ExtraData = field(default_factory=ExtraData)

    @property
    def case_names(self) -> tuple[str, ...]:
        """Throttled lookups have no candidate case names."""
        return ()


@dataclass(frozen=True, slots=True)
class LookupFailedCitationValidation:
    """A lookup that failed for a non-throttle reason (network, parse, upstream 5xx)."""

    status: ClassVar[ValidationStatus] = ValidationStatus.LOOKUP_FAILED
    citation_id: str
    locator: str
    source: str
    lookup_status: int | None
    lookup_cache: str | None
    lookup_key: str | None
    error_message: str | None
    failure_detail: ValidationFailureDetail | None
    extra_data: ExtraData = field(default_factory=ExtraData)

    @property
    def case_names(self) -> tuple[str, ...]:
        """Failed lookups have no candidate case names."""
        return ()


@dataclass(frozen=True, slots=True)
class SkippedCitationValidation:
    """A citation intentionally excluded from validation (e.g. not a FullCaseCitation)."""

    status: ClassVar[ValidationStatus] = ValidationStatus.SKIPPED
    citation_id: str
    source: str

    @property
    def case_names(self) -> tuple[str, ...]:
        """Skipped citations have no candidate case names."""
        return ()


CitationValidation: TypeAlias = (
    FoundCitationValidation
    | AmbiguousCitationValidation
    | NotFoundCitationValidation
    | InvalidCitationValidation
    | ThrottledCitationValidation
    | LookupFailedCitationValidation
    | SkippedCitationValidation
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedDocument(ExtractedDocument):
    """An extracted document with one validation outcome per citation."""

    validations: tuple[CitationValidation, ...]
    validation_metadata: ValidationMetadata

    @property
    def found(self) -> tuple[FoundCitationValidation, ...]:
        """Return citations found by the validation source."""
        return tuple(item for item in self.validations if isinstance(item, FoundCitationValidation))

    def __post_init__(self) -> None:
        ExtractedDocument.__post_init__(self)
        citation_ids = {item.citation_id for item in self.citations}
        validation_ids = [item.citation_id for item in self.validations]
        if any(not validation_id for validation_id in validation_ids):
            msg = "Citation validation identifiers must not be empty"
            raise ValueError(msg)
        if len(validation_ids) != len(set(validation_ids)):
            msg = "Citation validation identifiers must be unique within a document"
            raise ValueError(msg)
        if set(validation_ids) != citation_ids:
            msg = "Citation validation identifiers must exactly match extracted citation identifiers"
            raise ValueError(msg)
