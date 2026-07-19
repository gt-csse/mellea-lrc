"""Retrieval result types.

Retrieval is a deterministic (no-LLM) stage that resolves one citation against
CourtListener. Each outcome is one variant of the ``CitationRetrieval``
discriminated union. Found retrieval carries one structured
``CourtResolutionTrace``; ambiguous retrieval carries one trace per candidate.
Each records how the CL court was resolved (docket GET, cache hit, etc.). Retrieval only
retrieves data — it never compares the resolved court against the citation
court from extraction; that comparison belongs to the assessment stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.jurisdiction_inference.types import InferredDocument

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationRecord

RetrievalClientMode: TypeAlias = Literal["deployed", "sdk", "custom"]
HTTP_OK = 200
EXPECTED_CASE_NAME_PROBES = 2
MAX_CASE_NAME_SEARCH_CANDIDATES = 5


@dataclass(frozen=True, slots=True)
class CourtListenerRequestTrace:
    """Transport and cache metadata for one CourtListener request."""

    http_status: int | None = None
    cache: str | None = None
    key: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalFailureDetail:
    """Transport failure captured by retrieval after a client exception."""

    failure_type: str | None = None
    message: str | None = None
    retryable: bool | None = None
    upstream_status_code: int | None = None
    key: str | None = None
    url: str | None = None
    retry_after_seconds: float | None = None
    extra_data: ExtraData = field(default_factory=ExtraData)


class RetrievalStatus(str, Enum):
    """Canonical retrieval outcomes for citation existence checks."""

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
    """Traced court-resolution work for one found retrieval.

    Records how the CourtListener-side court slug was obtained (docket GET,
    cluster payload, or no resolution). Retrieval never compares this against
    the citation court from extraction; that comparison is the assessment
    stage's job. This is the per-citation "retrieval window" trace that
    mirrors the assessment stage's case-name followup provenance, restricted
    to deterministic (non-LLM) resolution.
    """

    courtlistener_court_id: str | None
    resolved_via: CourtResolutionSource
    docket_id: str | None
    docket_url: str | None
    request_trace: CourtListenerRequestTrace | None = None


class CaseNameSearchStatus(str, Enum):
    """Whether/why a case-name search ran for a not-found citation."""

    SEARCHED = "searched"
    PARTIAL = "partial"
    SKIPPED_NO_CASE_NAME = "skipped_no_case_name"
    SKIPPED_PARTIAL_CASE_NAME = "skipped_partial_case_name"
    SEARCH_UNAVAILABLE = "search_unavailable"
    SEARCH_FAILED = "search_failed"
    NOT_ATTEMPTED = "not_attempted"


class CaseNamePreparationStatus(str, Enum):
    """Whether/how case-name search inputs were prepared."""

    ACCEPTED = "accepted"
    EMPTY = "empty"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


class DateReextractionStatus(str, Enum):
    """Outcome of locator-bound decision-date re-extraction."""

    ACCEPTED = "accepted"
    NO_DATE = "no_date"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


class DecisionDatePrecision(str, Enum):
    """Precision of citation-bound decision-date evidence."""

    COMPLETE_DATE = "complete_date"
    YEAR_ONLY = "year_only"
    NO_DATE = "no_date"


@dataclass(frozen=True, slots=True)
class CaseNameSearchPreparation:
    """Validated party/date evidence and one normalized candidate-search plan.

    LLM-backed retrieval fills this from local document context before the
    locator.
    """

    status: CaseNamePreparationStatus = CaseNamePreparationStatus.NOT_ATTEMPTED
    original_case_name: str | None = None
    plaintiff: str | None = None
    defendant: str | None = None
    prepared_case_name: str | None = None
    extracted_decision_date: str | None = None
    decision_date: str | None = None
    decision_date_basis: str | None = None
    decision_year: str | None = None
    decision_date_precision: DecisionDatePrecision = DecisionDatePrecision.NO_DATE
    date_reextraction_status: DateReextractionStatus = DateReextractionStatus.NOT_ATTEMPTED
    date_error_message: str | None = None
    query_plaintiff: str | None = None
    query_defendant: str | None = None
    query_reason: str | None = None
    court: str | None = None
    locator: str | None = None
    source: str | None = None
    llm_classification: str | None = None
    llm_reason: str | None = None
    error_message: str | None = None


class CaseNameSearchCorpus(str, Enum):
    """CourtListener corpus searched by one case-name probe."""

    OPINIONS = "o"
    RECAP = "r"


class DocketEvidenceStatus(str, Enum):
    """Outcome of expanding one RECAP docket candidate."""

    ENRICHED = "enriched"
    NO_DECISIONAL_DOCUMENTS = "no_decisional_documents"
    SKIPPED_AFTER_CITED_YEAR = "skipped_after_cited_year"
    SKIPPED_PARTY_MISMATCH = "skipped_party_mismatch"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DocketDocumentEvidence:
    """One bounded docket document that may be the cited decision."""

    docket_entry_id: str | None = None
    recap_document_id: str | None = None
    entry_number: str | None = None
    document_number: str | None = None
    date_filed: str | None = None
    entry_description: str | None = None
    document_description: str | None = None
    page_count: int | None = None
    pacer_doc_id: str | None = None
    available: bool = False
    absolute_url: str | None = None
    decisional_cues: tuple[str, ...] = ()
    year_distance: int | None = None


@dataclass(frozen=True, slots=True)
class DocketCandidateEvidence:
    """Canonical docket metadata and ranked decisional-document evidence."""

    status: DocketEvidenceStatus
    docket_request: CourtListenerRequestTrace = field(default_factory=CourtListenerRequestTrace)
    entries_request: CourtListenerRequestTrace = field(default_factory=CourtListenerRequestTrace)
    case_name: str | None = None
    court_id: str | None = None
    docket_number: str | None = None
    date_filed: str | None = None
    date_terminated: str | None = None
    assigned_to: str | None = None
    referred_to: str | None = None
    nature_of_suit: str | None = None
    cause: str | None = None
    jurisdiction_type: str | None = None
    documents: tuple[DocketDocumentEvidence, ...] = ()
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class CaseNameSearchCandidate:
    """Small, stable projection of one CourtListener search result."""

    case_name: str | None = None
    court_id: str | None = None
    date_filed: str | None = None
    docket_number: str | None = None
    cluster_id: str | None = None
    docket_id: str | None = None
    absolute_url: str | None = None
    docket_evidence: DocketCandidateEvidence | None = None


@dataclass(frozen=True, slots=True)
class CaseNameSearchProbe:
    """Independent result from searching one CourtListener corpus."""

    corpus: CaseNameSearchCorpus
    status: CaseNameSearchStatus
    request_trace: CourtListenerRequestTrace
    case_count: int | None = None
    candidates: tuple[CaseNameSearchCandidate, ...] = ()

    def __post_init__(self) -> None:
        if self.status is CaseNameSearchStatus.SEARCHED:
            if self.request_trace.http_status != HTTP_OK:
                msg = "A searched case-name probe requires HTTP 200"
                raise ValueError(msg)
            if self.case_count is None or isinstance(self.case_count, bool) or self.case_count < 0:
                msg = "A searched case-name probe requires a non-negative case_count"
                raise ValueError(msg)
        elif self.case_count is not None:
            msg = "Only a searched case-name probe may carry case_count"
            raise ValueError(msg)
        if self.status is not CaseNameSearchStatus.SEARCHED and self.candidates:
            msg = "Only a searched case-name probe may carry candidates"
            raise ValueError(msg)
        if len(self.candidates) > MAX_CASE_NAME_SEARCH_CANDIDATES:
            msg = "A case-name probe carries at most five candidate summaries"
            raise ValueError(msg)
        if self.status not in {
            CaseNameSearchStatus.SEARCHED,
            CaseNameSearchStatus.SEARCH_FAILED,
            CaseNameSearchStatus.SEARCH_UNAVAILABLE,
        }:
            msg = "A probe requires a corpus-level search status"
            raise ValueError(msg)
        if (
            self.status is CaseNameSearchStatus.SEARCH_FAILED
            and self.request_trace.http_status == HTTP_OK
        ):
            msg = "A failed case-name probe cannot carry HTTP 200"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CaseNameSearchTrace:
    """Two-corpus case-name search attached to a not-found citation.

    When a reporter lookup 404s but both parties were extracted, the traced
    query combines party anchors with the extracted court when available. The
    same engineered query is sent independently to opinions and RECAP.
    Counts remain corpus-scoped: retrieval does not combine, rank, or interpret
    them, and expresses no opinion about whether a result is the cited case.
    """

    status: CaseNameSearchStatus = CaseNameSearchStatus.NOT_ATTEMPTED
    query: str | None = None
    probes: tuple[CaseNameSearchProbe, ...] = ()
    preparation: CaseNameSearchPreparation = field(default_factory=CaseNameSearchPreparation)

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
class RetrievalMetadata:
    """Provenance for the retrieval stage."""

    client_mode: RetrievalClientMode
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
class FoundCitationRetrieval:
    """A locator that retrieved exactly one candidate from CourtListener."""

    status: ClassVar[RetrievalStatus] = RetrievalStatus.FOUND
    citation_id: str
    locator: str
    source: str
    request_trace: CourtListenerRequestTrace
    candidate: RetrievedCandidate
    extra_data: ExtraData = field(default_factory=ExtraData)

    @property
    def case_names(self) -> tuple[str, ...]:
        """Return the retrieved candidate's case name when present."""
        return (self.candidate.record.case_name,) if self.candidate.record.case_name else ()


@dataclass(frozen=True, slots=True)
class AmbiguousCitationRetrieval:
    """A citation that resolves to multiple independently enriched candidates."""

    status: ClassVar[RetrievalStatus] = RetrievalStatus.AMBIGUOUS
    citation_id: str
    locator: str
    source: str
    request_trace: CourtListenerRequestTrace
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
class NotFoundCitationRetrieval:
    """A full citation that CourtListener does not have."""

    status: ClassVar[RetrievalStatus] = RetrievalStatus.NOT_FOUND
    citation_id: str
    locator: str
    source: str
    request_trace: CourtListenerRequestTrace
    candidate_search: CaseNameSearchTrace = field(default_factory=CaseNameSearchTrace)
    extra_data: ExtraData = field(default_factory=ExtraData)

    @property
    def case_names(self) -> tuple[str, ...]:
        """Not-found citations have no candidate case names."""
        return ()


@dataclass(frozen=True, slots=True)
class InvalidCitationRetrieval:
    """A citation missing the volume, reporter, or page required to look it up."""

    status: ClassVar[RetrievalStatus] = RetrievalStatus.INVALID
    citation_id: str
    source: str

    @property
    def case_names(self) -> tuple[str, ...]:
        """Invalid citations have no candidate case names."""
        return ()


@dataclass(frozen=True, slots=True)
class ThrottledCitationRetrieval:
    """A lookup that CourtListener rejected with a retryable throttle response."""

    status: ClassVar[RetrievalStatus] = RetrievalStatus.THROTTLED
    citation_id: str
    locator: str
    source: str
    request_trace: CourtListenerRequestTrace
    failure_detail: RetrievalFailureDetail | None
    extra_data: ExtraData = field(default_factory=ExtraData)

    @property
    def case_names(self) -> tuple[str, ...]:
        """Throttled lookups have no candidate case names."""
        return ()


@dataclass(frozen=True, slots=True)
class LookupFailedCitationRetrieval:
    """A lookup that failed for a non-throttle reason (network, parse, upstream 5xx)."""

    status: ClassVar[RetrievalStatus] = RetrievalStatus.LOOKUP_FAILED
    citation_id: str
    locator: str
    source: str
    request_trace: CourtListenerRequestTrace
    failure_detail: RetrievalFailureDetail | None
    extra_data: ExtraData = field(default_factory=ExtraData)

    @property
    def case_names(self) -> tuple[str, ...]:
        """Failed lookups have no candidate case names."""
        return ()


@dataclass(frozen=True, slots=True)
class SkippedCitationRetrieval:
    """A citation intentionally excluded from retrieval (e.g. not a FullCaseCitation)."""

    status: ClassVar[RetrievalStatus] = RetrievalStatus.SKIPPED
    citation_id: str
    source: str

    @property
    def case_names(self) -> tuple[str, ...]:
        """Skipped citations have no candidate case names."""
        return ()


CitationRetrieval: TypeAlias = (
    FoundCitationRetrieval
    | AmbiguousCitationRetrieval
    | NotFoundCitationRetrieval
    | InvalidCitationRetrieval
    | ThrottledCitationRetrieval
    | LookupFailedCitationRetrieval
    | SkippedCitationRetrieval
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievedDocument(InferredDocument):
    """An inferred document with one retrieval outcome per citation."""

    retrievals: tuple[CitationRetrieval, ...]
    retrieval_metadata: RetrievalMetadata

    @property
    def found(self) -> tuple[FoundCitationRetrieval, ...]:
        """Return citations found by the retrieval source."""
        return tuple(item for item in self.retrievals if isinstance(item, FoundCitationRetrieval))

    def __post_init__(self) -> None:
        InferredDocument.__post_init__(self)
        citation_ids = {item.citation_id for item in self.citations}
        retrieval_ids = [item.citation_id for item in self.retrievals]
        if any(not retrieval_id for retrieval_id in retrieval_ids):
            msg = "Citation retrieval identifiers must not be empty"
            raise ValueError(msg)
        if len(retrieval_ids) != len(set(retrieval_ids)):
            msg = "Citation retrieval identifiers must be unique within a document"
            raise ValueError(msg)
        if set(retrieval_ids) != citation_ids:
            msg = "Citation retrieval identifiers must exactly match extracted citation identifiers"
            raise ValueError(msg)
