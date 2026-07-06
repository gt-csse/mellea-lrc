"""Strict Pydantic DTOs for schema 17."""

# ruff: noqa: D101

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue


class ArtifactPayload(BaseModel):
    """Base transport model that rejects coercion and unknown fields."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class SourceMetadataPayload(ArtifactPayload):
    path: str | None
    format: Literal["pdf", "docx", "pptx", "xlsx", "html", "markdown", "text", "unknown"]
    header: str | None
    extra_data: dict[str, JsonValue]


class PreprocessingMetadataPayload(ArtifactPayload):
    backend: Literal["docling", "plain_text"]
    backend_version: str | None


class ExtractionMetadataPayload(ArtifactPayload):
    backend: Literal["eyecite"]
    backend_version: str | None


class RetrievalMetadataPayload(ArtifactPayload):
    client_mode: Literal["deployed", "sdk", "custom"]
    source: str
    duration_ms: float | None = None


class AssessmentMetadataPayload(ArtifactPayload):
    mellea_concurrency: int | None


class SpanPayload(ArtifactPayload):
    start: int
    end: int


class ReporterPayload(ArtifactPayload):
    edition_short_name: str
    root_short_name: str
    name: str
    cite_type: str
    is_scotus: bool
    source: str


class FullCaseCitationPayload(ArtifactPayload):
    type: Literal["FullCaseCitation"]
    plaintiff: str | None
    defendant: str | None
    volume: str | None
    page: str | None
    pin_cite: str | None
    extra: str | None
    year: str | None
    court: str | None
    parenthetical: str | None
    reporter: ReporterPayload | None = None


class FullLawCitationPayload(ArtifactPayload):
    type: Literal["FullLawCitation"]
    volume: str | None
    reporter: str | None
    page: str | None
    pin_cite: str | None
    year: str | None
    publisher: str | None
    parenthetical: str | None


class FullJournalCitationPayload(ArtifactPayload):
    type: Literal["FullJournalCitation"]
    volume: str | None
    reporter: str | None
    page: str | None
    pin_cite: str | None
    year: str | None
    parenthetical: str | None


class ShortCaseCitationPayload(ArtifactPayload):
    type: Literal["ShortCaseCitation"]
    volume: str | None
    page: str | None
    pin_cite: str | None
    court: str | None
    parenthetical: str | None
    reporter: ReporterPayload | None = None


class SupraCitationPayload(ArtifactPayload):
    type: Literal["SupraCitation"]
    pin_cite: str | None
    parenthetical: str | None


class IdCitationPayload(ArtifactPayload):
    type: Literal["IdCitation"]
    pin_cite: str | None
    parenthetical: str | None


class ReferenceCitationPayload(ArtifactPayload):
    type: Literal["ReferenceCitation"]
    plaintiff: str | None
    defendant: str | None


class UnknownCitationPayload(ArtifactPayload):
    type: Literal["UnknownCitation"]


CanonicalCitationPayload = Annotated[
    FullCaseCitationPayload
    | FullLawCitationPayload
    | FullJournalCitationPayload
    | ShortCaseCitationPayload
    | SupraCitationPayload
    | IdCitationPayload
    | ReferenceCitationPayload
    | UnknownCitationPayload,
    Field(discriminator="type"),
]


class ExtractedCitationPayload(ArtifactPayload):
    citation_id: str
    span: SpanPayload
    matched_text: str
    citation: CanonicalCitationPayload
    resolves_to: str | None


class CourtListenerCitationRecordPayload(ArtifactPayload):
    case_name: str | None
    date_filed: str | None
    court: str | None
    court_id: str | None = None
    docket_id: str | None = None
    extra_data: dict[str, JsonValue]


class RetrievalFailureDetailPayload(ArtifactPayload):
    failure_type: str | None
    message: str | None
    retryable: bool | None
    upstream_status_code: int | None
    key: str | None
    url: str | None
    retry_after_seconds: float | None
    extra_data: dict[str, JsonValue]


class CourtResolutionTracePayload(ArtifactPayload):
    courtlistener_court_id: str | None
    resolved_via: Literal[
        "cluster_provided",
        "docket_lookup",
        "no_docket_id",
        "docket_lookup_failed",
        "not_attempted",
    ]
    docket_id: str | None
    docket_url: str | None
    cached: bool
    error_message: str | None


class RetrievedCandidatePayload(ArtifactPayload):
    candidate_id: str
    record: CourtListenerCitationRecordPayload
    court_resolution: CourtResolutionTracePayload


class FoundCitationRetrievalPayload(ArtifactPayload):
    citation_id: str
    status: Literal["found"]
    locator: str
    source: str
    lookup_status: int
    lookup_cache: str | None
    lookup_key: str | None
    candidate: RetrievedCandidatePayload
    extra_data: dict[str, JsonValue]


class AmbiguousCitationRetrievalPayload(ArtifactPayload):
    citation_id: str
    status: Literal["ambiguous"]
    locator: str
    source: str
    lookup_status: int
    lookup_cache: str | None
    lookup_key: str | None
    candidates: list[RetrievedCandidatePayload]
    extra_data: dict[str, JsonValue]


class CaseNameSearchProbePayload(ArtifactPayload):
    corpus: Literal["o", "r"]
    status: Literal["searched", "search_unavailable", "search_failed"]
    case_count: int | None
    error_message: str | None
    http_status: int | None = None


class CaseNameSearchTracePayload(ArtifactPayload):
    status: Literal[
        "searched",
        "partial",
        "skipped_no_case_name",
        "skipped_partial_case_name",
        "search_unavailable",
        "search_failed",
        "not_attempted",
    ]
    query: str | None
    probes: list[CaseNameSearchProbePayload]


class NotFoundCitationRetrievalPayload(ArtifactPayload):
    citation_id: str
    status: Literal["not_found"]
    locator: str
    source: str
    lookup_status: int
    lookup_cache: str | None
    lookup_key: str | None
    candidate_search: CaseNameSearchTracePayload
    extra_data: dict[str, JsonValue]


class InvalidCitationRetrievalPayload(ArtifactPayload):
    citation_id: str
    status: Literal["invalid"]
    source: str


class ThrottledCitationRetrievalPayload(ArtifactPayload):
    citation_id: str
    status: Literal["throttled"]
    locator: str
    source: str
    lookup_status: int
    lookup_cache: str | None
    lookup_key: str | None
    error_message: str | None
    failure_detail: RetrievalFailureDetailPayload | None
    extra_data: dict[str, JsonValue]


class LookupFailedCitationRetrievalPayload(ArtifactPayload):
    citation_id: str
    status: Literal["lookup_failed"]
    locator: str
    source: str
    lookup_status: int | None
    lookup_cache: str | None
    lookup_key: str | None
    error_message: str | None
    failure_detail: RetrievalFailureDetailPayload | None
    extra_data: dict[str, JsonValue]


class SkippedCitationRetrievalPayload(ArtifactPayload):
    citation_id: str
    status: Literal["skipped"]
    source: str


CitationRetrievalPayload = Annotated[
    FoundCitationRetrievalPayload
    | AmbiguousCitationRetrievalPayload
    | NotFoundCitationRetrievalPayload
    | InvalidCitationRetrievalPayload
    | ThrottledCitationRetrievalPayload
    | LookupFailedCitationRetrievalPayload
    | SkippedCitationRetrievalPayload,
    Field(discriminator="status"),
]


class ChatTurnPayload(ArtifactPayload):
    role: str
    content: str
    extra_data: dict[str, JsonValue]


class CaseNameAssessmentPayload(ArtifactPayload):
    status: Literal[
        "exact_match",
        "semantic_match",
        "not_semantic_match",
        "different_case",
        "irregular_form",
        "unassessable",
    ]
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    message: str
    chat_history: list[ChatTurnPayload] | None


class YearAssessmentPayload(ArtifactPayload):
    status: Literal["exact_match", "mismatch", "missing"]
    extracted_year: str | None
    courtlistener_year: str | None
    message: str


class CourtsDBClassificationPayload(ArtifactPayload):
    court_id: str
    system: str | None
    jurisdiction: str | None
    type: str | None


class CourtAssessmentPayload(ArtifactPayload):
    status: Literal["exact_match", "mismatch", "missing"]
    extracted_court: str | None
    courtlistener_court_id: str | None
    message: str
    source: Literal["direct", "reporter_inferred"]





class ReextractedCaseNamePayload(ArtifactPayload):
    case_name: str
    case_name_span: SpanPayload


class CaseNameReassessmentNotRequiredPayload(ArtifactPayload):
    status: Literal["not_required"]


class CaseNameReextractionFailedPayload(ArtifactPayload):
    status: Literal["reextraction_failed"]
    error: str


class CaseNameReassessedPayload(ArtifactPayload):
    status: Literal["reassessed"]
    reextracted_case_name: ReextractedCaseNamePayload
    result: CaseNameAssessmentPayload


class CaseNameReassessmentFailedPayload(ArtifactPayload):
    status: Literal["reassessment_failed"]
    reextracted_case_name: ReextractedCaseNamePayload
    error: str


CaseNameFollowupPayload = Annotated[
    CaseNameReassessmentNotRequiredPayload
    | CaseNameReextractionFailedPayload
    | CaseNameReassessedPayload
    | CaseNameReassessmentFailedPayload,
    Field(discriminator="status"),
]


class CaseNameAssessmentRunPayload(ArtifactPayload):
    initial: CaseNameAssessmentPayload
    followup: CaseNameFollowupPayload


class ReporterJurisdictionEvidencePayload(ArtifactPayload):
    source: str
    statement: str


class ReporterInferencePayload(ArtifactPayload):
    reporter: ReporterPayload | None
    status: Literal[
        "unsupported",
        "missing_reporter",
        "unrecognized",
        "recognized",
    ]
    mlz_jurisdictions: list[str]


class CourtInferencePayload(ArtifactPayload):
    extracted_court: str | None
    status: Literal[
        "unsupported",
        "missing_court",
        "unrecognized",
        "resolved",
    ]
    courts_db_classification: CourtsDBClassificationPayload | None = None


class JurisdictionPayload(ArtifactPayload):
    reporter_inference: ReporterInferencePayload
    court_inference: CourtInferencePayload


class CitationAssessmentResultPayload(ArtifactPayload):
    case_name: CaseNameAssessmentRunPayload
    court: CourtAssessmentPayload
    year: YearAssessmentPayload


class WaitingCitationAssessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["waiting"]


class SkippedCitationAssessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["skipped"]
    reason: Literal["unsupported_citation_kind", "retrieval_not_eligible"]
    message: str


class AssessedCitationAssessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["assessed"]
    candidate_id: str
    result: CitationAssessmentResultPayload


class CandidateAssessmentPayload(ArtifactPayload):
    candidate_id: str
    result: CitationAssessmentResultPayload


class AmbiguousCitationAssessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["ambiguous"]
    candidates: list[CandidateAssessmentPayload]
    gated: bool = False
    message: str = ""


class FailedCitationAssessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["failed"]
    error: str


CitationAssessmentPayload = Annotated[
    WaitingCitationAssessmentPayload
    | SkippedCitationAssessmentPayload
    | AssessedCitationAssessmentPayload
    | AmbiguousCitationAssessmentPayload
    | FailedCitationAssessmentPayload,
    Field(discriminator="status"),
]


class _PreprocessedDocumentFields(ArtifactPayload):
    text: str
    source_metadata: SourceMetadataPayload
    preprocessing_metadata: PreprocessingMetadataPayload


class _ExtractedDocumentFields(_PreprocessedDocumentFields):
    extraction_metadata: ExtractionMetadataPayload
    citations: list[ExtractedCitationPayload]


class _InferredDocumentFields(_ExtractedDocumentFields):
    jurisdictions: list[JurisdictionPayload]


class _RetrievedDocumentFields(_InferredDocumentFields):
    retrieval_metadata: RetrievalMetadataPayload
    retrievals: list[CitationRetrievalPayload]


class PreprocessedDocumentPayload(_PreprocessedDocumentFields):
    schema_version: Literal[17]
    artifact_type: Literal["preprocessed_document"]


class ExtractedDocumentPayload(_ExtractedDocumentFields):
    schema_version: Literal[17]
    artifact_type: Literal["extracted_document"]


class InferredDocumentPayload(_InferredDocumentFields):
    schema_version: Literal[17]
    artifact_type: Literal["inferred_document"]


class RetrievedDocumentPayload(_RetrievedDocumentFields):
    schema_version: Literal[17]
    artifact_type: Literal["retrieved_document"]


class AssessedDocumentPayload(_RetrievedDocumentFields):
    schema_version: Literal[17]
    artifact_type: Literal["assessed_document"]
    assessment_metadata: AssessmentMetadataPayload
    assessments: list[CitationAssessmentPayload]
