"""Strict Pydantic DTOs for serialized artifact schema version 12."""

# ruff: noqa: D101

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


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


class ValidationMetadataPayload(ArtifactPayload):
    client_mode: Literal["deployed", "sdk", "custom"]
    source: str
    duration_ms: float | None = None


class AssessmentMetadataPayload(ArtifactPayload):
    mellea_concurrency: int | None


class SpanPayload(ArtifactPayload):
    start: int
    end: int


class FullCaseCitationPayload(ArtifactPayload):
    type: Literal["FullCaseCitation"]
    plaintiff: str | None
    defendant: str | None
    volume: str | None
    reporter: str | None
    page: str | None
    pin_cite: str | None
    extra: str | None
    year: str | None
    court: str | None
    parenthetical: str | None


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
    reporter: str | None
    page: str | None
    pin_cite: str | None
    court: str | None
    parenthetical: str | None


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


class CitationMatchPayload(ArtifactPayload):
    case_name: str | None
    date_filed: str | None
    court: str | None
    court_id: str | None = None
    docket_id: str | None = None
    extra_data: dict[str, JsonValue]


class ValidationFailureDetailPayload(ArtifactPayload):
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


class FoundCitationValidationPayload(ArtifactPayload):
    citation_id: str
    status: Literal["found"]
    locator: str
    source: str
    lookup_status: int
    lookup_cache: str | None
    lookup_key: str | None
    matches: list[CitationMatchPayload]
    court_resolution: CourtResolutionTracePayload
    extra_data: dict[str, JsonValue]


class AmbiguousCitationValidationPayload(ArtifactPayload):
    citation_id: str
    status: Literal["ambiguous"]
    locator: str
    source: str
    lookup_status: int
    lookup_cache: str | None
    lookup_key: str | None
    matches: list[CitationMatchPayload]
    extra_data: dict[str, JsonValue]


class CaseNameSearchTracePayload(ArtifactPayload):
    status: Literal[
        "searched",
        "skipped_no_case_name",
        "skipped_partial_case_name",
        "search_unavailable",
        "search_failed",
        "not_attempted",
    ]
    query: str | None
    case_count: int | None
    error_message: str | None


class NotFoundCitationValidationPayload(ArtifactPayload):
    citation_id: str
    status: Literal["not_found"]
    locator: str
    source: str
    lookup_status: int
    lookup_cache: str | None
    lookup_key: str | None
    candidate_search: CaseNameSearchTracePayload
    extra_data: dict[str, JsonValue]


class InvalidCitationValidationPayload(ArtifactPayload):
    citation_id: str
    status: Literal["invalid"]
    source: str


class ThrottledCitationValidationPayload(ArtifactPayload):
    citation_id: str
    status: Literal["throttled"]
    locator: str
    source: str
    lookup_status: int
    lookup_cache: str | None
    lookup_key: str | None
    error_message: str | None
    failure_detail: ValidationFailureDetailPayload | None
    extra_data: dict[str, JsonValue]


class LookupFailedCitationValidationPayload(ArtifactPayload):
    citation_id: str
    status: Literal["lookup_failed"]
    locator: str
    source: str
    lookup_status: int | None
    lookup_cache: str | None
    lookup_key: str | None
    error_message: str | None
    failure_detail: ValidationFailureDetailPayload | None
    extra_data: dict[str, JsonValue]


class SkippedCitationValidationPayload(ArtifactPayload):
    citation_id: str
    status: Literal["skipped"]
    source: str


CitationValidationPayload = Annotated[
    FoundCitationValidationPayload
    | AmbiguousCitationValidationPayload
    | NotFoundCitationValidationPayload
    | InvalidCitationValidationPayload
    | ThrottledCitationValidationPayload
    | LookupFailedCitationValidationPayload
    | SkippedCitationValidationPayload,
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


class CourtAssessmentPayload(ArtifactPayload):
    status: Literal["exact_match", "mismatch", "missing"]
    extracted_court: str | None
    courtlistener_court_id: str | None
    message: str


class CourtFollowupNotRequiredPayload(ArtifactPayload):
    status: Literal["not_required"]


class CourtInferredFromReporterPayload(ArtifactPayload):
    status: Literal["inferred_from_reporter"]
    reporter: str | None
    citation_court_before: str | None
    result: CourtAssessmentPayload


CourtFollowupPayload = Annotated[
    CourtFollowupNotRequiredPayload | CourtInferredFromReporterPayload,
    Field(discriminator="status"),
]


class CourtAssessmentRunPayload(ArtifactPayload):
    initial: CourtAssessmentPayload
    followup: CourtFollowupPayload


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


class CitationAssessmentResultPayload(ArtifactPayload):
    case_name: CaseNameAssessmentRunPayload
    court: CourtAssessmentRunPayload
    year: YearAssessmentPayload


class WaitingCitationAssessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["waiting"]


class SkippedCitationAssessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["skipped"]
    reason: Literal["unsupported_citation_kind", "validation_not_eligible"]
    message: str


class AssessedCitationAssessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["assessed"]
    result: CitationAssessmentResultPayload


class CandidateAssessmentPayload(ArtifactPayload):
    match: CitationMatchPayload
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


class _ValidatedDocumentFields(_ExtractedDocumentFields):
    validation_metadata: ValidationMetadataPayload
    validations: list[CitationValidationPayload]


class PreprocessedDocumentPayload(_PreprocessedDocumentFields):
    schema_version: Literal[12]
    artifact_type: Literal["preprocessed_document"]


class ExtractedDocumentPayload(_ExtractedDocumentFields):
    schema_version: Literal[12]
    artifact_type: Literal["extracted_document"]


class ValidatedDocumentPayload(_ValidatedDocumentFields):
    schema_version: Literal[12]
    artifact_type: Literal["validated_document"]


class AssessedDocumentPayload(_ValidatedDocumentFields):
    schema_version: Literal[12]
    artifact_type: Literal["assessed_document"]
    assessment_metadata: AssessmentMetadataPayload
    assessments: list[CitationAssessmentPayload]
