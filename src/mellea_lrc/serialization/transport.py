"""Strict Pydantic DTOs for serialized artifact schema version 4."""

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


class AssessmentMetadataPayload(ArtifactPayload):
    mellea_calls: int
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


class CitationValidationPayload(ArtifactPayload):
    citation_id: str
    locator: str | None
    status: Literal[
        "found",
        "ambiguous",
        "not_found",
        "invalid",
        "throttled",
        "lookup_failed",
        "skipped",
    ]
    source: str
    message: str
    lookup_status: int | None
    lookup_cache: str | None
    lookup_key: str | None
    error_message: str | None
    failure_detail: ValidationFailureDetailPayload | None
    matches: list[CitationMatchPayload]
    extra_data: dict[str, JsonValue]


class ChatTurnPayload(ArtifactPayload):
    role: str
    content: str
    extra_data: dict[str, JsonValue]


class CaseNameAssessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal[
        "exact_match",
        "semantic_match",
        "different_case",
        "irregular_form",
        "reextraction_fail",
        "needs_assessment",
    ]
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    message: str
    chat_history: list[ChatTurnPayload] | None


class YearAssessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["exact_match", "mismatch", "missing"]
    extracted_year: str | None
    courtlistener_year: str | None
    message: str


class CitationAssessmentResultPayload(ArtifactPayload):
    citation_id: str
    case_assess: CaseNameAssessmentPayload
    year_assess: YearAssessmentPayload


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


class FailedCitationAssessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["failed"]
    error: str


CitationAssessmentPayload = Annotated[
    WaitingCitationAssessmentPayload
    | SkippedCitationAssessmentPayload
    | AssessedCitationAssessmentPayload
    | FailedCitationAssessmentPayload,
    Field(discriminator="status"),
]


class ModifiedExtractedCitationPayload(ArtifactPayload):
    citation_id: str
    span: SpanPayload | None
    matched_text: str | None
    case_name: str | None
    extracted_case_name: str | None


class WaitingCitationReassessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["waiting"]


class SkippedCitationReassessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["skipped"]
    reason: Literal[
        "assessment_skipped",
        "assessment_failed",
        "reextraction_not_required",
    ]
    message: str


class ReassessedCitationReassessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["reassessed"]
    modified_citation: ModifiedExtractedCitationPayload
    result: CitationAssessmentResultPayload


class ReextractionFailedCitationReassessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["reextraction_failed"]
    error: str


class ReassessmentFailedCitationReassessmentPayload(ArtifactPayload):
    citation_id: str
    status: Literal["reassessment_failed"]
    modified_citation: ModifiedExtractedCitationPayload
    error: str


CitationReassessmentPayload = Annotated[
    WaitingCitationReassessmentPayload
    | SkippedCitationReassessmentPayload
    | ReassessedCitationReassessmentPayload
    | ReextractionFailedCitationReassessmentPayload
    | ReassessmentFailedCitationReassessmentPayload,
    Field(discriminator="status"),
]


class ExtractionCountsPayload(ArtifactPayload):
    total: int
    full: int
    by_type: dict[str, int]


class ValidationCountsPayload(ArtifactPayload):
    total: int
    found: int
    by_status: dict[str, int]


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
    schema_version: Literal[4]
    artifact_type: Literal["preprocessed_document"]


class ExtractedDocumentPayload(_ExtractedDocumentFields):
    schema_version: Literal[4]
    artifact_type: Literal["extracted_document"]
    counts: ExtractionCountsPayload


class ValidatedDocumentPayload(_ValidatedDocumentFields):
    schema_version: Literal[4]
    artifact_type: Literal["validated_document"]
    counts: ValidationCountsPayload


class AssessedDocumentPayload(_ValidatedDocumentFields):
    schema_version: Literal[4]
    artifact_type: Literal["assessed_document"]
    assessment_metadata: AssessmentMetadataPayload
    assessments: list[CitationAssessmentPayload]
    assessment_complete: bool
    assessment_status_counts: dict[str, int]
    reassessments: list[CitationReassessmentPayload]
    reassessment_status_counts: dict[str, int]
    case_name_counts: dict[str, int]
    year_counts: dict[str, int]
