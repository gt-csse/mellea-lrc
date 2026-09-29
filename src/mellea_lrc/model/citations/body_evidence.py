"""Citation-local evidence and decisions from other documents' citation text."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from mellea_lrc.model.citations.fields.case_name import CaseName
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.model.span import Span


class BodySource(str, Enum):
    COURTLISTENER_OPINION = "courtlistener_opinion"
    COURTLISTENER_RECAP = "courtlistener_recap"
    GOVINFO_OPINION = "govinfo_opinion"


class BodyCitationTreatment(str, Enum):
    """How the independent document treats its printed citation."""

    CITES_AS_AUTHORITY = "cites_as_authority"
    EXPLICITLY_DISPUTES = "explicitly_disputes"
    MENTIONS_ONLY = "mentions_only"


class BodyEvidenceFailure(BaseModel):
    """A search or body fetch that did not supply reviewable text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    item_id: str | None = None
    status_code: int | None = None


class BodySearchAttempt(BaseModel):
    """A provider query and its complete JSON search pages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1)
    pages: tuple[dict[str, JsonValue], ...] = ()
    failure: BodyEvidenceFailure | None = None


class BodyEvidence(BaseModel):
    """Exact source excerpt around a possible citation in another document.

    The excerpt is sufficient to resume review without fetching the document
    again. ``source_offset`` maps it to the fetched body; ``body_sha256`` and
    provider IDs identify the version that supplied it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    body_id: str = Field(min_length=1)
    parent_id: str | None = None
    url: str | None = None
    issued_on: date | None = None
    date_basis: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    body_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    excerpt: str = Field(min_length=1)
    source_offset: int = Field(ge=0)
    anchor_kind: Literal["locator"]
    anchor_span: Span

    @model_validator(mode="after")
    def _validate_excerpt(self) -> Self:
        if self.anchor_span.start == self.anchor_span.end or self.anchor_span.end > len(self.excerpt):
            raise ValueError("Body anchor span must lie inside the saved excerpt")
        if not self.excerpt[self.anchor_span.start : self.anchor_span.end].strip():
            raise ValueError("Body anchor match cannot be blank")
        return self


class BodySearch(BaseModel):
    """One provider stage's search, fetch outcomes, and reviewable excerpts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    source: BodySource
    retrospective_date: date | None
    attempts: tuple[BodySearchAttempt, ...] = ()
    discovery_pages: tuple[dict[str, JsonValue], ...] = ()
    evidence: tuple[BodyEvidence, ...] = ()
    failures: tuple[BodyEvidenceFailure, ...] = ()

    @model_validator(mode="after")
    def _validate_cutoff(self) -> Self:
        if self.retrospective_date is not None and any(
            item.issued_on is None or item.issued_on > self.retrospective_date for item in self.evidence
        ):
            raise ValueError("Body evidence must be dated on or before the retrospective cutoff")
        return self


class BodyCitationFields(BaseModel):
    """Fields literally read from one citation, without automatic equivalence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    locator: str | None
    case_name: str | None
    court: str | None
    date: str | None


class BodyFilingFields(BodyCitationFields):
    """Source-filing reread; a grounded name may carry model normalization."""

    normalized_case_name: CaseName | None

    @model_validator(mode="after")
    def _validate_name(self) -> Self:
        if (self.case_name is None) != (self.normalized_case_name is None):
            raise ValueError("A filing case name and its normalization must be supplied together")
        return self


class BodyFieldAssessment(BaseModel):
    """Semantic comparison of one source field with the cited third-party field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result: MatchResult
    reason: str = Field(min_length=1)


class BodyFieldComparisons(BaseModel):
    """Independent comparisons, including a complete locator assessment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    locator: BodyFieldAssessment
    case_name: BodyFieldAssessment
    court: BodyFieldAssessment
    date: BodyFieldAssessment


def compare_presence(decision: BodyCorroborationDecision) -> str | None:
    """Keep unavailable judgments tied to genuinely absent comparison values."""
    if decision.filing is None or decision.third_party is None or decision.comparisons is None:
        return None
    for field in ("locator", "case_name", "court", "date"):
        left = getattr(decision.filing, field)
        right = getattr(decision.third_party, field)
        result = getattr(decision.comparisons, field).result
        has_both = bool(left and left.strip()) and bool(right and right.strip())
        if has_both == (result is MatchResult.UNAVAILABLE):
            return f"{field}: use unavailable exactly when one side is absent"
    return None


class BodyCorroborationDecision(BaseModel):
    """One selected independent citation and the model's identity judgment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: BodySource | None
    evidence_index: int | None = Field(ge=0)
    citation_quote: str | None
    context_quote: str | None
    treatment: BodyCitationTreatment | None
    filing: BodyFilingFields | None
    third_party: BodyCitationFields | None
    comparisons: BodyFieldComparisons | None
    identity_verdict: IdentityVerdict
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        selected = self.source is not None
        details = (
            self.evidence_index,
            self.citation_quote,
            self.treatment,
            self.filing,
            self.third_party,
            self.comparisons,
        )
        if selected != all(value is not None for value in details):
            raise ValueError("A selected body citation needs both sides and every comparison")
        if not selected and any(value is not None for value in (*details, self.context_quote)):
            raise ValueError("A declined body citation cannot compare a candidate")
        if not selected and self.identity_verdict is not IdentityVerdict.DEFERRED:
            raise ValueError("Without a selected citation, identity must be deferred")
        if self.citation_quote is not None and not self.citation_quote.strip():
            raise ValueError("A selected citation quote cannot be blank")
        if self.context_quote is not None and not self.context_quote.strip():
            raise ValueError("A context quote cannot be blank")
        if self.treatment is BodyCitationTreatment.EXPLICITLY_DISPUTES and not self.context_quote:
            raise ValueError("An explicit challenge requires a quote from its surrounding discussion")
        if selected and (not self.filing.locator or not self.third_party.locator):
            raise ValueError("Both citations need a locator for body corroboration")
        if self.identity_verdict is IdentityVerdict.WRONG_IDENTITY and not (
            self.treatment is BodyCitationTreatment.EXPLICITLY_DISPUTES
            or any(
                getattr(self.comparisons, field).result is MatchResult.MISMATCH
                for field in ("locator", "case_name", "court", "date")
            )
        ):
            raise ValueError("A negative identity judgment needs a field conflict or explicit challenge")
        if self.identity_verdict is IdentityVerdict.CASE_IDENTITY_SUPPORTED and (
            self.treatment is not BodyCitationTreatment.CITES_AS_AUTHORITY
            or self.comparisons.locator.result is not MatchResult.MATCH
            or self.comparisons.case_name.result is not MatchResult.MATCH
        ):
            raise ValueError("Qualified case support needs a citing source, locator, and case name")
        if self.identity_verdict is IdentityVerdict.CORRECT_IDENTITY:
            if self.treatment is not BodyCitationTreatment.CITES_AS_AUTHORITY:
                raise ValueError("A mere mention or challenge cannot affirm citation identity")
            if (
                self.comparisons.locator.result is not MatchResult.MATCH
                or self.comparisons.case_name.result is not MatchResult.MATCH
            ):
                raise ValueError("An unqualified admission needs both locator and case-name support")
            if any(
                getattr(self.comparisons, field).result is MatchResult.MISMATCH
                for field in ("locator", "case_name", "court", "date")
            ):
                raise ValueError("A conflicting printed field cannot support an unqualified admission")
        return self


class BodyCorroborationReview(BaseModel):
    """Grounded selected quote and the complete model repair trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    decision: BodyCorroborationDecision | None = None
    grounded_quote: str | None = None
    quote_span: Span | None = None
    quote_similarity: float | None = Field(default=None, ge=0, le=100)
    grounded_context: str | None = None
    context_span: Span | None = None
    context_similarity: float | None = Field(default=None, ge=0, le=100)
    ivr: IvrRun | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> Self:
        decided = self.decision is not None
        if decided == (self.failure_reason is not None):
            raise ValueError("Body review needs either a decision or a failure")
        selected = decided and self.decision.source is not None
        if selected != all(
            value is not None for value in (self.grounded_quote, self.quote_span, self.quote_similarity)
        ):
            raise ValueError("A selected body review needs its grounded quote and span")
        if (decided and self.decision.context_quote is not None) != all(
            value is not None for value in (self.grounded_context, self.context_span, self.context_similarity)
        ):
            raise ValueError("A supplied context quote must ground in the saved excerpt")
        if self.ivr is not None and not self.ivr.success and decided:
            raise ValueError("A failed model run cannot provide a decision")
        return self
