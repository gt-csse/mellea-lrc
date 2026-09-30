"""Typed full citation occurrences independent of processing stage."""

from typing import Annotated, TypeAlias

from eyecite.models import Reporter
from pydantic import Field

from mellea_lrc.model.citations.citation import Citation
from mellea_lrc.model.citations.docket_lookup import (
    DocketLookup,
    DocketLookupAttempt,
    DocketLookupCandidate,
    DocketLookupCaseNameAssessment,
    DocketLookupFailure,
    DocketLookupFieldAssessment,
    DocketLookupReview,
    DocketLookupReviewDecision,
    DocketSearchSource,
)
from mellea_lrc.model.citations.field_body_evidence import (
    FieldBodySearch,
    IntendedCaseConfidence,
    IntendedCaseDecision,
    IntendedCaseReview,
)
from mellea_lrc.model.citations.fields import (
    CaseName,
    CaseNameField,
    CaseNameKind,
    CitationDate,
    CitationField,
    Court,
    CourtField,
    DateField,
    DocketEntryField,
    DocketLocatorValue,
    FullDocketLocator,
    FullReporterLocator,
    PinCiteField,
    PinCiteKind,
    PinCiteTarget,
    PinCiteValue,
    ReporterLocatorValue,
    ShortReporterLocator,
    ShortReporterLocatorValue,
)
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.full_docket import FullDocketCitation
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.citations.govinfo_lookup import (
    GovInfoDocketLookup,
    GovInfoDocketReview,
    GovInfoLookupAttempt,
    GovInfoLookupCandidate,
)
from mellea_lrc.model.citations.history import (
    Node,
    RelationshipUpdate,
    latest,
)
from mellea_lrc.model.citations.id import IdCitation
from mellea_lrc.model.citations.judgments import (
    IdentityJudgment,
    IdentityVerdict,
    MatchResult,
    ReporterExactCaseNameJudgment,
    ReporterExactCourtJudgment,
    ReporterExactDateJudgment,
)
from mellea_lrc.model.citations.kind import FullCitationKind, ShortCitationKind
from mellea_lrc.model.citations.leaf import (
    AttributionResult,
    LeafAttribution,
    LeafCitation,
    LeafReview,
    LeafReviewDecision,
)
from mellea_lrc.model.citations.reference import ReferenceCitation
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactLookup,
    ReporterExactLookupOutcome,
    ReporterExactLookupQuery,
    ReporterUniqueFieldAssessment,
    ReporterUniqueReview,
    ReporterUniqueReviewDecision,
)
from mellea_lrc.model.citations.short_reporter import ShortReporterCitation
from mellea_lrc.model.citations.supra import SupraCitation

FullCitationVariant: TypeAlias = Annotated[
    FullReporterCitation | FullDocketCitation,
    Field(discriminator="kind"),
]
CitationVariant: TypeAlias = Annotated[
    FullReporterCitation
    | FullDocketCitation
    | ShortReporterCitation
    | IdCitation
    | SupraCitation
    | ReferenceCitation,
    Field(discriminator="kind"),
]


__all__ = [
    "AttributionResult",
    "CaseName",
    "CaseNameField",
    "CaseNameKind",
    "Citation",
    "CitationDate",
    "CitationField",
    "CitationVariant",
    "Court",
    "CourtField",
    "DateField",
    "DocketEntryField",
    "DocketLocatorValue",
    "DocketLookup",
    "DocketLookupAttempt",
    "DocketLookupCandidate",
    "DocketLookupCaseNameAssessment",
    "DocketLookupFailure",
    "DocketLookupFieldAssessment",
    "DocketLookupReview",
    "DocketLookupReviewDecision",
    "DocketSearchSource",
    "FieldBodySearch",
    "FullCitation",
    "FullCitationKind",
    "FullCitationVariant",
    "FullDocketCitation",
    "FullDocketLocator",
    "FullReporterCitation",
    "FullReporterLocator",
    "GovInfoDocketLookup",
    "GovInfoDocketReview",
    "GovInfoLookupAttempt",
    "GovInfoLookupCandidate",
    "IdCitation",
    "IdentityJudgment",
    "IdentityVerdict",
    "IntendedCaseConfidence",
    "IntendedCaseDecision",
    "IntendedCaseReview",
    "LeafAttribution",
    "LeafCitation",
    "LeafReview",
    "LeafReviewDecision",
    "MatchResult",
    "Node",
    "PinCiteField",
    "PinCiteKind",
    "PinCiteTarget",
    "PinCiteValue",
    "ReferenceCitation",
    "RelationshipUpdate",
    "Reporter",
    "ReporterExactCaseNameJudgment",
    "ReporterExactCourtJudgment",
    "ReporterExactDateJudgment",
    "ReporterExactLookup",
    "ReporterExactLookupOutcome",
    "ReporterExactLookupQuery",
    "ReporterLocatorValue",
    "ReporterUniqueFieldAssessment",
    "ReporterUniqueReview",
    "ReporterUniqueReviewDecision",
    "ShortCitationKind",
    "ShortReporterCitation",
    "ShortReporterLocator",
    "ShortReporterLocatorValue",
    "SupraCitation",
    "latest",
]
