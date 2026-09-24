"""Typed full citation occurrences independent of processing stage."""

from typing import Annotated, TypeAlias

from eyecite.models import Reporter
from pydantic import Field

from mellea_lrc.model.citations.citation import Citation
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
from mellea_lrc.model.citations.history import (
    Node,
    RelationshipUpdate,
    latest,
)
from mellea_lrc.model.citations.kind import FullCitationKind, ShortCitationKind
from mellea_lrc.model.citations.short_reporter import ShortReporterCitation

FullCitationVariant: TypeAlias = Annotated[
    FullReporterCitation | FullDocketCitation,
    Field(discriminator="kind"),
]
CitationVariant: TypeAlias = Annotated[
    FullReporterCitation | FullDocketCitation | ShortReporterCitation,
    Field(discriminator="kind"),
]


__all__ = [
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
    "FullCitation",
    "FullCitationKind",
    "FullCitationVariant",
    "FullDocketCitation",
    "FullDocketLocator",
    "FullReporterCitation",
    "FullReporterLocator",
    "Node",
    "PinCiteField",
    "PinCiteKind",
    "PinCiteTarget",
    "PinCiteValue",
    "RelationshipUpdate",
    "Reporter",
    "ReporterLocatorValue",
    "ShortCitationKind",
    "ShortReporterCitation",
    "ShortReporterLocator",
    "ShortReporterLocatorValue",
    "latest",
]
