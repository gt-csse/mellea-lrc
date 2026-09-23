"""Typed full citation occurrences independent of processing stage."""

from typing import Annotated, TypeAlias

from pydantic import Field

from mellea_lrc.model.citations.date import CitationDate
from mellea_lrc.model.citations.fields import (
    CaseNameField,
    CitationField,
    CourtField,
    DateField,
    DocketEntryField,
    DocketNumberField,
    LocatorField,
    PageField,
    PinCiteField,
    ReporterField,
    VolumeField,
)
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.full_docket import FullDocketCitation
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.citations.history import (
    Node,
    RelationshipUpdate,
    latest,
)
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.citations.pin_cite import PinCiteKind, PinCiteTarget, PinCiteValue

FullCitationVariant: TypeAlias = Annotated[
    FullReporterCitation | FullDocketCitation,
    Field(discriminator="kind"),
]


__all__ = [
    "CaseNameField",
    "CitationDate",
    "CitationField",
    "CourtField",
    "DateField",
    "DocketEntryField",
    "DocketNumberField",
    "FullCitation",
    "FullCitationKind",
    "FullCitationVariant",
    "FullDocketCitation",
    "FullReporterCitation",
    "LocatorField",
    "Node",
    "PageField",
    "PinCiteField",
    "PinCiteKind",
    "PinCiteTarget",
    "PinCiteValue",
    "RelationshipUpdate",
    "ReporterField",
    "VolumeField",
    "latest",
]
