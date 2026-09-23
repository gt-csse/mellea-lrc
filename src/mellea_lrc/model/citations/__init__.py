"""Typed full citation occurrences independent of processing stage."""

from typing import Annotated, TypeAlias

from pydantic import Field

from mellea_lrc.model.citations.date import CitationDate
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.full_docket import FullDocketCitation
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.citations.history import (
    FieldUpdate,
    Node,
    latest,
)
from mellea_lrc.model.citations.kind import FullCitationKind

FullCitationVariant: TypeAlias = Annotated[
    FullReporterCitation | FullDocketCitation,
    Field(discriminator="kind"),
]


__all__ = [
    "CitationDate",
    "FieldUpdate",
    "FullCitation",
    "FullCitationKind",
    "FullCitationVariant",
    "FullDocketCitation",
    "FullReporterCitation",
    "Node",
    "latest",
]
