"""Domain fields, each with its own normalization and validation."""

from mellea_lrc.model.citations.fields.base import CitationField, NotStatedField, current_reading
from mellea_lrc.model.citations.fields.case_name import CaseName, CaseNameField, CaseNameKind
from mellea_lrc.model.citations.fields.court import Court, CourtField
from mellea_lrc.model.citations.fields.date import CitationDate, DateField
from mellea_lrc.model.citations.fields.docket import DocketEntryField, DocketLocatorValue, FullDocketLocator
from mellea_lrc.model.citations.fields.pin_cite import (
    PinCiteField,
    PinCiteKind,
    PinCiteTarget,
    PinCiteValue,
)
from mellea_lrc.model.citations.fields.reporter import FullReporterLocator, ReporterLocatorValue
from mellea_lrc.model.citations.fields.short_reporter import ShortReporterLocator, ShortReporterLocatorValue

__all__ = [
    "CaseName",
    "CaseNameField",
    "CaseNameKind",
    "CitationDate",
    "CitationField",
    "Court",
    "CourtField",
    "DateField",
    "DocketEntryField",
    "DocketLocatorValue",
    "FullDocketLocator",
    "FullReporterLocator",
    "NotStatedField",
    "PinCiteField",
    "PinCiteKind",
    "PinCiteTarget",
    "PinCiteValue",
    "ReporterLocatorValue",
    "ShortReporterLocator",
    "ShortReporterLocatorValue",
    "current_reading",
]
