"""Typed source-grounded citation fields and explicit normalization outcomes."""

from mellea_lrc.model.citations.fields.base import CitationField
from mellea_lrc.model.citations.fields.case_name import CaseName, CaseNameField, CaseNameKind
from mellea_lrc.model.citations.fields.court import Court, CourtField
from mellea_lrc.model.citations.fields.date import CitationDate, DateField
from mellea_lrc.model.citations.fields.docket import DocketEntryField, DocketLocatorValue, FullDocketLocator
from mellea_lrc.model.citations.fields.pin_cite import PinCiteField, PinCiteKind, PinCiteTarget, PinCiteValue
from mellea_lrc.model.citations.fields.reporter import FullReporterLocator, ReporterLocatorValue

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
    "PinCiteField",
    "PinCiteKind",
    "PinCiteTarget",
    "PinCiteValue",
    "ReporterLocatorValue",
]
