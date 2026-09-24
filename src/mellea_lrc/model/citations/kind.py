"""Distinct kinds of full and short citations in a document."""

from enum import Enum


class FullCitationKind(str, Enum):
    REPORTER = "reporter"
    DOCKET = "docket"


class ShortCitationKind(str, Enum):
    REPORTER = "short_reporter"
