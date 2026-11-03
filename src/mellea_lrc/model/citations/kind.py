"""Kinds of full citation occurrences."""

from enum import Enum


class FullCitationKind(str, Enum):
    REPORTER = "reporter"
    DOCKET = "docket"
