"""Kinds of full citation currently supported by the document."""

from enum import Enum


class FullCitationKind(str, Enum):
    REPORTER = "reporter"
    DOCKET = "docket"
