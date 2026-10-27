"""Formal types for Layer 2 preprocessed documents."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import model_validator

from mellea_lrc.model.source import DocumentBase
from mellea_lrc.model.span import Span


class PreprocessingBackend(str, Enum):
    """Engine that produced the preprocessed text."""

    DOCLING = "docling"
    PLAIN_TEXT = "plain_text"


@dataclass(frozen=True, slots=True)
class TableOfAuthoritiesComponent(Span):
    """One TOA component in the final exported source-text coordinates."""

    kind: Literal["table_of_authorities"] = "table_of_authorities"


class Rule(str, Enum):
    """Layout decisions applied to structured documents before text export."""

    MARGIN_LINE_NUMBERS = "margin_line_numbers"
    """The numbered left margin of pleading paper."""

    REPEATED_FURNITURE = "repeated_furniture"
    """Running heads and feet the converter labelled inconsistently."""

    DOCKET_STAMP = "docket_stamp"
    """The filing stamp a court prints across the top of every page."""

    TABLE_AS_TEXT = "table_as_text"
    """A table read in the order the page reads it, not rebuilt as a grid."""

    TABLE_OF_AUTHORITIES = "table_of_authorities"
    """The authority index: citations identify cases but supply no pinpoint proposition."""


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule.MARGIN_LINE_NUMBERS,
    Rule.REPEATED_FURNITURE,
    Rule.DOCKET_STAMP,
    Rule.TABLE_AS_TEXT,
    Rule.TABLE_OF_AUTHORITIES,
)
"""Default layout rules; callers may pass a shorter sequence or an empty one."""


@dataclass(frozen=True, slots=True)
class PreprocessingMetadata:
    """Backend provenance for the preprocessing substage."""

    backend: PreprocessingBackend = PreprocessingBackend.PLAIN_TEXT
    backend_version: str | None = None
    rules: tuple[Rule, ...] = ()
    """Which rules ran, in order. Empty means none ran."""


class PreprocessedDocument(DocumentBase):
    """Source text and the provenance of its rendering."""

    text: str
    preprocessing_metadata: PreprocessingMetadata
    index_spans: tuple[TableOfAuthoritiesComponent, ...] = ()
    """Table-of-authorities regions. Empty may mean the index is unknown."""

    @model_validator(mode="after")
    def _validate_text(self) -> "PreprocessedDocument":
        if not self.text:
            msg = "PreprocessedDocument.text must not be empty"
            raise ValueError(msg)
        previous_end = 0
        for component in self.index_spans:
            if component.start == component.end or component.end > len(self.text):
                raise ValueError("TOA component must occupy a nonempty range inside the source text")
            if component.start < previous_end:
                raise ValueError("TOA components must be ordered and nonoverlapping")
            previous_end = component.end
        return self
