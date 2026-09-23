"""Formal types for Layer 2 preprocessed documents."""

from dataclasses import dataclass
from enum import Enum

from pydantic import model_validator

from mellea_lrc.model.source import DocumentBase
from mellea_lrc.model.span import Span


class PreprocessingBackend(str, Enum):
    """Engine that produced the preprocessed text."""

    DOCLING = "docling"
    PLAIN_TEXT = "plain_text"


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
    """The index of cited cases, marked because it cites nothing."""


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
    """Backend provenance for the preprocessing stage."""

    backend: PreprocessingBackend = PreprocessingBackend.PLAIN_TEXT
    backend_version: str | None = None
    rules: tuple[Rule, ...] = ()
    """Which rules ran, in order. Empty means none ran."""


class PreprocessedDocument(DocumentBase):
    """Source text and the provenance of its rendering."""

    text: str
    preprocessing_metadata: PreprocessingMetadata
    index_spans: tuple[Span, ...] = ()
    """Table-of-authorities regions. Empty may mean the index is unknown."""

    @model_validator(mode="after")
    def _validate_text(self) -> "PreprocessedDocument":
        if not self.text:
            msg = "PreprocessedDocument.text must not be empty"
            raise ValueError(msg)
        return self
