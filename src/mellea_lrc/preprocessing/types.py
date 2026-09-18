"""Formal types for Layer 2 preprocessed documents."""

from dataclasses import dataclass
from enum import Enum

from mellea_lrc.core.documents import DocumentBase
from mellea_lrc.core.spans import Span


class PreprocessingBackend(str, Enum):
    """Engine that produced the preprocessed text."""

    DOCLING = "docling"
    PLAIN_TEXT = "plain_text"


class Rule(str, Enum):
    """One reading of the page this project makes rather than leaving to the converter.

    Each names one reading of the page that a converter would otherwise decide
    on its own: what is furniture rather than writing, how a table is read, and
    which region of the text cites nothing. Every one of them changes the text
    or what can be said about it, so which ran is recorded beside the text
    rather than assumed.

    Three of them take something out of the body: the numbered margin of
    pleading paper, a running head, the stamp an ECF system prints when a
    document is filed. `TABLE_AS_TEXT` takes nothing out and changes how a
    region is read. `TABLE_OF_AUTHORITIES` changes nothing at all and marks
    where the index sits, because an entry there names a case and claims
    nothing about it.

    Every rule reads the page -- where an item sits, whether its neighbours
    repeat, what a converter made of a region. A `.txt` file carries no
    geometry, so none of them applies to one, and a rendering made from text
    records that none ran.
    """

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
"""All of them. None of this is the document's text, and a rendering that keeps
it is wrong about the document -- a margin number landing inside a citation, a
page stamp read as part of a date, a table of authorities rebuilt into cells
that put a case name in a different one from its own citation. Pass a shorter
list to decline some of them, or an empty one to take the converter's own
reading whole."""


@dataclass(frozen=True, slots=True)
class PreprocessingMetadata:
    """Backend provenance for the preprocessing stage."""

    backend: PreprocessingBackend = PreprocessingBackend.PLAIN_TEXT
    backend_version: str | None = None
    rules: tuple[Rule, ...] = ()
    """Which rules ran, in the order they ran.

    Empty means none did, which is not the same as a rule finding nothing: the
    two produce different text and therefore different offsets, so a document
    has to record which reading it was rendered under.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class PreprocessedDocument(DocumentBase):
    """Layer 2 text output consumed by citation extraction."""

    text: str
    preprocessing_metadata: PreprocessingMetadata
    index_spans: tuple[Span, ...] = ()
    """Regions of `text` holding a table of authorities, which cites nothing.

    An index entry lists a case; it attaches no proposition to it and makes no
    claim about any page. What `Rule.TABLE_OF_AUTHORITIES` marked, and
    empty when that rule did not run or the backend cannot tell -- plain text
    carries no structure, so absence here means unknown rather than none, and
    `rules` is what says which.
    """

    def __post_init__(self) -> None:
        if not self.text:
            msg = "PreprocessedDocument.text must not be empty"
            raise ValueError(msg)
