"""Types shared by the active preprocessing layer."""

from mellea_lrc.model.documents import DocumentBase, SourceFormat, SourceMetadata
from mellea_lrc.model.extraction import (
    Citation,
    CitationDate,
    CitationField,
    CitationKind,
    Colocation,
    Document,
    Node,
    Operation,
    OperationKind,
)
from mellea_lrc.model.preprocessed import (
    DEFAULT_RULES,
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
    Rule,
)
from mellea_lrc.model.spans import Span

__all__ = [
    "DEFAULT_RULES",
    "Citation",
    "CitationDate",
    "CitationField",
    "CitationKind",
    "Colocation",
    "Document",
    "DocumentBase",
    "Node",
    "Operation",
    "OperationKind",
    "PreprocessedDocument",
    "PreprocessingBackend",
    "PreprocessingMetadata",
    "Rule",
    "SourceFormat",
    "SourceMetadata",
    "Span",
]
