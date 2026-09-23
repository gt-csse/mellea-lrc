"""Preprocessing layer public API.

One function and one list of rules: `preprocess(source, rules)`. Everything
else here is a type it returns.
"""

from mellea_lrc.model.documents import DocumentBase, SourceFormat, SourceMetadata
from mellea_lrc.model.preprocessed import (
    DEFAULT_RULES,
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
    Rule,
)
from mellea_lrc.preprocessing.pipeline import preprocess

__all__ = [
    "DEFAULT_RULES",
    "DocumentBase",
    "PreprocessedDocument",
    "PreprocessingBackend",
    "PreprocessingMetadata",
    "Rule",
    "SourceFormat",
    "SourceMetadata",
    "preprocess",
]
