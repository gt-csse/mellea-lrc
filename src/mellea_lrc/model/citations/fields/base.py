"""Source-grounded, citation-local field readings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mellea_lrc.parsing.span import Span

T = TypeVar("T")


def require_all_json_properties(schema: dict[str, object]) -> None:
    """Require nullable fields explicitly in structured model responses."""
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return
    schema["required"] = list(properties)
    for property_schema in properties.values():
        if isinstance(property_schema, dict):
            property_schema.pop("default", None)


def source_quote(source: str, span: Span) -> str:
    """Return an exact nonempty slice or fail at the quoting boundary."""
    if span.start == span.end or span.end > len(source):
        raise ValueError("Field span is outside the source or empty")
    quote = source[span.start : span.end]
    if not quote.strip():
        raise ValueError("Field quote is blank")
    return quote


def normalization_record(normalize: Callable[[], T]) -> dict[str, object]:
    """Keep a failed reading without mistaking it for a normalized value."""
    try:
        value = normalize()
    except ValueError as exc:
        return {
            "normalizable": False,
            "normalized": None,
            "normalization_error": str(exc) or type(exc).__name__,
        }
    return {"normalizable": True, "normalized": value, "normalization_error": None}


class CitationField(BaseModel, Generic[T]):
    """A field reading and the node that made it; subclasses own normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    node_id: str
    normalizable: bool
    # Pydantic needs a field for lossless checkpoints. This deliberately names
    # the unchecked storage; callers must use get_normalized().
    unchecked_normalized: T | None = Field(alias="normalized", repr=False)
    normalization_error: str | None
    quote: str | None = None
    span: Span | None = None

    def get_normalized(self) -> T:
        """Return a typed value or fail explicitly; never return None."""
        if not self.normalizable:
            raise ValueError(f"{type(self).__name__} is not normalizable: {self.normalization_error}")
        value = self.unchecked_normalized
        if value is None:
            raise ValueError("Normalizable field has no stored value")
        return value

    def validate_normalization(self, normalize: Callable[[], T]) -> None:
        """Check the stored outcome against the field's parsing rule."""
        if self.normalizable:
            if self.unchecked_normalized != normalize():
                raise ValueError("Field normalization does not match its quote")
            return
        # A failed reading is a historical fact. A later normalizer upgrade may
        # recognize its quote; loading the earlier checkpoint must still work.

    @model_validator(mode="after")
    def _validate_reading(self) -> Self:
        if self.normalizable:
            if self.unchecked_normalized is None or self.normalization_error is not None:
                raise ValueError("Normalizable field needs a value and no error")
            if isinstance(self.unchecked_normalized, str) and not self.unchecked_normalized.strip():
                raise ValueError("Normalized string is blank")
        elif self.unchecked_normalized is not None or not self.normalization_error:
            raise ValueError("Unnormalizable field needs an error and no value")
        if (self.quote is None) != (self.span is None):
            raise ValueError("Quote and span must be present together")
        if self.quote is not None and not self.quote.strip():
            raise ValueError("Field quote is blank")
        return self

    def validate_source(self, source: str) -> None:
        """Recheck exact source grounding after loading a document."""
        if self.span is not None and source_quote(source, self.span) != self.quote:
            raise ValueError("Field quote does not match its source span")


class NotStatedField(BaseModel):
    """An explicit absence reading that preserves every earlier field value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    kind: Literal["not_stated"] = "not_stated"

    @property
    def quote(self) -> None:
        return None

    @property
    def span(self) -> None:
        return None

    @property
    def normalizable(self) -> bool:
        return False

    def get_normalized(self) -> None:
        raise ValueError("Field is not stated; no normalized value is available")


FieldReading = TypeVar("FieldReading", bound=CitationField)


def current_reading(log: tuple[FieldReading | NotStatedField, ...] | None) -> FieldReading | None:
    """Read only the latest field entry; explicit absence never revives an older value."""
    if not log or isinstance(log[-1], NotStatedField):
        return None
    return log[-1]
