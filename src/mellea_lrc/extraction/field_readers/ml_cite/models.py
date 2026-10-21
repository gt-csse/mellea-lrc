"""Typed model configuration and source-grounded ML predictions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mellea_lrc.parsing.span import Span


class MLReaderConfig(BaseModel):
    """Explicit model settings, including the checkpoint's input window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    revision: str
    variant: str
    device: str
    context_before: int = Field(ge=0, strict=True)
    context_after: int = Field(ge=0, strict=True)
    batch_size: int = Field(ge=1, strict=True)

    @field_validator("model_id", "revision", "variant", "device")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Model settings must not be blank")
        return value


def validate_reader_inputs(source: str, locators: tuple[Span, ...]) -> None:
    """Reject an invalid batch before any encoding or inference."""
    if not isinstance(source, str):
        raise TypeError("ML source must be a string")
    if not isinstance(locators, tuple) or any(not isinstance(span, Span) for span in locators):
        raise TypeError("ML locators must be a tuple of Spans")
    for span in locators:
        if not 0 <= span.start < span.end <= len(source):
            raise ValueError("Locator must be a nonempty span inside the source")
        if not source[span.start : span.end].strip():
            raise ValueError("Target locator cannot be blank")
