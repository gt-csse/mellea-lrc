"""Source-only field predictions shared by the rule and ML readers."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mellea_lrc.parsing.span import Span


class MLReaderConfig(BaseModel):
    """Explicit model settings; this service does not read environment variables."""

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


class FieldPrediction(BaseModel):
    """Four quoted spans or absence, indexed to the supplied source text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    context_span: Span
    case_name: Span | None = None
    court: Span | None = None
    date: Span | None = None
    pin_cite: Span | None = None

    @model_validator(mode="after")
    def _validate_spans(self) -> Self:
        if self.context_span.start == self.context_span.end:
            raise ValueError("Field prediction context cannot be empty")
        for span in (self.case_name, self.court, self.date, self.pin_cite):
            if span is not None and not (
                self.context_span.start <= span.start < span.end <= self.context_span.end
            ):
                raise ValueError("Field spans must be nonempty and inside the prediction context")
        return self

    def validate_source(self, source: str, locator_span: Span) -> None:
        """Check every span, including fields a pipeline may choose not to use."""
        if not isinstance(source, str):
            raise TypeError("Field prediction source must be a string")
        if self.context_span.end > len(source):
            raise ValueError("Field prediction context is outside the source")
        if not (self.context_span.start <= locator_span.start < locator_span.end <= self.context_span.end):
            raise ValueError("Field prediction context must contain the target locator")
        if not source[locator_span.start : locator_span.end].strip():
            raise ValueError("Target locator cannot be blank")
        for span in (self.case_name, self.court, self.date, self.pin_cite):
            if span is not None and not source[span.start : span.end].strip():
                raise ValueError("Predicted field cannot be blank")


class MLFieldPrediction(FieldPrediction):
    """A field prediction with the identity of the model that produced it."""

    model_id: str
    revision: str
    variant: str

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if any(not value.strip() for value in (self.model_id, self.revision, self.variant)):
            raise ValueError("ML prediction needs a model, revision, and variant")
        return self
