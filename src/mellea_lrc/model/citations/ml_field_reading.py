"""The model prediction recorded at a citation decision node."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.parsing.span import Span


class MLFieldPrediction(BaseModel):
    """Four nullable source spans and the checkpoint that produced them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    revision: str
    variant: str
    context_span: Span
    case_name: Span | None = None
    court: Span | None = None
    date: Span | None = None
    pin_cite: Span | None = None

    @model_validator(mode="after")
    def _validate_prediction(self) -> Self:
        if any(not value.strip() for value in (self.model_id, self.revision, self.variant)):
            raise ValueError("ML prediction needs a model, revision, and variant")
        if self.context_span.start == self.context_span.end:
            raise ValueError("ML prediction context cannot be empty")
        for span in (self.case_name, self.court, self.date, self.pin_cite):
            if span is not None and not (
                self.context_span.start <= span.start < span.end <= self.context_span.end
            ):
                raise ValueError("Field spans must be nonempty and inside the prediction context")
        return self

    def validate_source(self, source: str, locator: Span) -> None:
        if not isinstance(source, str):
            raise TypeError("ML source must be a string")
        if not isinstance(locator, Span) or not 0 <= locator.start < locator.end <= len(source):
            raise ValueError("Locator must be a nonempty span inside the source")
        if not source[locator.start : locator.end].strip():
            raise ValueError("Target locator cannot be blank")
        if self.context_span.end > len(source):
            raise ValueError("ML prediction context is outside the source")
        if not self.context_span.start <= locator.start < locator.end <= self.context_span.end:
            raise ValueError("ML prediction context must contain the target locator")
        for span in (self.case_name, self.court, self.date, self.pin_cite):
            if span is not None and not source[span.start : span.end].strip():
                raise ValueError("Predicted field cannot be blank")


class MLFieldReading(BaseModel):
    """A single model invocation, separate from normalized field histories."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    prediction: MLFieldPrediction
