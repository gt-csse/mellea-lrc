"""Source-only field predictions shared by the rule and ML readers."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mellea_lrc.parsing.span import Span


class LocatorContext(BaseModel):
    """The complete, shared reader input, prepared by the prediction stage.

    Text is one exact source excerpt. Every span uses absolute source offsets;
    neither reader discovers neighboring locators or changes these boundaries.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    source_span: Span
    locator: Span
    kind: Literal["reporter", "docket", "short_reporter"]
    name_before: Span
    court_date_after: Span
    pin_region: Span
    known_names: tuple[str, ...] = ()

    @field_validator("known_names")
    @classmethod
    def _valid_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in names):
            raise ValueError("Known case names must be nonempty strings")
        return names

    @model_validator(mode="after")
    def _validate_context(self) -> Self:
        if not self.text or len(self.text) != self.source_span.end - self.source_span.start:
            raise ValueError("Locator context text must exactly fill its source span")
        for span in (self.locator, self.name_before, self.court_date_after, self.pin_region):
            if not self.source_span.start <= span.start <= span.end <= self.source_span.end:
                raise ValueError("Locator context boundaries must be inside its source span")
        start, end = self.locator.start - self.source_span.start, self.locator.end - self.source_span.start
        if start == end or not self.text[start:end].strip():
            raise ValueError("Target locator must be nonempty and nonblank")
        if self.name_before.end > self.locator.start:
            raise ValueError("Case-name prefix must end before the target locator")
        if self.court_date_after.start < self.locator.end:
            raise ValueError("Court/date suffix must start after the target locator")
        if self.kind == "short_reporter":
            if not self.locator.start <= self.pin_region.start <= self.pin_region.end <= self.locator.end:
                raise ValueError("Short reporter pinpoint region must be inside the target locator")
        elif self.pin_region.start < self.locator.end:
            raise ValueError("Full-citation pinpoint region must start after the target locator")
        return self


def validate_contexts(contexts: tuple[LocatorContext, ...]) -> None:
    """Fail before either reader starts a batch with malformed input."""
    if not isinstance(contexts, tuple) or any(
        not isinstance(context, LocatorContext) for context in contexts
    ):
        raise TypeError("Reader input must be a tuple of LocatorContexts")
    for context in contexts:
        context._validate_context()


class MLReaderConfig(BaseModel):
    """Explicit model settings; this service does not read environment variables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    revision: str
    variant: str
    device: str
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

    def validate_context(self, context: LocatorContext) -> None:
        """A reader must return spans within the exact context it received."""
        if self.context_span != context.source_span:
            raise ValueError("Field prediction must preserve the supplied locator context")
        for span in (self.case_name, self.court, self.date, self.pin_cite):
            if span is not None:
                start, end = span.start - context.source_span.start, span.end - context.source_span.start
                if not context.text[start:end].strip():
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
