"""Source-only field predictions shared by the rule and ML readers."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mellea_lrc.parsing.span import Span


class FieldContext(BaseModel):
    """Source-derived reading boundaries, without document or annotation state.

    Colocated locators share the prefix before their group and the suffix
    after it. Pinpoints remain local to each locator. Known short names are
    optional source-derived aliases, never normalized or gold field values.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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

    def validate_source(self, source: str, locator: Span) -> None:
        """Reject impossible boundaries rather than quietly changing context."""
        if any(span.end > len(source) for span in (self.name_before, self.court_date_after, self.pin_region)):
            raise ValueError("Field reading context is outside the source")
        if self.name_before.end > locator.start:
            raise ValueError("Case-name prefix must end before the target locator")
        if self.court_date_after.start < locator.end:
            raise ValueError("Court/date suffix must start after the target locator")
        if self.kind == "short_reporter":
            if not locator.start <= self.pin_region.start <= self.pin_region.end <= locator.end:
                raise ValueError("Short reporter pinpoint region must be inside the target locator")
        elif self.pin_region.start < locator.end:
            raise ValueError("Full-citation pinpoint region must start after the target locator")


def validate_reader_inputs(
    source: str, locators: tuple[Span, ...], contexts: tuple[FieldContext, ...] | None
) -> None:
    """Common fail-fast source contract used by both public readers."""
    if not isinstance(source, str):
        raise TypeError("Field prediction source must be a string")
    if not isinstance(locators, tuple) or any(not isinstance(locator, Span) for locator in locators):
        raise TypeError("Field locators must be a tuple of Spans")
    if contexts is not None and (
        not isinstance(contexts, tuple)
        or len(contexts) != len(locators)
        or any(not isinstance(context, FieldContext) for context in contexts)
    ):
        raise ValueError("Reading contexts must contain exactly one FieldContext for each locator")
    for index, locator in enumerate(locators):
        if (
            not 0 <= locator.start < locator.end <= len(source)
            or not source[locator.start : locator.end].strip()
        ):
            raise ValueError("Target locator is outside the source, empty, or blank")
        if contexts is not None:
            contexts[index].validate_source(source, locator)


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
