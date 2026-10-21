"""Read four source spans in one ML-cite inference, without normalizing fields."""

from __future__ import annotations

from functools import lru_cache

from mellea_lrc.field_extraction.models import (
    FieldContext,
    MLFieldPrediction,
    MLReaderConfig,
    validate_reader_inputs,
)
from mellea_lrc.parsing.span import Span

_FIELDS = ("case_name", "court", "date", "pin_cite")


def _load_runtime():
    """Keep torch and transformers optional for rule-only installations."""
    try:
        from . import _ml_cite_runtime
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] not in {
            "torch",
            "transformers",
            "huggingface_hub",
            "safetensors",
            "numpy",
        }:
            raise
        raise RuntimeError(
            "ML field extraction requires the ml-cite extra: uv sync --extra ml-cite"
        ) from error
    return _ml_cite_runtime


class MLCiteReader:
    """One cached checkpoint, producing grounded document-relative spans."""

    def __init__(self, config: MLReaderConfig):
        self.config = config
        runtime = _load_runtime()
        self._representation_error = runtime.RepresentationError
        self._model = runtime.load_model(
            repo_id=config.model_id,
            variant=config.variant,
            revision=config.revision,
            device=config.device,
        )

    def _encode_context(self, source: str, locator: Span):
        if not isinstance(locator, Span) or not 0 <= locator.start < locator.end <= len(source):
            raise ValueError("Locator must be a nonempty span inside the source")
        left = max(0, locator.start - self.config.context_before)
        right = min(len(source), locator.end + self.config.context_after)
        while True:
            context = source[left:right]
            record = {
                "context_text": context,
                "locator": {
                    "start": locator.start - left,
                    "end": locator.end - left,
                    "quote": source[locator.start : locator.end],
                },
            }
            try:
                return self._model.encode(record), context, Span(left, right)
            except self._representation_error as error:
                if left == locator.start and right == locator.end:
                    raise ValueError("Locator cannot fit the ML checkpoint's token limit") from error
                # Shrink source context, never the anchor or its source offsets.
                left = locator.start - (locator.start - left) // 2
                right = locator.end + (right - locator.end) // 2

    @staticmethod
    def _absolute_span(prediction, context: str, context_span: Span, field: str) -> Span | None:
        present, raw = getattr(prediction, "present", None), getattr(prediction, "span", None)
        if type(present) is not bool:
            raise ValueError(f"{field}: model presence must be boolean")
        if not present:
            if raw is not None:
                raise ValueError(f"{field}: absent prediction carries a span")
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"{field}: present prediction has no source span")
        start, end, quote = (raw.get(name) for name in ("start", "end", "quote"))
        if (
            type(start) is not int
            or type(end) is not int
            or not isinstance(quote, str)
            or not 0 <= start < end <= len(context)
            or context[start:end] != quote
        ):
            raise ValueError(f"{field}: model span does not match its source quote")
        return Span(context_span.start + start, context_span.start + end)

    def predict(
        self, source: str, locator: Span, *, context: FieldContext | None = None
    ) -> MLFieldPrediction:
        """Read one supplied source context, with the same contract as RuleFieldReader."""
        return self.predict_many(source, (locator,), contexts=(context,) if context is not None else None)[0]

    def predict_many(
        self,
        source: str,
        locators: tuple[Span, ...],
        *,
        contexts: tuple[FieldContext, ...] | None = None,
    ) -> tuple[MLFieldPrediction, ...]:
        """Batch locator-anchored inputs; each inference reads all four fields."""
        # Structural hints are the same source-only input accepted by rules.
        # ML still reads its trained context window and predicts all four heads;
        # rule boundaries do not suppress or rewrite model predictions.
        validate_reader_inputs(source, locators, contexts)
        encoded = [self._encode_context(source, span) for span in locators]
        results = []
        for start in range(0, len(encoded), self.config.batch_size):
            batch = encoded[start : start + self.config.batch_size]
            predictions = self._model.predict([item[0] for item in batch], [item[1] for item in batch])
            if not isinstance(predictions, list) or len(predictions) != len(batch):
                raise ValueError("ML checkpoint returned the wrong number of predictions")
            for fields, (_, context, context_span) in zip(predictions, batch, strict=True):
                if not isinstance(fields, dict) or set(fields) != set(_FIELDS):
                    raise ValueError("ML checkpoint must predict exactly the four citation fields")
                spans = {
                    name: self._absolute_span(fields[name], context, context_span, name) for name in _FIELDS
                }
                results.append(
                    MLFieldPrediction(
                        model_id=self.config.model_id,
                        revision=self.config.revision,
                        variant=self.config.variant,
                        context_span=context_span,
                        **spans,
                    )
                )
        for result, locator in zip(results, locators, strict=True):
            result.validate_source(source, locator)
        return tuple(results)


@lru_cache(maxsize=2)
def get_ml_reader(config: MLReaderConfig) -> MLCiteReader:
    """Reuse the loaded checkpoint across document and citation batches."""
    return MLCiteReader(config)
