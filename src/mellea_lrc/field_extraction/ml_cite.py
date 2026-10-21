"""Read four source spans in one ML-cite inference, without normalizing fields."""

from __future__ import annotations

from functools import lru_cache

from mellea_lrc.field_extraction.models import (
    LocatorContext,
    MLFieldPrediction,
    MLReaderConfig,
    validate_contexts,
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

    def _encode_context(self, context: LocatorContext):
        start = context.locator.start - context.source_span.start
        end = context.locator.end - context.source_span.start
        record = {
            "context_text": context.text,
            "locator": {"start": start, "end": end, "quote": context.text[start:end]},
        }
        try:
            return self._model.encode(record)
        except self._representation_error as error:
            raise ValueError(
                "Supplied locator context cannot fit the ML checkpoint's token limit; "
                "the reader will not crop or change the shared context"
            ) from error

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

    def predict(self, context: LocatorContext) -> MLFieldPrediction:
        """Read exactly the same context accepted by RuleFieldReader."""
        return self.predict_many((context,))[0]

    def predict_many(
        self,
        contexts: tuple[LocatorContext, ...],
    ) -> tuple[MLFieldPrediction, ...]:
        """Batch exact supplied contexts; each inference reads all four fields."""
        validate_contexts(contexts)
        encoded = [self._encode_context(context) for context in contexts]
        results = []
        for start in range(0, len(encoded), self.config.batch_size):
            batch = encoded[start : start + self.config.batch_size]
            batch_contexts = contexts[start : start + self.config.batch_size]
            predictions = self._model.predict(batch, [context.text for context in batch_contexts])
            if not isinstance(predictions, list) or len(predictions) != len(batch):
                raise ValueError("ML checkpoint returned the wrong number of predictions")
            for fields, context in zip(predictions, batch_contexts, strict=True):
                if not isinstance(fields, dict) or set(fields) != set(_FIELDS):
                    raise ValueError("ML checkpoint must predict exactly the four citation fields")
                spans = {
                    name: self._absolute_span(fields[name], context.text, context.source_span, name)
                    for name in _FIELDS
                }
                results.append(
                    MLFieldPrediction(
                        model_id=self.config.model_id,
                        revision=self.config.revision,
                        variant=self.config.variant,
                        context_span=context.source_span,
                        **spans,
                    )
                )
        for result, context in zip(results, contexts, strict=True):
            result.validate_context(context)
        return tuple(results)


@lru_cache(maxsize=2)
def get_ml_reader(config: MLReaderConfig) -> MLCiteReader:
    """Reuse the loaded checkpoint across document and citation batches."""
    return MLCiteReader(config)
