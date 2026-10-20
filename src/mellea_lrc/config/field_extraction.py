"""Explicit ML inference switch and independent citation-field span sources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mellea_lrc.configuration import read_env, required_setting

if TYPE_CHECKING:
    from mellea_lrc.field_extraction.models import MLFieldPrediction


class FieldExtractionConfig(BaseModel):
    """Inference and field selection are separate; no setting has a hidden policy default."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ml_enabled: bool = Field(strict=True)
    case_name: Literal["rule", "ml"]
    court: Literal["rule", "ml"]
    date: Literal["rule", "ml"]
    pin_cite: Literal["rule", "ml"]
    model_id: str | None = None
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    variant: str | None = None
    device: str | None = None
    context_before: int | None = Field(default=None, ge=0, strict=True)
    context_after: int | None = Field(default=None, ge=0, strict=True)
    batch_size: int | None = Field(default=None, ge=1, strict=True)

    @field_validator("model_id", "variant", "device")
    @classmethod
    def nonempty_setting(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("ML settings must not be blank")
        return value

    @model_validator(mode="after")
    def check_ml_configuration(self) -> Self:
        if not self.ml_enabled:
            selected = [
                name for name in ("case_name", "court", "date", "pin_cite") if getattr(self, name) == "ml"
            ]
            if selected:
                raise ValueError("ML field sources require ML_ENABLED=true: " + ", ".join(selected))
        else:
            missing = [
                name
                for name in (
                    "model_id",
                    "revision",
                    "variant",
                    "device",
                    "context_before",
                    "context_after",
                    "batch_size",
                )
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"ML field extraction requires: {', '.join(missing)}")
        return self

    def validate_prediction(self, prediction: MLFieldPrediction) -> MLFieldPrediction:
        """Reuse saved spans only when they belong to the requested checkpoint."""
        if (prediction.model_id, prediction.revision, prediction.variant) != (
            self.model_id,
            self.revision,
            self.variant,
        ):
            raise ValueError(
                "ML prediction model, revision, and variant must match the configured model; "
                "rewind before ML prediction to change checkpoints"
            )
        return prediction

    def ml_reader_config(self):
        """Pass only model settings into the standalone span reader."""
        from mellea_lrc.field_extraction.models import MLReaderConfig

        if not self.ml_enabled:
            raise ValueError("ML field reader requires ML_ENABLED=true")
        return MLReaderConfig(**{name: getattr(self, name) for name in MLReaderConfig.model_fields})

    def validate_runtime(self) -> None:
        """Fail before document work if an enabled checkpoint cannot be loaded."""
        if self.ml_enabled:
            from mellea_lrc.field_extraction.ml_cite import get_ml_reader

            get_ml_reader(self.ml_reader_config())

    @classmethod
    def from_env(cls, *, environ: Mapping[str, str | None] | None = None) -> Self:
        """Read all field sources explicitly; require checkpoint settings when enabled."""
        values = read_env() if environ is None else environ
        prefix = "MELLEA_LRC_FIELD_EXTRACTION_"
        enabled = required_setting(values, prefix + "ML_ENABLED")
        if enabled not in {"true", "false"}:
            raise ValueError(f"{prefix}ML_ENABLED must be exactly true or false")
        settings: dict[str, object] = {"ml_enabled": enabled == "true"}
        for name in ("case_name", "court", "date", "pin_cite"):
            settings[name] = required_setting(values, prefix + name.upper())
        if enabled == "true":
            for name in (
                "model_id",
                "revision",
                "variant",
                "device",
                "context_before",
                "context_after",
                "batch_size",
            ):
                value = required_setting(values, prefix + name.upper())
                settings[name] = (
                    int(value) if name in {"context_before", "context_after", "batch_size"} else value
                )
        return cls.model_validate(settings)
