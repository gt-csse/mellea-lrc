"""Named, complete model configurations selected explicitly by each stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.llm.config import LlmApiConfig, LlmOutputMode

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class LlmProfile:
    """One stage's model, transport, generation settings, and IVR budget.

    Credentials are referenced by environment variable and resolved lazily.
    Model settings never inherit from environment variables or another profile.
    """

    name: str
    model: str
    api_base: str
    api_key_env: str
    temperature: float = 0.0
    timeout_seconds: float = 300.0
    max_tokens: int = 8000
    max_attempts: int = 3
    output_mode: LlmOutputMode = LlmOutputMode.JSON_SCHEMA
    reasoning_effort: str | None = None
    service_tier: str | None = None

    def __post_init__(self) -> None:
        for name in ("name", "api_key_env"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("max_tokens", "max_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._binding("profile-validation")

    def _binding(self, api_key: str) -> LlmApiConfig:
        return LlmApiConfig(
            model=self.model,
            api_base=self.api_base,
            api_key=api_key,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
            output_mode=self.output_mode,
            reasoning_effort=self.reasoning_effort,
            service_tier=self.service_tier,
            profile_name=self.name,
        )

    def resolve(self, environ: Mapping[str, str] | None = None) -> LlmApiConfig:
        """Resolve only this profile's credential; never fall back to another key."""
        if environ is None:
            import os

            from dotenv import load_dotenv

            load_dotenv(override=False)
            environ = os.environ
        key = environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"Missing credential for profile {self.name}: {self.api_key_env}")
        return self._binding(key)


OPENROUTER_LUNA = LlmProfile(
    name="openrouter_luna",
    model="openai/gpt-6-luna",
    api_base="https://openrouter.ai/api/v1",
    api_key_env="MELLEA_LRC_OPENROUTER_API_KEY",
    reasoning_effort="low",
)

NRP_GLM = LlmProfile(
    name="nrp_glm",
    model="glm-5",
    api_base="https://ellm.nrp-nautilus.io/v1",
    api_key_env="MELLEA_LRC_NRP_API_KEY",
    reasoning_effort="low",
)

NRP_QWEN = LlmProfile(
    name="nrp_qwen",
    model="qwen3",
    api_base="https://ellm.nrp-nautilus.io/v1",
    api_key_env="MELLEA_LRC_NRP_API_KEY",
    reasoning_effort="low",
)

NRP_KIMI = LlmProfile(
    name="nrp_kimi",
    model="kimi",
    api_base="https://ellm.nrp-nautilus.io/v1",
    api_key_env="MELLEA_LRC_NRP_API_KEY",
    reasoning_effort="low",
)
