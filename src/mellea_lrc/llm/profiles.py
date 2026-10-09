"""Complete model packages and substage assignments read lazily from .env."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.configuration import read_env, required_setting
from mellea_lrc.llm.config import LlmApiConfig, LlmOutputMode

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class LlmProfile:
    """One substage's model, transport, generation settings, and IVR budget.

    Settings and credential references come from .env. There are no inherited
    profile settings or built-in model packages.
    """

    name: str
    model: str
    api_base: str
    api_key_env: str
    temperature: float
    timeout_seconds: float
    max_tokens: int
    max_attempts: int
    output_mode: LlmOutputMode
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

    @classmethod
    def from_env(cls, name: str, *, environ: Mapping[str, str | None] | None = None) -> LlmProfile:
        """Load one named package; explicit mappings allow isolated configuration tests."""
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError("Profile names must contain only ASCII letters, digits, and underscores")
        values = read_env() if environ is None else environ
        prefix = f"MELLEA_LRC_PROFILE_{name.upper()}_"

        def required(setting: str) -> str:
            return required_setting(values, prefix + setting)

        try:
            return cls(
                name=name,
                model=required("MODEL"),
                api_base=required("API_BASE"),
                api_key_env=required("API_KEY_ENV"),
                temperature=float(required("TEMPERATURE")),
                timeout_seconds=float(required("TIMEOUT_SECONDS")),
                max_tokens=int(required("MAX_TOKENS")),
                max_attempts=int(required("MAX_ATTEMPTS")),
                output_mode=LlmOutputMode(required("OUTPUT_MODE")),
                reasoning_effort=(values.get(prefix + "REASONING_EFFORT") or "").strip() or None,
                service_tier=(values.get(prefix + "SERVICE_TIER") or "").strip() or None,
            )
        except ValueError as error:
            raise ValueError(f"Invalid .env profile {name}: {error}") from error

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

    def resolve(self, environ: Mapping[str, str | None] | None = None) -> LlmApiConfig:
        """Resolve only this profile's credential; never fall back to another key."""
        values = read_env() if environ is None else environ
        key = (values.get(self.api_key_env) or "").strip()
        if not key:
            raise RuntimeError(f"Missing credential for profile {self.name}: {self.api_key_env}")
        return self._binding(key)


def load_profile(substage: str, *, environ: Mapping[str, str | None] | None = None) -> LlmProfile:
    """Select a substage's package using its .env assignment, with no fallback."""
    values = read_env() if environ is None else environ
    key = f"MELLEA_LRC_SUBSTAGE_{substage.replace('.', '_').upper()}_PROFILE"
    return LlmProfile.from_env(required_setting(values, key), environ=values)
