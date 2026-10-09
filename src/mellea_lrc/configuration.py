"""The shared reader for explicit runtime settings in the nearest .env file."""

from __future__ import annotations

from collections.abc import Mapping

from dotenv import dotenv_values, find_dotenv


def read_env() -> Mapping[str, str | None]:
    """Read .env afresh without changing or inheriting the process environment."""
    path = find_dotenv(usecwd=True)
    if not path:
        raise RuntimeError("No .env found; copy .env.example to .env and configure the required settings")
    return dotenv_values(path, interpolate=False)


def required_setting(values: Mapping[str, str | None], key: str) -> str:
    """Reject absent or blank required settings rather than inventing a default."""
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Missing .env setting: {key}")
    return value.strip()
