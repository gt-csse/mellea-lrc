"""Small JSON conversion helpers shared by serialization artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None


def serialize_dataclass(value: object) -> dict[str, JsonValue]:
    """Serialize every declared field of a dataclass instance."""
    if not is_dataclass(value) or isinstance(value, type):
        msg = f"Expected a dataclass instance, got {type(value).__name__}"
        raise TypeError(msg)
    return {field.name: serialize_value(getattr(value, field.name)) for field in fields(value)}


def serialize_value(value: object) -> JsonValue:
    """Convert supported immutable domain values into JSON-compatible values."""
    if isinstance(value, Enum):
        return serialize_value(value.value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return serialize_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [serialize_value(item) for item in value]
    msg = f"Cannot serialize {type(value).__name__} as JSON"
    raise TypeError(msg)


def require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    """Return a JSON object or raise a useful decoding error."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        msg = f"{name} must be a JSON object"
        raise ValueError(msg)
    return value


def require_list(value: object, *, name: str) -> list[object]:
    """Return a JSON array or raise a useful decoding error."""
    if not isinstance(value, list):
        msg = f"{name} must be a JSON array"
        raise ValueError(msg)
    return value
