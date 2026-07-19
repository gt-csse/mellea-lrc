"""Deeply immutable representations for JSON-shaped domain data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias, cast

JsonScalar: TypeAlias = str | int | float | bool | None
FrozenJsonValue: TypeAlias = JsonScalar | tuple["FrozenJsonValue", ...] | Mapping[str, "FrozenJsonValue"]
FrozenJsonObject: TypeAlias = Mapping[str, FrozenJsonValue]


@dataclass(frozen=True, slots=True, init=False)
class ExtraData:
    """Explicit, deeply immutable storage for unmodeled external fields."""

    values: FrozenJsonObject = field(default_factory=lambda: freeze_json_object({}))

    def __init__(self, values: Mapping[str, object] | None = None) -> None:
        object.__setattr__(self, "values", freeze_json_object(values or {}))

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-ready copy."""
        return thaw_json_object(self.values)

    def __bool__(self) -> bool:
        return bool(self.values)


def freeze_json_value(value: object) -> FrozenJsonValue:
    """Copy JSON-shaped data into tuples and read-only mappings."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = "JSON object keys must be strings"
                raise TypeError(msg)
            frozen[key] = freeze_json_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json_value(item) for item in value)
    msg = f"Unsupported JSON value: {type(value).__name__}"
    raise TypeError(msg)


def freeze_json_object(value: Mapping[str, object]) -> FrozenJsonObject:
    """Copy a JSON object into a deeply immutable mapping."""
    return cast("FrozenJsonObject", freeze_json_value(value))


def thaw_json_value(value: FrozenJsonValue) -> object:
    """Copy immutable JSON-shaped data into ordinary JSON containers."""
    if isinstance(value, Mapping):
        return {key: thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(item) for item in value]
    return value


def thaw_json_object(value: FrozenJsonObject) -> dict[str, object]:
    """Copy an immutable JSON object into a JSON-ready dictionary."""
    return cast("dict[str, object]", thaw_json_value(value))
