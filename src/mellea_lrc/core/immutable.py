"""Deeply immutable representations for JSON-shaped domain data."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeAlias, cast

JsonScalar: TypeAlias = str | int | float | bool | None
FrozenJsonValue: TypeAlias = (
    JsonScalar | tuple["FrozenJsonValue", ...] | Mapping[str, "FrozenJsonValue"]
)
FrozenJsonObject: TypeAlias = Mapping[str, FrozenJsonValue]
FrozenStringMap: TypeAlias = Mapping[str, str]


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


def freeze_string_map(value: Mapping[str, str]) -> FrozenStringMap:
    """Copy a string mapping into a read-only mapping."""
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        msg = "String mappings require string keys and values"
        raise TypeError(msg)
    return MappingProxyType(dict(value))
