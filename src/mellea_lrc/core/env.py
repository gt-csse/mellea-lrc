"""Small, strict helpers for project-local environment files."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path


def read_env_file(path: Path) -> dict[str, str]:
    """Read a simple dotenv file and reject duplicate variable names."""
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in values:
            msg = f"Duplicate environment variable in {path} at line {line_number}: {key}"
            raise RuntimeError(msg)
        values[key] = value.strip().strip("\"'")
    return values


def load_env_file(
    path: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> None:
    """Load a strict dotenv file into an environment mapping."""
    target = os.environ if environ is None else environ
    for key, value in read_env_file(path).items():
        if override or key not in target:
            target[key] = value
