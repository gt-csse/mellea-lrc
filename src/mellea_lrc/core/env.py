"""Strict project-local environment loading built on python-dotenv.

python-dotenv handles comment stripping, quoting, and escape sequences; this
module adds the two project-specific policies that python-dotenv does not
enforce on its own:

* duplicate keys are rejected rather than silently overwritten, and
* values are not written into the process environment unless ``override`` is
  explicitly requested.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import dotenv_values

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path


def _check_no_duplicate_keys(path: Path) -> None:
    """Raise ``RuntimeError`` if ``path`` defines any key more than once."""
    seen: dict[str, int] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key in seen:
            msg = f"Duplicate environment variable in {path} at line {line_number}: {key}"
            raise RuntimeError(msg)
        seen[key] = line_number


def read_env_file(path: Path) -> dict[str, str]:
    """Read a dotenv file into a mapping, rejecting duplicate keys."""
    if not path.exists():
        return {}

    _check_no_duplicate_keys(path)
    raw = dotenv_values(path)
    return {key: (value if value is not None else "") for key, value in raw.items()}


def load_env_file(
    path: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> None:
    """Load a dotenv file into an environment mapping.

    Process values are preserved unless ``override`` is ``True``.
    """
    target = os.environ if environ is None else environ
    for key, value in read_env_file(path).items():
        if override or key not in target:
            target[key] = value
