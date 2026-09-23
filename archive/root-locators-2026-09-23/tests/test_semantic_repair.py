"""Deferred semantic court/date repair has an explicit boundary."""

from __future__ import annotations

import asyncio

import pytest

from mellea_lrc.validation.field_checks.semantic_repair import repair_court_or_date_semantically


def test_semantic_court_repair_is_explicitly_deferred() -> None:
    """A field mismatch cannot silently turn into an unmeasured correction."""
    with pytest.raises(NotImplementedError, match="Semantic court repair is intentionally deferred"):
        asyncio.run(repair_court_or_date_semantically(field="court"))
