"""Tests for literal regex fragments compiled by the shared fuzzy service."""

from __future__ import annotations

import re

import pytest

from mellea_lrc.core.fuzziness import FuzzinessOption, fuzzy_literal


def test_fuzzy_literal_is_literal_under_a_perfect_match_policy() -> None:
    pattern = fuzzy_literal("Civil Action No.", FuzzinessOption.perfect_match())

    assert re.fullmatch(pattern, "Civil Action No.")
    assert not re.fullmatch(pattern, "Civil   Action  No.")
    assert not re.fullmatch(pattern, "CivilAction No.")
    assert not re.fullmatch(pattern, "Civil\nAction No.")


def test_fuzzy_literal_relaxes_horizontal_whitespace() -> None:
    pattern = fuzzy_literal("Civil Action No.", FuzzinessOption.whitespace_relaxation())

    assert re.fullmatch(pattern, "Civil Action No.")
    assert re.fullmatch(pattern, "CivilActionNo.")
    assert re.fullmatch(pattern, "Civil     Action\t\tNo.")
    assert re.fullmatch(pattern, "Civil Action No .")
    assert re.fullmatch(pattern, "Civil Action No\t.")
    assert not re.fullmatch(pattern, "Civil\nAction No.")


def test_fuzzy_literal_can_include_line_breaks_when_requested() -> None:
    pattern = fuzzy_literal("Civil Action No.", FuzzinessOption.whitespace_relaxation(), newline=True)

    assert re.fullmatch(pattern, "Civil\nAction No.")
    assert re.fullmatch(pattern, "Civil\r\n\n\nAction\tNo.")


def test_fuzzy_literal_escapes_non_whitespace_characters() -> None:
    pattern = fuzzy_literal("Civ. A. No.", FuzzinessOption.whitespace_relaxation())

    assert re.fullmatch(pattern, "Civ. A. No.")
    assert re.fullmatch(pattern, "Civ.A.No.")
    assert re.fullmatch(pattern, "Civ . A . No .")
    assert not re.fullmatch(pattern, "CivXA.No.")


def test_fuzzy_literal_rejects_edit_distance_discovery() -> None:
    with pytest.raises(ValueError, match="EDIT_DISTANCE"):
        fuzzy_literal("No.", FuzzinessOption.edit_distance(maximum_edits=1))
