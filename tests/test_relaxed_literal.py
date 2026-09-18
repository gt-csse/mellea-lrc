"""Tests for literal regex fragments with an explicit whitespace switch."""

from __future__ import annotations

import re

from mellea_lrc.extraction.reading.relaxation import relaxed_literal


def test_relaxed_literal_is_literal_when_whitespace_is_false() -> None:
    pattern = relaxed_literal("Civil Action No.", whitespace=False)

    assert re.fullmatch(pattern, "Civil Action No.")
    assert not re.fullmatch(pattern, "Civil   Action  No.")
    assert not re.fullmatch(pattern, "CivilAction No.")
    assert not re.fullmatch(pattern, "Civil\nAction No.")


def test_relaxed_literal_relaxes_horizontal_whitespace_by_default() -> None:
    pattern = relaxed_literal("Civil Action No.")

    assert re.fullmatch(pattern, "Civil Action No.")
    assert re.fullmatch(pattern, "CivilActionNo.")
    assert re.fullmatch(pattern, "Civil     Action\t\tNo.")
    assert not re.fullmatch(pattern, "Civil\nAction No.")


def test_relaxed_literal_can_include_line_breaks_when_requested() -> None:
    pattern = relaxed_literal("Civil Action No.", whitespace=True, newline=True)

    assert re.fullmatch(pattern, "Civil\nAction No.")
    assert re.fullmatch(pattern, "Civil\r\n\n\nAction\tNo.")


def test_relaxed_literal_escapes_non_whitespace_characters() -> None:
    pattern = relaxed_literal("Civ. A. No.", whitespace=True)

    assert re.fullmatch(pattern, "Civ. A. No.")
    assert re.fullmatch(pattern, "Civ.A.No.")
    assert not re.fullmatch(pattern, "CivXA.No.")
