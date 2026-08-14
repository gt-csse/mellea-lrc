"""Locator grammar for Mellea citation extraction.

A locator is the part of a citation that says where the opinion is reported: either a reporter locator (volume + reporter + first page)
or a docket number (e.g. No. CV-03-8920).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mellea_lrc.core.spans import Span

_REPORTER_TOKEN = r"(?:[A-Za-z][A-Za-z.'’]*|\.?\d+[a-z]+)"
_REPORTER = rf"{_REPORTER_TOKEN}(?:\s*{_REPORTER_TOKEN}){{0,3}}"

_REPORTER_LOCATOR = re.compile(
    rf"(?P<volume>\d+)\s+(?P<reporter>{_REPORTER})\s*(?P<page>\d+)(?!\d)",
)
_DOCKET_LOCATOR = re.compile(
    r"(?:Nos?|Case\s+Nos?|Civ\.?\s+A(?:ction)?\s+Nos?)\.?\s*"
    r"(?P<docket>[A-Za-z0-9](?:[A-Za-z0-9:.\-]*[A-Za-z0-9])?)",
    flags=re.IGNORECASE,
)

_REPORTER_NOISE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Locator:
    """One parsed locator and where it sat in the string it was parsed from."""

    text: str
    span: Span
    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    docket: str | None = None

    @property
    def key(self) -> tuple[str, ...]:
        """Return the identity used to decide whether two locators agree.

        Reporter spelling is normalized (case, spaces, periods)
        """
        if self.docket is not None:
            return ("docket", self.docket.casefold())
        return (
            "reporter",
            self.volume or "",
            _REPORTER_NOISE.sub("", (self.reporter or "").casefold()),
            self.page or "",
        )


def parse_locator(value: str) -> Locator | None:
    """Return the first locator in value, or None when it contains none."""
    reporter_match = _REPORTER_LOCATOR.search(value)
    if reporter_match is not None:
        return Locator(
            text=reporter_match.group(0),
            span=Span(reporter_match.start(), reporter_match.end()),
            volume=reporter_match.group("volume"),
            reporter=reporter_match.group("reporter").strip(),
            page=reporter_match.group("page"),
        )
    docket_match = _DOCKET_LOCATOR.search(value)
    if docket_match is not None:
        return Locator(
            text=docket_match.group(0),
            span=Span(docket_match.start(), docket_match.end()),
            docket=docket_match.group("docket"),
        )
    return None


def find_locators(value: str) -> tuple[Locator, ...]:
    """Return every reporter locator in value, in order of appearance."""
    return tuple(
        Locator(
            text=match.group(0),
            span=Span(match.start(), match.end()),
            volume=match.group("volume"),
            reporter=match.group("reporter").strip(),
            page=match.group("page"),
        )
        for match in _REPORTER_LOCATOR.finditer(value)
    )
