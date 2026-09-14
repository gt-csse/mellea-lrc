"""The roots as validation's identity stage left them.

The second growth is meant to run over roots that have been resolved: each one
looked up against the archives, its case name checked against the authority and
rewritten where the two disagree. That is what a leaf needs, because reaching a
root means matching a name, and it is the reason the leaves are grown after
validation rather than during the parse.

This replays a recorded identity run onto a fresh extraction, so the leaf pass
can be scored over the roots validation actually produced without calling the
stage again. A run's records are keyed by `citation_id`, which is a hash of a
citation's span and the characters at it, so the ids line up with any extraction
of the same text at the same relaxation. A record for a citation this run did
not read is skipped rather than guessed at.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from mellea_lrc.core.case_names import CaseName
from mellea_lrc.core.spans import Span

if TYPE_CHECKING:
    from pathlib import Path

    from mellea_lrc.extraction.types import Document

#: The fields an identity run corrects. `court` is on the citation like the
#: parties are; `case_name` is an object of its own.
_PARTY_FIELDS = ("plaintiff", "defendant", "court")


def _case_name(value: dict[str, Any]) -> CaseName:
    span = value.get("span") or {}
    return CaseName(
        span=Span(start=span.get("start", 0), end=span.get("end", 0)),
        text=value.get("text") or "",
        plaintiff=value.get("plaintiff"),
        defendant=value.get("defendant"),
    )


def corrections(artifact: Path) -> dict[str, list[tuple[str, Any]]]:
    """Each record's corrections, in the order the stage made them."""
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    made: dict[str, list[tuple[str, Any]]] = {}
    for record in payload.get("records", ()):
        for correction in record.get("corrections") or ():
            field = correction.get("field")
            if field == "case_name" or field in _PARTY_FIELDS:
                made.setdefault(record["citation_id"], []).append((field, correction.get("after")))
    return made


def settled(document: Document, artifact: Path) -> Document:
    """The document with an identity run's name corrections written onto `stated`.

    Only `stated` moves. `source` stays what the rules read, which is what makes
    the difference between the two arms visible: the same characters, read the
    same way, attached by a name that has since been checked.
    """
    if not artifact.exists():
        return document
    made = corrections(artifact)
    if not made:
        return document
    citations = []
    for record in document.citations:
        stated = record.stated
        for field, after in made.get(record.citation_id, ()):
            if after is None:
                continue
            value = _case_name(after) if field == "case_name" else after
            if hasattr(stated, field):
                stated = replace(stated, **{field: value})
        citations.append(record if stated is record.stated else replace(record, stated=stated))
    return replace(document, citations=tuple(citations))
