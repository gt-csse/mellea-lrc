r"""A locator whose reporter is set in capitals.

eyecite registers one extractor per reporter string and the reporter extractors
are case-sensitive, so `33 F.4TH 693` is not a citation while `33 F.4th 693` is.
Reporters that are already all capitals are unaffected -- `550 U.S. 544` reads
the same either way -- so the exposure is the 67 of 80 reporter abbreviations in
this project's corpora that carry lower-case letters.

**This is a candidate generator rather than a tokenizer fix on purpose.**
Hardening it means making every reporter spelling case-insensitive, which needs
a fix to the Aho-Corasick prefilter (moving all the string extractors to one
side leaves the other empty and pyahocorasick raises) and a precision
measurement nobody has. It would buy two citations. Proposing two candidates for
review costs two calls. See `exploration/notes/candidates-and-adjudication.md`.

Both candidates it found are real, and one of them matters twice over: the
missed full citation is what a later `Dalla-Longa, 33 F.4th at 695` needed to
resolve against, so one unread reporter costs a citation and an attribution.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING

from mellea_lrc.adjudication.types import Candidate, CandidateKind
from mellea_lrc.core.spans import Span

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mellea_lrc.extraction.types import ExtractedDocument

WINDOW = 120
_GENERATOR = "uppercase_reporters"


@lru_cache(maxsize=1)
def _patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """One pattern per reporter that has an upper-case form worth looking for.

    Built from the reporter database rather than a list, so a reporter this
    project has never seen is covered the day it appears.
    """
    from reporters_db import EDITIONS

    built = []
    for name in EDITIONS:
        if name.upper() == name:
            continue  # nothing to get wrong
        letters = r"\s*".join(re.escape(character) for character in name.upper() if not character.isspace())
        built.append((name, re.compile(rf"\b\d{{1,4}}\s+{letters}\s+\d{{1,4}}\b")))
    return tuple(built)


def uppercase_reporters(document: ExtractedDocument) -> Iterator[Candidate]:
    """Propose locator-shaped spans whose reporter is written in capitals."""
    text = document.text
    taken = [(item.locator_span.start, item.locator_span.end) for item in document.citations]
    seen: set[tuple[int, int]] = set()
    for name, pattern in _patterns():
        for match in pattern.finditer(text):
            if any(start <= match.start() < end for start, end in taken):
                continue
            if (match.start(), match.end()) in seen:
                continue
            seen.add((match.start(), match.end()))
            yield Candidate(
                generator=_GENERATOR,
                kind=CandidateKind.LOCATOR,
                span=Span(start=match.start(), end=match.end()),
                window=Span(
                    start=max(0, match.start() - WINDOW),
                    end=min(len(text), match.end() + WINDOW),
                ),
                note=(
                    f"reads as a locator in {name!r} if the reporter is not case-sensitive; "
                    f"extraction recorded nothing here"
                ),
            )
