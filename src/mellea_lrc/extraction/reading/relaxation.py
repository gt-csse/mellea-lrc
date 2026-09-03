r"""How much separator damage eyecite's reporter patterns will tolerate.

eyecite generates every reporter regex with **literal single spaces** joining
volume, reporter and page::

    (?P<volume>[1-9]\d*) (?P<reporter>WL),? (?P<page>...)
                        ^                  ^

Any damage to those separators -- a space lost to table extraction, a doubled
space from justified text, a page break landing mid-citation -- makes the
citation vanish entirely rather than degrade. That is the worst failure
available to a verification tool: a citation nobody extracted is a citation
nobody checked, so a filing full of them earns "nothing to report" rather than
"could not be read".

Relaxing those joins is reporter-agnostic. It is not a fix for a particular
reporter but for how the regexes are generated, so one substitution reaches all
~6,800 of them.

Relaxation is a spectrum, not a switch, because the widest setting is not the
safest one. :class:`Relaxation` names the three points on it:

============  ==================  ==============================  ============
level         volume -> reporter  reporter -> page                punctuation
                                                                  in reporter
============  ==================  ==============================  ============
``NONE``      literal space       literal space                   as generated
``BOUNDED``   any whitespace      any, stopping at a blank line   relaxed
``FULL``      any whitespace      any whitespace                  relaxed
============  ==================  ==============================  ============

The two joins are relaxed differently at ``BOUNDED``, and the asymmetry is the
whole point of the level.

Between volume and reporter, a break leaves reporter and page still adjacent on
the far side, so whatever page is captured is the citation's own. Blank lines
are always safe there, and they are needed: ``937\n\nS.W.2d 796`` is a real
citation split by a page break.

Between reporter and page, the page number is what lands beyond the break --
and on pleading paper, that is where the margin line numbers are. Allowing a
blank line there reads ``214 F.3d\n\n1\n\n2\n\n3`` as ``214 F.3d 1`` when the
citation is ``214 F.3d 1058``. Not a miss but a *wrong page*, which sends
validation to a different case and returns a confident verdict about it.

``FULL`` accepts that hazard in exchange for the citations only it can reach.
Measured over 103 documents and 2,603 citations, widening that join beyond
``BOUNDED`` changed the parse in six documents: two correct recoveries and four
errors, two of which destroyed a citation that had parsed correctly before.
Removing page margins first fixes one of the four and none of the other three,
so ``FULL`` is a deliberate choice and not something preprocessing earns.
"""

from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

import ahocorasick
from eyecite.models import TokenExtractor
from eyecite.tokenizers import EXTRACTORS, AhocorasickTokenizer, default_tokenizer

if TYPE_CHECKING:
    from eyecite.tokenizers import Tokenizer


class Relaxation(str, Enum):
    """How much whitespace damage a citation may carry and still be found."""

    NONE = "none"
    """eyecite's patterns as published. Any damaged separator loses the citation."""

    BOUNDED = "bounded"
    """Separators may be any whitespace, except that page may not cross a blank line."""

    FULL = "full"
    """Every separator may be any whitespace, blank lines included."""


# Whitespace a citation may be broken by. The bounded form allows any amount of
# horizontal space and at most one line ending -- a citation may be split across
# a line or a page, never across a paragraph.
_ACROSS_BLOCKS = r"\s*"
_WITHIN_BLOCK = r"[^\S\r\n]*(?:\r?\n[^\S\r\n]*)?"


# Reporter groups produced by eyecite's ``_relax_ws`` often end in ``\s*``
# themselves, so that variants like "U. S." still match. The original pattern's
# literal trailing space forces such a group to give the space back on
# backtracking; replacing that space with a relaxed join removes the pressure,
# and the group keeps it -- yielding reporter="U.S. " and a corrupted locator.
# The ``(?<!\s)`` assertion restores the pressure without requiring a space to
# be present, so the group still cannot end on whitespace. It also reaches the
# alternation-shaped groups, where lifting a trailing ``\s*`` out of the group
# body by string surgery cannot cover every branch.
def _joins(relaxation: Relaxation) -> tuple[tuple[str, str], ...]:
    """The two substitutions that relax a generated reporter pattern."""
    page_gap = _ACROSS_BLOCKS if relaxation is Relaxation.FULL else _WITHIN_BLOCK
    return (
        (r") (?P<reporter>", rf"){_ACROSS_BLOCKS}(?P<reporter>"),
        (r"),? (?P<page>", rf")(?<!\s),?{page_gap}(?P<page>"),
    )


# Inside a reporter, eyecite already allows whitespace *after* each period --
# `N\.\s*Y\.\s*2d` matches `N.Y.2d` and `N.Y. 2d`. It allows none before one,
# and none at all around an apostrophe, so `N.Y .2d` and `F. App ' x` match
# nothing. Extraction produces both: `58  N.Y .2d  916` in document 008 of
# false-citation-bench, and `777 F. App ' x 516` in a filing printed
# `777 F. App'x 516` on the page. Both are real citations no tokenizer reaches.
_REPORTER_GROUP = re.compile(r"\(\?P<reporter>((?:[^()\\]|\\.)*)\)")
# What a reporter is written with inside eyecite's own pattern: the escaped
# period it emits, and the apostrophes it leaves bare.
_TIGHT_PUNCTUATION = re.compile(r"\\\.|['’]")


def _relax(regex: str, joins: tuple[tuple[str, str], ...]) -> str:
    for old, new in joins:
        regex = regex.replace(old, new)
    return _REPORTER_GROUP.sub(_relax_reporter_punctuation, regex)


def _relax_reporter_punctuation(match: re.Match[str]) -> str:
    """Let whitespace sit on either side of the punctuation inside a reporter."""
    body = _TIGHT_PUNCTUATION.sub(lambda found: rf"\s*{found.group()}\s*", match.group(1))
    return f"(?P<reporter>{body})"


class _RelaxedTokenizer(AhocorasickTokenizer):
    """Prefiltered tokenizer that honours its own extractor list.

    ``AhocorasickTokenizer`` builds its prefilter from the module-level
    ``EXTRACTORS``, so a tokenizer constructed with replacement extractors
    silently never runs them. Rebuilding the filters from ``self.extractors``
    keeps the prefilter -- without it every one of the ~6,800 extractors runs
    against every document, which is far too slow to be usable.
    """

    def __post_init__(self) -> None:
        self.unfiltered_extractors = {e for e in self.extractors if not e.strings}
        self.case_sensitive_filter = self._filter(case_sensitive=True)
        self.case_insensitive_filter = self._filter(case_sensitive=False)

    def _filter(self, *, case_sensitive: bool) -> ahocorasick.Automaton:
        pairs = [
            (s.replace(" ", "") if case_sensitive else s.replace(" ", "").lower(), e)
            for e in self.extractors
            if e.strings and bool(e.flags & re.I) is not case_sensitive
            for s in e.strings
        ]
        return self.make_ahocorasick_filter(pairs)


@lru_cache(maxsize=len(Relaxation))
def tokenizer_for(relaxation: Relaxation) -> Tokenizer:
    """Build the tokenizer for one relaxation level.

    Rebuilding ~6,800 patterns is not cheap, so each level is built once and
    reused. ``NONE`` is eyecite's own shared tokenizer, untouched.
    """
    if relaxation is Relaxation.NONE:
        return default_tokenizer
    joins = _joins(relaxation)
    return _RelaxedTokenizer(
        extractors=[
            TokenExtractor(
                regex=_relax(extractor.regex, joins),
                constructor=extractor.constructor,
                extra=extractor.extra,
                flags=extractor.flags,
                strings=extractor.strings,
            )
            for extractor in EXTRACTORS
        ]
    )
