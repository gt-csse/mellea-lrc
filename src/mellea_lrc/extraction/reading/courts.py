r"""Resolve the court a citation's parenthetical names.

eyecite matches a court by spelling. It strips the punctuation out of the
parenthetical, lowercases it, and compares against one field in courts-db --
``citation_string`` -- taking an exact match if there is one and otherwise the
last court whose string merely *starts with* what was written.

Three things go wrong with that, and all three are visible on
``false-citation-bench``.

**courts-db carries one spelling per court and no aliases**, and it is not
consistent about ordinals::

    ca2    '2d Cir.'        the form the Bluebook prescribes
    ca3    '3rd Cir.'       not the form the Bluebook prescribes
    bap2   '2nd Cir. BAP'

So ``3d Cir.`` -- correct Bluebook, and what filings actually write -- resolves
to nothing, while ``3rd Cir.`` resolves. Six citations here lose their court to
that alone.

**The prefix fallback answers when it should decline.** ``2nd Cir.`` finds no
exact match, falls through, and lands on ``2nd Cir. BAP``: it returns *bap2*,
the Bankruptcy Appellate Panel of the Second Circuit, which is a different
court, with nothing to say a guess was made. Measured over the 107 distinct
court strings in this corpus, the fallback adds three answers that exact
matching does not: ``D. Minn.`` -> mnd, which is right; ``2 nd Cir.`` -> bap2,
which is wrong; and ``Ct. App.`` -> ctappindterr, the Court of Appeals of Indian
Territory, which is arbitrary -- ``Ct. App. Nev.`` matches the same prefix and
which one is returned depends on iteration order. One right, two wrong.

**The New York departments are not in the data at all.** courts-db models the
Appellate Division as a single court, ``nyappdiv``, named "The Four Departments
of the Appellate Division". A filing writes ``(2d Dep't 2017)``, and there is
nothing for that to match. It is 32 of the 51 courts this corpus states and
eyecite does not record -- the largest group by far.

## What this does instead

*   **Ordinals are normalised on both sides.** ``2d``, ``2nd`` and ``2`` become
    one key, so a court is found however the filing spells it. Applied across
    all 2,809 courts this creates **no new collisions**: the only key holding
    more than one court is the empty one, shared by the ~800 courts that carry
    no citation string, and those can never match anything anyway.
*   **A prefix match must be unique.** ``D. Minn.`` still reaches ``D.
    Minnesota`` because only one court starts that way. ``Ct. App.`` now returns
    nothing, because two do and neither is more right than the other. A court we
    cannot identify comes back empty rather than plausible.
*   **A department resolves to the Appellate Division**, with the department
    itself left in the text for whatever wants it. The reporter already says as
    much -- ``155 A.D.3d 781`` is an Appellate Division citation whatever the
    parenthetical holds.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

from eyecite.helpers import courts

# `2d`, `2nd` and `2 nd` are one ordinal. The space is allowed because
# extraction inserts it: `(2 nd Cir. 2009)` is in this corpus.
_ORDINAL = re.compile(r"\b(\d+)[^\S\r\n]*(?:st|nd|rd|d|th)\b", re.IGNORECASE)
_NOT_WORD = re.compile(r"[^\w]")

# `(2d Dep't 2017)` names a department of one court that courts-db holds whole.
_NEW_YORK_DEPARTMENT = re.compile(r"^\s*\d+[^\S\r\n]*(?:st|nd|rd|d|th)?\s*dep'?t\.?\s*$", re.IGNORECASE)
_NEW_YORK_APPELLATE_DIVISION = "nyappdiv"


def normalize(value: str) -> str:
    """The comparison key for a court string: no ordinal suffix, no punctuation."""
    return _NOT_WORD.sub("", _ORDINAL.sub(r"\1", value or "")).lower()


@lru_cache(maxsize=1)
def _index() -> dict[str, frozenset[str]]:
    """Every court in courts-db, keyed by its normalised citation string."""
    grouped: defaultdict[str, set[str]] = defaultdict(set)
    for court in courts:
        citation_string = court.get("citation_string")
        if not citation_string:
            continue
        grouped[normalize(citation_string)].add(str(court["id"]))
    return {key: frozenset(ids) for key, ids in grouped.items()}


def resolve_court(paren: str | None) -> str | None:
    """The court id the parenthetical names, or None when it names none clearly.

    None means "not identified", never "no court was written". A caller that
    needs to know which of those it is should read the text.
    """
    if not paren:
        return None
    if _NEW_YORK_DEPARTMENT.match(paren):
        return _NEW_YORK_APPELLATE_DIVISION
    key = normalize(paren)
    if not key:
        return None
    index = _index()
    exact = index.get(key)
    if exact and len(exact) == 1:
        return next(iter(exact))
    if exact:
        # Two courts share this spelling. Neither is more right than the other.
        return None
    prefixed = {court for stored, ids in index.items() if stored.startswith(key) for court in ids}
    if len(prefixed) == 1:
        return next(iter(prefixed))
    return None
