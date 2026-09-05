"""Characters a scanner reads for one another, and the two ways to use that.

The confusion set is shared by a generator and a reviewer, which is why it lives
here rather than in either. Both need the same set for the same reason -- an
optical read that turned ``183`` into ``l83`` or ``So.`` into ``S0.`` produces
text no exact matcher reaches -- but they need it in opposite directions:

*   A **generator** searches the document, so it needs the damaged *spellings* a
    reporter can appear as: :func:`variants`.
*   A **reviewer** compares a quote against parts, so it needs a *normal form*
    both sides can be folded to: :func:`fold`.

Only pairs that arise from letterform are included, and none that would let one
reporter turn into a different one. ``rn``/``m`` is deliberately absent: it is a
two-character-for-one confusion, which changes length, and length is what the
reviewer's checks use to tell a substitution from an insertion.
"""

from __future__ import annotations

import re

CONFUSABLE = {"o": "0", "l": "1", "i": "1", "s": "5", "b": "8", "z": "2", "g": "9"}
"""Letter to the digit it is read for. Applied to both sides of a comparison,
this is a symmetric fold: the digit is already its own image."""

_FOLD = str.maketrans(CONFUSABLE)
# Length-preserving ASCII lower-casing. `str.lower()` is not length-preserving
# for every code point, and a generator that scans a case-folded copy of the
# document reports offsets into the original, so a single character that folds
# to two would silently shift every span after it.
_LOWER = str.maketrans({chr(code): chr(code + 32) for code in range(ord("A"), ord("Z") + 1)})
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")

MAX_SUBSTITUTIONS = 1
"""How much optical damage one reporter spelling is assumed to carry.

Enumerating every subset of confusable positions is exponential in the spelling
and buys shapes nobody has seen: `Ohio` alone has three. One substitution is
what the damage in these corpora looks like, and it keeps the search finite.
"""


def lower(text: str) -> str:
    """ASCII-lower-case ``text`` without changing its length."""
    return text.translate(_LOWER)


def fold(text: str) -> str:
    """Lower-case ``text`` and fold every confusable character onto the digit.

    A normal form, for comparing two strings that may be damaged differently.
    Not for searching a document: ``So.`` folds to ``50.``, which the digits of
    a real page number also fold to.
    """
    return lower(text).translate(_FOLD)


def key(text: str) -> str:
    """The fold with punctuation and whitespace removed.

    What makes ``U,S,`` and ``U.S.`` the same reporter, and ``US`` too.
    """
    return _NON_ALPHANUMERIC.sub("", fold(text))


def variants(spelling: str) -> set[str]:
    """Lower-cased spelling, plus each single-character optical damage of it.

    A variant that keeps no letter is dropped. ``So.`` damaged twice is ``50.``,
    which is a number, and searching a document for it would flag every page
    number written with a full stop after it.
    """
    base = lower(spelling)
    found = {base}
    for index, character in enumerate(base):
        replacement = CONFUSABLE.get(character)
        if replacement is None:
            continue
        damaged = base[:index] + replacement + base[index + 1 :]
        if any(character.isalpha() for character in damaged):
            found.add(damaged)
    return found
