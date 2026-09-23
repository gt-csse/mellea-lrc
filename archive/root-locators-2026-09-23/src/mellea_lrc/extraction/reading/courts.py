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
*   **An abbreviation is read as an abbreviation.** courts-db spells some courts
    out where a filing abbreviates: it holds ``Bankr. S.D. Florida`` and the
    filing writes ``Bankr. S.D. Fla.``. That is not a prefix -- ``fla`` is not
    the start of ``florida`` -- so neither exact nor prefix matching reaches it.
    A legal abbreviation drops letters from the inside and keeps the rest in
    order, so a written word matches a stored word when it begins with the same
    letter and its letters appear in the stored word in order. Word counts must
    agree and the match must be unique; two candidates decline, as everywhere
    else here.
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


def _words(value: str) -> tuple[str, ...]:
    """The court string as words, ordinals normalised: `Bankr. S.D. Fla.` -> four."""
    return tuple(word for word in _NOT_WORD.split(_ORDINAL.sub(r"\1", value or "").lower()) if word)


def _abbreviates(written: str, stored: str) -> bool:
    """Whether one word is the other written short: same first letter, letters in order."""
    if written == stored:
        return True
    if not stored.startswith(written[:1]):
        return False
    remaining = iter(stored)
    return all(letter in remaining for letter in written)


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


@lru_cache(maxsize=1)
def _word_index() -> dict[tuple[str, ...], frozenset[str]]:
    """Every court in courts-db, keyed by the words of its citation string."""
    grouped: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
    for court in courts:
        citation_string = court.get("citation_string")
        if not citation_string:
            continue
        grouped[_words(citation_string)].add(str(court["id"]))
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
    written = _words(paren)
    abbreviated = {
        court
        for stored, ids in _word_index().items()
        if len(stored) == len(written)
        and all(_abbreviates(word, held) for word, held in zip(written, stored, strict=True))
        for court in ids
    }
    if len(abbreviated) == 1:
        return next(iter(abbreviated))
    return None


#: A reporter's series, which its court's abbreviation does not carry: the
#: Kansas Supreme Court is `Kan.` whether the case is in `Kan.` or `Kan. 2d`.
_SERIES = re.compile(r"\s*\d+\s*(?:st|nd|rd|d|th)\s*$", re.I)
_APPEALS = re.compile(r"\s*App\.?$")


@lru_cache(maxsize=1)
def _by_citation_string() -> dict[str, frozenset[str]]:
    """Every court by the abbreviation a citation to it is written with."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for court in courts:
        key = normalize(court.get("citation_string"))
        if key:
            grouped[key].add(str(court["id"]))
    return {key: frozenset(ids) for key, ids in grouped.items()}


def court_from_reporter(edition: str | None, *, cite_type: str | None, name: str | None) -> str | None:
    """The court a reporter is the reports of, when it is one court's.

    A filing writing `556 U.S. 662 (2009)` names no court and needs none: a
    reader knows the court from the reporter. This is that reading, and it is
    the bridge between the two databases the project already carries rather than
    a table someone maintains.

    **A state's official reports are abbreviated the way its court is.** The
    Bluebook writes the Kansas Supreme Court `Kan.` and its reports `Kan.`;
    North Carolina's Court of Appeals is `N.C. Ct. App.` and its reports
    `N.C. App.` So the edition is looked up against courts-db's own
    `citation_string`, with the series number dropped and `Ct.` tried where the
    reporter says `App.`, and a match is taken only when exactly one court has
    that abbreviation.

    **A reporter several courts publish in names none.** `P.3d`, `F.3d`,
    `So. 3d` and `A.2d` reach no court here and must not: nothing in
    `206 P. 327 (1922)` says which court decided it, and a guess is not a
    reading. The one federal exception is the Supreme Court's three reporters,
    which reporters-db names as its own.
    """
    if not edition:
        return None
    base = _SERIES.sub("", edition).strip()
    index = _by_citation_string()
    forms = [base]
    if _APPEALS.search(base):
        stem = _APPEALS.sub("", base)
        forms += [f"{stem} Ct. App.", f"{stem} App. Ct."]
    for form in forms:
        found = index.get(normalize(form))
        if found and len(found) == 1:
            return next(iter(found))
    if cite_type == "federal" and name:
        spelled = name.lower()
        if "supreme court" in spelled or "lawyer" in spelled:
            return _SUPREME_COURT
    return None


@lru_cache(maxsize=1)
def _courts_by_id() -> dict[str, dict]:
    return {str(court["id"]): court for court in courts}


def court_from_reporter_and_level(court_text: str, reporter_court_id: str | None) -> str | None:
    """Complete a bare appeals-court parenthetical with a reporter's state.

    ``Ct. App.`` alone identifies no jurisdiction. A state reporter may supply
    one, but the answer is accepted only when courts-db has a unique appeals
    court in that state. This does not change an explicitly resolved court.
    """
    if normalize(court_text) not in {"ctapp", "courtofappeal", "courtofappeals"}:
        return None
    source = _courts_by_id().get(reporter_court_id or "")
    if source is None or source.get("system") != "state" or not source.get("location"):
        return None
    if "court of appeal" in str(source.get("name", "")).casefold():
        return str(source["id"])
    matches = {
        str(candidate["id"])
        for candidate in courts
        if candidate.get("system") == "state"
        and candidate.get("location") == source["location"]
        and candidate.get("type") == "appellate"
        and normalize(str(candidate.get("citation_string") or "")).endswith("ctapp")
    }
    return next(iter(matches)) if len(matches) == 1 else None


#: The one court a federal reporter can be the reports of. Every other federal
#: reporter carries many courts, and the filing has to say which.
_SUPREME_COURT = "scotus"
