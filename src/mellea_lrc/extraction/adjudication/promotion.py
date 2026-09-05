r"""Turn a candidate a reader accepted into a citation, by re-reading it.

The question this answers: once someone has agreed that `33 F.4TH 693` is a
citation, how does it become an
:class:`~mellea_lrc.extraction.types.ExtractedCitation` like every other one?

Not by constructing one. A hand-built citation carries only the fields whoever
built it thought of, and would miss the court, the date, the pin cite and the
party names that `add_post_citation` and `find_case_name` produce -- so it would
be a second, poorer parser living beside the first.

Instead the window is **re-read with the same pipeline and a rule widened for
that span alone**. A rule too risky to apply to a whole corpus is safe applied
to one place a reader has confirmed, and the tokenizer used here is eyecite's
unfiltered one with every extractor made case-insensitive: far too slow for a
document and unremarkable on 240 characters.

Offsets add rather than remap. The window is a slice of the document's own text,
so a citation found at local offset `n` sits at `window.start + n`, and
``matched_text`` is still the source text as written. Nothing is rewritten,
which is the same property the extraction layer keeps.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import re
import uuid
from bisect import bisect_left, bisect_right
from functools import lru_cache
from typing import TYPE_CHECKING

from eyecite import get_citations
from eyecite.annotate import SpanUpdater
from eyecite.tokenizers import EXTRACTORS, Tokenizer

from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.eyecite_extractor import to_canonical
from mellea_lrc.extraction.types import ExtractedCitation

if TYPE_CHECKING:
    from mellea_lrc.extraction.adjudication.candidates.reporter_sites import SuspectedLocator
    from mellea_lrc.extraction.adjudication.review.locator import AdjudicatedLocator
    from mellea_lrc.extraction.adjudication.types import Candidate


@lru_cache(maxsize=1)
def _forgiving_tokenizer() -> Tokenizer:
    """Every extractor, case-insensitive, with no prefilter.

    `AhocorasickTokenizer` cannot be used case-insensitively: moving every
    string-bearing extractor to one side leaves the other automaton empty and
    pyahocorasick raises. The unfiltered tokenizer has no such split, and its
    cost is irrelevant on a window.
    """
    return Tokenizer(extractors=[dataclasses.replace(e, flags=e.flags | re.I) for e in EXTRACTORS])


def promote(text: str, candidate: Candidate) -> ExtractedCitation | None:
    """Re-read an accepted candidate and return it as an extracted citation.

    ``None`` when the widened read still finds nothing at that span, which is an
    answer worth keeping: the reader accepted something the pipeline cannot
    represent, and inventing a citation object would hide that.
    """
    window = text[candidate.window.start : candidate.window.end]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        found = get_citations(window, tokenizer=_forgiving_tokenizer())

    wanted = (
        candidate.span.start - candidate.window.start,
        candidate.span.end - candidate.window.start,
    )
    for citation in found:
        start, end = citation.span()
        if (start, end) != wanted:
            continue
        full_start, full_end = citation.full_span()
        return ExtractedCitation(
            citation_id=str(uuid.uuid4())[:8],
            full_span=Span(
                start=candidate.window.start + full_start,
                end=candidate.window.start + full_end,
            ),
            locator_span=Span(
                start=candidate.window.start + start,
                end=candidate.window.start + end,
            ),
            matched_text=citation.matched_text(),
            citation=to_canonical(citation),
        )
    return None


PROMOTION_WINDOW = 240
"""Characters either side of a locator that a re-read is given.

Enough for the case name in front and the court parenthetical behind, which is
what makes the promoted citation as complete as a parsed one, and short enough
that the unfiltered tokenizer's cost does not matter.
"""


def promote_locator(text: str, locator: AdjudicatedLocator) -> ExtractedCitation | None:
    """Re-read a reviewer-accepted locator by repairing it in place, then parsing.

    :func:`promote` re-reads text as written, so it recovers a citation only
    when the damage is something a widened *rule* can already forgive -- a lost
    capital, a missing space. It cannot recover ``556 U,S, 662``: no tokenizer
    matches ``U,S,``, however forgiving, because the reporter itself is not
    there.

    What a reviewer produces is exactly what closes that gap. It reports the
    three parts *repaired* alongside the damaged quote, so the repaired form can
    be substituted for the damaged characters and the result handed to eyecite,
    which then supplies the court, the date, the pin cite and the party names
    that no hand-built object would carry.

    Spans go back to the document either way:

    *   ``locator_span`` is the **damaged** span the reviewer grounded. The
        record points at the characters the filing actually contains, never at
        a repair.
    *   ``full_span`` is found in repaired coordinates and mapped back through
        :class:`~eyecite.annotate.SpanUpdater`, because substituting a repaired
        reporter for a damaged one changes length.

    Returns ``None`` when the repaired text still parses to nothing at that
    position, which is an answer worth keeping: the reviewer accepted something
    the pipeline cannot represent, and inventing a citation would hide that.
    """
    start = max(0, locator.span.start - PROMOTION_WINDOW)
    end = min(len(text), locator.span.end + PROMOTION_WINDOW)
    original = text[start:end]
    local_start = locator.span.start - start
    local_end = locator.span.end - start

    canonical = f"{locator.volume} {locator.reporter} {locator.page}"
    repaired = original[:local_start] + canonical + original[local_end:]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        found = get_citations(repaired, tokenizer=_forgiving_tokenizer())

    wanted = (local_start, local_start + len(canonical))
    updater = SpanUpdater(repaired, original) if repaired != original else None
    for citation in found:
        if citation.span() != wanted:
            continue
        full_start, full_end = citation.full_span()
        if updater is not None:
            # A repair is a net substitution, so the start binds right and the
            # end binds left, the same pairing the reviewer's grounding uses.
            full_start = updater.update(full_start, bisect_right)
            full_end = updater.update(full_end, bisect_left)
        return ExtractedCitation(
            citation_id=str(uuid.uuid4())[:8],
            full_span=Span(start=start + full_start, end=start + full_end),
            locator_span=Span(start=locator.span.start, end=locator.span.end),
            # The characters the document holds, not the ones that were parsed.
            matched_text=locator.text,
            citation=to_canonical(citation),
        )
    return None


def _lettered_page(citation: object) -> bool:
    """Whether a case citation was read with letters where its page should be.

    The unfiltered tokenizer is more permissive than the one extraction runs,
    and permissiveness has a cost: a footnote number in front of a procedural
    rule reads as a citation. `9 Fed. R. Civ. P. 8(a)(2)` becomes volume 9,
    reporter `Fed. R.`, page `Civ`, which is a false citation with every field
    filled in. A page is printed with numbers, so a page with no digit in it
    settles this without needing to know about that rule in particular.
    """
    groups = getattr(citation, "groups", None)
    if not isinstance(groups, dict):
        return False
    page = groups.get("page")
    return isinstance(page, str) and bool(page) and not any(c.isdigit() for c in page)


def reread_site(text: str, site: SuspectedLocator) -> ExtractedCitation | None:
    """Recover a suspected site with no model call, when re-reading is enough.

    A site is flagged because the extractor recorded nothing there, but the
    extractor reads the document with a *prefiltered, case-sensitive* tokenizer
    -- and the reason for that is speed over a corpus, not a judgement about
    what a citation is. On one window neither constraint is worth keeping, so
    ``33 F.4TH 693`` reads perfectly well here and needs nobody's opinion.

    This runs before the reviewer and takes those sites out of its queue. What
    is left is damage a widened *rule* cannot forgive -- a character read for
    another one, a reporter whose punctuation is wrong -- which is what
    :func:`promote_locator` and a reader are for.

    Returns the citation whose locator span covers the flagged reporter, or
    ``None``. Nothing is repaired and no span is remapped: the window is a slice
    of the document's own text, so offsets add.
    """
    start = max(0, site.span_start - PROMOTION_WINDOW)
    end = min(len(text), site.span_end + PROMOTION_WINDOW)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        found = get_citations(text[start:end], tokenizer=_forgiving_tokenizer())

    wanted_start = site.span_start - start
    wanted_end = site.span_end - start
    for citation in found:
        local_start, local_end = citation.span()
        if not (local_start <= wanted_start and wanted_end <= local_end):
            continue
        if _lettered_page(citation):
            continue
        full_start, full_end = citation.full_span()
        return ExtractedCitation(
            citation_id=str(uuid.uuid4())[:8],
            full_span=Span(start=start + full_start, end=start + full_end),
            locator_span=Span(start=start + local_start, end=start + local_end),
            matched_text=citation.matched_text(),
            citation=to_canonical(citation),
        )
    return None
