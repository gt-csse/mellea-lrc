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
from functools import lru_cache
from typing import TYPE_CHECKING

from eyecite import get_citations
from eyecite.tokenizers import EXTRACTORS, Tokenizer

from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.eyecite_extractor import to_canonical
from mellea_lrc.extraction.types import ExtractedCitation

if TYPE_CHECKING:
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
            span=Span(
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
