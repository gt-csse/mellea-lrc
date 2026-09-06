"""Give a citation an identifier its own content decides.

A citation's id used to be a fresh UUID, so reading the same document twice
produced two sets of ids for the same citations. That is not merely untidy: the
id is what `resolves_to` and `authority_id` point at, and what a later stage
keys its own records by, so re-running extraction silently orphaned everything
downstream of it -- an identity run over 26 filings, in this project's case,
three times.

The id is now a digest of what the citation *is*: where it sits and what the
document says there. Read the same text twice and the ids are the same, so a
later stage's records still point at something.

**A change to the text still invalidates them, and should.** Preprocessing that
removes different furniture produces different offsets, which is a different
citation in a different document, and a downstream record keyed by the old id
ought to stop matching rather than quietly attach to the wrong thing. What this
removes is churn with no cause, not churn with one.

Uniqueness is per document, which is the contract
:class:`~mellea_lrc.extraction.types.ExtractedDocument` enforces. Two citations
in one document cannot share a locator span, so they cannot share an id.
"""

from __future__ import annotations

from hashlib import blake2b
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mellea_lrc.core.spans import Span

_DIGEST_BYTES = 4
"""Four bytes, eight hex characters -- the shape the ids already had.

Per document, not globally: the most citations any filing in these corpora
carries is in the low hundreds, and a birthday collision over four billion
values needs tens of thousands before it is worth thinking about. The document
type raises on a duplicate either way.
"""


def citation_id(span: Span, matched_text: str) -> str:
    """The identifier for a citation at ``span`` reading ``matched_text``."""
    digest = blake2b(digest_size=_DIGEST_BYTES)
    digest.update(f"{span.start}:{span.end}:{matched_text}".encode())
    return digest.hexdigest()
