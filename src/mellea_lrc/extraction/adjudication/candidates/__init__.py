"""Generators: cheap, deterministic, and each answering one question.

Every generator takes an :class:`~mellea_lrc.extraction.types.ExtractedDocument`
and yields a proposal. None of them decides anything -- deciding is the
reviewer's job, and a generator that decides is a rule that should have been in
extraction instead.

They differ enormously in yield, and the difference is the design. A narrow
generator built from a pattern we cannot prove general is affordable to review
*because* it is narrow. A broad one is not: `reporter_sites` proposed 185
candidates across 77 mined documents, nearly all of them letterheads. See
`exploration/notes/candidates-and-adjudication.md`.

:mod:`~mellea_lrc.extraction.adjudication.candidates.reporter_sites` is the only
one with more than one stage, because "there is a reporter here" has more than
one answer: a spelling the gazetteer holds, and a run of letters that reduces to
one. Capitalisation and optical damage are folded away in both stages, which is
why there is no longer a separate generator for a reporter set in capitals --
that is a strict site whose written form differs from the gazetteer's, and
:func:`~mellea_lrc.extraction.adjudication.promotion.reread_site` recovers it
without asking anyone.
"""

from mellea_lrc.extraction.adjudication.candidates.ambiguous_editions import ambiguous_editions
from mellea_lrc.extraction.adjudication.candidates.orphan_short_forms import orphan_short_forms
from mellea_lrc.extraction.adjudication.candidates.reporter_sites import (
    SiteStage,
    SuspectedLocator,
    suspected_locators,
)

__all__ = [
    "SiteStage",
    "SuspectedLocator",
    "ambiguous_editions",
    "orphan_short_forms",
    "suspected_locators",
]
