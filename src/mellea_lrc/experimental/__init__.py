"""Extraction work that is not wired into the production pipeline.

Production is eyecite with its reporter patterns relaxed -- see
:class:`~mellea_lrc.extraction.Relaxation`, which used to live here and no
longer does. Everything remaining is an attempt to reach the citations a
relaxed pattern still cannot: the ones a filing's PDF extraction has damaged
past what any generated regex can match.

Two approaches live side by side, and they differ in where the model sits:

:mod:`~mellea_lrc.experimental.grounded_adjudication`
    A model adjudicates candidates. Citations already extracted are masked out,
    the remaining text is hunted for candidate sites, and a model is asked about
    each one -- reporting an identifier only when the text states one
    completely, quoting it verbatim so the answer can be grounded back into the
    document. **This is the sounder base to build on**: the model never decides
    what is in the document, only whether characters that are already there
    form a citation.

:mod:`~mellea_lrc.experimental.llm_only_extraction`
    An earlier prototype in which the model *is* the extractor, reading a whole
    document (or chunks of one) and listing the citations it finds. Retained for
    comparison and not maintained. Nothing constrains its output to text that
    exists, which is the property grounded adjudication is built around.
"""

from mellea_lrc.experimental.grounded_adjudication import (
    SuspectedLocator,
    mask_full_spans,
    mask_locator_spans,
    suspected_locators,
)

__all__ = [
    "SuspectedLocator",
    "mask_full_spans",
    "mask_locator_spans",
    "suspected_locators",
]
