"""Extraction work that is not wired into the production pipeline.

Production is eyecite plus a whitespace repair. Everything here is an attempt
to reach the citations that leaves behind -- the ones a filing's PDF extraction
has damaged past what a generated pattern can match.

Three approaches live side by side, and they differ in where the model sits:

:mod:`~mellea_lrc.experimental.relaxed_eyecite_extractor`
    No model at all. Rebuilds eyecite's patterns so the separators between
    volume, reporter and page tolerate any whitespace, recovering citations
    split across a line or page break. Cheap and deterministic, but the same
    relaxation lets PDF margin line numbers be read as a page.

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
    SuspectedSite,
    mask_full_spans,
    mask_locator_spans,
    suspected_sites,
)
from mellea_lrc.experimental.relaxed_eyecite_extractor import (
    extract_relaxed,
    extract_relaxed_citations,
    relaxed_tokenizer,
)

__all__ = [
    "SuspectedSite",
    "extract_relaxed",
    "extract_relaxed_citations",
    "mask_full_spans",
    "mask_locator_spans",
    "relaxed_tokenizer",
    "suspected_sites",
]
