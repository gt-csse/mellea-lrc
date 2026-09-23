"""Extraction work that is not wired into the pipeline.

What used to live here in two halves now lives in one place each.
:mod:`~mellea_lrc.extraction.adjudication` is the layer that follows the deterministic
rules -- candidate generators and the reviewers that judge them -- and is no
longer experimental in the sense this package means.

:mod:`~mellea_lrc.experimental.llm_only_extraction`
    An earlier prototype in which the model *is* the extractor, reading a whole
    document and listing the citations it finds. Retained for comparison and not
    maintained. Nothing constrains its output to text that exists, which is the
    property adjudication is built around: a reviewer is asked only whether
    characters already in the document mean what a generator proposed.
"""
