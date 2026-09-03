"""Whole-document citation extraction by a model, without a parser underneath.

An earlier prototype, kept for comparison and **not maintained**. Each strategy
hands a document -- whole, or in chunks -- to a model and takes the citations it
reports back.

Prefer :mod:`mellea_lrc.adjudication`, which puts the model somewhere
narrower: a deterministic extractor runs first, only the text it could not
parse is offered as candidate sites, and every answer must quote characters
that are in the document, so it can be grounded rather than trusted. Nothing
here constrains the output that way.

Every strategy implements the `MelleaExtractorBase` contract.
"""

from mellea_lrc.experimental.llm_only_extraction.base import MelleaExtractorBase
from mellea_lrc.experimental.llm_only_extraction.divide_conquer import MelleaDivideConquer
from mellea_lrc.experimental.llm_only_extraction.naive import MelleaNaive

__all__ = ["MelleaDivideConquer", "MelleaExtractorBase", "MelleaNaive"]
