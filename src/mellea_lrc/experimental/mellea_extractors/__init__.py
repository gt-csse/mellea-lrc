"""Mellea-backed citation extraction strategies.

Every strategy implements the `MelleaBase` contract
"""

from mellea_lrc.experimental.mellea_extractors.base import MelleaExtractorBase
from mellea_lrc.experimental.mellea_extractors.divide_conquer import MelleaDivideConquer
from mellea_lrc.experimental.mellea_extractors.naive import MelleaNaive

__all__ = ["MelleaDivideConquer", "MelleaExtractorBase", "MelleaNaive"]
