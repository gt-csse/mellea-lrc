"""Mellea case-name classification."""

from __future__ import annotations

from typing import Literal

from mellea import generative

SemanticMatchVerdict = Literal["semantic_match", "not_semantic_match"]
NonSemanticVerdict = Literal["different_case", "irregular_form"]
CASE_NAME_VERDICT_MAX_TOKENS = 128


@generative
async def semantic_match_case_name(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> SemanticMatchVerdict:
    """Classify whether an extracted name is a normal citation of the retrieved case.

    Consider only the case names and their local context. A semantic match denotes
    the same case using normal legal-citation abbreviation or party shortening, with
    both sides of "v." represented. Missing, garbled, materially incomplete, or
    different-case names are not semantic matches.
    """


@generative
async def classify_non_semantic_case_name(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> NonSemanticVerdict:
    """Classify why a re-extracted case name is not a semantic match.

    Consider only the case names and their local context. Return ``different_case``
    when they identify unrelated cases. Return ``irregular_form`` when they identify
    the same case but the extracted form is incomplete or garbled beyond normal
    legal-citation shortening.
    """
