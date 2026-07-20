"""Mellea case-name semantic classification."""

from typing import Literal

from mellea import generative

SemanticMatchVerdict = Literal["semantic_match", "not_semantic_match"]
CASE_NAME_VERDICT_MAX_TOKENS = 128


@generative
async def semantic_match_case_name(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> SemanticMatchVerdict:
    """Classify whether an extracted name normally cites the retrieved case.

    Consider only the case names and their local context. A semantic match is
    the same case using normal legal abbreviation or party shortening, with
    both sides of "v." represented. Missing, garbled, materially incomplete,
    or different-case names are not semantic matches.
    """
