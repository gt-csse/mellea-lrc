"""Citation-validation stages, each consuming and returning a Document."""

from mellea_lrc.validation.body_search.retrieve_courtlistener_intended_case_filings import (
    retrieve_courtlistener_intended_case_filings,
)
from mellea_lrc.validation.body_search.retrieve_courtlistener_intended_case_opinions import (
    retrieve_courtlistener_intended_case_opinions,
)
from mellea_lrc.validation.body_search.retrieve_courtlistener_locator_filings import (
    retrieve_courtlistener_locator_filings,
)
from mellea_lrc.validation.body_search.retrieve_courtlistener_locator_opinions import (
    retrieve_courtlistener_locator_opinions,
)
from mellea_lrc.validation.body_search.retrieve_govinfo_intended_case_opinions import (
    retrieve_govinfo_intended_case_opinions,
)
from mellea_lrc.validation.body_search.retrieve_govinfo_locator_opinions import (
    retrieve_govinfo_locator_opinions,
)
from mellea_lrc.validation.index_reporter_root_opinion_pages import index_reporter_root_opinion_pages
from mellea_lrc.validation.judge_ambiguous_reporter_roots_rule import (
    judge_ambiguous_reporter_roots_rule,
)
from mellea_lrc.validation.judge_reporter_citation_content import judge_reporter_citation_content
from mellea_lrc.validation.judge_root_identity_from_fields import judge_root_identity_from_fields
from mellea_lrc.validation.judge_unique_reporter_roots_rule import (
    judge_unique_reporter_roots_rule,
)
from mellea_lrc.validation.prepare_reporter_citation_evidence import (
    prepare_reporter_citation_evidence,
)
from mellea_lrc.validation.read_reporter_citation_propositions import read_reporter_citation_propositions
from mellea_lrc.validation.resolve_reporter_citation_pages import resolve_reporter_citation_pages
from mellea_lrc.validation.retrieve_courtlistener_docket_roots import (
    retrieve_courtlistener_docket_roots,
)
from mellea_lrc.validation.retrieve_govinfo_docket_roots import retrieve_govinfo_docket_roots
from mellea_lrc.validation.retrieve_reporter_root_clusters import (
    retrieve_reporter_root_clusters,
)
from mellea_lrc.validation.retrieve_reporter_root_dockets import (
    retrieve_reporter_root_dockets,
)
from mellea_lrc.validation.retrieve_reporter_root_opinions import retrieve_reporter_root_opinions
from mellea_lrc.validation.review_ambiguous_reporter_roots_llm import (
    review_ambiguous_reporter_roots_llm,
)
from mellea_lrc.validation.review_courtlistener_docket_roots_llm import (
    review_courtlistener_docket_roots_llm,
)
from mellea_lrc.validation.review_govinfo_docket_roots_llm import review_govinfo_docket_roots_llm
from mellea_lrc.validation.review_intended_cases_llm import review_intended_cases_llm
from mellea_lrc.validation.review_locator_bodies_llm import review_locator_bodies_llm
from mellea_lrc.validation.review_reporter_citation_full_opinions_llm import (
    review_reporter_citation_full_opinions_llm,
)
from mellea_lrc.validation.review_reporter_citation_pages_llm import (
    review_reporter_citation_pages_llm,
)
from mellea_lrc.validation.review_unique_reporter_roots_llm import (
    review_unique_reporter_roots_llm,
)
from mellea_lrc.validation.select_reporter_citation_opinions import select_reporter_citation_opinions

__all__ = [
    "index_reporter_root_opinion_pages",
    "judge_ambiguous_reporter_roots_rule",
    "judge_reporter_citation_content",
    "judge_root_identity_from_fields",
    "judge_unique_reporter_roots_rule",
    "prepare_reporter_citation_evidence",
    "read_reporter_citation_propositions",
    "resolve_reporter_citation_pages",
    "retrieve_courtlistener_docket_roots",
    "retrieve_courtlistener_intended_case_filings",
    "retrieve_courtlistener_intended_case_opinions",
    "retrieve_courtlistener_locator_filings",
    "retrieve_courtlistener_locator_opinions",
    "retrieve_govinfo_docket_roots",
    "retrieve_govinfo_intended_case_opinions",
    "retrieve_govinfo_locator_opinions",
    "retrieve_reporter_root_clusters",
    "retrieve_reporter_root_dockets",
    "retrieve_reporter_root_opinions",
    "review_ambiguous_reporter_roots_llm",
    "review_courtlistener_docket_roots_llm",
    "review_govinfo_docket_roots_llm",
    "review_intended_cases_llm",
    "review_locator_bodies_llm",
    "review_reporter_citation_full_opinions_llm",
    "review_reporter_citation_pages_llm",
    "review_unique_reporter_roots_llm",
    "select_reporter_citation_opinions",
]
