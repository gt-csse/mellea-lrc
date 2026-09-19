"""Terminal assessment projections for completed validation routes."""

from mellea_lrc.validation.aggregation.locator_found import (
    run_locator_candidate_assessment,
    run_locator_citation_summary,
)
from mellea_lrc.validation.aggregation.locator_identity import (
    requires_mellea_locator_candidate_choice,
    run_future_implementation_deferred_locator_identity_resolution,
    run_locator_identity_resolution,
    run_search_deferred_locator_identity_resolution,
)
from mellea_lrc.validation.aggregation.mellea_locator_candidate_choice import (
    run_mellea_locator_candidate_choice,
)
from mellea_lrc.validation.aggregation.opinion_search import (
    run_opinion_search_candidate_assessment,
)
from mellea_lrc.validation.aggregation.recap_search import (
    run_recap_search_candidate_assessment,
)
from mellea_lrc.validation.aggregation.search_citation_summary import (
    run_search_citation_summary,
)

__all__ = [
    "requires_mellea_locator_candidate_choice",
    "run_future_implementation_deferred_locator_identity_resolution",
    "run_locator_candidate_assessment",
    "run_locator_citation_summary",
    "run_locator_identity_resolution",
    "run_mellea_locator_candidate_choice",
    "run_opinion_search_candidate_assessment",
    "run_recap_search_candidate_assessment",
    "run_search_citation_summary",
    "run_search_deferred_locator_identity_resolution",
]
