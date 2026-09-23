"""Outer compositional API for Mellea-LRC stages.

This is the stable boundary for callers composing individual stages.  Every
public operation accepts a :class:`Document` and returns that same document
with more evidence written on it.  A caller may save it with
``document.model_dump(mode="json")``, restore it with ``Document.model_validate(...)``,
and resume at the next named stage.

The three composition helpers are intentionally small conveniences, rather
than an implicit end-to-end pipeline:

``grow_roots``
    Complete-locator discovery through field reading and filing-internal root
    formation.  Optional docket-site hunting belongs here, before co-location
    and before validation.

``validate_roots_identity``
    Docket retrieval and exact reporter lookup.  Both citation families then
    expose separate, bounded CourtListener and GovInfo metadata-discovery
    checkpoints before any later candidate-selection or body-corroboration
    work.

``grow_leaves``
    The deterministic second growth after root work.  Leaf-site hunting and
    pin-cite site review are separate, opt-in stages.

The command-line interface is the separate end-to-end entrypoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.extraction.adjudication import hunt_docket_locators
from mellea_lrc.extraction.adjudication.leaf_case_name_hunting import hunt_leaf_case_names
from mellea_lrc.extraction.adjudication.pin_cite_site_validation import validate_pin_cite_sites
from mellea_lrc.extraction.locator_stages import (
    find_docket_locators,
    find_full_reporter_locators,
    mark_full_reporter_locator_hunting_skipped,
    resolve_colocations,
)
from mellea_lrc.extraction.root_stages import form_roots
from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.extraction.stages import (
    CASE_NAME_STAGE,
    resolve_case_names,
    resolve_courts,
    resolve_dates,
    resolve_pin_cites,
)
from mellea_lrc.extraction.structure.leaf_growth import grow_leaves
from mellea_lrc.govinfo import GovInfoClient
from mellea_lrc.model.document import Document
from mellea_lrc.model.stage import isolated_stage
from mellea_lrc.validation.root_identity.body import (
    resolve_root_body_corroboration,
    review_shared_body_evidence,
    search_root_body_corroboration,
)
from mellea_lrc.validation.root_identity.docket import (
    lookup_govinfo_docket_roots,
    resolve_docket_root_ambiguities,
    resolve_docket_root_semantics,
    resolve_govinfo_docket_root_ambiguities,
    resolve_requeued_docket_root_ambiguities,
    review_and_requeue_unresolved_docket_roots,
    search_docket_roots,
    shortlist_docket_root_metadata_candidates,
    validate_unique_docket_root_identities,
    validate_unique_govinfo_docket_root_identities,
    validate_unique_requeued_docket_root_identities,
)
from mellea_lrc.validation.root_identity.docket_resolution import resolve_docket_root_identities
from mellea_lrc.validation.root_identity.open_web import (
    resolve_open_web_root_identities,
    search_open_web_roots,
)
from mellea_lrc.validation.root_identity.reporter import (
    lookup_full_reporter_locators_exact,
    resolve_full_reporter_locator_ambiguities,
    resolve_full_reporter_search_candidates,
    validate_unique_full_reporter_locator_identities,
)
from mellea_lrc.validation.root_identity.reporter_dates import review_full_reporter_exact_dates
from mellea_lrc.validation.search.docket import (
    resolve_docket_metadata_search_candidates,
    search_courtlistener_docket_roots,
    search_govinfo_docket_roots,
)
from mellea_lrc.validation.search.reporter import (
    search_courtlistener_full_reporter_roots,
    search_govinfo_full_reporter_roots,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient


__all__ = [
    "Document",
    "ExtractionRules",
    "find_docket_locators",
    "find_full_reporter_locators",
    "form_roots",
    "grow_leaves",
    "grow_roots",
    "hunt_docket_locators",
    "hunt_leaf_case_names",
    "lookup_full_reporter_locators_exact",
    "lookup_govinfo_docket_roots",
    "mark_full_reporter_locator_hunting_skipped",
    "resolve_case_names",
    "resolve_colocations",
    "resolve_courts",
    "resolve_dates",
    "resolve_docket_metadata_search_candidates",
    "resolve_docket_root_ambiguities",
    "resolve_docket_root_identities",
    "resolve_docket_root_semantics",
    "resolve_full_reporter_locator_ambiguities",
    "resolve_full_reporter_search_candidates",
    "resolve_govinfo_docket_root_ambiguities",
    "resolve_open_web_root_identities",
    "resolve_pin_cites",
    "resolve_requeued_docket_root_ambiguities",
    "resolve_root_body_corroboration",
    "review_and_requeue_unresolved_docket_roots",
    "review_full_reporter_exact_dates",
    "review_shared_body_evidence",
    "search_courtlistener_docket_roots",
    "search_courtlistener_full_reporter_roots",
    "search_docket_roots",
    "search_govinfo_docket_roots",
    "search_govinfo_full_reporter_roots",
    "search_open_web_roots",
    "search_root_body_corroboration",
    "shortlist_docket_root_metadata_candidates",
    "stable",
    "validate_pin_cite_sites",
    "validate_roots_identity",
    "validate_unique_docket_root_identities",
    "validate_unique_full_reporter_locator_identities",
    "validate_unique_govinfo_docket_root_identities",
    "validate_unique_requeued_docket_root_identities",
]


@isolated_stage
async def grow_roots(
    document: Document,
    *,
    rules: ExtractionRules | None = None,
    hunt_dockets: bool = False,
    session: MelleaSession | None = None,
) -> Document:
    """Grow complete locators into filing-internal roots.

    The explicit order is the extraction dependency order: every configured
    locator source finishes before co-location is projected; case names,
    courts, dates, and pin cites then read the final site boundaries; root
    formation is last.  ``hunt_dockets`` is deliberately opt-in because it
    incurs model calls.  It runs before co-location, so every newly admitted
    locator participates in the one final grouping pass.

    Full-reporter site hunting is not run here.  Its deliberately recorded
    skip leaves a checkpoint explaining that decision without pretending that
    the reporter rule pass found every possible reporter locator.
    """
    effective_rules = stable(rules)
    document = find_full_reporter_locators(document, rules=effective_rules)
    document = find_docket_locators(document, rules=effective_rules)
    document = mark_full_reporter_locator_hunting_skipped(
        document,
        reason="Full-reporter site hunting is disabled in the current root-growth profile.",
    )
    if hunt_dockets:
        document = await hunt_docket_locators(document, session=session, rules=effective_rules)
    document = resolve_colocations(document, rules=effective_rules)
    document = resolve_case_names(document, rules=effective_rules)
    document = resolve_courts(document, rules=effective_rules)
    document = resolve_dates(document, rules=effective_rules)
    document = resolve_pin_cites(document, rules=effective_rules)
    return form_roots(document)


@isolated_stage
async def validate_roots_identity(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    govinfo_client: GovInfoClient | None = None,
    session: MelleaSession | None = None,
) -> Document:
    """Validate formed docket roots first, then formed reporter roots.

    A docket root runs one ordered query-and-resolution plan: direct docket
    queries first, followed by compact case-name fragment queries when needed.
    Every query response follows the same fuzzy-docket shortlist,
    literal checks, and semantic candidate-choice drill. A serialized document
    retains each query, shortlist, comparison, and decision in its trace. The
    reporter path keeps its exact lookup, metadata discovery, candidate
    resolution, and body-corroboration stages separate.
    """
    document = await resolve_docket_root_identities(
        document,
        client=client,
        govinfo_client=govinfo_client,
        session=session,
    )
    document = await lookup_full_reporter_locators_exact(document, client=client)
    document = await validate_unique_full_reporter_locator_identities(
        document, client=client, session=session
    )
    document = await review_full_reporter_exact_dates(document, client=client)
    document = await resolve_full_reporter_locator_ambiguities(document, client=client, session=session)
    if CASE_NAME_STAGE in document.passes:
        # Exact reporter retrieval gets the first opportunity to settle the
        # citation. Metadata discovery is an independent fallback for roots
        # still unresolved after unique and ambiguous exact resolution.
        document = await search_courtlistener_full_reporter_roots(document, client=client, session=session)
        document = await search_govinfo_full_reporter_roots(document, client=govinfo_client, session=session)
        document = await resolve_full_reporter_search_candidates(document, session=session)
        document = await search_root_body_corroboration(
            document,
            client=client,
            govinfo_client=govinfo_client,
        )
        document = await resolve_root_body_corroboration(document, session=session)
        document = await review_shared_body_evidence(document, session=session)
    return document


# These are the individually composable stages. Their implementations may
# mutate records while applying several operations from one node, but each
# public call begins from its own snapshot so a saved in-memory checkpoint is
# as independent as a serialized checkpoint.
find_full_reporter_locators = isolated_stage(find_full_reporter_locators)
find_docket_locators = isolated_stage(find_docket_locators)
mark_full_reporter_locator_hunting_skipped = isolated_stage(mark_full_reporter_locator_hunting_skipped)
hunt_docket_locators = isolated_stage(hunt_docket_locators)
resolve_colocations = isolated_stage(resolve_colocations)
resolve_case_names = isolated_stage(resolve_case_names)
resolve_courts = isolated_stage(resolve_courts)
resolve_dates = isolated_stage(resolve_dates)
resolve_pin_cites = isolated_stage(resolve_pin_cites)
form_roots = isolated_stage(form_roots)
grow_leaves = isolated_stage(grow_leaves)
hunt_leaf_case_names = isolated_stage(hunt_leaf_case_names)
validate_pin_cite_sites = isolated_stage(validate_pin_cite_sites)

search_docket_roots = isolated_stage(search_docket_roots)
validate_unique_docket_root_identities = isolated_stage(validate_unique_docket_root_identities)
resolve_docket_root_ambiguities = isolated_stage(resolve_docket_root_ambiguities)
lookup_govinfo_docket_roots = isolated_stage(lookup_govinfo_docket_roots)
validate_unique_govinfo_docket_root_identities = isolated_stage(
    validate_unique_govinfo_docket_root_identities
)
resolve_govinfo_docket_root_ambiguities = isolated_stage(resolve_govinfo_docket_root_ambiguities)
review_and_requeue_unresolved_docket_roots = isolated_stage(review_and_requeue_unresolved_docket_roots)
validate_unique_requeued_docket_root_identities = isolated_stage(
    validate_unique_requeued_docket_root_identities
)
resolve_requeued_docket_root_ambiguities = isolated_stage(resolve_requeued_docket_root_ambiguities)
shortlist_docket_root_metadata_candidates = isolated_stage(shortlist_docket_root_metadata_candidates)
resolve_docket_root_semantics = isolated_stage(resolve_docket_root_semantics)
resolve_docket_root_identities = isolated_stage(resolve_docket_root_identities)

lookup_full_reporter_locators_exact = isolated_stage(lookup_full_reporter_locators_exact)
validate_unique_full_reporter_locator_identities = isolated_stage(
    validate_unique_full_reporter_locator_identities
)
resolve_full_reporter_locator_ambiguities = isolated_stage(resolve_full_reporter_locator_ambiguities)
review_full_reporter_exact_dates = isolated_stage(review_full_reporter_exact_dates)
search_courtlistener_docket_roots = isolated_stage(search_courtlistener_docket_roots)
search_govinfo_docket_roots = isolated_stage(search_govinfo_docket_roots)
resolve_docket_metadata_search_candidates = isolated_stage(resolve_docket_metadata_search_candidates)
search_courtlistener_full_reporter_roots = isolated_stage(search_courtlistener_full_reporter_roots)
search_govinfo_full_reporter_roots = isolated_stage(search_govinfo_full_reporter_roots)
resolve_full_reporter_search_candidates = isolated_stage(resolve_full_reporter_search_candidates)
search_root_body_corroboration = isolated_stage(search_root_body_corroboration)
resolve_root_body_corroboration = isolated_stage(resolve_root_body_corroboration)
review_shared_body_evidence = isolated_stage(review_shared_body_evidence)
search_open_web_roots = isolated_stage(search_open_web_roots)
resolve_open_web_root_identities = isolated_stage(resolve_open_web_root_identities)
