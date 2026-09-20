"""Outer compositional API for Mellea-LRC stages.

This is the stable boundary for callers composing individual stages.  Every
public operation accepts a :class:`Document` and returns that same document
with more evidence written on it.  A caller may save it with
``document.serialize()``, restore it with ``Document.from_serialized(...)``,
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
    leaf case-name validation are deliberately separate future stages.

The command-line interface is the separate end-to-end entrypoint.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mellea_lrc.extraction.adjudication import hunt_docket_locators
from mellea_lrc.extraction.eyecite_extractor import grow_leaves as _grow_leaves
from mellea_lrc.extraction.locator_stages import (
    find_docket_locators,
    find_full_reporter_locators,
    mark_full_reporter_locator_hunting_skipped,
    resolve_colocations,
)
from mellea_lrc.extraction.root_stages import ROOT_FORMATION_STAGE, form_roots
from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.extraction.stages import (
    CASE_NAME_STAGE,
    resolve_case_names,
    resolve_courts,
    resolve_dates,
    resolve_pin_cites,
)
from mellea_lrc.extraction.structure.attachment import Attachment
from mellea_lrc.extraction.types import Document
from mellea_lrc.govinfo import GovInfoClient
from mellea_lrc.validation.docket_roots import (
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
from mellea_lrc.validation.docket_search import (
    search_courtlistener_docket_roots,
    search_govinfo_docket_roots,
)
from mellea_lrc.validation.full_reporter_search import (
    search_courtlistener_full_reporter_roots,
    search_govinfo_full_reporter_roots,
)
from mellea_lrc.validation.roots import (
    lookup_full_reporter_locators_exact,
    resolve_full_reporter_locator_ambiguities,
    validate_unique_full_reporter_locator_identities,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient


LEAF_GROWTH_STAGE = "leaf_growth"

__all__ = [
    "Document",
    "ExtractionRules",
    "find_docket_locators",
    "find_full_reporter_locators",
    "form_roots",
    "grow_leaves",
    "grow_roots",
    "hunt_docket_locators",
    "lookup_full_reporter_locators_exact",
    "lookup_govinfo_docket_roots",
    "mark_full_reporter_locator_hunting_skipped",
    "resolve_case_names",
    "resolve_colocations",
    "resolve_courts",
    "resolve_dates",
    "resolve_docket_root_ambiguities",
    "resolve_docket_root_semantics",
    "resolve_full_reporter_locator_ambiguities",
    "resolve_govinfo_docket_root_ambiguities",
    "resolve_pin_cites",
    "resolve_requeued_docket_root_ambiguities",
    "review_and_requeue_unresolved_docket_roots",
    "search_courtlistener_docket_roots",
    "search_courtlistener_full_reporter_roots",
    "search_docket_roots",
    "search_govinfo_docket_roots",
    "search_govinfo_full_reporter_roots",
    "shortlist_docket_root_metadata_candidates",
    "stable",
    "validate_roots_identity",
    "validate_unique_docket_root_identities",
    "validate_unique_full_reporter_locator_identities",
    "validate_unique_govinfo_docket_root_identities",
    "validate_unique_requeued_docket_root_identities",
]


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


async def validate_roots_identity(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    govinfo_client: GovInfoClient | None = None,
    session: MelleaSession | None = None,
) -> Document:
    """Validate formed docket roots first, then formed reporter roots.

    This convenience never merges the individual checkpoints.  A serialized
    document still records every checkpoint in order: initial CourtListener
    docket search, unique identity, ambiguity; the GovInfo fallback lookup for
    CourtListener misses, with its own unique identity and ambiguity checks;
    one extraction review and any resulting
    requeued docket lookup; then its unique identity and ambiguity checkpoints;
    one deterministic CourtListener-metadata shortlist; then bounded
    CourtListener and GovInfo metadata discovery, each retaining all query
    attempts and candidates; then reporter exact
    lookup; bounded CourtListener and GovInfo metadata discovery for reporter
    exact misses; then reporter unique identity and ambiguity. The review may
    correct only a docket number grounded in the filing, and a correction gets
    exactly one requeued lookup rather than an implicit repair loop. Docket
    lookup is first because it remains useful even where no court was read.
    """
    document = await search_docket_roots(document, client=client)
    document = await validate_unique_docket_root_identities(document)
    document = await resolve_docket_root_ambiguities(document)
    document = await lookup_govinfo_docket_roots(document, client=govinfo_client)
    document = await validate_unique_govinfo_docket_root_identities(document)
    document = await resolve_govinfo_docket_root_ambiguities(document)
    document = await review_and_requeue_unresolved_docket_roots(document, client=client, session=session)
    document = await validate_unique_requeued_docket_root_identities(document)
    document = await resolve_requeued_docket_root_ambiguities(document)
    document = await shortlist_docket_root_metadata_candidates(document)
    document = await resolve_docket_root_semantics(document, session=session)
    # Discovery is deliberately a pair of provider-level stages, each with its
    # own bounded internal query exploration. Candidate selection remains a
    # later identity operation, and body-text corroboration remains separate.
    # The two provider metadata routes depend on the explicitly readable
    # case-name field. Keep direct identity validation usable for manually
    # formed roots that have not entered field reading; callers can resume
    # from that Document after resolving case names.
    if CASE_NAME_STAGE in document.passes:
        document = await search_courtlistener_docket_roots(document, client=client, session=session)
        document = await search_govinfo_docket_roots(document, client=govinfo_client, session=session)
    document = await lookup_full_reporter_locators_exact(document, client=client)
    if CASE_NAME_STAGE in document.passes:
        document = await search_courtlistener_full_reporter_roots(document, client=client, session=session)
        document = await search_govinfo_full_reporter_roots(document, client=govinfo_client, session=session)
    document = await validate_unique_full_reporter_locator_identities(
        document, client=client, session=session
    )
    return await resolve_full_reporter_locator_ambiguities(document, client=client, session=session)


async def grow_leaves(
    document: Document,
    *,
    attach: Attachment = Attachment.STATED,
) -> Document:
    """Attach deterministic short-form leaves to the roots this document holds.

    Root identity validation is not a prerequisite: this stage is useful for
    structural evaluation immediately after :func:`form_roots`, and it also
    respects roots withdrawn by a later identity stage.  It reuses the
    established deterministic reader and attachment policy, but writes the
    explicit ``leaf_growth`` checkpoint used by the compositional API.

    TODO: add a separate leaf-site-hunting stage and a separate check that a
    leaf's stated case name agrees with its root and retrieved record.  Neither
    concern belongs in this structural attachment stage, and neither should
    make leaf growth depend on co-location.
    """
    if ROOT_FORMATION_STAGE not in document.passes:
        msg = "Leaf growth requires root_formation before attachment."
        raise ValueError(msg)
    if LEAF_GROWTH_STAGE in document.passes:
        return document

    # The native reader still marks its historical ``leaves`` pass.  Replace
    # that marker at the outer boundary so every new public stage has one
    # unambiguous, checkpointable name.
    grown = _grow_leaves(document, attach=attach)
    passes = tuple(pass_name for pass_name in grown.passes if pass_name != "leaves")
    return replace(grown, passes=(*passes, LEAF_GROWTH_STAGE))
