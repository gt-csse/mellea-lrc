"""Corroborate root identity with citations in other archived documents.

A search hit belongs to the citing document. Its metadata cannot be copied to
the cited authority, but an actual citation in its body can independently
confirm what the source filing says about that authority.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date as CalendarDate
from hashlib import sha256
from html.parser import HTMLParser
from math import ceil
from typing import TYPE_CHECKING, Literal

from eyecite import get_citations
from eyecite.models import FullCaseCitation as EyeciteFullCaseCitation
from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerError
from mellea_lrc.courtlistener.client import courtlistener_retry_delay
from mellea_lrc.extraction.reading.courts import resolve_court
from mellea_lrc.govinfo import GovInfoClient, govinfo_package_candidate, govinfo_uscourts_body_query
from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.llm.grounding import EvidenceCandidate, GroundingEvidence, fuzzy_find
from mellea_lrc.model.case_names import CaseName
from mellea_lrc.model.citations import CitationDate, CitationField, DocketCitation, FullCaseCitation
from mellea_lrc.model.fuzziness import FuzzinessOption
from mellea_lrc.model.operations import (
    attribute_authority,
    judge_citation,
    mark_extraction_reviewed,
    observe_citation,
    resolve_citation,
    update_fields,
)
from mellea_lrc.model.record import Node, Question, Reads, Resolution
from mellea_lrc.model.spans import Span
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.serialization.validated_document import deserialize_validation_node
from mellea_lrc.validation.candidates.selection import CANDIDATE_SELECTION_LIMIT
from mellea_lrc.validation.root_identity.docket_resolution import DOCKET_ROOT_IDENTITY_STAGE
from mellea_lrc.validation.root_identity.reporter import FULL_REPORTER_SEARCH_CANDIDATE_RESOLUTION_STAGE
from mellea_lrc.validation.types import (
    BodySearchAttempt,
    CitationValidation,
    LocatorIdentityResolutionNode,
    LocatorIdentityResolutionOutcome,
    RootBodySearchNode,
    RootBodySearchOutcome,
    RootBodySearchSource,
    ValidationNode,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.model.document import Document
    from mellea_lrc.model.record import CitationRecord


ROOT_BODY_CORROBORATION_SEARCH_STAGE = "root_body_corroboration_search"
ROOT_BODY_CORROBORATION_RESOLUTION_STAGE = "root_body_corroboration_resolution"
SHARED_BODY_EVIDENCE_REVIEW_STAGE = "shared_body_evidence_review"
_MADE_BY = "mellea_lrc.validation.root_identity.body"
_BODY_REVIEW_MAX_TOKENS = 384
_THIRD_PARTY_FETCH_LIMIT = 20
_THIRD_PARTY_EVIDENCE_LIMIT = 12
_THIRD_PARTY_CONTEXT_CHARS = 480
_MAX_CITATION_QUOTE_CHARS = 400


@dataclass(frozen=True, slots=True)
class _DirectBodyCandidate:
    """A body-search result that identifies itself as the cited authority."""

    index: int
    source: RootBodySearchSource
    record: Mapping[str, object]
    anchor: str


class _BodyIdentityProposal(BaseModel):
    """One grounded choice among self-identifying body-search candidates."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["select_candidate", "defer"]
    candidate_index: int | None
    anchor_quote: str | None
    rationale: str


@dataclass(frozen=True, slots=True)
class _ThirdPartyEvidence:
    index: int
    source: RootBodySearchSource
    citing_id: str
    citing_case_name: str | None
    excerpt: str
    excerpt_start: int
    origin: str


class _CitationFieldValues(BaseModel):
    """One side's values as literally read from its own document."""

    model_config = ConfigDict(extra="forbid")

    locator: str | None
    case_name: str | None
    court: str | None
    date: str | None


class _ThirdPartyComparisons(BaseModel):
    """The model's semantic judgment for each independently read field."""

    model_config = ConfigDict(extra="forbid")

    locator: Literal["match", "mismatch", "unavailable"]
    case_name: Literal["match", "mismatch", "unavailable"]
    court: Literal["match", "mismatch", "unavailable"]
    date: Literal["match", "mismatch", "unavailable"]


class _ThirdPartyProposal(BaseModel):
    """Judge one independently quoted citation against the source citation."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["corroborates", "does_not_corroborate", "defer"]
    evidence_index: int | None
    citation_quote: str | None
    source_fields: _CitationFieldValues
    cited_fields: _CitationFieldValues
    comparisons: _ThirdPartyComparisons
    reason: str = Field(min_length=1)


async def search_root_body_corroboration(
    document: Document,
    *,
    client: CourtListenerServiceClient | None = None,
    govinfo_client: GovInfoClient | None = None,
    retrospective_date: CalendarDate | None = None,
) -> Document:
    """Search the three provider body indexes for each unresolved root.

    CourtListener opinions yield cluster-shaped records, CourtListener RECAP
    yields docket-shaped records, and GovInfo yields USCOURTS packages.  The
    source locator is the sole query input, so query construction is general
    and the filing never contributes unstated search facts.

    ``retrospective_date`` is the source filing's drafting date. With a cutoff,
    undated and later opinions and RECAP child filings cannot enter the saved
    candidate set. GovInfo opinion granules are filtered when their text is
    fetched because package dates do not date individual opinions.
    """
    _require_stage(
        document,
        DOCKET_ROOT_IDENTITY_STAGE,
        "Root body corroboration search",
    )
    _require_stage(
        document,
        FULL_REPORTER_SEARCH_CANDIDATE_RESOLUTION_STAGE,
        "Root body corroboration search",
    )
    if ROOT_BODY_CORROBORATION_SEARCH_STAGE in document.passes:
        if retrospective_date is not None:
            for record in _body_searched_roots(document):
                saved = {node.retrospective_date for node in _saved_body_nodes(record)}
                if saved != {retrospective_date.isoformat()}:
                    raise ValueError("Body search was already run without this retrospective date")
        return document
    _reject_partial_stage(document, ROOT_BODY_CORROBORATION_SEARCH_STAGE)

    courtlistener = client if client is not None else CourtListenerClient()
    govinfo = govinfo_client if govinfo_client is not None else GovInfoClient()
    for record in _unresolved_roots(document):
        locator = record.matched_text.strip()
        nodes = (
            _courtlistener_body_node(
                record,
                locator=locator,
                source=RootBodySearchSource.COURTLISTENER_CLUSTER,
                search_type="o",
                client=courtlistener,
                retrospective_date=retrospective_date,
            ),
            _courtlistener_body_node(
                record,
                locator=locator,
                source=RootBodySearchSource.COURTLISTENER_DOCKET,
                search_type="r",
                client=courtlistener,
                retrospective_date=retrospective_date,
            ),
            _govinfo_body_node(
                record, locator=locator, client=govinfo, retrospective_date=retrospective_date
            ),
        )
        nodes = tuple(
            replace(node, retrospective_date=retrospective_date.isoformat())
            if retrospective_date is not None and node.retrospective_date is None
            else node
            for node in nodes
        )
        for node in nodes:
            trace = _trace_node(node, stage=ROOT_BODY_CORROBORATION_SEARCH_STAGE)
            observe_citation(record, trace)
            judge_citation(
                record,
                trace,
                _lookup_question(record),
                _lookup_outcome(node.outcome),
                message=node.outcome_message,
            )
    return document.evolve(passes=(*document.passes, ROOT_BODY_CORROBORATION_SEARCH_STAGE))


async def resolve_root_body_corroboration(
    document: Document,
    *,
    session: MelleaSession | None = None,
    client: CourtListenerServiceClient | None = None,
    govinfo_client: GovInfoClient | None = None,
    retrospective_date: CalendarDate | None = None,
) -> Document:
    """Resolve dated direct records and independent citations available by the cutoff.

    A saved search checkpoint carries its cutoff into this stage; an explicit
    different date is rejected. Older checkpoints may receive a cutoff here,
    in which case candidates are filtered before any body or model review.
    """
    _require_stage(document, ROOT_BODY_CORROBORATION_SEARCH_STAGE, "Root body corroboration resolution")
    if ROOT_BODY_CORROBORATION_RESOLUTION_STAGE in document.passes:
        if retrospective_date is not None:
            for record in _body_searched_roots(document):
                saved = {node.retrospective_date for node in _saved_body_nodes(record)}
                if saved != {retrospective_date.isoformat()}:
                    raise ValueError("Body resolution was already run without this retrospective date")
        return document
    _reject_partial_stage(document, ROOT_BODY_CORROBORATION_RESOLUTION_STAGE)

    courtlistener = client if client is not None else CourtListenerClient()
    govinfo = govinfo_client if govinfo_client is not None else GovInfoClient()
    for record in _unresolved_roots(document):
        nodes = _saved_body_nodes(record)
        cutoff = _stored_retrospective_date(nodes, retrospective_date)
        direct = _direct_candidates(record, nodes, retrospective_date=cutoff)
        if direct:
            node = await _resolve_direct_candidates(record, direct, session=session)
            observe_citation(record, node)
            if node.outcome == "resolved":
                candidate = _selected_direct_candidate(node, direct)
                resolution = _resolution_from_body_candidate(record, candidate, node)
                resolve_citation(record, node, resolution)
                attribute_authority(record, node, _body_authority_id(candidate))
                judge_citation(
                    record,
                    node,
                    Question.IDENTITY,
                    LocatorIdentityResolutionOutcome.RESOLVED.value,
                    message=node.message,
                )
                continue
            if node.outcome == "no_match":
                judge_citation(
                    record,
                    node,
                    Question.IDENTITY,
                    LocatorIdentityResolutionOutcome.NO_MATCH.value,
                    message=node.message,
                )
                continue
        fetches: list[dict[str, object]] = []
        third_party = _third_party_evidence(
            document,
            record,
            nodes,
            courtlistener,
            govinfo,
            retrospective_date=cutoff,
            fetches=fetches,
        )
        if third_party:
            node = await _review_third_party(document, record, third_party, fetches=fetches, session=session)
            observe_citation(record, node)
            if node.outcome == "resolved":
                cited_name = node.details["cited_case_name"]
                assert isinstance(cited_name, str)
                resolve_citation(
                    record,
                    node,
                    Resolution(
                        cluster_id=None,
                        case_name=cited_name,
                        date_filed=None,
                        court_id=None,
                        node_id=node.node_id,
                        citations=(_locator_for_body_search(record),),
                    ),
                )
                attribute_authority(record, node, _third_party_authority_id(record, cited_name))
                judge_citation(
                    record, node, Question.IDENTITY, "resolved", type="third_party", message=node.message
                )
                continue
            if node.outcome == "no_match":
                judge_citation(
                    record, node, Question.IDENTITY, "no_match", type="third_party", message=node.message
                )
                continue
        else:
            observe_citation(
                record,
                _third_party_node(
                    record,
                    (),
                    outcome="no_body_citation",
                    message="No reviewable independent citation was available from fetched body text.",
                    fetches=fetches,
                ),
            )
        progression = _resolve_body_candidates(record, nodes=nodes)
        _write_identity_progression(record, progression, stage=ROOT_BODY_CORROBORATION_RESOLUTION_STAGE)
    return document.evolve(passes=(*document.passes, ROOT_BODY_CORROBORATION_RESOLUTION_STAGE))


async def review_shared_body_evidence(
    document: Document, *, session: MelleaSession | None = None
) -> Document:
    """Review a deferred root against already grounded independent evidence.

    A neighboring root may have found a complete third-party citation that
    prints both locators. Its decision is not transferred: the deferred root
    receives its own source reread, field comparisons, and identity judgment.
    Grouping only routes the saved evidence; it is not described to the model.
    """
    _require_stage(document, ROOT_BODY_CORROBORATION_RESOLUTION_STAGE, "Shared body evidence review")
    if SHARED_BODY_EVIDENCE_REVIEW_STAGE in document.passes:
        return document
    _reject_partial_stage(document, SHARED_BODY_EVIDENCE_REVIEW_STAGE)

    roots = tuple(record for record in document.active_citations if record.is_root and record.colocation_id)
    for record in roots:
        if record.judgement(Question.IDENTITY).outcome != "deferred_to_open_web_search":
            continue
        evidence: list[_ThirdPartyEvidence] = []
        sibling_by_index: dict[int, CitationRecord] = {}
        depends_on: list[str] = []
        seen_origins: set[str] = set()
        for sibling in roots:
            if sibling is record or sibling.colocation_id != record.colocation_id:
                continue
            if sibling.judgement(Question.IDENTITY).outcome != "resolved":
                continue
            saved = _selected_grounded_body_evidence(sibling)
            if saved is None:
                continue
            source_node, selected = saved
            if selected.origin in seen_origins:
                continue
            seen_origins.add(selected.origin)
            index = len(evidence) + 1
            evidence.append(replace(selected, index=index))
            sibling_by_index[index] = sibling
            depends_on.append(source_node.node_id)
            if len(evidence) >= _THIRD_PARTY_EVIDENCE_LIMIT:
                break
        if not evidence:
            continue
        node = await _review_third_party(
            document,
            record,
            tuple(evidence),
            fetches=[],
            session=session,
            stage=SHARED_BODY_EVIDENCE_REVIEW_STAGE,
            extra_depends_on=tuple(depends_on),
        )
        observe_citation(record, node)
        if node.outcome == "resolved":
            cited_name = node.details["cited_case_name"]
            assert isinstance(cited_name, str)
            resolve_citation(
                record,
                node,
                Resolution(
                    cluster_id=None,
                    case_name=cited_name,
                    date_filed=None,
                    court_id=None,
                    node_id=node.node_id,
                    citations=(_locator_for_body_search(record),),
                ),
            )
            selected_index = node.details.get("selected_evidence_index")
            sibling = sibling_by_index.get(selected_index) if isinstance(selected_index, int) else None
            authority_id = sibling.authority_id if sibling is not None else None
            attribute_authority(record, node, authority_id or _third_party_authority_id(record, cited_name))
            judge_citation(
                record, node, Question.IDENTITY, "resolved", type="third_party", message=node.message
            )
        elif node.outcome == "no_match":
            judge_citation(
                record, node, Question.IDENTITY, "no_match", type="third_party", message=node.message
            )
    return document.evolve(passes=(*document.passes, SHARED_BODY_EVIDENCE_REVIEW_STAGE))


def _selected_grounded_body_evidence(
    record: CitationRecord,
) -> tuple[Node, _ThirdPartyEvidence] | None:
    """Reconstruct one previously highlighted citation from its saved trace."""
    for node in reversed(record.trace):
        if (
            node.stage != ROOT_BODY_CORROBORATION_RESOLUTION_STAGE
            or node.outcome != "resolved"
            or node.details.get("judgement_type") != "third_party"
        ):
            continue
        selected_index = node.details.get("selected_evidence_index")
        grounding = node.details.get("citation_grounding")
        quote = node.details.get("citation_quote")
        saved = node.details.get("evidence")
        if not isinstance(selected_index, int) or not isinstance(grounding, dict):
            continue
        if not isinstance(quote, str) or not isinstance(saved, list):
            continue
        item = next(
            (entry for entry in saved if isinstance(entry, dict) and entry.get("index") == selected_index),
            None,
        )
        if item is None:
            continue
        try:
            selected = _ThirdPartyEvidence(
                index=selected_index,
                source=RootBodySearchSource(item["source"]),
                citing_id=str(item["citing_id"]),
                citing_case_name=item.get("citing_case_name"),
                excerpt=item["excerpt"],
                excerpt_start=item["excerpt_start"],
                origin=item["origin"],
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(selected.excerpt, str) or not isinstance(selected.excerpt_start, int):
            continue
        if selected.origin != grounding.get("origin"):
            continue
        grounded = GroundingEvidence((EvidenceCandidate(selected.excerpt, selected),)).find_fragment(
            quote,
            FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True),
            line_number_aware=True,
        )
        if grounded is not None and grounded.text == grounding.get("matched_text"):
            return node, selected
    return None


def _direct_candidates(
    record: CitationRecord,
    nodes: tuple[RootBodySearchNode, ...],
    *,
    retrospective_date: CalendarDate | None = None,
) -> tuple[_DirectBodyCandidate, ...]:
    """Keep only body results whose own identifier equals the source locator.

    The search engine guarantees merely that a document contains the queried
    characters.  This second check prevents a later opinion or brief from
    becoming the authority simply because it quoted the citation.
    """
    found: list[_DirectBodyCandidate] = []
    index = 0
    for node in nodes:
        for candidate in node.candidates:
            index += 1
            if retrospective_date is not None and (
                node.source
                in {
                    RootBodySearchSource.COURTLISTENER_DOCKET,
                    RootBodySearchSource.GOVINFO_PACKAGE,
                }
                or _retrospective_candidate(candidate, node.source, retrospective_date) is None
            ):
                # A docket's filing date does not date its mutable caption;
                # GovInfo package dates do not date individual opinion granules.
                # Only a dated body document can corroborate either one here.
                continue
            anchor = _self_identifier(record, candidate)
            if anchor is not None:
                found.append(
                    _DirectBodyCandidate(index=index, source=node.source, record=candidate, anchor=anchor)
                )
    return tuple(found)


def _self_identifier(record: CitationRecord, candidate: Mapping[str, object]) -> str | None:
    citation = record.fields
    if isinstance(citation, DocketCitation):
        docket = candidate.get("docketNumber")
        if not isinstance(docket, str) or not citation.docket_number:
            return None
        evidence = GroundingEvidence([EvidenceCandidate(docket, docket)])
        match = evidence.resolve(citation.docket_number, FuzzinessOption.edit_distance(similarity_percent=90))
        return docket if match is not None else None
    if not isinstance(citation, FullCaseCitation):
        return None
    source = _canonical_locator(_full_reporter_locator(citation))
    values = candidate.get("citation")
    if not source or not isinstance(values, (list, tuple)):
        return None
    return next(
        (value for value in values if isinstance(value, str) and _canonical_locator(value) == source),
        None,
    )


def _validate_direct_candidate_grounding(
    ctx: Context, candidates: tuple[_DirectBodyCandidate, ...]
) -> ValidationResult:
    """Give IVR actionable feedback when a selected anchor cannot be reproduced."""
    try:
        proposal = _BodyIdentityProposal.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        # The wrapper's schema requirement supplies the repair message.
        return ValidationResult(result=True)
    if proposal.decision == "defer":
        return ValidationResult(result=True)
    selected = next((item for item in candidates if item.index == proposal.candidate_index), None)
    if selected is None:
        return ValidationResult(
            result=False,
            reason="Select the candidate_index of one selection_eligible reviewed candidate, or defer.",
        )
    if not proposal.anchor_quote or not proposal.anchor_quote.strip():
        return ValidationResult(
            result=False,
            reason=(
                f"For selected candidate_index {selected.index}, copy its self_identifier "
                "as anchor_quote, or defer."
            ),
        )
    grounded = GroundingEvidence([EvidenceCandidate(selected.anchor, selected)]).resolve(
        proposal.anchor_quote,
        FuzzinessOption.edit_distance(similarity_percent=90),
    )
    if grounded is None:
        return ValidationResult(
            result=False,
            reason=(
                f"anchor_quote {proposal.anchor_quote!r} does not reproduce the selected "
                f"candidate_index {selected.index} self_identifier {selected.anchor!r} within 90% "
                "edit-distance similarity. Copy that candidate's own identifier, or defer."
            ),
        )
    return ValidationResult(result=True)


async def _resolve_direct_candidates(
    record: CitationRecord,
    candidates: tuple[_DirectBodyCandidate, ...],
    *,
    session: MelleaSession | None,
) -> Node:
    viable = tuple(candidate for candidate in candidates if not _field_conflicts(record, candidate))
    if not viable:
        # Docket numbers are not globally unique. A conflicting court or date
        # excludes these hits, but cannot establish that the cited case is
        # wrong; another docket with the same number may still be the target.
        if isinstance(record.fields, DocketCitation):
            return _body_identity_node(
                record,
                outcome="deferred",
                candidates=candidates,
                message="No same-number docket hit agrees with the stated fields; continue evidence search.",
            )
        return _body_identity_node(
            record,
            outcome="no_match",
            candidates=candidates,
            message="Every self-identifying body candidate contradicts a stated court or date.",
        )
    if len(viable) == 1 and _same_case_name(_stated_case_name(record), _candidate_case_name(viable[0])):
        return _body_identity_node(
            record,
            outcome="resolved",
            candidates=candidates,
            selected=viable[0],
            message="One self-identifying body candidate agrees with the stated fields.",
        )
    try:
        run = await run_instruct_ivr(
            session or start_mellea_session_from_env(),
            InstructIvrSpec(
                prefix=(
                    "Choose an authority only from self-identifying corpus hits. A self-identifying hit has "
                    "the citation's own reporter locator or docket number in its metadata. Do not choose a "
                    "document merely because it quotes the locator. Case-name variants can be equivalent, "
                    "but a stated court or date conflict cannot be repaired here. For a selection, copy "
                    "the selected candidate's self_identifier as anchor_quote."
                ),
                description=(
                    "source_locator:\n{{source_locator}}\n\nstated_case_name:\n{{stated_case_name}}"
                    "\n\nreviewed_candidates_json:\n{{reviewed_candidates_json}}"
                ),
                user_variables={
                    "source_locator": record.matched_text,
                    "stated_case_name": _stated_case_name(record),
                    "reviewed_candidates_json": json.dumps(
                        [
                            _direct_candidate_payload(candidate, viable=candidate in viable)
                            for candidate in candidates
                        ],
                        ensure_ascii=False,
                    ),
                },
                output_format=_BodyIdentityProposal,
                requirements=[
                    req(
                        "Ground a selected candidate in its own self_identifier.",
                        validation_fn=lambda ctx: _validate_direct_candidate_grounding(ctx, viable),
                    ),
                ],
            ),
            strategy=MultiTurnStrategy(loop_budget=2),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(
                max_tokens=_BODY_REVIEW_MAX_TOKENS
            ),
        )
        if not run.success:
            return _body_identity_node(
                record,
                outcome="deferred",
                candidates=candidates,
                message="Body identity review exhausted its repair budget.",
                error=run.failure_reason,
                run=run,
            )
        proposal = _BodyIdentityProposal.model_validate_json(run.output)
    except (Exception, ValidationError) as exc:
        return _body_identity_node(
            record,
            outcome="deferred",
            candidates=candidates,
            message="Body identity review did not produce a usable decision.",
            error=f"{type(exc).__name__}: {exc}",
        )
    selected = next((candidate for candidate in viable if candidate.index == proposal.candidate_index), None)
    grounded = (
        selected is not None
        and proposal.anchor_quote is not None
        and GroundingEvidence([EvidenceCandidate(selected.anchor, selected)]).resolve(
            proposal.anchor_quote,
            FuzzinessOption.edit_distance(similarity_percent=90),
        )
        is not None
    )
    if proposal.decision == "select_candidate" and grounded and selected is not None:
        return _body_identity_node(
            record,
            outcome="resolved",
            candidates=candidates,
            selected=selected,
            message=proposal.rationale,
            run=run,
        )
    return _body_identity_node(
        record,
        outcome="deferred",
        candidates=candidates,
        message=proposal.rationale,
        run=run,
    )


def _third_party_evidence(
    document: Document,
    record: CitationRecord,
    nodes: tuple[RootBodySearchNode, ...],
    client: CourtListenerServiceClient,
    govinfo_client: GovInfoClient,
    *,
    retrospective_date: CalendarDate | None = None,
    fetches: list[dict[str, object]],
) -> tuple[_ThirdPartyEvidence, ...]:
    """Read actual citing text; a search hit or its copied query is not evidence."""
    locator = _locator_for_body_search(record)
    if not locator:
        return ()
    found: list[_ThirdPartyEvidence] = []
    fetched = 0
    # Search sources have independent rankings. Alternate their candidates so
    # one large corpus cannot spend the entire fetch budget before another
    # corpus has a chance to provide a citing document.
    for rank in range(max((len(node.candidates) for node in nodes), default=0)):
        for node in nodes:
            if rank >= len(node.candidates):
                continue
            candidate = node.candidates[rank]
            if retrospective_date is not None:
                eligible = _retrospective_candidate(candidate, node.source, retrospective_date)
                if eligible is None:
                    continue
                candidate = eligible
            if _self_identifier(record, candidate) is not None:
                continue
            if _same_source_docket(document, node.source, candidate):
                fetches.append(
                    {
                        "origin": f"{node.source.value}:{_citing_identifier(node.source, candidate)}",
                        "outcome": "same_source_docket",
                        "source_docket_id": document.source_metadata.courtlistener_docket_id,
                    }
                )
                continue
            if node.source is not RootBodySearchSource.GOVINFO_PACKAGE and any(
                fetch.get("outcome") == "failed" and fetch.get("failure_type") == "api_limit"
                for fetch in fetches
            ):
                continue
            citing_id = _citing_identifier(node.source, candidate)
            if citing_id is None:
                continue
            if fetched >= _THIRD_PARTY_FETCH_LIMIT:
                continue
            fetched += 1
            for origin, text in _fetch_citing_texts(
                node.source,
                candidate,
                client,
                govinfo_client,
                retrospective_date=retrospective_date,
                fetches=fetches,
            ):
                if _same_source_document(document.text, text):
                    _set_fetch_outcome(fetches, origin, "same_source_document")
                    continue
                context = _locator_context(
                    locator, text, require_reporter_digits=isinstance(record.fields, FullCaseCitation)
                )
                if context is None:
                    _set_fetch_outcome(fetches, origin, "locator_absent")
                    continue
                excerpt, excerpt_start = context
                _set_fetch_outcome(fetches, origin, "citation_context_found")
                found.append(
                    _ThirdPartyEvidence(
                        index=len(found) + 1,
                        source=node.source,
                        citing_id=citing_id,
                        citing_case_name=_optional_string(candidate.get("caseName")),
                        excerpt=excerpt,
                        excerpt_start=excerpt_start,
                        origin=origin,
                    )
                )
                break
            if len(found) >= _THIRD_PARTY_EVIDENCE_LIMIT:
                return tuple(found)
    return tuple(found)


def _same_source_docket(
    document: Document, source: RootBodySearchSource, candidate: Mapping[str, object]
) -> bool:
    """A later filing in the source case is not independent third-party evidence."""
    if source is RootBodySearchSource.GOVINFO_PACKAGE:
        return False
    source_id = document.source_metadata.courtlistener_docket_id
    candidate_id = _optional_string(candidate.get("docket_id"))
    return source_id is not None and candidate_id is not None and source_id == candidate_id


def _locator_for_body_search(record: CitationRecord) -> str:
    citation = record.fields
    if isinstance(citation, DocketCitation):
        return citation.docket_number or record.matched_text
    if isinstance(citation, FullCaseCitation):
        return _full_reporter_locator(citation) or record.matched_text
    return record.matched_text


def _docket_body_fallback_query(record: CitationRecord, original: str) -> str | None:
    """Retry a zero-hit labeled docket using its parsed identifier alone."""
    if not isinstance(record.fields, DocketCitation):
        return None
    parsed = " ".join((record.fields.docket_number or "").split())
    return parsed if parsed and parsed != original else None


def _citing_identifier(source: RootBodySearchSource, candidate: Mapping[str, object]) -> str | None:
    key = {
        RootBodySearchSource.COURTLISTENER_CLUSTER: "cluster_id",
        RootBodySearchSource.COURTLISTENER_DOCKET: "docket_id",
        RootBodySearchSource.GOVINFO_PACKAGE: "govinfo_package_id",
    }[source]
    return _optional_string(candidate.get(key))


def _fetch_citing_texts(
    source: RootBodySearchSource,
    candidate: Mapping[str, object],
    client: CourtListenerServiceClient,
    govinfo_client: GovInfoClient,
    *,
    retrospective_date: CalendarDate | None = None,
    fetches: list[dict[str, object]],
) -> tuple[tuple[str, str], ...]:
    if source is RootBodySearchSource.GOVINFO_PACKAGE:
        package_id = _optional_string(candidate.get("govinfo_package_id"))
        if package_id is None:
            return ()
        origin = f"govinfo_package:{package_id}:full_text"
        try:
            content = (
                govinfo_client.get_package_text(package_id, retrospective_date=retrospective_date)
                if retrospective_date is not None
                else govinfo_client.get_package_text(package_id)
            )
        except Exception as exc:
            fetches.append({"origin": origin, "outcome": "failed", "error": f"{type(exc).__name__}: {exc}"})
            return ()
        fetches.append({"origin": origin, "outcome": "fetched" if content else "empty"})
        return ((origin, content),) if content else ()
    nested_key = "opinions" if source is RootBodySearchSource.COURTLISTENER_CLUSTER else "recap_documents"
    nested = candidate.get(nested_key)
    if not isinstance(nested, (list, tuple)):
        return ()
    texts: list[tuple[str, str]] = []
    for item in nested[:2]:
        if not isinstance(item, Mapping):
            continue
        item_id = _optional_string(item.get("id"))
        if item_id is None:
            continue
        origin = f"{nested_key}:{item_id}:full_text"
        content = _read_courtlistener_body(source, item_id, client, origin=origin, fetches=fetches)
        if content is None:
            if any(
                fetch.get("origin") == origin
                and fetch.get("outcome") == "failed"
                and fetch.get("failure_type") == "api_limit"
                for fetch in fetches
            ):
                break
            continue
        fetches.append({"origin": origin, "outcome": "fetched" if content else "empty"})
        if content:
            texts.append((origin, content))
    return tuple(texts)


def _read_courtlistener_body(
    source: RootBodySearchSource,
    item_id: str,
    client: CourtListenerServiceClient,
    *,
    origin: str,
    fetches: list[dict[str, object]],
) -> str | None:
    """Retry short quota windows and transient failures, retaining each failure."""
    waited_seconds = 0.0
    for attempt in range(3):
        try:
            if source is RootBodySearchSource.COURTLISTENER_CLUSTER:
                return _plain_html(client.get_opinion(item_id).html_with_citations)
            return client.get_recap_document(item_id).plain_text
        except CourtListenerError as exc:
            delay = courtlistener_retry_delay(exc, retry_number=attempt + 1, waited_seconds=waited_seconds)
            fetches.append(
                {
                    "origin": origin,
                    "outcome": "retry" if delay is not None else "failed",
                    "attempt": attempt + 1,
                    "failure_type": exc.failure_type,
                    "http_status": exc.upstream_status_code,
                    "retryable": exc.retryable,
                    "retry_after_seconds": exc.retry_after_seconds,
                    "wait_seconds": delay,
                    "error": str(exc),
                }
            )
            if delay is not None:
                time.sleep(delay)
                waited_seconds += delay
                continue
            return None
        except Exception as exc:
            fetches.append({"origin": origin, "outcome": "failed", "error": f"{type(exc).__name__}: {exc}"})
            return None
    return None


def _set_fetch_outcome(fetches: list[dict[str, object]], origin: str, outcome: str) -> None:
    for fetch in reversed(fetches):
        if fetch.get("origin") == origin:
            fetch["outcome"] = outcome
            return


class _BodyTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_html(html: str) -> str:
    parser = _BodyTextParser()
    parser.feed(html)
    return " ".join(parser.parts)


def _locator_context(
    locator: str, text: str, *, require_reporter_digits: bool = False
) -> tuple[str, int] | None:
    if not locator or not text:
        return None
    start = 0
    # Fuzzy matching helps with copied punctuation and OCR, but a reporter
    # citation with a different year or serial is a different locator. Keep
    # looking after a near miss in case the correct citation occurs later.
    while start < len(text):
        match = fuzzy_find(locator, text[start:], FuzzinessOption.edit_distance(similarity_percent=90))
        if match is None:
            return None
        relative = text[start:].find(match)
        if relative < 0:
            return None
        offset = start + relative
        if not require_reporter_digits or _locator_digits_agree(locator, match):
            left = max(0, offset - _THIRD_PARTY_CONTEXT_CHARS)
            right = min(len(text), offset + len(match) + _THIRD_PARTY_CONTEXT_CHARS)
            return text[left:right], left
        start = offset + max(1, len(match))
    return None


def _source_citation_context(document: Document, record: CitationRecord) -> tuple[str, int]:
    """A bounded view of this citation in the filing, with document offset."""
    start = max(0, record.locator_span.start - 320)
    end = min(len(document.text), record.locator_span.end + 320)
    return document.text[start:end], start


def _same_source_document(source_text: str, candidate_text: str) -> bool:
    """Reject an archive copy of the very filing being validated.

    A later opinion may quote one passage, but an archived copy of the filing
    shares many passages throughout. Sample short word runs across the source
    so line wrapping, OCR changes, and offset-preserving noise masks do not
    make a copied filing look independent.
    """
    source_words = re.findall(r"[a-z0-9]+", source_text.casefold())
    candidate_words = re.findall(r"[a-z0-9]+", candidate_text.casefold())
    if len(source_words) < 120 or len(candidate_words) < 120:
        return False
    candidate_flat = " ".join(candidate_words)
    step = max(16, len(source_words) // 32)
    windows = [" ".join(source_words[index : index + 16]) for index in range(0, len(source_words) - 15, step)]
    matches = sum(window in candidate_flat for window in windows)
    return matches >= max(4, ceil(len(windows) * 0.35))


async def _review_third_party(
    document: Document,
    record: CitationRecord,
    evidence: tuple[_ThirdPartyEvidence, ...],
    *,
    fetches: list[dict[str, object]],
    session: MelleaSession | None,
    stage: str = ROOT_BODY_CORROBORATION_RESOLUTION_STAGE,
    extra_depends_on: tuple[str, ...] = (),
) -> Node:
    source_context, source_start = _source_citation_context(document, record)
    locator_start = record.locator_span.start - source_start
    locator_end = record.locator_span.end - source_start
    marked_source_context = (
        source_context[:locator_start]
        + "<target_locator>"
        + source_context[locator_start:locator_end]
        + "</target_locator>"
        + source_context[locator_end:]
    )
    try:
        run = await run_instruct_ivr(
            session or start_mellea_session_from_env(),
            InstructIvrSpec(
                prefix=(
                    "Compare the citation in the source filing with a citation written inside an independent "
                    "opinion or filing. That document's own case name, court, date, and docket "
                    "describe the citing document; do not assign them to the cited case. Select one citation "
                    "written in a body excerpt. Copy only that entire citation as citation_quote, starting "
                    "with its case name and ending with its citation's court/date if present. Do not quote "
                    "the surrounding sentence, argument, explanatory parenthetical, or the whole excerpt. "
                    "Read source_fields only from source_context near the marked locator; the target_locator "
                    "tags are boundary markers, not citation text. This is another "
                    "chance to correct extraction, so copy the filing's literal case name, locator (without "
                    "an introductory label such as Case No.), court, "
                    "and date even where source_citation_json differs. Never copy a third-party value into "
                    "source_fields. Read cited_fields only from citation_quote, copying those four values "
                    "literally. For each side, locator is only the specific identifier being compared, without "
                    "a parallel locator or pin cite; citation_quote still contains the full citation. Use "
                    "null for a field that its own document does not state. Compare each pair "
                    "semantically and independently as match, mismatch, or unavailable; the comparison is "
                    "your judgment, not a literal-string test. In reason, explain the overall decision, any "
                    "actual source-field correction, and any conflict. A different display spelling of an "
                    "already correct court or date is not a correction. Do not let a court or date conflict change the "
                    "case-name or locator judgment. "
                    "Ordinary abbreviations and docket display variations can agree, but a misspelled "
                    "party or title is a case-name mismatch even if the intended case is recognizable. "
                    "Return corroborates "
                    "when the grounded citation supports the same identity. Return does_not_corroborate when "
                    "it uses the same locator but affirmatively identifies a different case name, court, or "
                    "decision date. A docket identifies a case that may have several opinions or orders: a "
                    "different date on the same docket is a contradiction only when both citations identify "
                    "the same opinion. Check any parallel reporter citation for that distinction. Return defer "
                    "for a date difference between distinct or unidentified opinions, a mere locator mention, an "
                    "unclear citation, a copied filing, a discussion of a nonexistent citation, or a quote "
                    "offered as an example of error. For defer without a complete citation, use null quote, "
                    "null cited_fields, and unavailable comparisons. Compare every digit of reporter citations."
                ),
                description=(
                    "source_citation_json:\n{{source_citation_json}}\n\n"
                    "source_context:\n{{source_context}}\n\n"
                    "independent_body_evidence_json:\n{{evidence_json}}"
                ),
                user_variables={
                    "source_citation_json": json.dumps(
                        {
                            "case_name": _stated_case_name(record),
                            "locator": _locator_for_body_search(record),
                            "court": getattr(record.fields, "court", None),
                            "date": _stated_date_value(record),
                        },
                        ensure_ascii=False,
                    ),
                    "source_context": marked_source_context,
                    "evidence_json": json.dumps(
                        [
                            {
                                "evidence_index": item.index,
                                "citing_record": f"{item.source.value}:{item.citing_id}",
                                "citing_case_name": item.citing_case_name,
                                "body_excerpt": item.excerpt,
                            }
                            for item in evidence
                        ],
                        ensure_ascii=False,
                    ),
                },
                output_format=_ThirdPartyProposal,
                requirements=[
                    req(
                        "Ground the reviewed source locator at the marked target locator.",
                        validation_fn=lambda ctx: _validate_third_party_source_locator(ctx, document, record),
                    ),
                    req(
                        "Ground the independent citation and every proposed field in its own document.",
                        validation_fn=lambda ctx: _validate_third_party_grounding(
                            ctx, source_context, evidence
                        ),
                    ),
                ],
            ),
            strategy=MultiTurnStrategy(loop_budget=2),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=900),
        )
        if not run.success:
            return _third_party_node(
                record,
                evidence,
                message="Third-party review exhausted repairs.",
                fetches=fetches,
                run=run,
                stage=stage,
                extra_depends_on=extra_depends_on,
            )
        proposal = _ThirdPartyProposal.model_validate_json(run.output)
    except Exception as exc:
        return _third_party_node(
            record,
            evidence,
            message="Third-party review failed.",
            fetches=fetches,
            error=f"{type(exc).__name__}: {exc}",
            stage=stage,
            extra_depends_on=extra_depends_on,
        )
    selected = next((item for item in evidence if item.index == proposal.evidence_index), None)
    quote = proposal.citation_quote
    policy = FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True)
    quote_match = (
        GroundingEvidence((EvidenceCandidate(selected.excerpt, selected),)).find_fragment(
            quote, policy, line_number_aware=True
        )
        if selected is not None and isinstance(quote, str) and quote.strip()
        else None
    )
    actual_quote = quote_match.text if quote_match is not None else None
    source_fields, source_offsets, source_grounded = _ground_review_fields(
        proposal.source_fields, source_context, source_start, policy
    )
    # A repeated identifier elsewhere in the context cannot stand in for this
    # citation. Its source-side reading must overlap the locator's own span.
    if proposal.source_fields.locator is not None:
        locator_start = max(0, record.locator_span.start - 24)
        locator_end = min(len(document.text), record.locator_span.end + 24)
        local_locator = GroundingEvidence(
            (EvidenceCandidate(document.text[locator_start:locator_end], record),)
        ).find_fragment(proposal.source_fields.locator, policy)
        if local_locator is not None:
            matched_start = locator_start + local_locator.start
            matched_end = locator_start + local_locator.end
            if matched_start < record.locator_span.end and matched_end > record.locator_span.start:
                source_fields["locator"] = local_locator.text
                source_offsets["locator"] = (matched_start, matched_end)
            else:
                source_fields["locator"] = None
                source_offsets.pop("locator", None)
                source_grounded = False
        else:
            source_fields["locator"] = None
            source_offsets.pop("locator", None)
            source_grounded = False
    # The quote is highlighted against original bytes, while field values are
    # read from the verified margin-number-free view of that same quote.
    normalized_quote = quote_match.normalized_text if quote_match is not None else ""
    cited_fields, cited_offsets, cited_grounded = _ground_review_fields(
        proposal.cited_fields, normalized_quote, 0, policy
    )
    name = cited_fields["case_name"]
    actual_locator = cited_fields["locator"]
    focused_quote = (
        actual_quote is not None
        and len(actual_quote) <= _MAX_CITATION_QUOTE_CHARS
        and not normalized_quote[: cited_offsets.get("case_name", (len(normalized_quote), 0))[0]].strip()
    )
    grounded = (
        selected is not None
        and quote_match is not None
        and source_grounded
        and cited_grounded
        and source_fields["locator"] is not None
        and isinstance(name, str)
        and isinstance(actual_locator, str)
        and focused_quote
        and _locator_digits_agree(proposal.cited_fields.locator or "", actual_locator)
    )
    comparisons = _third_party_comparison_payload(record, proposal.comparisons, source_fields, cited_fields)
    opinion_locators = _docket_opinion_locator_comparison(document, record, actual_quote)
    citation_grounding = (
        {
            "origin": selected.origin,
            "matched_text": quote_match.text,
            "excerpt_span": {"start": quote_match.start, "end": quote_match.end},
            "document_span": {
                "start": selected.excerpt_start + quote_match.start,
                "end": selected.excerpt_start + quote_match.end,
            },
            "match_type": quote_match.match_type.value,
            "similarity_percent": quote_match.similarity_percent,
            "edits": quote_match.edits,
        }
        if selected is not None and quote_match is not None and focused_quote
        else None
    )
    outcome = "deferred"
    if grounded and proposal.comparisons.locator == "match":
        field_outcomes = (
            proposal.comparisons.case_name,
            proposal.comparisons.court,
            proposal.comparisons.date,
        )
        if proposal.decision == "corroborates" and "mismatch" not in field_outcomes:
            outcome = "resolved"
        elif proposal.decision == "does_not_corroborate" and "mismatch" in field_outcomes:
            # A docket names the case, not one decision in it. A date-only
            # conflict cannot reject that case when the two citations identify
            # different opinions, or no opinion-specific locator is available.
            other_conflict = any(
                getattr(proposal.comparisons, field) == "mismatch" for field in ("case_name", "court")
            )
            if other_conflict or opinion_locators is None or opinion_locators["shared"]:
                outcome = "no_match"
    node = _third_party_node(
        record,
        evidence,
        outcome=outcome,
        selected=selected if focused_quote else None,
        cited_case_name=name if outcome != "deferred" else None,
        citation_quote=actual_quote if focused_quote else None,
        cited_locator=actual_locator if outcome != "deferred" else None,
        message=proposal.reason,
        fetches=fetches,
        run=run,
        field_comparisons=comparisons,
        source_fields=source_fields,
        cited_fields=cited_fields,
        citation_grounding=citation_grounding,
        opinion_locator_comparison=opinion_locators,
        stage=stage,
        extra_depends_on=extra_depends_on,
    )
    observe_citation(record, node)
    _apply_source_review(record, document, node, proposal, source_fields, source_offsets, source_grounded)
    return node


def _validate_third_party_source_locator(
    ctx: Context, document: Document, record: CitationRecord
) -> ValidationResult:
    """Repair a model reading a parallel locator instead of this root's locator."""
    try:
        proposal = _ThirdPartyProposal.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        # The IVR wrapper supplies schema errors; this checks a valid proposal.
        return ValidationResult(result=True)
    if proposal.decision == "defer":
        return ValidationResult(result=True)
    locator = proposal.source_fields.locator
    expected = _locator_for_body_search(record)
    if not locator:
        return ValidationResult(
            result=False,
            reason=(
                "For corroborates or does_not_corroborate, source_fields.locator must copy "
                f"the marked target_locator {expected!r}, not a parallel identifier."
            ),
        )
    start = max(0, record.locator_span.start - 24)
    end = min(len(document.text), record.locator_span.end + 24)
    policy = FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True)
    matched = GroundingEvidence((EvidenceCandidate(document.text[start:end], record),)).find_fragment(
        locator, policy
    )
    if matched is None or (
        start + matched.start >= record.locator_span.end or start + matched.end <= record.locator_span.start
    ):
        return ValidationResult(
            result=False,
            reason=(
                f"source_fields.locator {locator!r} is not the marked target_locator {expected!r}. "
                "Copy the identifier between the target_locator tags; a nearby parallel docket or "
                "reporter identifier belongs to a different locator."
            ),
        )
    return ValidationResult(result=True)


def _validate_third_party_grounding(
    ctx: Context, source_context: str, evidence: tuple[_ThirdPartyEvidence, ...]
) -> ValidationResult:
    """Give the model actionable repair feedback before an ungrounded verdict."""
    try:
        proposal = _ThirdPartyProposal.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        return ValidationResult(result=True)
    if proposal.decision == "defer":
        return ValidationResult(result=True)
    selected = next((item for item in evidence if item.index == proposal.evidence_index), None)
    if selected is None or not proposal.citation_quote:
        return ValidationResult(
            result=False,
            reason="Select an evidence_index and copy its complete citation as citation_quote, or defer.",
        )
    policy = FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True)
    quote = GroundingEvidence((EvidenceCandidate(selected.excerpt, selected),)).find_fragment(
        proposal.citation_quote, policy, line_number_aware=True
    )
    if quote is None:
        return ValidationResult(
            result=False,
            reason=(
                "citation_quote does not match the selected body_excerpt. Copy the citation's actual "
                "characters, including any OCR artifact, or select another complete citation; defer if neither works."
            ),
        )
    if len(quote.text) > _MAX_CITATION_QUOTE_CHARS:
        return ValidationResult(
            result=False,
            reason="citation_quote includes too much context. Copy only the complete citation, or defer.",
        )
    if not proposal.cited_fields.case_name or not proposal.cited_fields.locator:
        return ValidationResult(
            result=False,
            reason=(
                "A non-deferred comparison needs cited_fields.case_name and cited_fields.locator "
                "copied from citation_quote; defer if it is not a complete citation."
            ),
        )
    for label, fields, text in (
        ("source_fields", proposal.source_fields, source_context),
        ("cited_fields", proposal.cited_fields, quote.normalized_text),
    ):
        for field_name in _REVIEW_FIELDS:
            value = getattr(fields, field_name)
            if value is None:
                continue
            found = GroundingEvidence((EvidenceCandidate(text, text),)).find_fragment(value, policy)
            if found is None:
                return ValidationResult(
                    result=False,
                    reason=(
                        f"{label}.{field_name}={value!r} is not grounded in its own document. "
                        "Copy that document's visible text or use null when the field is not stated; "
                        "then reconsider that field's comparison."
                    ),
                )
    cited_name = GroundingEvidence((EvidenceCandidate(quote.normalized_text, quote),)).find_fragment(
        proposal.cited_fields.case_name, policy
    )
    if cited_name is None or quote.normalized_text[: cited_name.start].strip():
        return ValidationResult(
            result=False,
            reason="citation_quote must begin with the cited case name, without surrounding prose.",
        )
    return ValidationResult(result=True)


def _docket_opinion_locator_comparison(
    document: Document, record: CitationRecord, citation_quote: str | None
) -> dict[str, object] | None:
    """Check whether two citations on a docket identify the same opinion.

    Eyecite reads only the citation's own source span and the grounded quoted
    citation, so unrelated reporter citations elsewhere in either document do
    not make a docket date disagreement decisive.
    """
    if not isinstance(record.fields, DocketCitation):
        return None
    span = record.fields.span
    source_text = document.text[span.start : span.end] if span is not None else ""
    source = _opinion_specific_locators(source_text)
    cited = _opinion_specific_locators(citation_quote or "")
    return {
        "source": sorted(source),
        "cited": sorted(cited),
        "shared": bool(source & cited),
    }


def _opinion_specific_locators(citation_text: str) -> frozenset[str]:
    """Reporter identifiers inside one citation, normalized for display only."""
    if not citation_text.strip():
        return frozenset()
    return frozenset(
        _canonical_locator(citation.matched_text())
        for citation in get_citations(citation_text)
        if isinstance(citation, EyeciteFullCaseCitation)
    )


def _stated_date_value(record: CitationRecord) -> str | None:
    date = getattr(record.fields, "date", None)
    if date is None:
        return None
    return " ".join(str(part) for part in (date.month, date.day, date.year) if part is not None)


_REVIEW_FIELDS = ("locator", "case_name", "court", "date")


def _ground_review_fields(
    proposed: _CitationFieldValues,
    text: str,
    offset: int,
    policy: FuzzinessOption,
) -> tuple[dict[str, str | None], dict[str, tuple[int, int]], bool]:
    """Use each document, not the other document, to verify its own readings."""
    evidence = GroundingEvidence((EvidenceCandidate(text, text),))
    values: dict[str, str | None] = {}
    spans: dict[str, tuple[int, int]] = {}
    all_grounded = True
    for field_name in _REVIEW_FIELDS:
        value = getattr(proposed, field_name)
        if value is None:
            values[field_name] = None
            continue
        match = evidence.find_fragment(value, policy)
        values[field_name] = match.normalized_text if match is not None else None
        if match is None:
            all_grounded = False
        else:
            spans[field_name] = (offset + match.start, offset + match.end)
    return values, spans, all_grounded


def _third_party_comparison_payload(
    record: CitationRecord,
    comparisons: _ThirdPartyComparisons,
    source_fields: Mapping[str, str | None],
    cited_fields: Mapping[str, str | None],
) -> dict[str, dict[str, str | None]]:
    stated = {
        "locator": _locator_for_body_search(record),
        "case_name": _stated_case_name(record) or None,
        "court": getattr(record.fields, "court", None),
        "date": _stated_date_value(record),
    }
    return {
        field_name: {
            "original_stated_value": stated[field_name],
            "source_value": source_fields.get(field_name),
            "cited_value": cited_fields.get(field_name),
            "outcome": getattr(comparisons, field_name),
        }
        for field_name in _REVIEW_FIELDS
    }


def _apply_source_review(
    record: CitationRecord,
    document: Document,
    review_node: Node,
    proposal: _ThirdPartyProposal,
    source_fields: Mapping[str, str | None],
    spans: Mapping[str, tuple[int, int]],
    all_grounded: bool,
) -> None:
    """Write grounded filing rereads as one batch of field updates.

    The one model call reads two documents, but its source-side corrections need
    DOCUMENT provenance. Its third-party identity verdict has RECORD provenance.
    Keeping distinct nodes prevents archive evidence from rewriting the filing.
    """
    locator = source_fields.get("locator")
    if not all_grounded or locator is None or "locator" not in spans:
        return
    document_node = Node(
        node_id=(
            f"{record.citation_id}:root_body:source_reread"
            if review_node.stage == ROOT_BODY_CORROBORATION_RESOLUTION_STAGE
            else f"{record.citation_id}:shared_body:source_reread"
        ),
        reads=Reads.DOCUMENT,
        stage=review_node.stage,
        made_by=_MADE_BY,
        outcome="grounded_reextract",
        message=proposal.reason,
        depends_on=(review_node.node_id,),
        details={
            "source_fields": dict(source_fields),
            "source_spans": {key: {"start": span[0], "end": span[1]} for key, span in spans.items()},
            "review_node_id": review_node.node_id,
        },
    )
    reason = proposal.reason
    citation = record.fields
    changes: dict[CitationField, object] = {}

    if isinstance(citation, DocketCitation):
        # The docket identifier is plain text and can be reread from its own span.
        # Reporter locators carry a structured Reporter and stay read-only here.
        loc_start, loc_end = spans["locator"]
        if (
            loc_start >= record.locator_span.start - 24
            and loc_end <= record.locator_span.end + 24
            and locator != citation.docket_number
        ):
            changes[CitationField.DOCKET_NUMBER] = locator

    name = source_fields.get("case_name")
    name_span = spans.get("case_name")
    if name is not None and name_span is not None and name_span[1] <= record.locator_span.start:
        plaintiff, defendant = _parties_from_reviewed_name(name)
        reread = CaseName(
            span=Span(*name_span),
            text=document.text[name_span[0] : name_span[1]],
            plaintiff=plaintiff,
            defendant=defendant,
        )
        if record.fields.case_name != reread:
            changes[CitationField.CASE_NAME] = reread
        for field_name, value in ((CitationField.PLAINTIFF, plaintiff), (CitationField.DEFENDANT, defendant)):
            if getattr(record.fields, field_name.value) != value:
                changes[field_name] = value

    court_text = source_fields.get("court")
    if court_text is not None:
        court_id = resolve_court(court_text)
        if court_id is not None and court_id != record.fields.court:
            changes[CitationField.COURT] = court_id
            if isinstance(record.fields, DocketCitation):
                if record.fields.court_text != court_text:
                    changes[CitationField.COURT_TEXT] = court_text
                if record.fields.court_name is not None:
                    changes[CitationField.COURT_NAME] = None

    date_text = source_fields.get("date")
    parsed_date = _parse_reviewed_date(date_text) if date_text is not None else None
    existing_date = record.fields.date
    would_lose_precision = (
        existing_date is not None
        and existing_date.is_exact
        and not (parsed_date is not None and parsed_date.is_exact)
        and parsed_date is not None
        and parsed_date.year == existing_date.year
    )
    if parsed_date is not None and parsed_date != existing_date and not would_lose_precision:
        changes[CitationField.DATE] = parsed_date
    update_fields(record, document_node, changes, reason=reason)
    mark_extraction_reviewed(record, document_node)


def _parties_from_reviewed_name(name: str) -> tuple[str | None, str | None]:
    parts = re.split(r"\s+v(?:s)?\.?\s+", name, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    if re.match(r"(?:In\s+re|Ex\s+parte)\s+", name, flags=re.IGNORECASE):
        return None, re.sub(r"^(?:In\s+re|Ex\s+parte)\s+", "", name, flags=re.IGNORECASE)
    return None, None


_REVIEW_DATE = re.compile(
    r"(?:(?P<month>(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?)"
    r"\s+(?P<day>\d{1,2}),?\s+)?(?P<year>\d{4})",
    re.IGNORECASE,
)


def _parse_reviewed_date(value: str) -> CitationDate | None:
    match = _REVIEW_DATE.fullmatch(value.strip())
    if match is None:
        return None
    month = match.group("month")
    day = match.group("day")
    return CitationDate(year=match.group("year"), month=month, day=day)


def _third_party_node(
    record: CitationRecord,
    evidence: tuple[_ThirdPartyEvidence, ...],
    *,
    message: str,
    outcome: str = "deferred",
    selected: _ThirdPartyEvidence | None = None,
    cited_case_name: str | None = None,
    citation_quote: str | None = None,
    cited_locator: str | None = None,
    field_comparisons: Mapping[str, object] | None = None,
    source_fields: Mapping[str, str | None] | None = None,
    cited_fields: Mapping[str, str | None] | None = None,
    citation_grounding: Mapping[str, object] | None = None,
    opinion_locator_comparison: Mapping[str, object] | None = None,
    fetches: list[dict[str, object]] | None = None,
    error: str | None = None,
    run: object | None = None,
    stage: str = ROOT_BODY_CORROBORATION_RESOLUTION_STAGE,
    extra_depends_on: tuple[str, ...] = (),
) -> Node:
    return Node(
        node_id=(
            f"{record.citation_id}:root_body:third_party_review"
            if stage == ROOT_BODY_CORROBORATION_RESOLUTION_STAGE
            else f"{record.citation_id}:shared_body:third_party_review"
        ),
        reads=Reads.RECORD,
        stage=stage,
        made_by=_MADE_BY,
        outcome=outcome,
        message=message,
        depends_on=(
            *(f"{record.citation_id}:root_body:{source.value}" for source in RootBodySearchSource),
            *extra_depends_on,
        ),
        details={
            "judgement_type": "third_party",
            "evidence": [serialize_dataclass(item) for item in evidence],
            "fetches": fetches or [],
            "selected_evidence_index": selected.index if selected else None,
            "cited_case_name": cited_case_name,
            "citation_quote": citation_quote,
            "cited_locator": cited_locator,
            "field_comparisons": dict(field_comparisons) if field_comparisons is not None else None,
            "source_fields": dict(source_fields) if source_fields is not None else None,
            "cited_fields": dict(cited_fields) if cited_fields is not None else None,
            "citation_grounding": dict(citation_grounding) if citation_grounding is not None else None,
            "opinion_locator_comparison": (
                dict(opinion_locator_comparison) if opinion_locator_comparison is not None else None
            ),
            "error": error,
            "ivr": serialize_dataclass(run) if run is not None else None,
        },
    )


def _third_party_authority_id(record: CitationRecord, case_name: str) -> str:
    """Stable local identity handle; it never pretends to be the citing record ID."""
    key = "|".join(
        (
            type(record.fields).__name__,
            _canonical_locator(_locator_for_body_search(record)),
            _normalized_name(case_name),
            str(getattr(record.fields, "court", None) or ""),
            str(getattr(getattr(record.fields, "date", None), "year", "") or ""),
        )
    )
    return f"third_party:{sha256(key.encode()).hexdigest()[:20]}"


def _locator_digits_agree(cited_locator: str, actual_locator: str) -> bool:
    """A fuzzy copy may vary punctuation; it cannot change printed digits."""
    cited_digits = re.findall(r"\d+", cited_locator)
    return cited_digits == re.findall(r"\d+", actual_locator)


def _field_conflicts(record: CitationRecord, candidate: _DirectBodyCandidate) -> bool:
    citation = record.fields
    stated_court = citation.court if isinstance(citation, (DocketCitation, FullCaseCitation)) else None
    candidate_court = _optional_string(candidate.record.get("court_id"))
    if stated_court and candidate_court and stated_court != candidate_court:
        return True
    stated_date = citation.date if isinstance(citation, (DocketCitation, FullCaseCitation)) else None
    if stated_date is None:
        return False
    if candidate.source is RootBodySearchSource.COURTLISTENER_CLUSTER:
        decision_date = _optional_string(candidate.record.get("decisionDate"))
        return decision_date is not None and decision_date[:4] != stated_date.year
    if candidate.source is RootBodySearchSource.COURTLISTENER_DOCKET:
        filed = _optional_string(candidate.record.get("dateFiled"))
        return _filing_date_after_citation(filed, stated_date.year)
    return False


def _filing_date_after_citation(value: str | None, stated_year: str) -> bool:
    if value is None:
        return False
    try:
        return CalendarDate.fromisoformat(value).year > int(stated_year)
    except ValueError:
        return False


def _selected_direct_candidate(
    node: Node,
    candidates: tuple[_DirectBodyCandidate, ...],
) -> _DirectBodyCandidate:
    selected_index = node.details.get("selected_candidate_index")
    selected = next((candidate for candidate in candidates if candidate.index == selected_index), None)
    if selected is None:
        msg = "A resolved body node must identify one direct candidate."
        raise ValueError(msg)
    return selected


def _resolution_from_body_candidate(
    record: CitationRecord,
    candidate: _DirectBodyCandidate,
    node: Node,
) -> Resolution:
    citations = candidate.record.get("citation")
    return Resolution(
        cluster_id=_optional_string(candidate.record.get("cluster_id")),
        case_name=_candidate_case_name(candidate),
        date_filed=_optional_string(candidate.record.get("decisionDate"))
        or _optional_string(candidate.record.get("dateFiled")),
        court_id=_optional_string(candidate.record.get("court_id")),
        node_id=node.node_id,
        citations=tuple(value for value in citations if isinstance(value, str))
        if isinstance(citations, (list, tuple))
        else (),
        docket_id=_optional_string(candidate.record.get("docket_id")),
        govinfo_package_id=_optional_string(candidate.record.get("govinfo_package_id")),
    )


def _body_authority_id(candidate: _DirectBodyCandidate) -> str:
    if candidate.source is RootBodySearchSource.COURTLISTENER_CLUSTER:
        identifier = _optional_string(candidate.record.get("cluster_id"))
        if identifier:
            return f"courtlistener:cluster:{identifier}"
    if candidate.source is RootBodySearchSource.COURTLISTENER_DOCKET:
        identifier = _optional_string(candidate.record.get("docket_id"))
        if identifier:
            return f"courtlistener:docket:{identifier}"
    identifier = _optional_string(candidate.record.get("govinfo_package_id"))
    if identifier:
        return f"govinfo:package:{identifier}"
    return f"body_corroboration:{candidate.source.value}:{candidate.index}"


def _body_identity_node(
    record: CitationRecord,
    *,
    outcome: str,
    candidates: tuple[_DirectBodyCandidate, ...],
    message: str,
    selected: _DirectBodyCandidate | None = None,
    error: str | None = None,
    run: object | None = None,
) -> Node:
    return Node(
        node_id=f"{record.citation_id}:root_body:identity_review",
        reads=Reads.RECORD,
        stage=ROOT_BODY_CORROBORATION_RESOLUTION_STAGE,
        made_by=_MADE_BY,
        outcome=outcome,
        message=message,
        details={
            "direct_candidates": [
                _direct_candidate_payload(candidate, viable=not _field_conflicts(record, candidate))
                for candidate in candidates
            ],
            "selected_candidate_index": selected.index if selected else None,
            "error": error,
            "ivr": serialize_dataclass(run) if run is not None else None,
        },
    )


def _direct_candidate_payload(candidate: _DirectBodyCandidate, *, viable: bool) -> dict[str, object]:
    return {
        "candidate_index": candidate.index,
        "source": candidate.source.value,
        "self_identifier": candidate.anchor,
        "selection_eligible": viable,
        "case_name": _candidate_case_name(candidate),
        "court_id": _optional_string(candidate.record.get("court_id")),
        "decision_date": _optional_string(candidate.record.get("decisionDate")),
        "date_filed": _optional_string(candidate.record.get("dateFiled")),
        "cluster_id": _optional_string(candidate.record.get("cluster_id")),
        "docket_id": _optional_string(candidate.record.get("docket_id")),
        "govinfo_package_id": _optional_string(candidate.record.get("govinfo_package_id")),
    }


def _candidate_case_name(candidate: _DirectBodyCandidate) -> str | None:
    return _optional_string(candidate.record.get("caseName"))


def _stated_case_name(record: CitationRecord) -> str:
    citation = record.fields
    if citation.case_name is not None and citation.case_name.text.strip():
        return citation.case_name.text.strip()
    return " v. ".join(
        part.strip() for part in (citation.plaintiff, citation.defendant) if part and part.strip()
    )


def _same_case_name(left: str, right: str | None) -> bool:
    return bool(right) and _normalized_name(left) == _normalized_name(right)


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _full_reporter_locator(citation: FullCaseCitation) -> str:
    if citation.volume is None or citation.reporter is None or citation.page is None:
        return ""
    return f"{citation.volume} {citation.reporter} {citation.page}"


def _canonical_locator(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _optional_string(value: object) -> str | None:
    return str(value) if isinstance(value, str | int) else None


def _courtlistener_body_node(
    record: CitationRecord,
    *,
    locator: str,
    source: RootBodySearchSource,
    search_type: str,
    client: CourtListenerServiceClient,
    retrospective_date: CalendarDate | None = None,
) -> RootBodySearchNode:
    if not locator:
        return _body_node(
            record,
            source=source,
            status=ValidationNodeStatus.SKIPPED,
            outcome=RootBodySearchOutcome.UNAVAILABLE,
            locator=locator,
            query=None,
            status_message="Skipped body search because this root has no locator text.",
            outcome_message="Body corroboration is unavailable without a source locator.",
        )
    retry_diagnostics: list[str] = []
    search_attempts: list[BodySearchAttempt] = []
    waited_seconds = 0.0
    try:
        for attempt in range(3):
            try:
                result = client.search(locator, search_type)  # type: ignore[arg-type]
                break
            except CourtListenerError as exc:
                delay = courtlistener_retry_delay(
                    exc, retry_number=attempt + 1, waited_seconds=waited_seconds
                )
                retry_diagnostics.append(
                    f"attempt {attempt + 1}: {exc.failure_type} "
                    f"({exc.upstream_status_code or 'no HTTP response'}), "
                    f"wait {delay:g}s"
                    if delay is not None
                    else f"attempt {attempt + 1}: {exc.failure_type} "
                    f"({exc.upstream_status_code or 'no HTTP response'}), no retry"
                )
                if delay is None:
                    raise
                time.sleep(delay)
                waited_seconds += delay
        outcome = _body_outcome(result.count, len(result.results))
        search_attempts.append(
            BodySearchAttempt(
                kind="raw_locator",
                query=result.query,
                status=(
                    ValidationNodeStatus.FAILED
                    if outcome is RootBodySearchOutcome.FAILED
                    else ValidationNodeStatus.SUCCEEDED
                ),
                candidate_count=result.count,
                returned_count=len(result.results),
                continuation=result.next_cursor,
                used_for_candidates=outcome is not RootBodySearchOutcome.NOT_FOUND,
            )
        )
        alternate = (
            _docket_body_fallback_query(record, locator)
            if outcome is RootBodySearchOutcome.NOT_FOUND
            else None
        )
        if alternate is not None:
            try:
                fallback = client.search(alternate, search_type)  # type: ignore[arg-type]
                fallback_outcome = _body_outcome(fallback.count, len(fallback.results))
                use_fallback = fallback_outcome in {
                    RootBodySearchOutcome.FOUND,
                    RootBodySearchOutcome.EXCEEDS_REVIEW_LIMIT,
                }
                search_attempts.append(
                    BodySearchAttempt(
                        kind="parsed_docket",
                        query=fallback.query,
                        status=(
                            ValidationNodeStatus.FAILED
                            if fallback_outcome is RootBodySearchOutcome.FAILED
                            else ValidationNodeStatus.SUCCEEDED
                        ),
                        candidate_count=fallback.count,
                        returned_count=len(fallback.results),
                        continuation=fallback.next_cursor,
                        used_for_candidates=use_fallback,
                    )
                )
                if use_fallback:
                    result, outcome = fallback, fallback_outcome
            except Exception as exc:
                search_attempts.append(
                    BodySearchAttempt(
                        kind="parsed_docket",
                        query=alternate,
                        status=ValidationNodeStatus.FAILED,
                        candidate_count=None,
                        returned_count=0,
                        continuation=None,
                        used_for_candidates=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        raw_candidates = result.results
        selected_results = raw_candidates
        if result.count >= CANDIDATE_SELECTION_LIMIT:
            # A quoted locator is a general refinement of a broad body query.
            # The raw page remains available if the phrase has no bounded hits,
            # is incomplete, or fails. Only a complete bounded phrase page gets
            # priority within the existing body-fetch budget.
            phrase = _locator_for_body_search(record).strip()
            if phrase:
                phrase_query = '"' + phrase.replace('"', r"\"") + '"'
                try:
                    refined = client.search(phrase_query, search_type)  # type: ignore[arg-type]
                    refined_outcome = _body_outcome(refined.count, len(refined.results))
                    use_refinement = refined_outcome is RootBodySearchOutcome.FOUND
                    search_attempts.append(
                        BodySearchAttempt(
                            kind="quoted_locator",
                            query=refined.query,
                            status=(
                                ValidationNodeStatus.FAILED
                                if refined_outcome is RootBodySearchOutcome.FAILED
                                else ValidationNodeStatus.SUCCEEDED
                            ),
                            candidate_count=refined.count,
                            returned_count=len(refined.results),
                            continuation=refined.next_cursor,
                            used_for_candidates=use_refinement,
                            error=(
                                "Quoted search returned an incomplete bounded result."
                                if refined_outcome is RootBodySearchOutcome.FAILED
                                else None
                            ),
                        )
                    )
                    if use_refinement:
                        selected_results = _deduplicate_body_search_hits(
                            (*refined.results, *raw_candidates), source=source
                        )
                except Exception as exc:
                    search_attempts.append(
                        BodySearchAttempt(
                            kind="quoted_locator",
                            query=phrase_query,
                            status=ValidationNodeStatus.FAILED,
                            candidate_count=None,
                            returned_count=0,
                            continuation=None,
                            used_for_candidates=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
        candidates = tuple(_body_candidate(item, locator=locator) for item in selected_results)
        if retrospective_date is not None:
            candidates = tuple(
                eligible
                for candidate in candidates
                if (eligible := _retrospective_candidate(candidate, source, retrospective_date)) is not None
            )
        excluded_body_items = 0
        if retrospective_date is not None and source is RootBodySearchSource.COURTLISTENER_DOCKET:
            excluded_body_items = sum(
                len(item.get("recap_documents", ()))
                for item in selected_results
                if isinstance(item.get("recap_documents"), (list, tuple))
            ) - sum(len(item["recap_documents"]) for item in candidates)
        return _body_node(
            record,
            source=source,
            status=(
                ValidationNodeStatus.FAILED
                if outcome is RootBodySearchOutcome.FAILED
                else ValidationNodeStatus.SUCCEEDED
            ),
            outcome=outcome,
            locator=locator,
            query=result.query,
            candidate_count=result.count,
            candidates=candidates,
            search_attempts=tuple(search_attempts),
            retrospective_date=retrospective_date,
            excluded_by_date=len(selected_results) - len(candidates),
            excluded_body_items_by_date=excluded_body_items,
            continuation=result.next_cursor,
            status_message=(
                "CourtListener corpus body search completed."
                if not retry_diagnostics
                else f"CourtListener corpus body search completed after {len(retry_diagnostics) + 1} "
                f"attempts ({'; '.join(retry_diagnostics)})."
            ),
            outcome_message=_body_message(source, outcome, result.count),
            error=(
                f"Search reported {result.count} candidates but returned {len(result.results)} below the review limit."
                if outcome is RootBodySearchOutcome.FAILED
                else None
            ),
        )
    except Exception as exc:
        return _body_node(
            record,
            source=source,
            status=ValidationNodeStatus.FAILED,
            outcome=RootBodySearchOutcome.FAILED,
            locator=locator,
            query=locator,
            search_attempts=(
                *search_attempts,
                BodySearchAttempt(
                    kind="raw_locator",
                    query=locator,
                    status=ValidationNodeStatus.FAILED,
                    candidate_count=None,
                    returned_count=0,
                    continuation=None,
                    used_for_candidates=False,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            if not search_attempts
            else tuple(search_attempts),
            status_message="CourtListener corpus body search failed.",
            outcome_message="No CourtListener body candidates were available for this route.",
            error=(
                f"{type(exc).__name__}: {exc}"
                + (f"; retries: {'; '.join(retry_diagnostics)}" if retry_diagnostics else "")
            ),
        )


def _govinfo_body_node(
    record: CitationRecord,
    *,
    locator: str,
    client: GovInfoClient,
    retrospective_date: CalendarDate | None = None,
) -> RootBodySearchNode:
    source = RootBodySearchSource.GOVINFO_PACKAGE
    if not locator:
        return _body_node(
            record,
            source=source,
            status=ValidationNodeStatus.SKIPPED,
            outcome=RootBodySearchOutcome.UNAVAILABLE,
            locator=locator,
            query=None,
            status_message="Skipped GovInfo body search because this root has no locator text.",
            outcome_message="Body corroboration is unavailable without a source locator.",
        )
    query = govinfo_uscourts_body_query(locator)
    search_attempts: list[BodySearchAttempt] = []
    try:
        result = client.search_uscourts(query, page_size=CANDIDATE_SELECTION_LIMIT)
        outcome = _body_outcome(result.count, len(result.results))
        search_attempts.append(
            BodySearchAttempt(
                kind="raw_locator",
                query=result.query,
                status=(
                    ValidationNodeStatus.FAILED
                    if outcome is RootBodySearchOutcome.FAILED
                    else ValidationNodeStatus.SUCCEEDED
                ),
                candidate_count=result.count,
                returned_count=len(result.results),
                continuation=result.next_offset_mark,
                used_for_candidates=outcome is not RootBodySearchOutcome.NOT_FOUND,
            )
        )
        alternate = (
            _docket_body_fallback_query(record, locator)
            if outcome is RootBodySearchOutcome.NOT_FOUND
            else None
        )
        if alternate is not None:
            alternate_query = govinfo_uscourts_body_query(alternate)
            try:
                fallback = client.search_uscourts(alternate_query, page_size=CANDIDATE_SELECTION_LIMIT)
                fallback_outcome = _body_outcome(fallback.count, len(fallback.results))
                use_fallback = fallback_outcome in {
                    RootBodySearchOutcome.FOUND,
                    RootBodySearchOutcome.EXCEEDS_REVIEW_LIMIT,
                }
                search_attempts.append(
                    BodySearchAttempt(
                        kind="parsed_docket",
                        query=fallback.query,
                        status=(
                            ValidationNodeStatus.FAILED
                            if fallback_outcome is RootBodySearchOutcome.FAILED
                            else ValidationNodeStatus.SUCCEEDED
                        ),
                        candidate_count=fallback.count,
                        returned_count=len(fallback.results),
                        continuation=fallback.next_offset_mark,
                        used_for_candidates=use_fallback,
                    )
                )
                if use_fallback:
                    result, outcome = fallback, fallback_outcome
            except Exception as exc:
                search_attempts.append(
                    BodySearchAttempt(
                        kind="parsed_docket",
                        query=alternate_query,
                        status=ValidationNodeStatus.FAILED,
                        candidate_count=None,
                        returned_count=0,
                        continuation=None,
                        used_for_candidates=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        candidates = tuple(
            _body_candidate(govinfo_package_candidate(dict(item)), locator=locator) for item in result.results
        )
        if retrospective_date is not None:
            candidates = tuple(
                eligible
                for candidate in candidates
                if (eligible := _retrospective_candidate(candidate, source, retrospective_date)) is not None
            )
        return _body_node(
            record,
            source=source,
            status=(
                ValidationNodeStatus.FAILED
                if outcome is RootBodySearchOutcome.FAILED
                else ValidationNodeStatus.SUCCEEDED
            ),
            outcome=outcome,
            locator=locator,
            query=result.query,
            candidate_count=result.count,
            candidates=candidates,
            search_attempts=tuple(search_attempts),
            retrospective_date=retrospective_date,
            excluded_by_date=len(result.results) - len(candidates),
            continuation=result.next_offset_mark,
            status_message="GovInfo USCOURTS body search completed.",
            outcome_message=_body_message(source, outcome, result.count),
            error=(
                f"Search reported {result.count} packages but returned {len(result.results)} below the review limit."
                if outcome is RootBodySearchOutcome.FAILED
                else None
            ),
        )
    except Exception as exc:
        return _body_node(
            record,
            source=source,
            status=ValidationNodeStatus.FAILED,
            outcome=RootBodySearchOutcome.FAILED,
            locator=locator,
            query=query,
            search_attempts=tuple(search_attempts),
            status_message="GovInfo USCOURTS body search failed.",
            outcome_message="No GovInfo body candidates were available for this route.",
            error=f"{type(exc).__name__}: {exc}",
        )


def _body_outcome(count: int, returned_count: int) -> RootBodySearchOutcome:
    if count == 0:
        return RootBodySearchOutcome.NOT_FOUND
    if count >= CANDIDATE_SELECTION_LIMIT:
        return RootBodySearchOutcome.EXCEEDS_REVIEW_LIMIT
    # A successful bounded candidate review requires every candidate.  This
    # protects the earlier CourtListener 200/partial-page failure mode from
    # silently becoming a selection among incomplete results.
    if returned_count != count:
        return RootBodySearchOutcome.FAILED
    return RootBodySearchOutcome.FOUND


def _body_message(source: RootBodySearchSource, outcome: RootBodySearchOutcome, count: int) -> str:
    label = {
        RootBodySearchSource.COURTLISTENER_CLUSTER: "CourtListener opinion",
        RootBodySearchSource.COURTLISTENER_DOCKET: "CourtListener RECAP",
        RootBodySearchSource.GOVINFO_PACKAGE: "GovInfo USCOURTS",
    }[source]
    return {
        RootBodySearchOutcome.FOUND: f"{label} body search returned {count} bounded candidate(s).",
        RootBodySearchOutcome.NOT_FOUND: f"{label} body search returned no candidates.",
        RootBodySearchOutcome.EXCEEDS_REVIEW_LIMIT: (
            f"{label} body search returned {count} candidates, at or above the review limit."
        ),
        RootBodySearchOutcome.UNAVAILABLE: f"{label} body search is unavailable.",
        RootBodySearchOutcome.FAILED: f"{label} body search returned an incomplete or failed result.",
    }[outcome]


def _body_node(
    record: CitationRecord,
    *,
    source: RootBodySearchSource,
    status: ValidationNodeStatus,
    outcome: RootBodySearchOutcome,
    locator: str,
    query: str | None,
    candidate_count: int = 0,
    candidates: tuple[Mapping[str, object], ...] = (),
    continuation: str | None = None,
    status_message: str | None,
    outcome_message: str | None,
    error: str | None = None,
    retrospective_date: CalendarDate | None = None,
    excluded_by_date: int = 0,
    excluded_body_items_by_date: int = 0,
    search_attempts: tuple[BodySearchAttempt, ...] = (),
) -> RootBodySearchNode:
    return RootBodySearchNode(
        node_id=f"{record.citation_id}:root_body:{source.value}",
        status=status,
        outcome=outcome,
        source=source,
        root_locator=locator,
        query=query,
        candidate_count=candidate_count,
        candidates=candidates,
        continuation=continuation,
        depends_on=(),
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
        retrospective_date=retrospective_date.isoformat() if retrospective_date is not None else None,
        excluded_by_date=excluded_by_date,
        excluded_body_items_by_date=excluded_body_items_by_date,
        search_attempts=search_attempts,
    )


def _deduplicate_body_search_hits(
    results: tuple[Mapping[str, object], ...], *, source: RootBodySearchSource
) -> tuple[Mapping[str, object], ...]:
    """Keep phrase hits first, retaining every raw hit without a duplicate ID."""
    id_fields = (
        ("cluster_id", "clusterId", "cluster", "id")
        if source is RootBodySearchSource.COURTLISTENER_CLUSTER
        else ("docket_id", "docketId", "id")
    )
    seen: set[str] = set()
    kept: list[Mapping[str, object]] = []
    for item in results:
        identifier = _first_string(item, *id_fields)
        if identifier is not None:
            if identifier in seen:
                continue
            seen.add(identifier)
        kept.append(item)
    return tuple(kept)


def _body_candidate(result: Mapping[str, object], *, locator: str) -> dict[str, object]:
    """Preserve raw provider output and add only common field-check aliases."""
    return {
        **dict(result),
        "cluster_id": _first_string(result, "cluster_id", "clusterId", "cluster"),
        "docket_id": _first_string(result, "docket_id", "docketId"),
        "caseName": _first_string(result, "caseName", "case_name", "case_name_short"),
        "court_id": _first_string(result, "court_id", "court"),
        "docketNumber": _first_string(result, "docketNumber", "docket_number"),
        "dateFiled": _first_string(result, "dateFiled", "date_filed"),
        "decisionDate": _first_string(result, "decisionDate", "dateFiled", "date_filed"),
        "body_locator": locator,
    }


def _parse_evidence_date(value: object) -> CalendarDate | None:
    """Accept a provider's ISO calendar date, not a search/index timestamp."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(?:T.*)?", value.strip())
    if match is None:
        return None
    try:
        return CalendarDate.fromisoformat(match.group(1))
    except ValueError:
        return None


def _retrospective_candidate(
    candidate: Mapping[str, object],
    source: RootBodySearchSource,
    cutoff: CalendarDate,
) -> dict[str, object] | None:
    """Keep only body material provably available by the source draft date.

    A RECAP search hit is a mutable docket: its parent ``dateFiled`` does not
    date the child filing. The child ``entry_date_filed`` is required instead.
    Missing or malformed provenance dates are ineligible under a cutoff.
    """
    if source is RootBodySearchSource.COURTLISTENER_DOCKET:
        nested = candidate.get("recap_documents")
        if not isinstance(nested, (list, tuple)):
            return None
        eligible = tuple(
            item
            for item in nested
            if isinstance(item, Mapping)
            and (filed := _parse_evidence_date(item.get("entry_date_filed"))) is not None
            and filed <= cutoff
        )
        if not eligible:
            return None
        return {**candidate, "recap_documents": eligible, "caseName": None}
    if source is RootBodySearchSource.GOVINFO_PACKAGE:
        # One package can contain opinions with different issue dates. It is
        # filtered per granule when its text is fetched, never by package date.
        return dict(candidate)
    issued = _parse_evidence_date(candidate.get("decisionDate"))
    if issued is None or issued > cutoff:
        return None
    return dict(candidate)


def _stored_retrospective_date(
    nodes: tuple[RootBodySearchNode, ...], requested: CalendarDate | None
) -> CalendarDate | None:
    saved = {node.retrospective_date for node in nodes}
    if len(saved) > 1:
        raise ValueError("Body-search nodes disagree on the retrospective date")
    value = next(iter(saved), None)
    stored = CalendarDate.fromisoformat(value) if value is not None else None
    if requested is not None and stored is not None and requested != stored:
        raise ValueError("Resolution retrospective date differs from the saved body-search cutoff")
    return requested if requested is not None else stored


def _first_string(result: Mapping[str, object], *names: str) -> str | None:
    for name in names:
        value = result.get(name)
        if isinstance(value, str | int):
            return str(value)
    return None


def _resolve_body_candidates(
    record: CitationRecord,
    *,
    nodes: tuple[RootBodySearchNode, ...],
) -> CitationValidation:
    """Defer roots whose direct and independent body evidence did not resolve."""
    validation = CitationValidation(citation=record, nodes=nodes)
    total = sum(len(node.candidates) for node in nodes)
    if any(node.outcome is RootBodySearchOutcome.EXCEEDS_REVIEW_LIMIT for node in nodes):
        return validation.append(
            _open_web_resolution(
                validation,
                depends_on=tuple(node.node_id for node in nodes),
                reason="At least one corpus body search reached the candidate-review limit.",
            )
        )
    if total == 0:
        return validation.append(
            _open_web_resolution(
                validation,
                depends_on=tuple(node.node_id for node in nodes),
                reason="No provider body search returned a candidate for the source locator.",
            )
        )
    return validation.append(
        _open_web_resolution(
            validation,
            depends_on=tuple(node.node_id for node in nodes),
            reason=(
                f"The provider corpora returned {total} document(s), but no grounded independent citation "
                "was admitted from their text; continue with open-web retrieval."
            ),
        )
    )


def _open_web_resolution(
    validation: CitationValidation,
    *,
    depends_on: tuple[str, ...],
    reason: str,
) -> LocatorIdentityResolutionNode:
    return LocatorIdentityResolutionNode(
        node_id=f"{validation.citation_id}:root_body:identity_resolution",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorIdentityResolutionOutcome.DEFERRED_TO_OPEN_WEB_SEARCH,
        selected_candidate_index=None,
        selected_assessment_node_id=None,
        matching_candidate_indices=(),
        selection_evidence_node_id=None,
        depends_on=depends_on,
        status_message="Root identity deferred to open-web search.",
        outcome_message=reason,
    )


def _write_identity_progression(
    record: CitationRecord, progression: CitationValidation, *, stage: str
) -> None:
    projected = {node.node_id: _trace_node(node, stage=stage) for node in progression.nodes}
    for node in projected.values():
        observe_citation(record, node)
    resolution = progression.identity_resolution
    if resolution is None:
        msg = f"Body corroboration for {record.citation_id!r} ended without an identity decision"
        raise ValueError(msg)
    resolution_node = projected[resolution.node_id]
    judge_citation(
        record,
        resolution_node,
        Question.IDENTITY,
        resolution.outcome.value,
        message=resolution.outcome_message,
    )


def _saved_body_nodes(record: CitationRecord) -> tuple[RootBodySearchNode, ...]:
    restored: list[RootBodySearchNode] = []
    for source in RootBodySearchSource:
        matches = [
            node
            for node in record.trace
            if node.stage == ROOT_BODY_CORROBORATION_SEARCH_STAGE
            and node.node_id == f"{record.citation_id}:root_body:{source.value}"
            and node.details.get("validation_node_type") == RootBodySearchNode.__name__
        ]
        if len(matches) != 1:
            msg = f"Expected one {source.value} body node for {record.citation_id!r}, found {len(matches)}"
            raise ValueError(msg)
        raw = matches[0].details.get("validation")
        if not isinstance(raw, dict):
            msg = f"Saved body node for {record.citation_id!r} has no validation payload"
            raise ValueError(msg)
        restored_node = deserialize_validation_node({"node_type": RootBodySearchNode.__name__, **raw})
        if not isinstance(restored_node, RootBodySearchNode):
            msg = f"Saved body node decoded as {type(restored_node).__name__}"
            raise ValueError(msg)
        restored.append(restored_node)
    return tuple(restored)


def _body_searched_roots(document: Document) -> tuple[CitationRecord, ...]:
    """Include roots resolved after search when checking checkpoint provenance."""
    return tuple(
        record
        for record in document.citations
        if any(node.stage == ROOT_BODY_CORROBORATION_SEARCH_STAGE for node in record.trace)
    )


def _unresolved_roots(document: Document) -> tuple[CitationRecord, ...]:
    retrieval_deferrals = {
        LocatorIdentityResolutionOutcome.DEFERRED_TO_SEARCH.value,
        LocatorIdentityResolutionOutcome.DEFERRED_TO_SEMANTIC_REVIEW.value,
        LocatorIdentityResolutionOutcome.DEFERRED_TO_FUTURE_IMPLEMENTATION.value,
    }
    return tuple(
        record
        for record in document.active_citations
        if record.is_root
        and isinstance(record.fields, (DocketCitation, FullCaseCitation))
        and record.judgement(Question.IDENTITY).outcome in retrieval_deferrals
    )


def _lookup_question(record: CitationRecord) -> Question:
    return Question.DOCKET_LOOKUP if isinstance(record.fields, DocketCitation) else Question.LOCATOR_LOOKUP


def _lookup_outcome(outcome: RootBodySearchOutcome) -> str:
    return {
        RootBodySearchOutcome.FOUND: "body_search_found",
        RootBodySearchOutcome.NOT_FOUND: "body_search_not_found",
        RootBodySearchOutcome.EXCEEDS_REVIEW_LIMIT: "body_search_exceeds_candidate_limit",
        RootBodySearchOutcome.UNAVAILABLE: "body_search_unavailable",
        RootBodySearchOutcome.FAILED: "body_search_failed",
    }[outcome]


def _require_stage(document: Document, required_stage: str, stage_name: str) -> None:
    if required_stage not in document.passes:
        msg = f"{stage_name} requires {required_stage} before validation."
        raise ValueError(msg)


def _reject_partial_stage(document: Document, stage: str) -> None:
    if any(node.stage == stage for record in document.citations for node in record.trace):
        msg = f"Cannot resume partial {stage}; restart from the document before this stage."
        raise ValueError(msg)


def _trace_node(node: ValidationNode, *, stage: str) -> Node:
    payload = serialize_dataclass(node)
    return Node(
        node_id=node.node_id,
        reads=Reads.RECORD,
        stage=stage,
        made_by=_MADE_BY,
        outcome=str(payload["outcome"]),
        message=node.outcome_message,
        depends_on=node.depends_on,
        details={"validation_node_type": type(node).__name__, "validation": payload},
    )
