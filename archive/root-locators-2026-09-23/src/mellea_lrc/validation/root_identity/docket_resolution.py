"""One ordered query-and-resolution plan for docket-root identity.

A docket search is not a set of unrelated validation paths.  A direct docket
query is simply the first, narrowest query.  Each later query is evaluated by
the same evidence boundary and may settle the root before further provider
work is spent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import TYPE_CHECKING, Literal

from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.extraction.root_stages import ROOT_FORMATION_STAGE
from mellea_lrc.govinfo import GovInfoClient, govinfo_uscourts_case_name_query, govinfo_uscourts_docket_query
from mellea_lrc.model.citations import DocketCitation
from mellea_lrc.model.operations import (
    judge_citation,
    mark_extraction_reviewed,
    observe_citation,
)
from mellea_lrc.model.record import Node, Question, Reads
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.serialization.validated_document import deserialize_validation_node
from mellea_lrc.validation.aggregation.locator_found import run_locator_citation_summary
from mellea_lrc.validation.aggregation.locator_identity import run_locator_identity_resolution
from mellea_lrc.validation.aggregation.mellea_docket_metadata_choice import (
    run_mellea_docket_metadata_choice,
)
from mellea_lrc.validation.candidates.evaluation import (
    run_docket_search_candidate_evaluation,
    run_govinfo_docket_search_candidate_evaluation,
)
from mellea_lrc.validation.field_checks.court_check import run_court_check
from mellea_lrc.validation.field_checks.docket_number_check import run_docket_number_check
from mellea_lrc.validation.field_checks.exact_case_name_check import run_exact_case_name_check
from mellea_lrc.validation.field_checks.mellea_docket_citation_reextraction import (
    run_mellea_docket_citation_reextraction,
)
from mellea_lrc.validation.field_checks.year_check import run_year_check
from mellea_lrc.validation.root_identity.docket import (
    MAX_DOCKET_CANDIDATE_REVIEW,
    _docket_candidate_assessment,
    _write_identity_progression,
    build_docket_metadata_shortlist,
)
from mellea_lrc.validation.root_identity.docket_source_review import apply_docket_source_review
from mellea_lrc.validation.search.common import (
    MetadataQuery,
    courtlistener_case_name_query,
    prepare_case_name_terms,
    run_courtlistener_metadata_attempt,
    run_govinfo_metadata_attempt,
)
from mellea_lrc.validation.types import (
    CitationValidation,
    DocketMetadataShortlistNode,
    DocketMetadataShortlistOutcome,
    DocketRootSearchNode,
    DocketRootSearchOutcome,
    LocatorCandidateAssessmentOutcome,
    LocatorIdentityResolutionNode,
    LocatorIdentityResolutionOutcome,
    MelleaDocketCitationReextractionNode,
    MelleaLocatorCandidateChoiceNode,
    MelleaLocatorCandidateChoiceOutcome,
    ValidationNode,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.model.document import Document
    from mellea_lrc.model.record import CitationRecord


DOCKET_ROOT_IDENTITY_STAGE = "docket_root_identity"
_MADE_BY = "mellea_lrc.validation.root_identity.docket_resolution"
_Provider = Literal["courtlistener", "govinfo"]


@dataclass(frozen=True, slots=True)
class _DocketQuery:
    """One provider request in the source-grounded docket query plan."""

    provider: _Provider
    kind: str
    query: str
    court_id: str | None
    case_name_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _QueryReview:
    """Programmatic evidence from one provider query, held for final review."""

    validation: CitationValidation
    shortlist: DocketMetadataShortlistNode


@dataclass(frozen=True, slots=True)
class _SourceReview:
    """The re-read node and whether a direct query must be repeated."""

    node: MelleaDocketCitationReextractionNode
    requires_direct_recheck: bool


async def resolve_docket_root_identities(
    document: Document,
    *,
    retrospective_date: date | None = None,
    client: CourtListenerServiceClient | None = None,
    govinfo_client: GovInfoClient | None = None,
    session: MelleaSession | None = None,
) -> Document:
    """Resolve docket roots through one query-and-resolution drill.

    The order is deliberately fixed:

    1. direct provider retrieval and programmatic comparison;
    2. one source-grounded combined citation re-extraction, then direct
       retrieval again if it corrected a source field used by the query;
    3. source-grounded case-name discovery queries and programmatic comparison;
    4. bounded semantic representative review of only the candidates that
       remain plausible after every programmatic pass.

    Thus a model cannot choose a record before it has been given one chance to
    correct a missed case name, court, or docket spelling in the filing.  The
    query family is shared across direct and discovery retrieval; only the
    query text differs.
    """
    if ROOT_FORMATION_STAGE not in document.passes:
        msg = "Docket-root identity requires root_formation before retrieval."
        raise ValueError(msg)
    if retrospective_date is not None and not isinstance(retrospective_date, date):
        msg = "retrospective_date must be a datetime.date or None"
        raise TypeError(msg)
    if DOCKET_ROOT_IDENTITY_STAGE in document.passes:
        if retrospective_date is not None:
            _require_matching_saved_cutoff(document, retrospective_date)
        return document
    if any(
        node.stage == DOCKET_ROOT_IDENTITY_STAGE for record in document.citations for node in record.trace
    ):
        msg = "Cannot resume partial docket_root_identity; restart from the preceding Document."
        raise ValueError(msg)

    courtlistener = client if client is not None else CourtListenerClient()
    govinfo = govinfo_client if govinfo_client is not None else GovInfoClient()
    for record in _docket_roots(document):
        if record.judgement(Question.IDENTITY).outcome == LocatorIdentityResolutionOutcome.RESOLVED.value:
            continue
        reviews: list[_QueryReview] = []
        last_node_id: str | None = None
        next_query_ordinal = 1

        # The direct docket lookup is the narrowest query. Court is not a
        # retrieval filter: a conflicting court is still evidence to judge.
        direct_queries = _direct_query_plan(record)
        resolved, last_node_id = _run_programmatic_queries(
            record,
            queries=direct_queries,
            ordinal_start=next_query_ordinal,
            courtlistener=courtlistener,
            govinfo=govinfo,
            reviews=reviews,
            retrospective_date=retrospective_date,
        )
        next_query_ordinal += len(direct_queries)
        if resolved:
            continue

        # A completed source re-read is recorded as first-class document
        # evidence and flips ``extraction_reviewed_by_llm`` so this loop cannot
        # silently repeat.  A corrected docket or court changes the query
        # state, or a newly recovered source case name changes the comparison
        # evidence, so direct retrieval and comparison are then repeated once.
        source_review = await _review_unresolved_docket_citation(
            record,
            document=document,
            trigger_node_id=last_node_id,
            session=session,
        )
        if source_review is not None:
            last_node_id = source_review.node.node_id
            if source_review.requires_direct_recheck:
                resolved, last_node_id = _run_programmatic_queries(
                    record,
                    queries=_direct_query_plan(record),
                    ordinal_start=next_query_ordinal,
                    courtlistener=courtlistener,
                    govinfo=govinfo,
                    reviews=reviews,
                    retrospective_date=retrospective_date,
                )
                next_query_ordinal += len(direct_queries)
                if resolved:
                    continue

        terms = await prepare_case_name_terms(
            record,
            stage=DOCKET_ROOT_IDENTITY_STAGE,
            made_by=_MADE_BY,
            source_case_name=_source_case_name(record),
            session=session,
        )
        observe_citation(record, terms.node)
        last_node_id = terms.node.node_id
        resolved, last_node_id = _run_programmatic_queries(
            record,
            queries=_case_name_query_plan(record, terms.terms),
            ordinal_start=next_query_ordinal,
            courtlistener=courtlistener,
            govinfo=govinfo,
            reviews=reviews,
            retrospective_date=retrospective_date,
        )
        if resolved:
            continue

        semantic_resolution = await _resolve_remaining_semantic_candidates(
            record, reviews=reviews, session=session
        )
        if semantic_resolution:
            continue
        _defer_to_body_corroboration(record, depends_on=last_node_id)
    return document.evolve(passes=(*document.passes, DOCKET_ROOT_IDENTITY_STAGE))


def _require_matching_saved_cutoff(document: Document, retrospective_date: date) -> None:
    """Do not reuse an identity stage checked against another evidence horizon."""
    requested = retrospective_date.isoformat()
    for record in _docket_roots(document):
        searches = tuple(
            node
            for node in record.trace
            if node.stage == DOCKET_ROOT_IDENTITY_STAGE
            and node.details.get("validation_node_type") == DocketRootSearchNode.__name__
        )
        if not searches:
            msg = (
                f"Completed docket identity for {record.citation_id!r} has no saved search cutoff; "
                f"cannot reuse it for {requested}."
            )
            raise ValueError(msg)
        for node in searches:
            saved = node.details.get("validation")
            saved_cutoff = saved.get("retrospective_date") if isinstance(saved, Mapping) else None
            if saved_cutoff != requested:
                msg = (
                    f"Saved docket search {node.node_id!r} used cutoff {saved_cutoff!r}; "
                    f"cannot reuse it for {requested}."
                )
                raise ValueError(msg)


def _direct_query_plan(record: CitationRecord) -> tuple[_DocketQuery, ...]:
    """Return the shared first query: the stated docket at each provider."""
    citation = _citation(record)
    if citation.docket_number is None:
        return ()
    return (
        _DocketQuery("courtlistener", "direct_docket", citation.docket_number, None),
        _DocketQuery(
            "govinfo",
            "direct_docket",
            govinfo_uscourts_docket_query(citation.docket_number, court_id=None),
            None,
        ),
    )


def _case_name_query_plan(record: CitationRecord, terms: tuple[str, ...]) -> tuple[_DocketQuery, ...]:
    """Discover candidates with short source-grounded caption fragments.

    The independent one-term queries are the recall-oriented path: a database
    caption can legitimately omit another party or title word.  When the
    source reader supplied two terms, one bounded pair query complements them.
    It is not a reconstructed full-caption query; it gives the semantic stage
    a compact candidate set when a distinctive pair is needed to avoid an
    unbounded one-term result page.
    """
    citation = _citation(record)
    queries: list[_DocketQuery] = []
    for term in terms:
        if citation.court is not None:
            queries.extend(
                (
                    _DocketQuery(
                        "courtlistener",
                        "case_name_fragment_with_stated_court",
                        courtlistener_case_name_query((term,), citation.court),
                        citation.court,
                        (term,),
                    ),
                    _DocketQuery(
                        "govinfo",
                        "case_name_fragment_with_stated_court",
                        govinfo_uscourts_case_name_query((term,), court_id=citation.court),
                        citation.court,
                        (term,),
                    ),
                )
            )
        queries.extend(
            (
                _DocketQuery(
                    "courtlistener",
                    "case_name_fragment",
                    courtlistener_case_name_query((term,), None),
                    None,
                    (term,),
                ),
                _DocketQuery(
                    "govinfo",
                    "case_name_fragment",
                    govinfo_uscourts_case_name_query((term,), court_id=None),
                    None,
                    (term,),
                ),
            )
        )
    # Limit the complement to the first two source-grounded retrieval cues.
    # This is intentionally a bounded pair, never an AND over every word in
    # the extracted case name, so normal caption variation still has the
    # independent-fragment queries above.
    pair = terms[:2]
    if len(pair) == 2:
        if citation.court is not None:
            queries.extend(
                (
                    _DocketQuery(
                        "courtlistener",
                        "case_name_pair_with_stated_court",
                        courtlistener_case_name_query(pair, citation.court),
                        citation.court,
                        pair,
                    ),
                    _DocketQuery(
                        "govinfo",
                        "case_name_pair_with_stated_court",
                        govinfo_uscourts_case_name_query(pair, court_id=citation.court),
                        citation.court,
                        pair,
                    ),
                )
            )
        queries.extend(
            (
                _DocketQuery(
                    "courtlistener",
                    "case_name_pair",
                    courtlistener_case_name_query(pair, None),
                    None,
                    pair,
                ),
                _DocketQuery(
                    "govinfo",
                    "case_name_pair",
                    govinfo_uscourts_case_name_query(pair, court_id=None),
                    None,
                    pair,
                ),
            )
        )
    return tuple(_deduplicate_queries(queries))


def _query_plan(record: CitationRecord, terms: tuple[str, ...]) -> tuple[_DocketQuery, ...]:
    """Compatibility helper exposing the complete query family for inspection."""
    return _direct_query_plan(record) + _case_name_query_plan(record, terms)


def _run_query(
    record: CitationRecord,
    *,
    query: _DocketQuery,
    ordinal: int,
    courtlistener: CourtListenerServiceClient,
    govinfo: GovInfoClient,
    retrospective_date: date | None,
) -> DocketRootSearchNode:
    """Run one provider query and preserve its complete response boundary."""
    spec = MetadataQuery(kind=query.kind, query=query.query, court_id=query.court_id)
    attempt = (
        run_courtlistener_metadata_attempt(spec, courtlistener)
        if query.provider == "courtlistener"
        else run_govinfo_metadata_attempt(spec, govinfo)
    )
    raw_count = attempt.candidate_count or 0
    raw_returned_count = len(attempt.candidates)
    candidates = (
        attempt.candidates
        if retrospective_date is None
        else tuple(
            candidate
            for candidate in attempt.candidates
            if _candidate_available_by(candidate, query.provider, retrospective_date)
        )
    )
    excluded_count = raw_returned_count - len(candidates)
    # The raw provider count still determines whether a response was bounded.
    # Filtering a partial page cannot certify that unseen results are old enough.
    outcome = (
        DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT
        if attempt.status is ValidationNodeStatus.SUCCEEDED and raw_count >= MAX_DOCKET_CANDIDATE_REVIEW
        else _search_outcome(attempt.status, len(candidates))
    )
    attempt = replace(attempt, candidates=candidates)
    return DocketRootSearchNode(
        node_id=f"{record.citation_id}:{DOCKET_ROOT_IDENTITY_STAGE}:query:{ordinal}:{query.provider}",
        status=attempt.status,
        outcome=outcome,
        docket_number=_citation(record).docket_number,
        query=query.query,
        candidate_count=len(candidates),
        candidates=candidates,
        next_cursor=attempt.continuation,
        depends_on=(),
        status_message=f"{query.provider.title()} {query.kind} docket query completed.",
        outcome_message=(
            _query_message(query, outcome, raw_count)
            if retrospective_date is None
            else (
                f"{_query_message(query, outcome, raw_count)} "
                f"Retrospective cutoff {retrospective_date.isoformat()} retained {len(candidates)} "
                f"and excluded {excluded_count} returned candidate(s)."
            )
        ),
        error=attempt.error,
        attempts=(attempt,),
        retrospective_date=retrospective_date.isoformat() if retrospective_date else None,
        raw_candidate_count=attempt.candidate_count,
        excluded_candidate_count=excluded_count,
    )


def _candidate_available_by(candidate: Mapping[str, object], provider: _Provider, cutoff: date) -> bool:
    """Reject evidence whose source date cannot establish existence by cutoff.

    CourtListener's docket ``dateFiled`` establishes the docket's inception,
    not a historical snapshot of its mutable caption.  The latter limitation
    must remain explicit when interpreting retrospective identity results.
    GovInfo package dates cannot establish the dates of every opinion in a
    multi-opinion package, so package candidates are withheld until we can
    inspect dated granules individually.
    """
    if provider == "govinfo":
        return False
    filed = candidate.get("dateFiled", candidate.get("date_filed"))
    if not isinstance(filed, str):
        return False
    try:
        return date.fromisoformat(filed) <= cutoff
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class _QueryResolution:
    terminal: bool
    last_node_id: str | None
    review: _QueryReview | None = None


def _run_programmatic_queries(
    record: CitationRecord,
    *,
    queries: tuple[_DocketQuery, ...],
    ordinal_start: int,
    courtlistener: CourtListenerServiceClient,
    govinfo: GovInfoClient,
    reviews: list[_QueryReview],
    retrospective_date: date | None,
) -> tuple[bool, str | None]:
    """Run a query family and retain unresolved evidence for one later choice."""
    last_node_id: str | None = None
    for offset, query in enumerate(queries):
        ordinal = ordinal_start + offset
        search = _run_query(
            record,
            query=query,
            ordinal=ordinal,
            courtlistener=courtlistener,
            govinfo=govinfo,
            retrospective_date=retrospective_date,
        )
        trace = _trace(search)
        observe_citation(record, trace)
        judge_citation(
            record,
            trace,
            Question.DOCKET_LOOKUP,
            _lookup_outcome(search.outcome),
            message=search.outcome_message,
        )
        last_node_id = search.node_id
        result = _resolve_query_programmatically(record, search=search, provider=query.provider)
        if result.review is not None:
            reviews.append(result.review)
        if result.terminal:
            return True, result.last_node_id
        if result.last_node_id is not None:
            last_node_id = result.last_node_id
    return False, last_node_id


def _resolve_query_programmatically(
    record: CitationRecord,
    *,
    search: DocketRootSearchNode,
    provider: _Provider,
) -> _QueryResolution:
    """Apply the deterministic half of the shared identity drill.

    Semantic selection has a later, single boundary.  Keeping it out of this
    helper means a candidate discovered before source re-extraction cannot be
    selected merely because it resembles an incomplete initial parse.
    """
    if search.outcome in {DocketRootSearchOutcome.NOT_FOUND, DocketRootSearchOutcome.FAILED}:
        return _QueryResolution(False, search.node_id)

    shortlist = build_docket_metadata_shortlist(
        record,
        candidates=search.candidates,
        candidate_count=search.candidate_count,
        # A successful GovInfo response can retain an opaque offset marker even
        # after it returned every result in its reported count.  Completeness
        # is therefore defined by the persisted count and payload, as it is in
        # the common provider adapter; a marker alone must not make bounded
        # evidence appear partial.
        complete_result_set=(
            search.outcome is not DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT
            and search.candidate_count == len(search.candidates)
        ),
        search_node_id=search.node_id,
        depends_on=(search.node_id,),
        node_id=f"{search.node_id}:shortlist",
    )
    observe_citation(record, _trace(shortlist))
    if shortlist.outcome is DocketMetadataShortlistOutcome.NO_CANDIDATES:
        return _QueryResolution(False, shortlist.node_id)
    if len(shortlist.candidates) >= MAX_DOCKET_CANDIDATE_REVIEW:
        return _QueryResolution(False, shortlist.node_id)

    validation = CitationValidation(citation=record, nodes=(search, shortlist))
    for shortlisted in shortlist.candidates:
        index = shortlisted.candidate_index
        candidate = (
            run_docket_search_candidate_evaluation(
                validation,
                result=search.candidates[index - 1],
                candidate_index=index,
                depends_on=(search.node_id, shortlist.node_id),
                node_prefix=search.node_id,
            )
            if provider == "courtlistener"
            else run_govinfo_docket_search_candidate_evaluation(
                validation,
                result=search.candidates[index - 1],
                candidate_index=index,
                depends_on=(search.node_id, shortlist.node_id),
                node_prefix=search.node_id,
            )
        )
        validation = validation.append(candidate)
        docket = run_docket_number_check(validation, candidate=candidate)
        case_name = run_exact_case_name_check(
            validation,
            candidate=candidate,
            extracted_case_name=_source_case_name(record),
        )
        year = run_year_check(validation, candidate=candidate)
        court = run_court_check(validation, evidence=candidate)
        validation = validation.append(docket).append(case_name).append(year).append(court)
        validation = validation.append(
            _docket_candidate_assessment(
                validation,
                candidate=candidate,
                docket=docket,
                case_name=case_name,
                year_outcome=year.outcome,
                court_outcome=court.outcome,
            )
        )
    summary = run_locator_citation_summary(validation)
    validation = validation.append(summary)
    matches = tuple(
        candidate
        for candidate in summary.candidates
        if candidate.outcome is LocatorCandidateAssessmentOutcome.MATCH
    )
    if len(matches) == 1:
        validation = validation.append(run_locator_identity_resolution(validation, summary=summary))
        _write_identity_progression(record, validation, stage=DOCKET_ROOT_IDENTITY_STAGE)
        return _QueryResolution(
            True, validation.identity_resolution.node_id if validation.identity_resolution else None
        )

    _record_validation(record, validation)
    return _QueryResolution(False, summary.node_id, _QueryReview(validation=validation, shortlist=shortlist))


async def _review_unresolved_docket_citation(
    record: CitationRecord,
    *,
    document: Document,
    trigger_node_id: str | None,
    session: MelleaSession | None,
) -> _SourceReview | None:
    """Re-read one unresolved citation once and write grounded source fields."""
    if record.extraction_reviewed_by_llm:
        saved = _saved_reextraction(record)
        return _SourceReview(node=saved, requires_direct_recheck=False) if saved is not None else None
    source_case_name_before = _source_case_name(record)
    review = await run_mellea_docket_citation_reextraction(
        record,
        document=document,
        trigger_node_id=trigger_node_id or f"{record.citation_id}:{ROOT_FORMATION_STAGE}",
        session=session,
    )
    trace = _trace(review, reads=Reads.DOCUMENT)
    if review.status is ValidationNodeStatus.SUCCEEDED:
        mark_extraction_reviewed(record, trace)

    changes = apply_docket_source_review(record, document, review, trace)
    judge_citation(
        record,
        trace,
        Question.EXTRACTION_REVIEW,
        review.outcome.value,
        message=review.outcome_message or review.reason,
    )
    # A direct query has to be reassessed if it ran before the source reader
    # recovered a missing case name.  The query string may be identical, but
    # its candidate checks now have material source evidence that was absent
    # before.  Docket and court corrections also change the query state.
    source_case_name_after = _source_case_name(record)
    return _SourceReview(
        node=review,
        requires_direct_recheck=changes or source_case_name_before != source_case_name_after,
    )


async def _resolve_remaining_semantic_candidates(
    record: CitationRecord,
    *,
    reviews: list[_QueryReview],
    session: MelleaSession | None,
) -> bool:
    """Ask for a representative only after all programmatic retrieval ends.

    Every 40%-similar docket candidate reaches model review, including those
    with independent court or date contradictions. Case-name and docket
    spelling equivalence remain semantic questions. Court and chronology
    findings stay in their own field nodes and affect only the final verdict.
    """
    for review in reviews:
        summary = review.validation.aggregation
        if summary is None or not hasattr(summary, "candidates"):
            msg = "Programmatic docket review must end in a locator-candidate summary"
            raise ValueError(msg)
        eligible = tuple(candidate.candidate_index for candidate in summary.candidates)
        if not eligible:
            continue
        narrowed = replace(
            review.shortlist,
            candidates=tuple(
                candidate
                for candidate in review.shortlist.candidates
                if candidate.candidate_index in eligible
            ),
        )
        if not narrowed.candidates:
            continue
        choice = await run_mellea_docket_metadata_choice(
            review.validation,
            summary=summary,
            shortlist=narrowed,
            stated_case_name=_source_case_name(record),
            session=session,
        )
        progression = review.validation.append(choice)
        if choice.outcome is not MelleaLocatorCandidateChoiceOutcome.SELECTED:
            _record_validation(record, progression)
            continue
        has_stated_name = _source_case_name(record) is not None
        if choice.docket_equivalent is not True or (
            has_stated_name and choice.case_name_equivalent is not True
        ):
            _record_validation(record, progression)
            continue
        progression = progression.append(
            run_locator_identity_resolution(progression, summary=summary, choice=choice)
        )
        _write_identity_progression(record, progression, stage=DOCKET_ROOT_IDENTITY_STAGE)
        return progression.identity_resolution is not None
    return False


def _record_validation(record: CitationRecord, validation: CitationValidation) -> None:
    """Persist a nonterminal query review without pretending it settled identity."""
    for node in validation.nodes:
        observe_citation(
            record,
            _trace(
                node,
                reads=Reads.DOCUMENT if isinstance(node, MelleaLocatorCandidateChoiceNode) else Reads.RECORD,
            ),
        )


def _defer_to_body_corroboration(record: CitationRecord, *, depends_on: str | None) -> None:
    resolution = LocatorIdentityResolutionNode(
        node_id=f"{record.citation_id}:{DOCKET_ROOT_IDENTITY_STAGE}:deferred",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorIdentityResolutionOutcome.DEFERRED_TO_FUTURE_IMPLEMENTATION,
        selected_candidate_index=None,
        selected_assessment_node_id=None,
        matching_candidate_indices=(),
        selection_evidence_node_id=None,
        depends_on=(depends_on,) if depends_on is not None else (),
        status_message="Docket-root query plan completed without an admitted identity.",
        outcome_message="Every bounded docket query was exhausted; body corroboration remains available.",
    )
    trace = _trace(resolution)
    judge_citation(
        record, trace, Question.IDENTITY, resolution.outcome.value, message=resolution.outcome_message
    )


def _source_case_name(record: CitationRecord) -> str | None:
    citation = _citation(record)
    if citation.case_name is not None and citation.case_name.text.strip():
        return citation.case_name.text
    parties = tuple(
        value.strip() for value in (citation.plaintiff, citation.defendant) if value and value.strip()
    )
    if len(parties) == 2:
        return f"{parties[0]} v. {parties[1]}"
    return parties[0] if parties else None


def _saved_reextraction(record: CitationRecord) -> MelleaDocketCitationReextractionNode | None:
    """Load the one source re-read stored in this record's trace."""
    matches = [
        node
        for node in record.trace
        if node.details.get("validation_node_type") == MelleaDocketCitationReextractionNode.__name__
    ]
    if not matches:
        return None
    if len(matches) != 1:
        msg = f"Expected at most one docket-citation re-extraction for {record.citation_id!r}"
        raise ValueError(msg)
    raw = matches[0].details.get("validation")
    if not isinstance(raw, dict):
        msg = f"Saved docket-citation re-extraction for {record.citation_id!r} has no validation payload"
        raise ValueError(msg)
    decoded = deserialize_validation_node({"node_type": MelleaDocketCitationReextractionNode.__name__, **raw})
    if not isinstance(decoded, MelleaDocketCitationReextractionNode):
        msg = f"Unexpected saved re-extraction type: {type(decoded).__name__}"
        raise TypeError(msg)
    return decoded


def _citation(record: CitationRecord) -> DocketCitation:
    if not isinstance(record.fields, DocketCitation):
        msg = f"Expected a docket root, got {type(record.fields).__name__}"
        raise TypeError(msg)
    return record.fields


def _docket_roots(document: Document) -> tuple[CitationRecord, ...]:
    return tuple(
        record
        for record in document.active_citations
        if record.is_root and isinstance(record.fields, DocketCitation)
    )


def _search_outcome(status: ValidationNodeStatus, count: int) -> DocketRootSearchOutcome:
    if status is ValidationNodeStatus.FAILED:
        return DocketRootSearchOutcome.FAILED
    if count == 0:
        return DocketRootSearchOutcome.NOT_FOUND
    if count == 1:
        return DocketRootSearchOutcome.FOUND
    if count < MAX_DOCKET_CANDIDATE_REVIEW:
        return DocketRootSearchOutcome.AMBIGUOUS
    return DocketRootSearchOutcome.EXCEEDS_REVIEW_LIMIT


def _lookup_outcome(outcome: DocketRootSearchOutcome) -> str:
    return f"docket_query_{outcome.value}"


def _query_message(query: _DocketQuery, outcome: DocketRootSearchOutcome, count: int) -> str:
    return (
        f"{query.provider.title()} {query.kind} query {outcome.value}; "
        f"provider reported {count} candidate(s)."
    )


def _deduplicate_queries(queries: list[_DocketQuery]) -> tuple[_DocketQuery, ...]:
    seen: set[tuple[str, str]] = set()
    return tuple(
        query
        for query in queries
        if not ((query.provider, query.query) in seen or seen.add((query.provider, query.query)))
    )


def _trace(node: ValidationNode, *, reads: Reads = Reads.RECORD) -> Node:
    payload = serialize_dataclass(node)
    message = next(
        (
            value
            for key in ("outcome_message", "status_message", "error")
            if isinstance(value := payload.get(key), str)
        ),
        None,
    )
    return Node(
        node_id=node.node_id,
        reads=reads,
        stage=DOCKET_ROOT_IDENTITY_STAGE,
        made_by=_MADE_BY,
        outcome=str(payload["outcome"]),
        message=message,
        depends_on=node.depends_on,
        details={"validation_node_type": type(node).__name__, "validation": payload},
    )
