"""Opinion retrieval counts admitted reporter roots, with no annotation dependency."""

from __future__ import annotations

import pytest

from evaluations import validate_pincite as evaluation
from mellea_lrc.model import Document, FullReporterCitation, Span
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactAmbiguityOutcome,
    ReporterExactAmbiguityResolution,
    ReporterExactLookup,
    ReporterExactLookupOutcome,
    ReporterExactLookupQuery,
)
from mellea_lrc.model.citations.reporter_opinion import (
    OpinionRetrievalOutcome,
    ReporterRootOpinionRetrieval,
    ReporterRootOpinionSource,
    RetrievedReporterOpinion,
)
from mellea_lrc.providers.courtlistener import CourtListenerCitationLookup
from mellea_lrc.validation.reporter_root_opinion_retrieval import STAGE

LOCATORS = ("550 U.S. 544", "347 U.S. 483", "410 U.S. 113", "505 U.S. 833")
LOOKUP_STAGE = "12.1_reporter_root_lookup_cluster_retrieval"
AMBIGUITY_STAGE = "13.2_reporter_root_lookup_ambiguous_rule_judgment"


def _with_opinions(citation, *, cluster_id, bundle, candidate_index=0, outcomes=None):
    citation = citation.record(STAGE)
    citation = citation.with_reporter_root_opinion_source(
        ReporterRootOpinionSource(
            node_id=citation.nodes[-1].id,
            cluster=citation.reporter_exact_lookup.response.clusters[candidate_index],
        )
    )
    opinions = []
    for identifier in bundle:
        outcome = (outcomes or {}).get(identifier, OpinionRetrievalOutcome.RETRIEVED)
        response = (
            None
            if outcome is OpinionRetrievalOutcome.NOT_FOUND
            else {
                "id": identifier,
                "cluster": cluster_id,
                "plain_text": "Opinion text" if outcome is OpinionRetrievalOutcome.RETRIEVED else "",
            }
        )
        opinions.append(
            RetrievedReporterOpinion(
                opinion_id=identifier, cluster_id=cluster_id, outcome=outcome, response=response
            )
        )
    return citation.with_reporter_root_opinion_retrieval(
        ReporterRootOpinionRetrieval(
            node_id=citation.nodes[-1].id,
            cluster_id=cluster_id,
            sub_opinion_ids=bundle,
            opinions=tuple(opinions),
        )
    )


def _document(*bundles: tuple[str, ...] | None, outcomes=None, fetch=True) -> Document:
    """None is an unadmitted root without an original cluster."""
    document = Document.from_source("; ".join(LOCATORS[: len(bundles)]))
    for index in range(len(bundles)):
        quote = LOCATORS[index]
        start = document.text.index(quote)
        document = document.add_citation(
            FullReporterCitation.from_locator(
                citation_id=f"citation-{index}",
                stage="1_full_reporter_locators",
                source=document.text,
                span=Span(start=start, end=start + len(quote)),
            )
        )
    document = document.complete("1_full_reporter_locators")
    for citation, bundle in zip(document.citations, bundles, strict=True):
        citation = citation.record("10_roots").with_root(citation.id)
        if bundle is not None:
            citation = citation.with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
        document = document.replace_citation(citation)
    document = document.complete("10_roots")
    for index, (citation, bundle) in enumerate(zip(document.roots, bundles, strict=True)):
        if bundle is None:
            continue
        citation = citation.record(LOOKUP_STAGE)
        volume, _, page = LOCATORS[index].split()
        citation = citation.with_reporter_exact_lookup(
            ReporterExactLookup(
                node_id=citation.nodes[-1].id,
                outcome=ReporterExactLookupOutcome.UNIQUE,
                query=ReporterExactLookupQuery(volume=int(volume), edition="U.S.", page=page),
                response=CourtListenerCitationLookup.model_validate(
                    {
                        "citation": LOCATORS[index],
                        "status": 200,
                        "clusters": [{"id": index + 1, "sub_opinions": list(bundle)}],
                    }
                ),
            )
        )
        document = document.replace_citation(citation)
    document = document.complete(LOOKUP_STAGE)
    if fetch:
        for index, (citation, bundle) in enumerate(zip(document.roots, bundles, strict=True)):
            if bundle is not None:
                citation = _with_opinions(
                    citation, cluster_id=str(index + 1), bundle=bundle, outcomes=outcomes
                )
                document = document.replace_citation(citation)
    return document.complete(STAGE)


def _ambiguous_document(selected_index: int | None) -> Document:
    document = _document(("20",)).get_stage("10_roots")
    citation = document.roots[0].record(LOOKUP_STAGE)
    citation = citation.with_reporter_exact_lookup(
        ReporterExactLookup(
            node_id=citation.nodes[-1].id,
            outcome=ReporterExactLookupOutcome.AMBIGUOUS,
            query=ReporterExactLookupQuery(volume=550, edition="U.S.", page="544"),
            response=CourtListenerCitationLookup.model_validate(
                {
                    "citation": LOCATORS[0],
                    "status": 200,
                    "clusters": [{"id": 1, "sub_opinions": [20]}, {"id": 2, "sub_opinions": [21]}],
                }
            ),
        )
    )
    document = document.replace_citation(citation).complete(LOOKUP_STAGE)
    citation = citation.record(AMBIGUITY_STAGE)
    citation = citation.with_reporter_exact_ambiguity_resolution(
        ReporterExactAmbiguityResolution(
            node_id=citation.nodes[-1].id,
            outcome=(
                ReporterExactAmbiguityOutcome.NO_UNIQUE_RULE_MATCH
                if selected_index is None
                else ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH
            ),
            passing_candidate_indices=() if selected_index is None else (selected_index,),
            selected_candidate_index=selected_index,
        )
    )
    document = document.replace_citation(citation).complete(AMBIGUITY_STAGE)
    if selected_index is not None:
        citation = _with_opinions(
            citation,
            cluster_id=str(selected_index + 1),
            bundle=(str(20 + selected_index),),
            candidate_index=selected_index,
        )
        document = document.replace_citation(citation)
    return document.complete(STAGE)


def test_unique_original_clusters_use_one_root_each_without_gold_or_opinion_object_denominators():
    document = _document(("20", "21", "22"), ("30",), None)

    score = evaluation.score_reporter_root_opinion_retrieval(document)

    assert document.source_path is None
    assert score.reporter_roots_opinion_retrievals == score.reporter_roots_correct_identity == 2
    assert score.as_dict() == {
        "stage": STAGE,
        "reporter_roots_opinion_retrievals": 2,
        "reporter_roots_correct_identity": 2,
        "ratio": 1.0,
    }


@pytest.mark.parametrize("selected_index", [0, 1])
def test_selected_ambiguous_original_cluster_is_eligible(selected_index):
    document = _ambiguous_document(selected_index)

    score = evaluation.score_reporter_root_opinion_retrieval(document)

    assert score.reporter_roots_opinion_retrievals == score.reporter_roots_correct_identity == 1


def test_ambiguous_lookup_without_a_selected_cluster_is_not_eligible():
    score = evaluation.score_reporter_root_opinion_retrieval(_ambiguous_document(None))

    assert score.reporter_roots_opinion_retrievals == score.reporter_roots_correct_identity == 0
    assert score.as_dict()["ratio"] is None


@pytest.mark.parametrize(
    "verdict",
    [
        None,
        IdentityVerdict.WRONG_IDENTITY,
        IdentityVerdict.PARTIALLY_CORROBORATED,
        IdentityVerdict.UNDETERMINED,
    ],
)
def test_only_current_correct_identity_qualifies_even_when_a_cluster_is_selected(verdict):
    document = _document(("20",)).get_stage(LOOKUP_STAGE)
    if verdict is None:
        lookup = document.roots[0].reporter_exact_lookup
        document = document.get_stage("1_full_reporter_locators")
        citation = document.citations[0].record("10_roots").with_root(document.citations[0].id)
        document = document.replace_citation(citation).complete("10_roots")
        citation = citation.record(LOOKUP_STAGE).with_reporter_exact_lookup(lookup)
        document = document.replace_citation(citation).complete(LOOKUP_STAGE)
    else:
        citation = document.roots[0].record("38_identity_review").with_identity_judgment(verdict)
        document = document.replace_citation(citation).complete("38_identity_review")
    document = document.complete(STAGE)

    score = evaluation.score_reporter_root_opinion_retrieval(document)

    assert score.reporter_roots_opinion_retrievals == score.reporter_roots_correct_identity == 0


def test_body_only_admission_without_original_cluster_does_not_expand_the_ratio():
    document = _document(("20",), None).get_stage(LOOKUP_STAGE)
    citation = (
        document.roots[1].record("38_body_review").with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
    )
    document = document.replace_citation(citation).complete("38_body_review")
    citation = _with_opinions(document.roots[0], cluster_id="1", bundle=("20",))
    document = document.replace_citation(citation).complete(STAGE)

    score = evaluation.score_reporter_root_opinion_retrieval(document)

    assert all(
        root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY for root in document.roots
    )
    assert score.reporter_roots_opinion_retrievals == score.reporter_roots_correct_identity == 1


def test_not_found_and_empty_text_remain_eligible_but_are_not_successful_retrievals():
    document = _document(
        ("20",),
        ("21",),
        ("22",),
        outcomes={"20": OpinionRetrievalOutcome.NOT_FOUND, "21": OpinionRetrievalOutcome.EMPTY_TEXT},
    )

    score = evaluation.score_reporter_root_opinion_retrieval(document)

    assert score.reporter_roots_correct_identity == 3
    assert score.reporter_roots_opinion_retrievals == 1
    assert score.as_dict()["ratio"] == pytest.approx(1 / 3)


def test_selected_cluster_without_saved_retrieval_or_subopinions_is_not_silently_excluded():
    no_saved_result = evaluation.score_reporter_root_opinion_retrieval(_document(("20",), fetch=False))
    no_subopinions = evaluation.score_reporter_root_opinion_retrieval(_document(()))

    for score in (no_saved_result, no_subopinions):
        assert score.reporter_roots_correct_identity == 1
        assert score.reporter_roots_opinion_retrievals == 0


def test_roundtrip_scoring_recovers_stage_39_before_a_later_withdrawal():
    checkpoint = _document(("20",))
    citation = checkpoint.roots[0].record("40_later").with_root(WITHDRAWN_ROOT_ID)
    later = checkpoint.replace_citation(citation).complete("40_later")
    later = Document.model_validate_json(later.model_dump_json())

    assert later.roots == ()
    assert evaluation.score_reporter_root_opinion_retrieval(
        later
    ) == evaluation.score_reporter_root_opinion_retrieval(checkpoint)


def test_workflow_aggregation_and_rendering_expose_only_the_requested_ratio():
    score = evaluation.score_validate_pincite(_document(("20",), ("21",)))
    combined = score + score

    assert combined.as_dict() == {
        "stages": [
            {
                "stage": STAGE,
                "reporter_roots_opinion_retrievals": 4,
                "reporter_roots_correct_identity": 4,
                "ratio": 1.0,
            }
        ]
    }
    report = evaluation.render_validate_pincite(score)
    assert "| reporter_roots_opinion_retrievals / reporter_roots_correct_identity | 2/2 (100.0%) |" in report
    assert report.count("| reporter_roots_") == 1
    assert "precision" not in report.lower() and "recall" not in report.lower()
    assert "annotation" not in report.lower()


def test_empty_cohort_ratio_is_unavailable_and_impossible_counts_raise():
    score = evaluation.score_reporter_root_opinion_retrieval(_document(None))

    assert score.as_dict()["ratio"] is None
    assert "0/0 (—)" in evaluation.render_reporter_root_opinion_retrieval(score)
    with pytest.raises(ValueError, match="eligible reporter-root cohort"):
        evaluation.OpinionRetrievalScore(
            reporter_roots_opinion_retrievals=2, reporter_roots_correct_identity=1
        )
