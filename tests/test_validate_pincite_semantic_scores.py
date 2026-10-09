"""Pinpoint accuracy uses native settled annotations under correct-identity roots."""

from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from evaluations import validate_pincite as evaluation
from evaluations.score_types import substage_heading
from mellea_lrc.model import Document, FullReporterCitation, Span
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_opinion import (
    OpinionRetrievalOutcome,
    ReporterRootOpinionRetrieval,
    ReporterRootOpinionSource,
    RetrievedReporterOpinion,
)
from mellea_lrc.model.citations.reporter_page_resolution import (
    ReporterCitationPageResolution,
    ReporterPageResolutionOutcome,
)
from mellea_lrc.model.citations.reporter_pages import IndexedReporterOpinion, ReporterRootOpinionPageIndex
from mellea_lrc.model.citations.reporter_pinpoint import (
    OpinionEvidenceQuote,
    OpinionReviewScope,
    OpinionSupportResult,
    PinpointEvidenceOutcome,
    ReporterCitationPinpointEvidence,
    ReporterCitationSupportReview,
    ReporterOpinionEvidence,
    ReporterPinpointJudgment,
    ReporterPinpointVerdict,
    ReporterSupportDecision,
)
from mellea_lrc.providers.courtlistener import CourtListenerCluster

LOCATORS = ("550 U.S. 544", "347 U.S. 483", "410 U.S. 113", "550 U.S. 544", "505 U.S. 833", "500 U.S. 100")
PINS = ("570", "490", "120", "560", "840", None)
OPINION = "Actual opinion supports the proposition."


def _document(tmp_path, *, wrong_root_identity=False, docket_gold=False):
    dataset = tmp_path / "primary"
    textdir = dataset / "documents_txt"
    textdir.mkdir(parents=True)
    (dataset / "documents").mkdir()
    source = textdir / "filing.txt"
    text = "; ".join(locator + (f", {pin}" if pin else "") for locator, pin in zip(LOCATORS, PINS))
    if docket_gold:
        text += "; No. 23-123; Id. at 7"
    source.write_text(text)
    document = Document.from_source(source)
    starts = []
    offset = 0
    for index, locator in enumerate(LOCATORS):
        start = text.index(locator, offset)
        starts.append(start)
        offset = start + len(locator)
        if index != 1:  # A settled gold root the extractor never created.
            document = document.add_citation(
                FullReporterCitation.from_locator(
                    citation_id=f"c{index}",
                    substage="grow_roots.locator_discovery.full_reporter_locators",
                    source=text,
                    span=Span(start=start, end=start + len(locator)),
                )
            )
    document = document.complete_substage("grow_roots.locator_discovery.full_reporter_locators")
    for citation in document.citations:
        index = int(citation.id[1:])
        if PINS[index] is not None:
            start = starts[index] + len(LOCATORS[index]) + 2
            document = document.replace_citation(
                citation.record("grow_roots.field_reading.pin_cites").with_pin_cite(
                    text, Span(start=start, end=start + len(PINS[index]))
                )
            )
    document = document.complete_substage("grow_roots.field_reading.pin_cites")
    for citation in document.citations:
        root_id = "c0" if citation.id == "c3" else citation.id
        citation = citation.record("grow_roots.root_formation.rule").with_root(root_id)
        if root_id == citation.id:
            citation = citation.with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
        document = document.replace_citation(citation)
    document = document.complete_substage("grow_roots.root_formation.rule")
    for root in document.roots:
        root = root.record(evaluation.SUBSTAGE)
        cluster = CourtListenerCluster.model_validate({"id": 1, "sub_opinions": [20]})
        root = root.with_reporter_root_opinion_source(
            ReporterRootOpinionSource(node_id=root.nodes[-1].id, cluster=cluster)
        )
        body = "" if root.id == "c2" else OPINION
        root = root.with_reporter_root_opinion_retrieval(
            ReporterRootOpinionRetrieval(
                node_id=root.nodes[-1].id,
                cluster_id="1",
                sub_opinion_ids=("20",),
                opinions=(
                    RetrievedReporterOpinion(
                        opinion_id="20",
                        cluster_id="1",
                        outcome=OpinionRetrievalOutcome.RETRIEVED
                        if body
                        else OpinionRetrievalOutcome.EMPTY_TEXT,
                        response={"id": 20, "cluster": 1, "plain_text": body},
                    ),
                ),
            )
        )
        document = document.replace_citation(root)
    document = document.complete_substage(evaluation.SUBSTAGE)
    for root in document.roots:
        root = root.record(evaluation.PAGE_INDEX_SUBSTAGE)
        root = root.with_reporter_root_opinion_page_index(
            ReporterRootOpinionPageIndex(
                node_id=root.nodes[-1].id,
                cluster_id="1",
                opinions=(
                    IndexedReporterOpinion(
                        opinion_id="20",
                        text_field=None if root.id == "c2" else "plain_text",
                        text="" if root.id == "c2" else OPINION,
                    ),
                ),
            )
        )
        document = document.replace_citation(root)
    document = document.complete_substage(evaluation.PAGE_INDEX_SUBSTAGE)
    for citation in document.citations:
        citation = citation.record(evaluation.PAGE_RESOLUTION_SUBSTAGE)
        citation = citation.with_reporter_page_resolution(
            ReporterCitationPageResolution(
                node_id=citation.nodes[-1].id,
                root_id="c0" if citation.id == "c3" else citation.id,
                locator_citation_id=citation.id,
                locator_reading_index=0,
                pin_citation_id=citation.id if citation.pin_cite else None,
                pin_reading_index=0 if citation.pin_cite else None,
                outcome=ReporterPageResolutionOutcome.UNLOCATED
                if citation.pin_cite
                else ReporterPageResolutionOutcome.NO_PIN,
                reason="The opinion lacks reporter pages.",
            )
        )
        document = document.replace_citation(citation)
    document = (
        document.complete_substage(evaluation.PAGE_RESOLUTION_SUBSTAGE)
        .complete_substage(evaluation.OPINION_SELECTION_SUBSTAGE)
        .complete_substage(evaluation.PROPOSITION_SUBSTAGE)
    )
    for citation in document.citations:
        citation = citation.record(evaluation.PINPOINT_EVIDENCE_SUBSTAGE)
        citation = citation.with_reporter_pinpoint_evidence(
            ReporterCitationPinpointEvidence(
                node_id=citation.nodes[-1].id,
                root_id="c0" if citation.id == "c3" else citation.id,
                resolution_index=0,
                proposition_index=None,
                pages=(),
                outcome=PinpointEvidenceOutcome.MISSING_PAGES,
                reason="The full opinion remains available where pages are missing.",
            )
        )
        document = document.replace_citation(citation)
    document = document.complete_substage(evaluation.PINPOINT_EVIDENCE_SUBSTAGE)
    header = {
        "unit": "header",
        "dataset": "primary",
        "document": source.name,
        "text": {
            "path": "primary/documents_txt/filing.txt",
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "length": len(text),
        },
    }
    rows = []
    for index, locator in enumerate(LOCATORS):
        span = {"start": starts[index], "end": starts[index] + len(locator)}
        row = {
            "unit": "citation",
            "id": f"g{index}",
            "is_root": index != 3,
            "root_id": "g0" if index == 3 else f"g{index}",
            "kind": "FullCaseCitation",
            "locator": {
                "source": {"kind": "quoted", **span},
                "normalization": {"kind": "value", "value": {"kind": "reporter"}},
            },
            "validation": {
                "identity": {
                    "label": "WRONG_IDENTITY" if wrong_root_identity and index == 0 else "CORRECT_IDENTITY"
                },
                "opinion_target": {
                    "state": "identified",
                    "exhaustive": False,
                    "targets": [{"provider": "cap", "id": "9999"}],
                },
            },
        }
        if PINS[index] is not None:
            label = "SKIPPED" if index == 4 else "CORRECT_PINCITE" if index in (0, 1) else "WRONG_PINCITE"
            row["validation"]["pincite"] = {"label": label}
            if label != "SKIPPED":
                row["validation"]["pincite"].update(
                    pagination_available=index in (0, 1),
                    correct_page=True if index in (0, 1) else None,
                    found_pages=[],
                )
            if index == 4:
                row["validation"]["pincite"]["skip_type"] = "TOA"
        rows.append(row)
    if docket_gold:
        start = text.index("No. 23-123")
        rows.append(
            {
                "unit": "citation",
                "id": "gd",
                "is_root": True,
                "root_id": "gd",
                "kind": "DocketCitation",
                "locator": {
                    "source": {"kind": "quoted", "start": start, "end": start + len("No. 23-123")},
                    "normalization": {"kind": "value", "value": {"kind": "docket"}},
                },
                "validation": {"identity": {"label": "CORRECT_IDENTITY"}},
            }
        )
        start = text.index("Id. at 7")
        rows.append(
            {
                "unit": "citation",
                "id": "gdl",
                "is_root": False,
                "root_id": "gd",
                "kind": "IdCitation",
                "cited_as": {"start": start, "end": start + len("Id. at 7")},
                "pin_cite": {"start": start + 7, "end": start + 8, "quote": "7"},
                "validation": {"pincite": {"label": "CORRECT_PINCITE"}},
            }
        )
    (dataset / "documents" / "filing.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (header, *rows)) + "\n"
    )
    return document


def _judgments(document, verdicts):
    if evaluation.PAGE_SUPPORT_SUBSTAGE not in document.substage_runs:
        document = document.complete_substage(evaluation.PAGE_SUPPORT_SUBSTAGE).complete_substage(
            evaluation.FULL_OPINION_SUBSTAGE
        )
    for citation in document.citations:
        if citation.id not in verdicts:
            continue
        citation = citation.record(evaluation.JUDGMENT_SUBSTAGE)
        citation = citation.with_reporter_pinpoint_judgment(
            ReporterPinpointJudgment(
                node_id=citation.nodes[-1].id,
                evidence_index=0,
                review_index=None,
                verdict=verdicts[citation.id],
                pagination_available=False,
                correct_page=None,
                found_pages=(),
                reason="Opinion support and pinpoint location are evaluated separately.",
            )
        )
        document = document.replace_citation(citation)
    return document.complete_substage(evaluation.JUDGMENT_SUBSTAGE)


def test_fixed_gold_keeps_missing_roots_and_empty_opinions_and_excludes_skips(tmp_path):
    document = _judgments(
        _document(tmp_path),
        {
            "c0": ReporterPinpointVerdict.CORRECT_PINCITE,
            "c3": ReporterPinpointVerdict.CORRECT_PINCITE,
            "c4": ReporterPinpointVerdict.WRONG_PINCITE,
            "c5": ReporterPinpointVerdict.WRONG_PINCITE,
        },
    )
    score = evaluation.score_validate_pincite(document)
    assert score.pinpoint["reporter_roots"].gold == 3
    assert score.pinpoint["reporter_leaves"].gold == 1
    assert score.pinpoint["total"].as_dict() == {
        "correct": 1,
        "predicted": 2,
        "gold": 4,
        "wrong": 1,
        "undetermined": 0,
        "missing_judgments": 2,
        "missing_opinions": 2,
        "precision": 0.5,
        "recall": 0.25,
    }
    assert score.substages[-1].unscored_definitive == 2
    assert score.as_dict()["gold_cohort"]["total"] == 4
    assert "| total | 1/2 (50.0%) | 1/4 (25.0%) |" in evaluation.render_validate_pincite(score)


def test_undetermined_is_a_recall_miss_and_has_no_definitive_precision(tmp_path):
    document = _judgments(
        _document(tmp_path),
        {"c0": ReporterPinpointVerdict.UNDETERMINED, "c3": ReporterPinpointVerdict.WRONG_PINCITE},
    )
    score = evaluation.score_validate_pincite(document).pinpoint["total"].as_dict()
    assert score["correct"] == score["predicted"] == score["undetermined"] == 1
    assert score["gold"] == 4 and score["precision"] == 1 and score["recall"] == 0.25


def test_wrong_gold_root_excludes_family_even_when_pipeline_and_leaf_identity_agree(tmp_path):
    document = _judgments(
        _document(tmp_path, wrong_root_identity=True),
        {
            "c0": ReporterPinpointVerdict.CORRECT_PINCITE,
            "c3": ReporterPinpointVerdict.WRONG_PINCITE,
            "c2": ReporterPinpointVerdict.WRONG_PINCITE,
        },
    )
    assert document.roots[0].identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY

    score = evaluation.score_validate_pincite(document)

    assert tuple(score.pinpoint) == evaluation.GROUPS
    assert score.pinpoint["reporter_roots"].gold == 2  # Includes the missing root and empty opinion.
    assert score.pinpoint["reporter_leaves"].gold == 0  # The leaf's own label cannot override its root.
    assert score.pinpoint["total"].correct == score.pinpoint["total"].predicted == 1
    assert score.pinpoint["total"].as_dict()["precision"] == 1
    assert score.pinpoint["total"].as_dict()["recall"] == 0.5
    assert score.substages[-1].unscored_definitive == 2


def test_unimplemented_docket_leaf_stays_in_common_gold_denominator(tmp_path):
    document = _judgments(
        _document(tmp_path, docket_gold=True),
        {"c0": ReporterPinpointVerdict.CORRECT_PINCITE, "c3": ReporterPinpointVerdict.WRONG_PINCITE},
    )
    score = evaluation.score_validate_pincite(document)

    assert score.as_dict()["gold_cohort"] == {
        "scope": "Native settled pin cites under gold CORRECT_IDENTITY roots, including reporter and docket roots and leaves",
        "labels": ["CORRECT_PINCITE", "WRONG_PINCITE"],
        "reporter_roots": 3,
        "reporter_leaves": 1,
        "docket_roots": 0,
        "docket_leaves": 1,
        "total": 5,
    }
    docket = score.pinpoint["docket_leaves"].as_dict()
    assert docket["gold"] == docket["missing_judgments"] == 1
    assert docket["correct"] == docket["predicted"] == 0
    assert score.pinpoint["total"].correct == 2
    assert score.pinpoint["total"].as_dict()["recall"] == 0.4
    assert "| docket_leaves | 0/0 (—) | 0/1 (0.0%) |" in evaluation.render_validate_pincite(score)


def _review(document, citation_id, substage, result):
    citation = next(c for c in document.citations if c.id == citation_id).record(substage)
    evidence = ()
    indices = ()
    if result in {OpinionSupportResult.SUPPORTED, OpinionSupportResult.CONTRADICTED}:
        citation = citation.with_reporter_opinion_evidence(
            ReporterOpinionEvidence(
                node_id=citation.nodes[-1].id,
                root_id="c0",
                opinion_id="20",
                quote=OPINION,
                span=Span(start=0, end=len(OPINION)),
            )
        )
        evidence = (OpinionEvidenceQuote(opinion_id="20", quote=OPINION),)
        indices = (len(citation.reporter_opinion_evidence) - 1,)
    citation = citation.with_reporter_support_review(
        ReporterCitationSupportReview(
            node_id=citation.nodes[-1].id,
            evidence_index=0,
            scope=OpinionReviewScope.CITED_PAGES
            if substage == evaluation.PAGE_SUPPORT_SUBSTAGE
            else OpinionReviewScope.FULL_OPINION,
            decision=ReporterSupportDecision(
                result=result,
                evidence=evidence,
                pagination_available=False,
                correct_page=None,
                found_pages=(),
                reason="Read the opinion for the attributed proposition.",
            ),
            opinion_evidence_indices=indices,
        )
    )
    return document.replace_citation(citation)


def test_page_contradiction_is_provisional_until_full_opinion_review(tmp_path):
    document = _review(
        _document(tmp_path), "c0", evaluation.PAGE_SUPPORT_SUBSTAGE, OpinionSupportResult.CONTRADICTED
    ).complete_substage(evaluation.PAGE_SUPPORT_SUBSTAGE)
    score = evaluation.score_reporter_citation_pinpoint_page_review(document)
    assert score.counts == {"contradicted": 1}
    assert score.judgments["total"].predicted == 0
    assert score.judgments["total"].undetermined == 1
    document = _review(
        document, "c0", evaluation.FULL_OPINION_SUBSTAGE, OpinionSupportResult.CONTRADICTED
    ).complete_substage(evaluation.FULL_OPINION_SUBSTAGE)
    score = evaluation.score_reporter_citation_full_opinion_review(document)
    assert score.judgments["total"].predicted == 1
    assert score.judgments["total"].correct == 0


def test_stage_precision_is_incremental_and_later_roundtrip_recovers_prior_checkpoint(tmp_path):
    document = _review(
        _document(tmp_path), "c0", evaluation.PAGE_SUPPORT_SUBSTAGE, OpinionSupportResult.SUPPORTED
    )
    document = _review(document, "c3", evaluation.PAGE_SUPPORT_SUBSTAGE, OpinionSupportResult.NOT_FOUND)
    document = document.complete_substage(evaluation.PAGE_SUPPORT_SUBSTAGE)
    page_score = evaluation.score_reporter_citation_pinpoint_page_review(document)
    assert page_score.judgments["total"].predicted == page_score.judgments["total"].correct == 1
    assert page_score.judgments["total"].undetermined == 1  # A page miss is not whole-opinion absence.
    document = _review(
        document, "c3", evaluation.FULL_OPINION_SUBSTAGE, OpinionSupportResult.NOT_FOUND
    ).complete_substage(evaluation.FULL_OPINION_SUBSTAGE)
    full_score = evaluation.score_reporter_citation_full_opinion_review(document)
    assert full_score.judgments["reporter_roots"].predicted == 0
    assert (
        full_score.judgments["reporter_leaves"].correct
        == full_score.judgments["reporter_leaves"].predicted
        == 1
    )
    assert evaluation.score_validate_pincite(document).pinpoint["total"].correct == 2
    document = _judgments(
        document, {"c0": ReporterPinpointVerdict.CORRECT_PINCITE, "c3": ReporterPinpointVerdict.WRONG_PINCITE}
    )
    restored = Document.model_validate_json(document.model_dump_json())
    assert evaluation.score_reporter_citation_pinpoint_page_review(restored) == page_score
    assert evaluation.score_reporter_citation_full_opinion_review(restored) == full_score
    total = evaluation.score_validate_pincite(restored).pinpoint["total"]
    assert total.correct == total.predicted == 2 and total.gold == 4


def test_dataset_inventory_is_independent_of_missing_roots_and_failed_retrieval(tmp_path):
    document = _document(tmp_path, wrong_root_identity=True).get_substage(evaluation.SUBSTAGE)
    successful = evaluation.score_validate_pincite(document)
    failed = document.model_copy(
        update={
            "citations": tuple(
                citation.model_copy(update={"reporter_root_opinion_retrieval": None})
                for citation in document.citations
            )
        }
    )
    unavailable = evaluation.score_validate_pincite(failed)

    assert successful.substages[0].reporter_roots_opinion_retrievals == 3
    assert unavailable.substages[0].reporter_roots_opinion_retrievals == 0
    assert successful.dataset == unavailable.dataset
    inventory = successful.dataset
    assert inventory.all_annotations["total_pincites"] == 5
    assert inventory.all_annotations["settled"] == 4
    assert inventory.all_annotations["SKIPPED_TOA"] == 1
    assert inventory.under_gold_correct_identity_roots["total_pincites"] == 3
    assert inventory.under_gold_correct_identity_roots["settled"] == 2
    assert inventory.settled_under_gold_correct_identity_roots["reporter_leaves"]["WRONG_PINCITE"] == 0
    doubled = successful + successful
    assert doubled.dataset.all_annotations["total_pincites"] == 10
    assert doubled.as_dict()["dataset"]["all_annotations"]["settled"] == 8
    report = evaluation.render_validate_pincite(successful)
    assert report.index("## Dataset annotations") < report.index(substage_heading(evaluation.SUBSTAGE))
    assert "explicit native correct_page Boolean, independently of the content label" in report
    assert all(
        legacy not in report for legacy in ("page_unlocated", "other_page", "cited_target", "other_target")
    )


def test_dataset_inventory_counts_native_unknowns_and_four_families_only(monkeypatch):
    def row(identifier, root_id, kind, *, label=None, identity=None, pin=False):
        validation = {}
        if label:
            validation["pincite"] = {"label": label}
            if label == "SKIPPED":
                validation["pincite"]["skip_type"] = "UNSETTLED"
        if identity:
            validation["identity"] = {"label": identity}
        return {
            "id": identifier,
            "root_id": root_id,
            "is_root": identifier == root_id,
            "kind": kind,
            "validation": validation,
            "pin_cite": {"start": 0, "end": 1} if pin else {"source": {"kind": "not_stated"}},
        }

    rows = (
        row("r", "r", "FullCaseCitation", label="CORRECT_PINCITE", identity="CORRECT_IDENTITY"),
        row("w", "w", "FullCaseCitation", label="WRONG_PINCITE", identity="WRONG_IDENTITY"),
        row("u", "u", "FullCaseCitation", label="SKIPPED", identity="CORRECT_IDENTITY"),
        row("d", "d", "DocketCitation", label="CORRECT_PINCITE", identity="CORRECT_IDENTITY"),
        row("wl", "w", "ShortCaseCitation", label="WRONG_PINCITE", identity="CORRECT_IDENTITY"),
        row("dl", "d", "IdCitation", label="CORRECT_PINCITE"),
        row("missing", "r", "ShortCaseCitation", pin=True),
        row("no-pin", "r", "IdCitation"),
    )
    monkeypatch.setattr(evaluation, "citation_annotations", lambda _document: rows)
    inventory = evaluation._dataset_inventory(Document.from_source("Native annotations only."))

    assert inventory.all_annotations["total_pincites"] == 7
    assert inventory.all_annotations["settled"] == 5
    assert (
        inventory.all_annotations["SKIPPED_UNSETTLED"] == inventory.all_annotations["missing_pin_label"] == 1
    )
    assert inventory.under_gold_correct_identity_roots["total_pincites"] == 5
    assert inventory.under_gold_correct_identity_roots["settled"] == 3
    assert inventory.settled_by_family == {
        "reporter_roots": {"CORRECT_PINCITE": 1, "WRONG_PINCITE": 1},
        "reporter_leaves": {"CORRECT_PINCITE": 0, "WRONG_PINCITE": 1},
        "docket_roots": {"CORRECT_PINCITE": 1, "WRONG_PINCITE": 0},
        "docket_leaves": {"CORRECT_PINCITE": 1, "WRONG_PINCITE": 0},
    }
    assert inventory.settled_under_gold_correct_identity_roots["reporter_leaves"]["WRONG_PINCITE"] == 0


def test_gold_source_hash_mismatch_fails_instead_of_narrowing_the_cohort(tmp_path):
    document = _judgments(_document(tmp_path), {"c0": ReporterPinpointVerdict.CORRECT_PINCITE})
    annotation = tmp_path / "primary" / "documents" / "filing.jsonl"
    lines = annotation.read_text().splitlines()
    header = json.loads(lines[0])
    header["text"]["sha256"] = "wrong"
    annotation.write_text(json.dumps(header) + "\n" + "\n".join(lines[1:]))
    with pytest.raises(ValueError, match="does not match"):
        evaluation.score_validate_pincite(document)


def test_proposition_checkpoint_counts_need_no_annotation_or_source_path():
    document = (
        Document.from_source("No citations here.")
        .complete_substage(evaluation.SUBSTAGE)
        .complete_substage(evaluation.PROPOSITION_SUBSTAGE)
    )

    score = evaluation.score_validate_pincite(document)

    assert document.source_path is None
    assert score.pinpoint is None
    assert score.substages[-1].as_dict() == {"substage": evaluation.PROPOSITION_SUBSTAGE, "counts": {}}
    assert "gold_cohort" not in score.as_dict()


@pytest.mark.parametrize(
    "suffix",
    [
        "reporter_root_opinion_page_index",
        "reporter_citation_page_resolution",
        "reporter_citation_opinion_review",
        "reporter_citation_propositions",
        "reporter_citation_pinpoint_evidence",
        "reporter_citation_pinpoint_page_review",
        "reporter_citation_full_opinion_review",
        "reporter_citation_pinpoint_judgment",
    ],
)
def test_stage_scorers_and_renderers_are_independently_callable_with_one_argument(tmp_path, suffix):
    document = _judgments(_document(tmp_path), {"c0": ReporterPinpointVerdict.CORRECT_PINCITE})
    scorer = getattr(evaluation, f"score_{suffix}")
    renderer = getattr(evaluation, f"render_{suffix}")

    assert tuple(inspect.signature(scorer).parameters) == ("document",)
    assert tuple(inspect.signature(renderer).parameters) == ("score",)
    stage_score = scorer(document)
    assert evaluation._SUBSTAGE_SCORERS[stage_score.substage] is scorer
    assert renderer(stage_score).startswith(f"{substage_heading(stage_score.substage)}\n")
    workflow = evaluation.score_validate_pincite(document)
    assert stage_score == next(
        score for score in workflow.substages if score.substage == stage_score.substage
    )
    assert renderer(stage_score) in evaluation.render_validate_pincite(workflow)
