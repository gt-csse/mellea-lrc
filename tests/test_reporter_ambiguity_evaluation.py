from __future__ import annotations

import asyncio

from evaluations.render_reporter_ambiguity_report import render_report
from evaluations.score_reporter_ambiguity import _summary, score_document
from mellea_lrc.api import Document, grow_roots, reporter_root_exact_lookup
from mellea_lrc.courtlistener import CourtListenerCitationLookup
from mellea_lrc.model.citations.judgments import IdentityNextStep, IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactAmbiguityOutcome,
    ReporterExactAmbiguityResolution,
)
from mellea_lrc.validation.reporter_root_exact_ambiguity import STAGE


class AmbiguousClient:
    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        return CourtListenerCitationLookup.model_validate({
            "citation": f"{volume} {reporter} {page}",
            "status": 300,
            "clusters": [
                {"id": 1, "caseNameFull": "Bell Atlantic Corporation v. Twombly", "court_id": "scotus", "dateFiled": "2007-05-21"},
                {"id": 2, "caseNameFull": "Bell Atlantic Corporation v. Jones", "court_id": "scotus", "dateFiled": "2007-05-21"},
            ],
        })


def _stage_document(outcome: ReporterExactAmbiguityOutcome, label: str):
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."
    roots = asyncio.run(grow_roots(Document.from_source(source), hunt_dockets=False))
    exact = reporter_root_exact_lookup(roots, client=AmbiguousClient())
    root = exact.roots[0].record(STAGE)
    passing = (0,) if outcome is ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH else ()
    selected = 0 if passing else None
    root = root.with_reporter_exact_ambiguity_resolution(ReporterExactAmbiguityResolution(
        node_id=root.nodes[-1].id,
        outcome=outcome,
        passing_candidate_indices=passing,
        selected_candidate_index=selected,
    ))
    if passing:
        root = root.with_case_name_judgment(len(root.case_name) - 1, 0, MatchResult.MATCH)
    root = (
        root.with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
        if passing
        else root.with_identity_judgment(IdentityVerdict.DEFERRED, IdentityNextStep.AMBIGUITY)
    )
    document = exact.replace_citation(root).complete(STAGE)
    gold = {
        "id": "gold-1", "root_id": "gold-1", "is_root": True,
        "kind": "FullCaseCitation", "identifier": {"kind": "reporter"},
        "locator": {"start": root.locator_span.start, "end": root.locator_span.end},
        "validation": {"identity": {"label": label}},
    }
    return document, (gold,)


def test_ambiguity_score_uses_only_root_identity_for_admission():
    document, rows = _stage_document(
        ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH, "CORRECT_IDENTITY"
    )
    counts, details = score_document(document, rows)
    summary = _summary(counts)

    assert summary["admission_precision"] == 1.0
    assert summary["admission_recall"] == 1.0
    assert details[0]["selected_candidate_index"] == 0
    assert details[0]["candidates"][0]["field_judgments"]["case_name"] == [
        {"reading_index": 0, "candidate_index": 0, "result": "match"}
    ]
    assert details[0]["candidates"][1]["field_judgments"]["case_name"] == []


def test_admitting_a_wrong_identity_is_an_incorrect_admission():
    document, rows = _stage_document(
        ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH, "WRONG_IDENTITY"
    )
    counts, _ = score_document(document, rows)
    summary = _summary(counts)

    assert summary["admission_precision"] == 0.0
    assert summary["admission_recall"] is None


def test_annotated_repeated_occurrence_uses_its_canonical_root_label():
    document, (canonical,) = _stage_document(
        ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH, "CORRECT_IDENTITY"
    )
    canonical = {**canonical, "id": "gold-root", "root_id": "gold-root"}
    repeated = {
        "id": "gold-leaf",
        "root_id": "gold-root",
        "is_root": False,
        "kind": "FullCaseCitation",
        "locator": canonical["locator"],
    }

    counts, details = score_document(document, (repeated,), identity_roots=(canonical,))

    assert counts["gold_ambiguous_roots"] == 1
    assert counts["correct_admissions"] == 1
    assert details[0]["gold_root_id"] == "gold-root"


def test_ambiguity_report_shows_only_stage_admission_and_route_counts():
    counts = {
        "admission_precision": 0.75,
        "admission_recall": 0.5,
        "gold_ambiguous_roots": 4,
        "route_outcomes": {
            "unique_rule_match": 3,
            "review_required": 1,
            "too_many_candidates": 2,
        },
    }
    report = render_report(
        {"stage": STAGE, "sets": {"primary": counts}, "totals": counts},
        source_label="saved/summary.json",
    )

    assert "| primary | 75.0% | 50.0% | 4 | 3 | 1 | 2 |" in report
    assert "Field judgment precision" not in report
