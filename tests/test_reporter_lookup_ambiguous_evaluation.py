from __future__ import annotations

import asyncio

from evaluations.render_reporter_lookup_ambiguous import render_report
from evaluations.score_reporter_lookup_ambiguous import _summary, score_document
from mellea_lrc.api import Document, grow_roots, reporter_root_lookup, reporter_root_lookup_ambiguous
from mellea_lrc.courtlistener import CourtListenerCitationLookup
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactAmbiguityOutcome,
    ReporterExactAmbiguityResolution,
)
from mellea_lrc.validation.reporter_root_lookup_ambiguous import STAGE
from mellea_lrc.validation.stage_names import REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM


class AmbiguousClient:
    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        return CourtListenerCitationLookup.model_validate(
            {
                "citation": f"{volume} {reporter} {page}",
                "status": 300,
                "clusters": [
                    {
                        "id": 1,
                        "caseNameFull": "Bell Atlantic Corporation v. Twombly",
                        "court_id": "scotus",
                        "dateFiled": "2007-05-21",
                    },
                    {
                        "id": 2,
                        "caseNameFull": "Bell Atlantic Corporation v. Jones",
                        "court_id": "scotus",
                        "dateFiled": "2007-05-21",
                    },
                ],
            }
        )


def _stage_document(outcome: ReporterExactAmbiguityOutcome, label: str):
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."
    roots = asyncio.run(grow_roots(Document.from_source(source), hunt_dockets=False))
    exact = reporter_root_lookup(roots, client=AmbiguousClient())
    root = exact.roots[0].record(STAGE)
    passing = (0,) if outcome is ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH else ()
    selected = 0 if passing else None
    root = root.with_reporter_exact_ambiguity_resolution(
        ReporterExactAmbiguityResolution(
            node_id=root.nodes[-1].id,
            outcome=outcome,
            passing_candidate_indices=passing,
            selected_candidate_index=selected,
        )
    )
    if passing:
        root = root.with_case_name_judgment(len(root.case_name) - 1, 0, MatchResult.MATCH)
    root = (
        root.with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
        if passing
        else root.with_identity_judgment(IdentityVerdict.DEFERRED, REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM)
    )
    document = exact.replace_citation(root).complete(STAGE)
    field_gold = {}
    for field in ("case_name", "court", "date"):
        readings = getattr(root, field)
        if not readings:
            continue
        reading = readings[-1]
        span = reading.span
        field_gold[field] = {"label": "agrees"}
        if span is not None:
            field_gold[field].update(start=span.start, end=span.end, quote=reading.quote)
        if field == "court":
            field_gold[field]["id"] = reading.get_normalized().id
        if field == "date":
            normalized = reading.get_normalized()
            value = f"{normalized.year:04d}"
            if normalized.month is not None:
                value += f"-{normalized.month:02d}"
            if normalized.day is not None:
                value += f"-{normalized.day:02d}"
            field_gold[field]["normalized"] = value
    gold = {
        "id": "gold-1",
        "root_id": "gold-1",
        "is_root": True,
        "kind": "FullCaseCitation",
        "identifier": {"kind": "reporter"},
        "locator": {"start": root.locator_span.start, "end": root.locator_span.end},
        "case_name": field_gold.get("case_name"),
        "court": field_gold.get("court"),
        "date": field_gold.get("date"),
        "validation": {
            "identity": {
                "label": label,
                "fields": {key: {"label": value["label"]} for key, value in field_gold.items()},
            }
        },
    }
    return document, (gold,)


def test_ambiguity_score_reports_only_selected_candidate_field_precision():
    document, rows = _stage_document(ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH, "CORRECT_IDENTITY")
    counts, details = score_document(document, rows)
    summary = _summary(counts)

    assert summary == {
        "field_precision": {
            "case_name": {"value": 1.0, "correct": 1, "scored": 1},
            "court": {"value": None, "correct": 0, "scored": 0},
            "date": {"value": None, "correct": 0, "scored": 0},
        }
    }
    assert set(counts) == {"case_name_scored", "case_name_correct"}
    assert details[0]["selected_candidate_index"] == 0
    assert details[0]["candidates"][0]["field_judgments"]["case_name"] == [
        {"reading_index": 0, "candidate_index": 0, "result": "match"}
    ]
    assert details[0]["candidates"][1]["field_judgments"]["case_name"] == []


def test_identity_label_does_not_change_field_precision():
    document, rows = _stage_document(ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH, "WRONG_IDENTITY")
    counts, _ = score_document(document, rows)
    summary = _summary(counts)

    assert set(summary) == {"field_precision"}
    assert summary["field_precision"]["case_name"] == {"value": 1.0, "correct": 1, "scored": 1}


def test_annotated_repeated_occurrence_uses_its_canonical_root_label():
    document, (canonical,) = _stage_document(
        ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH, "CORRECT_IDENTITY"
    )
    canonical_name = canonical["case_name"]
    canonical = {
        **canonical,
        "id": "gold-root",
        "root_id": "gold-root",
        "case_name": {
            **canonical_name,
            "start": canonical_name["start"] + 100,
            "end": canonical_name["end"] + 100,
        },
    }
    repeated = {
        "id": "gold-leaf",
        "root_id": "gold-root",
        "is_root": False,
        "kind": "FullCaseCitation",
        "locator": canonical["locator"],
    }

    counts, details = score_document(document, (repeated,), identity_roots=(canonical,))

    assert set(counts) == {"case_name_scored", "case_name_correct"}
    assert _summary(counts)["field_precision"]["case_name"] == {"value": 1.0, "correct": 1, "scored": 1}
    assert details[0]["gold_root_id"] == "gold-root"


def test_ambiguity_score_uses_only_citations_routed_to_the_stage():
    roots = asyncio.run(
        grow_roots(
            Document.from_source("Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."),
            hunt_dockets=False,
        )
    )
    exact = reporter_root_lookup(roots, client=AmbiguousClient())
    other_route = (
        exact.roots[0]
        .record("route_setup")
        .with_identity_judgment(IdentityVerdict.DEFERRED, REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM)
    )
    before = exact.replace_citation(other_route).complete("route_setup")
    after = reporter_root_lookup_ambiguous(before)

    counts, details = score_document(after, ())

    assert details == []
    assert _summary(counts) == {
        "field_precision": {
            field: {"value": None, "correct": 0, "scored": 0} for field in ("case_name", "court", "date")
        }
    }


def test_ambiguity_report_shows_precision_and_explicit_denominators_only():
    counts = {
        "admission_precision": {"value": 0.75, "correct": 3, "scored": 4},
        "route_outcomes": {"unique_rule_match": 4},
        "field_precision": {
            "case_name": {"value": 1.0, "correct": 3, "scored": 3},
            "court": {"value": None, "correct": 0, "scored": 0},
            "date": {"value": 0.5, "correct": 1, "scored": 2},
        },
    }
    report = render_report(
        {"stage": STAGE, "sets": {"primary": counts}, "totals": counts},
        source_label="saved/summary.json",
    )

    assert "| Set | Case name | Court | Date |" in report
    assert "| primary | 3/3 (100.0%) | 0/0 (—) | 1/2 (50.0%) |" in report
    assert "admission" not in report.lower()
    assert "identity" not in report.lower()
    assert "route" not in report.lower()
    assert "recall" not in report.lower()
    assert "gold ambiguous roots" not in report.lower()
