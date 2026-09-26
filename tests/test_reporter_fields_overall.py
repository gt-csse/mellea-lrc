"""Combined reporter reviews and overall field scores use saved, aligned evidence."""

from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy

import pytest

from evaluations.combine_reporter_reviews import combine_documents
from evaluations.render_reporter_fields_overall import render_report
from evaluations.score_reporter_fields_overall import NAME, _summary, score_document
from mellea_lrc.api import Document, grow_roots, reporter_root_lookup, reporter_root_lookup_ambiguous
from mellea_lrc.courtlistener import CourtListenerCitationLookup
from mellea_lrc.model import FullReporterCitation
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterAmbiguousReviewDecision,
    ReporterUniqueReviewDecision,
)
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm import reporter_root_lookup_ambiguous_llm
from mellea_lrc.validation.reporter_root_lookup_unique_llm import reporter_root_lookup_unique_llm
from mellea_lrc.validation.stage_names import (
    REPORTER_ROOT_LOOKUP,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
    REPORTER_ROOT_LOOKUP_UNIQUE_LLM,
)

SOURCE = (
    "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007). "
    "Roe v. Wade, 410 U.S. 113 (1973). "
    "Brown v. Board of Education, 347 U.S. 483 (1954)."
)


class _LookupClient:
    def __init__(self, *, twombly_id: int = 1) -> None:
        self.twombly_id = twombly_id

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        assert reporter == "U.S."
        assert (volume, page) in {("550", "544"), ("410", "113"), ("347", "483")}
        if volume == "550":
            status = 200
            clusters = [
                {
                    "id": self.twombly_id,
                    "caseName": "Bell Atlantic Corporation v. Twombly",
                    "court_id": "scotus",
                    "dateFiled": "2007-05-21",
                }
            ]
        elif volume == "410":
            status = 300
            clusters = [
                {
                    "id": cluster_id,
                    "caseName": "Roe v. Wade",
                    "court_id": "scotus",
                    "dateFiled": "1973-01-22",
                }
                for cluster_id in (11, 22)
            ]
        else:
            status = 404
            clusters = []
        for cluster in clusters:
            cluster["citations"] = [{"volume": int(volume), "reporter": reporter, "page": page}]
        return CourtListenerCitationLookup.model_validate(
            {"citation": f"{volume} {reporter} {page}", "status": status, "clusters": clusters}
        )


class _UniqueReviewer:
    async def __call__(self, _context: object) -> ReporterUniqueReviewDecision:
        return ReporterUniqueReviewDecision.model_validate(
            {
                **{
                    field: {
                        "propose_replacement": False,
                        "quote": None,
                        "result": "match",
                        "reason": "The filing and saved record agree.",
                    }
                    for field in ("case_name", "court", "date")
                },
                "reason": "The saved candidate agrees with the filing.",
            }
        )


class _AmbiguousReviewer:
    async def __call__(self, _context: object) -> ReporterAmbiguousReviewDecision:
        return ReporterAmbiguousReviewDecision.model_validate(
            {
                "selected_candidate_index": 0,
                **{
                    field: {
                        "propose_replacement": False,
                        "quote": None,
                        "result": "match",
                        "reason": "The filing and saved record agree.",
                    }
                    for field in ("case_name", "court", "date")
                },
                "reason": "The first saved candidate is selected.",
            }
        )


def _lookup(*, twombly_id: int = 1) -> Document:
    roots = asyncio.run(grow_roots(Document.from_source(SOURCE), hunt_dockets=False))
    return reporter_root_lookup(roots, client=_LookupClient(twombly_id=twombly_id))


def _branches(base: Document) -> tuple[Document, Document]:
    ambiguous_rules = reporter_root_lookup_ambiguous(base)
    ambiguous = asyncio.run(
        reporter_root_lookup_ambiguous_llm(ambiguous_rules, reviewer=_AmbiguousReviewer())
    )
    unique = asyncio.run(reporter_root_lookup_unique_llm(base, reviewer=_UniqueReviewer()))
    return ambiguous, unique


def _gold(root: FullReporterCitation, document: Document, cluster_id: str) -> dict:
    locator = root.locator_span
    case_name = root.case_name[-1]
    court = root.court[-1]
    date = root.date[-1]
    return {
        "id": root.id,
        "root_id": root.id,
        "is_root": True,
        "kind": "FullCaseCitation",
        "identifier": {"kind": "reporter"},
        "locator": {
            "start": locator.start,
            "end": locator.end,
            "quote": document.text[locator.start : locator.end],
        },
        "case_name": {
            "start": case_name.span.start,
            "end": case_name.span.end,
            "quote": case_name.quote,
        },
        "court": {"id": court.get_normalized().id},
        "date": {
            "start": date.span.start,
            "end": date.span.end,
            "quote": date.quote,
            "normalized": date.quote,
        },
        "validation": {
            "identity": {
                "fields": {field: {"label": "agrees"} for field in ("case_name", "court", "date")},
                "evidence": [{"source": {"kind": "cluster", "id": cluster_id}}],
            }
        },
    }


def test_merge_preserves_both_branch_checkpoints() -> None:
    lookup = _lookup()
    ambiguous, unique = _branches(lookup)

    combined = combine_documents(ambiguous, unique)
    restored = Document.model_validate_json(combined.model_dump_json())

    assert restored.stage_runs[-2:] == (
        REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
        REPORTER_ROOT_LOOKUP_UNIQUE_LLM,
    )
    assert restored.get_stage(REPORTER_ROOT_LOOKUP) == lookup
    assert restored.get_stage(REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM) == ambiguous
    assert restored.roots[0].reporter_unique_review is not None
    assert restored.roots[1].reporter_ambiguous_review is not None
    assert not restored.roots[2].reporter_exact_lookup.response.clusters


def test_merge_rejects_different_lookup_checkpoints() -> None:
    ambiguous, _ = _branches(_lookup())
    _, unique = _branches(_lookup(twombly_id=999))

    with pytest.raises(ValueError, match="identical lookup checkpoint"):
        combine_documents(ambiguous, unique)


def test_overall_recall_includes_unresolved_gold_and_precision_requires_alignment() -> None:
    ambiguous, unique = _branches(_lookup())
    combined = combine_documents(ambiguous, unique)
    rows = tuple(
        _gold(root, combined, cluster_id)
        for root, cluster_id in zip(combined.roots, ("1", "22", "missing"), strict=True)
    )

    counts, details = score_document(combined, rows)
    fields = _summary(counts)["fields"]

    assert counts["gold_canonical_reporter_roots"] == 3
    for field in ("case_name", "court", "date"):
        assert fields[field] == {
            "correct": 1,
            "scored": 1,
            "gold": 3,
            "precision": 1.0,
            "recall": 0.3333,
        }
        assert [item["outcome"] for item in details if item["field"] == field] == [
            "correct",
            "candidate_alignment_unverified",
            "no_selected_candidate",
        ]


def test_repeated_reporter_occurrence_counts_as_locator_but_not_field_gold() -> None:
    ambiguous, unique = _branches(_lookup())
    combined = combine_documents(ambiguous, unique)
    roots = tuple(combined.roots)
    canonical = tuple(
        _gold(root, combined, cluster_id)
        for root, cluster_id in zip(roots, ("1", "11", "missing"), strict=True)
    )
    repeated = {
        "id": "repeated-occurrence",
        "root_id": canonical[0]["id"],
        "is_root": False,
        "kind": "FullCaseCitation",
        "locator": {"start": 0, "end": 4, "quote": combined.text[:4]},
    }

    counts, _ = score_document(combined, (*canonical, repeated))
    population = _summary(counts)["population"]

    assert population == {
        "reporter_locator_occurrences": 4,
        "reporter_identities": 3,
        "full_reporter_roots": 3,
        "table_of_authorities_roots": 0,
    }
    assert _summary(counts)["fields"]["case_name"]["gold"] == 3


def test_overall_report_shows_root_population_separately_from_field_denominator() -> None:
    summary = _summary(
        Counter(
            gold_reporter_locator_occurrences=4,
            gold_reporter_identities=3,
            gold_canonical_reporter_roots=2,
            gold_table_of_authorities_roots=1,
            case_name_correct=1,
            case_name_scored=1,
            case_name_gold=1,
        )
    )

    report = render_report(
        {"name": NAME, "sets": {"primary": summary}, "totals": summary},
        source_label="saved-score.json",
    )

    assert "| Set | Field | Full reporter locator roots | Precision | Recall |" in report
    assert "| primary | case name | 2 | 1/1 (100.0%) | 1/1 (100.0%) |" in report
    assert "4 annotated reporter locator occurrences represent 3 full-reporter roots" in report
    assert "including 1 first cited in a table of authorities" in report


def test_toa_root_uses_same_field_on_body_representative_but_not_changed_field() -> None:
    ambiguous, unique = _branches(_lookup())
    combined = combine_documents(ambiguous, unique)
    body = _gold(combined.roots[0], combined, "1")
    toa = deepcopy(body)
    toa["id"] = toa["root_id"] = "toa-root"
    toa["locator"] = {"start": 0, "end": 1, "quote": combined.text[:1]}
    toa["in_table_of_authorities"] = True
    body["id"] = "body-repeat"
    body["root_id"] = "toa-root"
    body["is_root"] = False
    body.pop("validation")

    counts, details = score_document(combined, (toa, body))
    assert counts["gold_canonical_reporter_roots"] == 1
    assert counts["gold_table_of_authorities_roots"] == 1
    assert all(_summary(counts)["fields"][field]["correct"] == 1 for field in ("case_name", "court", "date"))
    assert all(item["outcome"] == "correct" for item in details)

    toa["case_name"]["quote"] = "Different v. Name"
    counts, details = score_document(combined, (toa, body))
    assert _summary(counts)["fields"]["case_name"]["correct"] == 0
    assert next(item for item in details if item["field"] == "case_name")["outcome"] == "changed_occurrence"
