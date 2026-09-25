"""Identity scoring uses saved stage decisions and stable gold denominators."""

from __future__ import annotations

import asyncio

from evaluations.render_identity_report import render_identity_report
from evaluations.score_identity import _summary, score_document
from mellea_lrc.api import Document, grow_roots, reporter_root_exact_lookup
from mellea_lrc.courtlistener import CourtListenerCitationLookup
from mellea_lrc.model.citations.judgments import IdentityVerdict

SOURCE = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."


class FakeLookupClient:
    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        assert (volume, reporter, page) == ("550", "U.S.", "544")
        return CourtListenerCitationLookup.model_validate(
            {
                "citation": "550 U.S. 544",
                "status": 200,
                "clusters": [
                    {
                        "id": 1,
                        "caseNameFull": "Bell Atlantic Corporation v. Twombly",
                        "court_id": "scotus",
                        "dateFiled": "2007-05-21",
                        "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
                    }
                ],
            }
        )


def _document(source: str = SOURCE) -> Document:
    roots = asyncio.run(grow_roots(Document.from_source(source), hunt_dockets=False))
    return reporter_root_exact_lookup(roots, client=FakeLookupClient())


def _gold(document: Document, *, label: str = "CORRECT_IDENTITY", gold_id: str = "gold-1") -> dict:
    (root,) = document.roots
    return {
        "id": gold_id,
        "root_id": gold_id,
        "is_root": True,
        "kind": "FullCaseCitation",
        "identifier": {"kind": "reporter"},
        "locator": {"start": root.locator_span.start, "end": root.locator_span.end},
        "validation": {"identity": {"label": label}},
    }


def test_exact_identity_score_uses_uppercase_gold_and_includes_missing_gold() -> None:
    document = _document()
    missed = {
        **_gold(document, gold_id="gold-2"),
        "locator": {"start": 0, "end": 5},
    }
    counts, details = score_document(document, (_gold(document), missed))

    assert counts["gold_reporter_roots"] == 2
    assert counts["gold_correct_reporter_roots"] == 2
    assert counts["correct_decisions"] == 1
    assert counts["scored_decisions"] == 1
    assert counts["missing_gold_reporter_roots"] == 1
    assert _summary(counts)["decision_precision"] == 1.0
    assert _summary(counts)["reporter_root_decision_recall"] == 0.5
    assert _summary(counts)["admission_recall"] == 0.5
    assert any(row.get("gold_root_id") == "gold-2" and row.get("reason") == "missing_root" for row in details)


def test_wrong_gold_is_a_false_admission() -> None:
    document = _document()
    counts, _ = score_document(document, (_gold(document, label="WRONG_IDENTITY"),))

    assert counts["incorrect_decisions"] == 1
    assert counts["gold_wrong_reporter_roots"] == 1
    assert counts["scored_admissions"] == 1
    assert counts["correct_admissions"] == 0
    assert _summary(counts)["decision_precision"] == 0.0


def test_unmatched_prediction_counts_against_precision_only_in_labeled_set() -> None:
    document = _document()
    labeled, _ = score_document(document, (), identity_labeled=True)
    unlabeled, _ = score_document(document, (), identity_labeled=False)

    assert labeled["scored_decisions"] == 1
    assert labeled["incorrect_decisions"] == 1
    assert unlabeled["scored_decisions"] == 0
    assert unlabeled["unscored_decisions"] == 1


def test_later_judgment_cannot_change_exact_stage_score() -> None:
    document = _document()
    (root,) = document.roots
    later = document.replace_citation(
        root.record("later_review").with_identity_judgment(IdentityVerdict.WRONG_IDENTITY)
    ).complete("later_review")

    counts, _ = score_document(later, (_gold(document),))
    assert counts["correct_decisions"] == 1
    assert later.roots[0].identity_judgments[-1].verdict is IdentityVerdict.WRONG_IDENTITY


def test_unannotated_representative_cannot_gain_credit_from_attached_leaf() -> None:
    document = _document(f"{SOURCE} {SOURCE}")
    (_, leaf) = document.full_locators
    gold = {
        **_gold(document),
        "locator": {"start": leaf.locator_span.start, "end": leaf.locator_span.end},
    }

    counts, details = score_document(document, (gold,))

    assert counts["unmatched_locator"] == 1
    assert counts["incorrect_decisions"] == 1
    assert counts["missing_gold_reporter_roots"] == 1
    assert any(row.get("reason") == "missing_root" for row in details)


def test_annotated_leaf_can_be_predicted_representative_of_same_gold_root() -> None:
    document = _document(f"{SOURCE} {SOURCE}")
    root, leaf = document.full_locators
    gold_leaf = {
        "id": "gold-leaf",
        "root_id": "gold-root",
        "is_root": False,
        "kind": "FullCaseCitation",
        "identifier": {"kind": "reporter"},
        "locator": {"start": root.locator_span.start, "end": root.locator_span.end},
    }
    gold_root = {
        **_gold(document, gold_id="gold-root"),
        "locator": {"start": leaf.locator_span.start, "end": leaf.locator_span.end},
    }

    counts, _ = score_document(document, (gold_leaf, gold_root))

    assert counts["correct_decisions"] == 1
    assert counts["reached_gold_reporter_roots"] == 1


def test_masked_gold_root_label_applies_to_unmasked_full_citation() -> None:
    document = _document()
    gold_leaf = {
        **_gold(document),
        "id": "body-leaf",
        "root_id": "masked-root",
        "is_root": False,
        "validation": {},
    }
    masked_root = {
        **_gold(document, gold_id="masked-root"),
        "locator": {"start": 0, "end": 5},
    }

    counts, _ = score_document(document, (gold_leaf,), identity_roots=(masked_root,))

    assert counts["gold_reporter_roots"] == 1
    assert counts["correct_decisions"] == 1
    assert counts["missing_gold_reporter_roots"] == 0


def test_merge_of_two_gold_roots_is_false_decision_and_two_misses() -> None:
    document = _document(f"{SOURCE} {SOURCE}")
    (_, leaf) = document.full_locators
    gold_second = {
        **_gold(document, gold_id="gold-2"),
        "locator": {"start": leaf.locator_span.start, "end": leaf.locator_span.end},
    }

    counts, _ = score_document(document, (_gold(document), gold_second))

    assert counts["conflicting_gold_roots"] == 1
    assert counts["incorrect_decisions"] == 1
    assert counts["missing_gold_reporter_roots"] == 2


def test_two_decided_roots_for_one_gold_root_receive_only_one_credit() -> None:
    before = asyncio.run(grow_roots(Document.from_source(f"{SOURCE} {SOURCE}"), hunt_dockets=False))
    (_, second) = before.full_locators
    split = before.replace_citation(second.record("split").with_root(second.id)).complete("split")
    document = reporter_root_exact_lookup(split, client=FakeLookupClient())
    first, second = document.full_locators
    gold_root = {
        "id": "gold-root",
        "root_id": "gold-root",
        "is_root": True,
        "kind": "FullCaseCitation",
        "identifier": {"kind": "reporter"},
        "locator": {"start": first.locator_span.start, "end": first.locator_span.end},
        "validation": {"identity": {"label": "CORRECT_IDENTITY"}},
    }
    gold_leaf = {
        "id": "gold-leaf",
        "root_id": gold_root["id"],
        "is_root": False,
        "kind": "FullCaseCitation",
        "identifier": {"kind": "reporter"},
        "locator": {"start": second.locator_span.start, "end": second.locator_span.end},
    }

    counts, _ = score_document(document, (gold_root, gold_leaf))

    assert first.id != second.id
    assert counts["correct_decisions"] == 1
    assert counts["duplicate_decision"] == 1
    assert counts["scored_decisions"] == 2
    assert counts["gold_reporter_roots"] == 1


def test_report_renders_saved_counts_without_rescoring() -> None:
    document = _document()
    counts, _ = score_document(document, (_gold(document),))
    summary = _summary(counts)
    report = render_identity_report(
        {"stage": "reporter_root_exact_lookup", "sets": {"primary": summary}, "totals": summary},
        source_label="saved/summary.json",
    )

    assert "1/1 (100.0%)" in report
    assert "saved/summary.json" in report
    assert "Decision recall" in report
