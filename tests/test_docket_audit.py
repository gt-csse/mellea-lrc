"""Docket admission is a separate pass between grouping and court writing."""

from dataclasses import replace

import pytest

from mellea_lrc.core.citations import DocketCitation
from mellea_lrc.extraction import audit_dockets, grow_roots, resolve_courts, stable
from mellea_lrc.extraction.reading import post_citation
from mellea_lrc.extraction.reading.docket_audit import audit_docket_citations
from mellea_lrc.extraction.structure.citation_tree import build_citation_tree
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.serialization import deserialize_document, serialize_document
from mellea_lrc.validation import initialize_validation


def _before_audit(text):
    return grow_roots(preprocess(text), rules=replace(stable(), docket_auditor=None, court_reader=None))


def test_audit_reads_court_without_writing_it() -> None:
    original = _before_audit("Smith v. Jones, No. 1:24-cv-00123 (D. Ariz. 2024).")
    audited = audit_dockets(original, stable())

    assert len(audited.active_citations) == 1
    assert audited.citations[0].stated.court is None
    assert audited.citations[0].source == original.citations[0].source
    assert original.citations[0].trace == ()
    assert audited.citations[0].trace[-1].details["reason"] == "explicit_court"

    resolved = resolve_courts(audited, stable())
    assert resolved.citations[0].stated.court == "azd"


@pytest.mark.parametrize(
    "text",
    [
        "Case No. 1:24-cv-00123 (MG) (Joint Administration Requested)",
        "Case 1:24-cv-00123 Document 10 Filed 01/01/24 Page 1 of 8",
        "Jennifer Smith (State Bar No. 1124201)",
        "The filing was submitted in Case No. 1:24-cv-00123.",
    ],
)
def test_unsupported_dockets_are_withdrawn_but_keep_their_locators(text: str) -> None:
    document = grow_roots(preprocess(text), rules=stable())

    assert len(document.locators) == 1
    assert document.active_citations == ()
    assert document.citations[0].withdrawn
    assert document.citations[0].trace[-1].details["reason"] == "no_citation_context"
    assert build_citation_tree(document).roots == ()
    assert initialize_validation(document).citations == ()
    recovered = deserialize_document(serialize_document(document))
    assert recovered.locators == document.locators
    assert recovered.citations == document.citations


def test_colocation_preserves_a_courtless_docket() -> None:
    document = _before_audit("Smith v. Jones, No. 1:24-cv-00123, 2024 WL 123456 (2024).")
    # The audit consumes the grouping layer's ids, independently of how that
    # layer discovered the group.
    grouped = tuple(
        replace(record, colocation_id=document.citations[0].citation_id) for record in document.citations
    )
    audited = audit_docket_citations(document.text, grouped)

    docket = next(record for record in audited if isinstance(record.stated, DocketCitation))
    assert not docket.withdrawn
    assert docket.stated.court is None
    assert docket.trace[-1].details["reason"] == "colocated_reporter"


def test_audit_and_court_writer_scan_independently(monkeypatch) -> None:
    calls = []
    original = post_citation.court_for_docket

    def scan(text, end, *, stop=None):
        calls.append(end)
        return original(text, end, stop=stop)

    monkeypatch.setattr(post_citation, "court_for_docket", scan)
    document = grow_roots(preprocess("Smith v. Jones, No. 1:24-cv-00123 (D. Ariz. 2024)."), rules=stable())
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert document.active_citations[0].stated.court == "azd"


def test_court_scan_starts_after_the_last_group_member() -> None:
    text = (
        "Smith v. Jones, No. 1:24-cv-00123, 390 U.S. 727, 88 S.Ct. 1323, "
        "20 L.Ed.2d 262, 2024 WL 123456789, 2024 U.S. Dist. LEXIS 123456 (D. Ariz. 2024)."
    )
    document = _before_audit(text)
    grouped = tuple(
        replace(record, colocation_id=document.citations[0].citation_id) for record in document.citations
    )
    docket = next(record for record in grouped if isinstance(record.stated, DocketCitation))
    assert text.index("(D. Ariz.") - docket.locator_span.end > 70
    court = post_citation.docket_court(document.text, docket, grouped)
    assert court is not None and court.court_id == "azd"


def test_audit_does_not_borrow_the_next_unrelated_citations_court() -> None:
    document = _before_audit("Smith, No. 1:24-cv-00123; Jones, 556 U.S. 662 (D. Ariz. 2024).")
    ungrouped = tuple(replace(record, colocation_id=None) for record in document.citations)
    audited = audit_docket_citations(document.text, ungrouped)

    docket = next(record for record in audited if isinstance(record.stated, DocketCitation))
    assert docket.withdrawn


def test_audit_is_opt_in_and_can_be_overridden() -> None:
    document = _before_audit("Case No. 1:24-cv-00123")
    assert audit_dockets(document) is document
    rules = replace(stable(), docket_auditor=lambda text, citations: tuple(citations))
    assert audit_dockets(document, rules).active_citations == document.citations


def test_replaying_an_audit_does_not_duplicate_its_trace() -> None:
    document = grow_roots(preprocess("Case No. 1:24-cv-00123"), rules=stable())
    assert audit_dockets(document, stable()).citations == document.citations
