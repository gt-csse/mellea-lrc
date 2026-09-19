"""A citation's record: what the rules read, what it is now, and the evidence.

`source` never moves, `stated` is the current reading, and every change to it
lives on the node that justified it -- so a change with no evidence is not a
thing that can be built, rather than a thing that is checked for.
"""

from __future__ import annotations

import pytest

from mellea_lrc.core.case_names import CaseName
from mellea_lrc.core.citations import FullCaseCitation, placed
from mellea_lrc.core.record import CitationRecord, Correction, Node, Reads
from mellea_lrc.core.spans import Span

READ = CaseName(span=Span(start=0, end=6), text="Hassan", defendant="Hassan")
FULLER = CaseName(
    span=Span(start=0, end=23),
    text="United States v. Hassan",
    plaintiff="United States",
    defendant="Hassan",
)


def _record() -> CitationRecord:
    return CitationRecord(
        citation_id="c1",
        source=placed(
            FullCaseCitation(volume="742", reporter="F.3d", page="104"),
            span=Span(start=0, end=30),
            locator_span=Span(start=9, end=21),
            matched_text="742 F.3d 104",
            case_name=READ,
        ),
    )


def _node(**extra: object) -> Node:
    return Node(
        node_id="case_name:0-6",
        reads=Reads.DOCUMENT,
        stage="case_name",
        made_by="adjudicate_case_name",
        outcome="names_a_citation",
        **extra,
    )


def test_a_record_starts_as_what_the_rules_read() -> None:
    record = _record()
    assert record.stated is record.source
    assert record.stated.case_name == READ
    assert record.corrections == ()


def test_a_correction_moves_stated_and_leaves_source_alone() -> None:
    record = _record()
    record.correct(_node(), "case_name", FULLER, reason="named in the sentence")

    assert record.stated.case_name == FULLER
    assert record.source.case_name == READ
    assert [(c.field, c.before, c.after) for c in record.corrections] == [("case_name", READ, FULLER)]


def test_a_correction_names_the_node_it_rests_on() -> None:
    """The change is on the record and the evidence is one lookup away.

    Both are written by the same call, so neither can go missing, and reading
    the history never means walking the trace's graph.
    """
    record = _record()
    record.correct(_node(), "case_name", FULLER, reason="named in the sentence")

    correction = record.corrections[-1]
    assert correction.reason == "named in the sentence"
    assert correction.node_id == "case_name:0-6"
    node = next(item for item in record.trace if item.node_id == correction.node_id)
    assert node.outcome == "names_a_citation"


def test_a_node_that_read_a_record_cannot_correct_what_the_filing_states() -> None:
    """An archive's answer belongs on `found`, and `correct` is the only door."""
    record = _record()
    with pytest.raises(ValueError, match="cannot correct what the filing states"):
        record.correct(
            Node(
                node_id="lookup",
                reads=Reads.RECORD,
                stage="identity",
                made_by="courtlistener",
                outcome="found",
            ),
            "case_name",
            FULLER,
            reason="the archive says so",
        )
    assert record.stated.case_name == READ
    assert record.corrections == ()


def test_a_correction_that_changes_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="must change the value"):
        Correction(field="case_name", before=READ, after=READ, reason="why", node_id="n")


def test_a_root_is_what_extraction_read_and_an_authority_is_what_a_lookup_found() -> None:
    record = _record()
    record.root_id = "c1"
    assert record.authority == "c1"
    record.authority_id = "c9"
    assert record.authority == "c9"
    assert record.root_id == "c1"


def test_model_reparse_provenance_is_a_direct_citation_field() -> None:
    record = _record()

    record.mark_stated_fields_reparsed_by_model()

    assert record.stated_fields_reparsed_by_model
