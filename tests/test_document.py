"""One document object, appended to by every stage.

`docs/Document.md` is the design these assert: a record is a citation and
nothing else holds a citation's state; what belongs to no citation is a
finding; what has run is written down; and nothing is ever deleted.
"""

from __future__ import annotations

import contextlib
import io
import json

import pytest

from mellea_lrc.core.citations import is_leaf
from mellea_lrc.core.findings import FindingKind
from mellea_lrc.core.record import WITHDRAWN, Node, Reads
from mellea_lrc.extraction import Relaxation, extract_from_plain_text, grow_leaves
from mellea_lrc.serialization import deserialize_extracted_document, serialize_extracted_document

ORPHAN = "The court disagreed. DCD Programs , 833 F.2d at 186. That principle applies."
WHOLE = (
    "Ashcroft v. Iqbal , 556 U.S. 662 (2009). Id. at 678. "
    "Bell Atl. Corp. v. Twombly , 550 U.S. 544 (2007). Twombly , 550 U.S. at 570."
)


def _read(text: str):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text, relaxation=Relaxation.FULL)


def _through_the_artifact(document):
    return deserialize_extracted_document(json.loads(json.dumps(serialize_extracted_document(document))))


def test_the_document_says_which_passes_have_run() -> None:
    """A document with roots and no leaves is unfinished, not short of short forms."""
    roots = _read(WHOLE)
    assert roots.passes == ("roots",)
    assert grow_leaves(roots).passes == ("roots", "leaves")


def test_growing_the_leaves_twice_is_one_pass_that_ran_again() -> None:
    assert grow_leaves(grow_leaves(_read(WHOLE))).passes == ("roots", "leaves")


def test_a_leaf_that_reaches_no_root_is_reported_rather_than_dropped() -> None:
    """`833 F.2d at 186` states a page of a case this filing never identifies.

    It cannot be built -- a leaf with no root is a citation of nothing -- and it
    is a defect the project reports, so it leaves the pass as a finding.
    """
    grown = grow_leaves(_read(ORPHAN))
    assert not [c for c in grown.citations if is_leaf(c.stated)]
    assert [f.kind for f in grown.findings] == [FindingKind.UNGROWN_LEAF]
    finding = grown.findings[0]
    assert finding.citation is not None
    assert finding.citation.matched_text == "833 F.2d at 186"
    assert finding.stage == "extraction"


def test_a_document_whose_leaves_have_not_grown_reports_nothing_yet() -> None:
    """The absence of a finding is not a claim that there is nothing to find."""
    assert _read(ORPHAN).findings == ()


def test_findings_survive_the_artifact() -> None:
    grown = grow_leaves(_read(ORPHAN))
    assert _through_the_artifact(grown) == grown


def test_a_node_carries_whatever_the_stage_that_made_it_keeps() -> None:
    """`details` is written back verbatim and nothing in `core` reads it."""
    document = _read(WHOLE)
    record = document.citations[0]
    record.observe(
        Node(
            node_id="lookup",
            reads=Reads.RECORD,
            stage="identity",
            made_by="exact_locator_lookup",
            outcome="found",
            details={"cluster_id": "145875", "citations": [{"volume": "556"}]},
        )
    )
    back = _through_the_artifact(document)
    assert back.citations[0].trace[0].details == {
        "cluster_id": "145875",
        "citations": [{"volume": "556"}],
    }


def test_a_withdrawn_citation_stays_in_the_document() -> None:
    """Reclassifying is a node, not a deletion.

    `root_id` and `authority_id` name citation ids, so a record that goes away
    takes every reference to it with it.
    """
    document = _read(WHOLE)
    record = document.citations[0]
    assert not record.withdrawn
    record.observe(
        Node(
            node_id="identity",
            reads=Reads.RECORD,
            stage="identity",
            made_by="identity_scope",
            outcome=WITHDRAWN,
            message="the span is a record entry, not a citation to a case",
        )
    )
    assert record.withdrawn
    back = _through_the_artifact(document)
    assert [c.citation_id for c in back.citations] == [c.citation_id for c in document.citations]
    assert back.citations[0].withdrawn


def test_a_node_that_read_a_record_still_cannot_correct_the_filing() -> None:
    """`details` changes nothing about what a node may write."""
    with pytest.raises(ValueError, match="cannot correct what the filing states"):
        _read(WHOLE).citations[0].correcting(
            Node(
                node_id="lookup",
                reads=Reads.RECORD,
                stage="identity",
                made_by="exact_locator_lookup",
                outcome="found",
                details={"cluster_id": "145875"},
            ),
            "court",
            "ca6",
            reason="the archive says so",
        )
