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

from mellea_lrc.extraction.eyecite_extractor import extract_from_plain_text, grow_leaves
from mellea_lrc.extraction.structure.withdrawal import withdraw_leaves_of_withdrawn_roots
from mellea_lrc.model.citations import CitationField, is_leaf
from mellea_lrc.model.document import Document
from mellea_lrc.model.extraction_metadata import Relaxation
from mellea_lrc.model.findings import FindingKind
from mellea_lrc.model.operations import (
    judge_citation,
    observe_citation,
    resolve_citation,
    update_field,
    withdraw_citation,
    withdraw_subtree,
)
from mellea_lrc.model.record import (
    UNJUDGED,
    WITHDRAWN,
    WITHDRAWN_HEAD_ID,
    Node,
    OperationKind,
    Question,
    Reads,
    Resolution,
)

ORPHAN = "The court disagreed. DCD Programs , 833 F.2d at 186. That principle applies."
WHOLE = (
    "Ashcroft v. Iqbal , 556 U.S. 662 (2009). Id. at 678. "
    "Bell Atl. Corp. v. Twombly , 550 U.S. 544 (2007). Twombly , 550 U.S. at 570."
)


def _read(text: str):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text, relaxation=Relaxation.FULL)


def _through_the_artifact(document):
    return Document.model_validate(json.loads(json.dumps(document.model_dump(mode="json"))))


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
    assert not [c for c in grown.citations if is_leaf(c.fields)]
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
    observe_citation(
        record,
        Node(
            node_id="lookup",
            reads=Reads.RECORD,
            stage="identity",
            made_by="exact_locator_lookup",
            outcome="found",
            details={"cluster_id": "145875", "citations": [{"volume": "556"}]},
        ),
    )
    back = _through_the_artifact(document)
    assert next(node for node in back.citations[0].trace if node.node_id == "lookup").details == {
        "cluster_id": "145875",
        "citations": [{"volume": "556"}],
    }


def test_snapshot_isolates_citation_state_and_nested_evidence() -> None:
    document = _read(WHOLE)
    source_details = {"candidates": [{"cluster_id": "145875"}]}
    observe_citation(
        document.citations[0],
        Node(
            node_id="snapshot_lookup",
            reads=Reads.RECORD,
            stage="identity",
            made_by="exact_locator_lookup",
            outcome="found",
            details=source_details,
        ),
    )
    checkpoint = document.snapshot()

    document.citations[0].authority_id = "new-authority"
    source_details["candidates"][0]["cluster_id"] = "changed"

    assert checkpoint.citations[0] is not document.citations[0]
    assert checkpoint.citations[0].authority_id is None
    assert checkpoint.citations[0].trace[-1].details == {"candidates": [{"cluster_id": "145875"}]}


def test_a_withdrawn_citation_stays_in_the_document() -> None:
    """Reclassifying is a node, not a deletion.

    `root_id` and `authority_id` name citation ids, so a record that goes away
    takes every reference to it with it.
    """
    document = _read(WHOLE)
    record = document.citations[0]
    assert not record.withdrawn
    withdraw_citation(
        record,
        Node(
            node_id="identity",
            reads=Reads.RECORD,
            stage="identity",
            made_by="identity_scope",
            outcome=WITHDRAWN,
            message="the span is a record entry, not a citation to a case",
        ),
    )
    assert record.withdrawn
    assert record.root_id == WITHDRAWN_HEAD_ID
    assert record.root_link_node_id == "identity"
    back = _through_the_artifact(document)
    assert [c.citation_id for c in back.citations] == [c.citation_id for c in document.citations]
    assert back.citations[0].withdrawn


def test_a_node_that_read_a_record_still_cannot_correct_the_filing() -> None:
    """`details` changes nothing about what a node may write."""
    with pytest.raises(ValueError, match="cannot update filing fields"):
        update_field(
            _read(WHOLE).citations[0],
            Node(
                node_id="lookup",
                reads=Reads.RECORD,
                stage="identity",
                made_by="exact_locator_lookup",
                outcome="found",
                details={"cluster_id": "145875"},
            ),
            CitationField.COURT,
            "ca6",
            reason="the archive says so",
        )


def test_every_citation_carries_a_judgement_from_the_moment_it_is_read() -> None:
    """Absent and unmade are different, and a reader should never have to guess."""
    document = _read(WHOLE)
    for record in document.citations:
        assert [record.judgement(q).outcome for q in Question] == [UNJUDGED] * len(Question)
    assert document.citations[0].judgement(Question.IDENTITY).node_id is None


def test_a_judgement_is_written_with_the_node_that_reached_it() -> None:
    """Not found by searching the trace for whichever node was the aggregation."""
    document = _read(WHOLE)
    record = document.citations[0]
    judge_citation(
        record,
        Node(
            node_id="identity:aggregate",
            reads=Reads.RECORD,
            stage="identity",
            made_by="identity_aggregation",
            outcome="resolved",
        ),
        Question.IDENTITY,
        "reaches_the_authority_it_names",
        message="the locator resolved and the name matches",
    )
    back = _through_the_artifact(document).citations[0]
    identity = back.judgement(Question.IDENTITY)
    assert identity.outcome == "reaches_the_authority_it_names"
    assert identity.node_id == "identity:aggregate"
    assert identity.message == "the locator resolved and the name matches"
    # The other questions are untouched, which is the point of one slot each.
    assert back.judgement(Question.PINPOINT).outcome == UNJUDGED


def test_field_updates_are_a_list_on_the_record_and_survive_the_artifact() -> None:
    document = _read(WHOLE)
    record = document.citations[0]
    before = record.fields.court
    update_field(
        record,
        Node(
            node_id="name",
            reads=Reads.DOCUMENT,
            stage="identity",
            made_by="mellea_case_name_check",
            outcome="corrected",
        ),
        CitationField.COURT,
        "ca7",
        reason="the filing writes the Seventh Circuit",
    )
    back = _through_the_artifact(document).citations[0]
    assert [(c.field, c.before, c.after, c.node_id) for c in back.field_updates if c.node_id == "name"] == [
        (CitationField.COURT, before, "ca7", "name")
    ]
    assert back.fields.court == "ca7"


def test_what_an_archive_found_survives_the_artifact() -> None:
    """`found` is what the leaf pass must not drop when it reads a document back."""
    document = _read(WHOLE)
    record = document.citations[0]
    node = Node(
        node_id="lookup",
        reads=Reads.RECORD,
        stage="identity",
        made_by="exact_locator_lookup",
        outcome="found",
    )
    resolve_citation(
        record,
        node,
        Resolution(
            cluster_id="145875",
            case_name="Ashcroft v. Iqbal",
            date_filed="2009-05-18",
            court_id="scotus",
            node_id=node.node_id,
            opinion_ids=("145875",),
            citations=("556 U.S. 662",),
        ),
    )
    back = _through_the_artifact(document).citations[0]
    assert back.found is not None
    assert back.found.cluster_id == "145875"
    assert back.found.citations == ("556 U.S. 662",)
    assert back.found.node_id == "lookup"


def test_growing_the_leaves_keeps_what_validation_settled() -> None:
    """The leaf pass rebuilds nothing about a root; it only adds leaves."""
    document = _read(WHOLE)
    root = next(c for c in document.citations if c.fields.page == "662")
    node = Node(
        node_id="lookup",
        reads=Reads.RECORD,
        stage="identity",
        made_by="exact_locator_lookup",
        outcome="found",
    )
    resolve_citation(
        root,
        node,
        Resolution(
            cluster_id="145875",
            case_name="Ashcroft v. Iqbal",
            date_filed="2009-05-18",
            court_id="scotus",
            node_id=node.node_id,
        ),
    )
    judge_citation(
        root,
        Node(
            node_id="identity:aggregate",
            reads=Reads.RECORD,
            stage="identity",
            made_by="identity_aggregation",
            outcome="resolved",
        ),
        Question.IDENTITY,
        "reaches_the_authority_it_names",
    )

    grown = grow_leaves(_through_the_artifact(document))
    kept = next(c for c in grown.citations if c.citation_id == root.citation_id)
    assert kept.found is not None
    assert kept.found.cluster_id == "145875"
    assert kept.judgement(Question.IDENTITY).outcome == "reaches_the_authority_it_names"
    assert root.citation_id in {c.root_id for c in grown.citations if is_leaf(c.fields)}


def test_pinpoint_answers_its_own_question_without_overwriting_identity() -> None:
    """A root that states a page is judged twice, on two different questions.

    A citation can reach the right case and misstate the page. One slot would
    make the second stage erase the first finding.
    """
    document = _read(WHOLE)
    record = document.citations[0]
    judge_citation(
        record,
        Node(
            node_id="identity:aggregate",
            reads=Reads.RECORD,
            stage="identity",
            made_by="identity_aggregation",
            outcome="resolved",
        ),
        Question.IDENTITY,
        "reaches_the_authority_it_names",
    )
    judge_citation(
        record,
        Node(
            node_id="pinpoint:check",
            reads=Reads.RECORD,
            stage="pinpoint",
            made_by="mellea_pinpoint_check",
            outcome="unsupported",
            details={"page": 678},
        ),
        Question.PINPOINT,
        "the_page_does_not_say_it",
    )

    back = _through_the_artifact(document).citations[0]
    assert back.judgement(Question.IDENTITY).outcome == "reaches_the_authority_it_names"
    assert back.judgement(Question.PINPOINT).outcome == "the_page_does_not_say_it"
    assert {"identity:aggregate", "pinpoint:check"} <= {node.node_id for node in back.trace}


def test_a_withdrawn_root_takes_its_leaves_and_retains_old_links_in_history() -> None:
    """Withdrawal reattaches a whole tree and retains its old links as events."""
    grown = grow_leaves(_read(WHOLE))
    root = next(c for c in grown.citations if c.fields.page == "662")
    leaves = [c for c in grown.citations if is_leaf(c.fields) and c.root_id == root.citation_id]
    assert leaves

    withdrawn = withdraw_subtree(
        grown,
        root.citation_id,
        Node(
            node_id="identity:scope",
            reads=Reads.RECORD,
            stage="identity",
            made_by="identity_scope",
            outcome=WITHDRAWN,
            message="the locator reaches no case",
        ),
    )
    assert not any(c.withdrawn for c in grown.citations)
    assert withdraw_leaves_of_withdrawn_roots(withdrawn.citations) == 0

    leaf_ids = {item.citation_id for item in leaves}
    for leaf in (c for c in withdrawn.citations if c.citation_id in leaf_ids):
        assert leaf.withdrawn
        assert leaf.root_id == WITHDRAWN_HEAD_ID
        assert any(
            operation.kind is OperationKind.ROOT_LINK and operation.after == root.citation_id
            for operation in leaf.operations
        )
        node = next(n for n in leaf.trace if n.node_id == leaf.root_link_node_id)
        assert node.depends_on == ("identity:scope",)
    # Every other citation is untouched.
    others = [c for c in withdrawn.citations if c.citation_id not in leaf_ids | {root.citation_id}]
    assert not any(c.withdrawn for c in others)


def test_the_sweep_is_idempotent_and_survives_the_artifact() -> None:
    grown = grow_leaves(_read(WHOLE))
    root = next(c for c in grown.citations if c.fields.page == "662")
    withdrawn = withdraw_subtree(
        grown,
        root.citation_id,
        Node(
            node_id="identity:scope",
            reads=Reads.RECORD,
            stage="identity",
            made_by="identity_scope",
            outcome=WITHDRAWN,
        ),
    )
    assert withdraw_leaves_of_withdrawn_roots(withdrawn.citations) == 0
    assert sum(c.withdrawn for c in withdrawn.citations) > 1

    back = _through_the_artifact(withdrawn)
    assert sum(1 for c in back.citations if c.withdrawn) == sum(c.withdrawn for c in withdrawn.citations)
    assert withdraw_leaves_of_withdrawn_roots(back.citations) == 0


def test_a_leaf_cannot_grow_onto_a_root_already_withdrawn() -> None:
    """A later pass reports the orphan instead of reviving a withdrawn root."""
    roots = _read(WHOLE)
    root = next(c for c in roots.citations if c.fields.page == "662")
    roots = withdraw_subtree(
        roots,
        root.citation_id,
        Node(
            node_id="identity:scope",
            reads=Reads.RECORD,
            stage="identity",
            made_by="identity_scope",
            outcome=WITHDRAWN,
        ),
    )
    grown = grow_leaves(roots)
    assert not any(c.root_id == root.citation_id for c in grown.active_citations)
    assert any(f.kind is FindingKind.UNGROWN_LEAF for f in grown.findings)


def test_a_withdrawn_citation_is_not_reported_to_the_evaluation() -> None:
    """Withdrawing has to cost nothing, or no stage will do it.

    The record stays so that nothing pointing at it breaks, and the arm is not
    claiming it any more, so the score does not see it.
    """
    from evaluations.extraction.tree import _from_rules

    document = grow_leaves(_read(WHOLE))
    root = next(c for c in document.citations if c.fields.page == "662")
    before = _from_rules(document, dockets=True)
    document = withdraw_subtree(
        document,
        root.citation_id,
        Node(
            node_id="identity:scope",
            reads=Reads.RECORD,
            stage="identity",
            made_by="identity_scope",
            outcome=WITHDRAWN,
            message="the span is a record entry, not a citation to a case",
        ),
    )
    withdraw_leaves_of_withdrawn_roots(document.citations)
    after = _from_rules(document, dockets=True)

    assert len(after) < len(before)
    gone = set(before) - set(after)
    assert (root.locator_span.start, root.locator_span.end) in gone
