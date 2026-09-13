"""The second growth: leaves onto the roots that came back from validation.

Extraction grows roots; validation removes what reaches no case and settles what survives;
extraction grows leaves onto what it is handed. The contract between the two
halves is this module's subject -- what a removed root takes with it, and what a
surviving root keeps.
"""

from __future__ import annotations

import contextlib
import io
import json

from mellea_lrc.core.citations import is_leaf
from mellea_lrc.core.case_names import CaseName
from mellea_lrc.core.record import Node, Reads
from mellea_lrc.extraction import Relaxation, extract_from_plain_text, grow_leaves
from mellea_lrc.serialization import deserialize_extracted_document, serialize_extracted_document

TEXT = (
    "Bell Atl. Corp. v. Twombly , 550 U.S. 544 (2007). Id. at 570. "
    "The court went on. Twombly , 550 U.S. at 563. "
    "Ashcroft v. Iqbal , 556 U.S. 662 (2009). Iqbal , 556 U.S. at 678."
)


def _roots():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(TEXT, relaxation=Relaxation.FULL)


def _through_the_artifact(document, removed: set[str] = frozenset()):
    """Out as an artifact and back, which is how validation returns it."""
    payload = json.loads(json.dumps(serialize_extracted_document(document)))
    payload["citations"] = [c for c in payload["citations"] if c["citation_id"] not in removed]
    return deserialize_extracted_document(payload)


def test_the_first_growth_is_roots_only() -> None:
    document = _roots()

    assert not [c for c in document.citations if is_leaf(c.stated)]
    assert len(document.citations) == 2


def test_leaves_grow_onto_the_roots_that_came_back() -> None:
    returned = _through_the_artifact(_roots())

    grown = grow_leaves(returned)
    leaves = [c for c in grown.citations if is_leaf(c.stated)]

    assert len(leaves) == 3
    assert {c.root_id for c in leaves} <= {c.citation_id for c in returned.citations}


def test_a_root_removed_by_validation_takes_its_leaves() -> None:
    """A leaf whose root validation removed reaches nothing, and a leaf that
    reaches nothing cannot be built. So removing a root removes its leaves."""
    document = _roots()
    twombly = next(c for c in document.citations if c.stated.page == "544")

    grown = grow_leaves(_through_the_artifact(document, removed={twombly.citation_id}))
    leaves = [c for c in grown.citations if is_leaf(c.stated)]

    assert twombly.citation_id not in {c.citation_id for c in grown.citations}
    assert [c.matched_text for c in leaves] == ["556 U.S. at 678"]


def test_a_surviving_root_keeps_everything_validation_wrote_on_it() -> None:
    """The roots the document holds are kept as they are, not re-read.

    A citation's identifier is a hash of its span and the characters at it, so
    the second read produces the same ids -- which is what lets the roots
    already here be the ones the leaves point at, with their authority and their
    trace intact.
    """
    document = _roots()
    root = document.citations[0]
    root.observe(
        Node(
            node_id="identity",
            reads=Reads.RECORD,
            stage="identity",
            made_by="exact_locator_lookup",
            outcome="resolved",
        )
    )
    object.__setattr__(root, "authority_id", "authority-1")

    grown = grow_leaves(_through_the_artifact(document))
    same = next(c for c in grown.citations if c.citation_id == root.citation_id)

    assert same.authority_id == "authority-1"
    assert [node.stage for node in same.trace] == ["identity"]


def test_a_leaf_finds_a_root_by_the_name_stated_not_the_name_parsed() -> None:
    """Why the leaves are grown in a second pass.

    eyecite reads the party of `Huri v. Office of the Chief Judge of the Cir.
    Ct. of Cook Cnty. , 804 F.3d 826` as `Cnty.`, because the name search stops
    at the `Cnty.` in front of the citation. A later `Huri , supra` states no
    volume and no reporter, so the name is the only thing that can attach it,
    and against the parse there is no `Huri` to attach to. Correcting `stated`
    is what puts the name back, and attachment reads `stated`.
    """
    text = (
        "The elements are set out in Huri v. Office of the Chief Judge of the Cir. Ct. of "
        "Cook Cnty. , 804 F.3d 826, 833 (7th Cir. 2015). A different panel decided "
        "Yancick v. Hanna Steel Corp. , 653 F.3d 532 (7th Cir. 2011). "
        "See Huri , supra , at 834."
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
    huri = next(c for c in document.citations if c.stated.page == "826")
    assert "Huri" not in (huri.stated.case_name.text if huri.stated.case_name else "")

    assert not [c for c in grow_leaves(document).citations if is_leaf(c.stated)]

    huri.observe(
        huri.correcting(
            Node(
                node_id="name",
                reads=Reads.DOCUMENT,
                stage="identity",
                made_by="mellea_case_name_check",
                outcome="corrected",
            ),
            "case_name",
            CaseName(
                span=huri.stated.case_name.span if huri.stated.case_name else None,
                text="Huri v. Office of the Chief Judge of the Cir. Ct. of Cook Cnty.",
                plaintiff="Huri",
                defendant="Office of the Chief Judge of the Cir. Ct. of Cook Cnty.",
            ),
            reason="the parse stopped at the `Cnty.` in front of the citation",
        )
    )

    leaves = [c for c in grow_leaves(document).citations if is_leaf(c.stated)]
    assert [c.stated.antecedent for c in leaves] == ["Huri"]
    assert leaves[0].root_id == huri.citation_id
