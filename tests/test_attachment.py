"""Which root a leaf points at, decided from `stated`.

Each test is one of the three ways `root_for` decides, and the refusals in
between: a leaf that matches nothing, or matches two roots and cannot be
narrowed, gets no root and is not grown at all.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import replace

from mellea_lrc.extraction import Relaxation
from mellea_lrc.extraction.eyecite_extractor import _read as _extract
from mellea_lrc.extraction.structure.attachment import root_for
from mellea_lrc.preprocessing import preprocess


def _read(text: str):
    """The roots of `text`, and every leaf in it, attached or not.

    `_read` is the parse both growths share, and it hands back the leaves
    before anything decides where they go -- including the ones nothing can
    attach, which the grown document does not hold.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document, leaves = _extract(preprocess(text), Relaxation.FULL)
    return document.citations, [citation for _, citation, _ in leaves]


def _page_of(roots, citation_id: str) -> str:
    return next(r.stated.page for r in roots if r.citation_id == citation_id)


def test_one_root_states_this_volume_so_the_short_form_is_its() -> None:
    roots, leaves = _read(
        "Ashcroft v. Iqbal , 556 U.S. 662 (2009). Bell Atl. Corp. v. Twombly , "
        "550 U.S. 544 (2007). The pleading standard. 556 U.S. at 678."
    )
    assert _page_of(roots, root_for(leaves[0], roots)) == "662"


def test_two_roots_share_a_volume_and_the_page_claimed_tells_them_apart() -> None:
    """Neither the volume nor the reporter separates these; 254 does."""
    roots, leaves = _read(
        "Anderson v. Liberty Lobby, Inc. , 477 U.S. 242 (1986). "
        "Celotex Corp. v. Catrett , 477 U.S. 317 (1986). "
        "A dispute is genuine. 477 U.S. at 254. The movant bears it. 477 U.S. at 323."
    )
    assert [_page_of(roots, root_for(leaf, roots)) for leaf in leaves] == ["242", "317"]


def test_a_short_form_naming_no_root_in_its_volume_is_refused() -> None:
    roots, leaves = _read(
        "Ashcroft v. Iqbal , 556 U.S. 662 (2009). See 410 U.S. at 116."
    )
    assert [root_for(leaf, roots) for leaf in leaves] == [None]


def test_id_takes_the_root_of_the_citation_before_it() -> None:
    roots, leaves = _read(
        "Bell Atl. Corp. v. Twombly , 550 U.S. 544 (2007). Id. at 570."
    )
    attached = []
    for leaf in leaves:
        attached.append(root_for(leaf, roots, before=roots + tuple(attached)))
    assert _page_of(roots, attached[0]) == "544"


def test_id_claiming_a_page_its_antecedent_cannot_hold_is_refused() -> None:
    """`Id. at 2011` after a case reported at 544 is not that case's page."""
    roots, leaves = _read(
        "Bell Atl. Corp. v. Twombly , 550 U.S. 544 (2007). Id. at 2011."
    )
    assert [root_for(leaf, roots, before=roots) for leaf in leaves] == [None]


def test_supra_is_matched_by_name_alone() -> None:
    roots, leaves = _read(
        "Ashcroft v. Iqbal , 556 U.S. 662 (2009). Bell Atl. Corp. v. Twombly , "
        "550 U.S. 544 (2007). See Iqbal , supra , at 678."
    )
    assert _page_of(roots, root_for(leaves[0], roots)) == "662"


def test_supra_whose_name_matches_no_root_is_refused() -> None:
    roots, leaves = _read(
        "Ashcroft v. Iqbal , 556 U.S. 662 (2009). See Matsushita , supra , at 587."
    )
    assert [root_for(leaf, roots) for leaf in leaves] == [None]


def test_nothing_is_read_from_source() -> None:
    """The parse is not consulted, so a name only `source` holds attracts nothing.

    `stated` starts equal to `source`, so this is visible only once they differ.
    Here the name is taken off `stated` alone; `source` still reads `Ashcroft v.
    Iqbal`, and `Iqbal , supra` no longer reaches it.
    """
    roots, leaves = _read(
        "Ashcroft v. Iqbal , 556 U.S. 662 (2009). See Iqbal , supra , at 678."
    )
    root = roots[0]
    assert root_for(leaves[0], roots) == root.citation_id

    object.__setattr__(
        root,
        "stated",
        replace(root.stated, case_name=None, plaintiff=None, defendant=None),
    )
    assert root.source.case_name is not None
    assert root_for(leaves[0], roots) is None


def test_a_case_written_in_full_twice_is_one_root_and_the_first_is_it() -> None:
    """Four full citations of `Burrell` are one authority stated four times.

    Volume and reporter reach all four, the name reaches all four, and the page
    cannot separate them because they all begin at 408. They disagree about
    nothing, so nothing is being guessed: the occurrence that introduced the
    case is the root and the rest are returns to it.
    """
    roots, leaves = _read(
        "Burrell v. Dr. Pepper/Seven Up Bottling Grp., L.P. , 482 F.3d 408, 412 (5th Cir. 2007). "
        "The court went on. Burrell v. Dr. Pepper/Seven Up Bottling Grp., L.P. , 482 F.3d 408 "
        "(5th Cir. 2007). And later still, 482 F.3d at 414."
    )
    first = min(roots, key=lambda record: record.full_span.start)
    assert root_for(leaves[0], roots) == first.citation_id


def test_a_pin_cite_no_root_can_hold_does_not_refuse_the_only_case_in_the_volume() -> None:
    """`482 F.3d at 41215` is page 412 with a margin line number stuck to it.

    No root begins within reach of 41215, but every root in the volume states
    `482 F.3d 408`, so which case is meant was never in doubt. The page narrows
    between candidates; it does not veto the only answer.
    """
    roots, leaves = _read(
        "Burrell v. Dr. Pepper/Seven Up Bottling Grp., L.P. , 482 F.3d 408 (5th Cir. 2007). "
        "Burrell v. Dr. Pepper/Seven Up Bottling Grp., L.P. , 482 F.3d 408 (5th Cir. 2007). "
        "See 482 F.3d at 41215."
    )
    assert root_for(leaves[0], roots) is not None


def test_id_means_the_citation_before_it_and_not_the_last_case() -> None:
    """A filing writing `§ 2529 , Id., at 300` points at the section.

    eyecite reads `§ 2529` as a citation of its own, and it reaches no case, so
    the `Id.` after it reaches no case either. Walking past it to the case two
    sentences back would invent an attribution the filing never made.
    """
    roots, leaves = _read(
        "Anderson v. Liberty Lobby, Inc. , 477 U.S. 242 (1986). "
        "See 10A Charles Alan Wright & Arthur R. Miller, Federal Practice and Procedure "
        "§ 2529, Id., at 300."
    )
    assert [type(leaf).__name__ for leaf in leaves] == ["IdCitation"]
    before = [r for r in roots if r.full_span.start < leaves[0].span.start]
    assert root_for(leaves[0], roots, before=before) is None
