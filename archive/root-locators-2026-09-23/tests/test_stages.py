"""Tests for the order the passes over a citation list run in.

A stated constraint that nothing checks is a comment. These assert the one the
sequence actually has, by running the stages the wrong way round and showing
what it costs -- which is not an error but a plausible wrong answer, the kind
this project is meant to be bad at producing.
"""

from __future__ import annotations

import contextlib
import io

from mellea_lrc.extraction.eyecite_extractor import extract_from_plain_text
from mellea_lrc.extraction.stages import STAGES, refine
from mellea_lrc.model.citations import FullCaseCitation
from mellea_lrc.model.extraction_metadata import Relaxation
from mellea_lrc.model.operations import record_colocation
from mellea_lrc.model.record import Node, Reads

# One decision in three reporters, with the single date after the last of them.
_PARALLEL = "St. Amant v. Thompson, 390 U.S. 727, 731, 88 S.Ct. 1323, 20 L.Ed.2d 262 (1968)."


def _citations(text: str):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text, relaxation=Relaxation.FULL, with_leaves=True).citations


def test_the_sequence_is_the_one_the_docstring_describes() -> None:
    assert [stage.name for stage in STAGES] == [
        "colocation",
        "case_names",
        "docket_audit",
        "courts",
        "dates",
        "pin_cites",
        "root",
    ]


def test_every_stage_says_why_it_runs_where_it_does() -> None:
    """The reason is the thing worth keeping; a stage without one is a call."""
    assert all(stage.why.strip() for stage in STAGES)


def test_in_order_a_parallel_citation_keeps_the_year_it_reaches_for() -> None:
    dates = {
        item.fields.date.year for item in _citations(_PARALLEL) if isinstance(item.fields, FullCaseCitation)
    }

    assert dates == {"1968"}


def test_without_colocation_first_the_parallel_citation_loses_it() -> None:
    """post_citation bounds each search by co-location, so it cannot run first.

    Running the stages reversed is simulated by clearing the ids, because that
    is the state the list is in before `colocation` has run: no citation knows
    it has a parallel neighbour. Each member then stops at the next citation
    instead of reading across it, and the two that are not last in the run reach
    no date at all.
    """
    unmarked = tuple(
        record_colocation(
            item,
            None,
            Node(
                f"test:ungroup:{item.citation_id}",
                Reads.RECORD,
                "stage_order_test",
                __name__,
                "ungrouped",
            ),
        )
        for item in _citations(_PARALLEL)
    )
    dates = next(stage for stage in STAGES if stage.name == "dates")
    without = dates.run(_PARALLEL, unmarked)

    dates = {
        item.fields.date.year if item.fields.date else None
        for item in without
        if isinstance(item.fields, FullCaseCitation)
    }

    assert None in dates
    assert dates != {"1968"}


def test_refine_runs_every_stage() -> None:
    """Whatever the sequence holds, `refine` applies all of it."""
    raw = _citations(_PARALLEL)
    stepped = tuple(raw)
    for stage in STAGES:
        stepped = stage.run(_PARALLEL, stepped)

    assert refine(_PARALLEL, raw) == stepped


def test_the_authority_is_written_onto_every_citation_that_has_one() -> None:
    """The chain's answer, recorded rather than left to be recomputed."""
    text = "Doe v. Megless, 654 F.3d 404, 408 (3d Cir. 2011). Id. at 409."
    citations = _citations(text)
    full = next(c for c in citations if isinstance(c.fields, FullCaseCitation))
    reference = next(c for c in citations if c.fields.kind.value == "IdCitation")

    assert full.root_id == full.citation_id
    assert reference.root_id == full.citation_id


def test_a_leaf_with_nothing_before_it_is_not_grown() -> None:
    """Not attributed used to be an answer the field carried; now it is absence.

    An `Id.` opening a document points at nothing, and a leaf is built from a
    root or not at all -- so what the record says about it is that it is not
    there. See `docs/Extraction.md`, "Roots first, leaves after validation".
    """
    citations = _citations("The rule is settled. Id. at 409.")

    assert not [c for c in citations if c.fields.kind.value == "IdCitation"]
