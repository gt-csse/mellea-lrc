"""A logged field keeps every value it has held, and who set it.

`case_name` is the only field logged today. What is tested here is the contract
rather than that one field: the log opens when the citation is built, the last
touch is the value, and nothing a reader writes erases what it replaced.
"""

from __future__ import annotations

from dataclasses import replace

from mellea_lrc.core.case_names import CaseName
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.core.field_log import RULES
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.types import CASE_NAME, ExtractedCitation


def _name(
    start: int, end: int, text: str, plaintiff: str | None = None, defendant: str | None = None
) -> CaseName:
    return CaseName(span=Span(start=start, end=end), text=text, plaintiff=plaintiff, defendant=defendant)


def _citation(case_name: CaseName | None = None, **extra: object) -> ExtractedCitation:
    return ExtractedCitation(
        citation_id="c1",
        full_span=Span(start=0, end=30),
        locator_span=Span(start=10, end=22),
        matched_text="550 U.S. 544",
        citation=FullCaseCitation(volume="550", reporter="U.S.", page="544"),
        case_name_read=case_name,
        **extra,
    )


def test_the_log_opens_with_what_built_the_citation() -> None:
    name = _name(0, 9, "Twombly", defendant="Twombly")
    citation = _citation(name)
    assert citation.case_name == name
    assert citation.case_name_span == name.span
    assert [(touch.by, touch.value) for touch in citation.field_log.history(CASE_NAME)] == [(RULES, name)]


def test_a_citation_with_no_name_read_opens_its_log_with_none() -> None:
    """`None` is a value, so writing a name over it is visibly an overwrite."""
    citation = _citation(None)
    assert citation.case_name_span is None
    assert citation.field_log.history(CASE_NAME) == (citation.field_log.latest(CASE_NAME),)
    assert citation.field_log.latest(CASE_NAME).value is None


def test_a_reader_writing_a_name_over_none_is_an_overwrite_on_the_record() -> None:
    citation = _citation(None)
    found = _name(40, 62, "Bell Atl. Corp. v. Twombly", "Bell Atl. Corp.", "Twombly")
    citation.record_case_name(found, by="adjudicate_case_name", reason="names 550 U.S. 544")
    assert citation.case_name == found
    assert citation.case_name.plaintiff == "Bell Atl. Corp."
    history = citation.field_log.history(CASE_NAME)
    assert [(touch.by, touch.value) for touch in history] == [
        (RULES, None),
        ("adjudicate_case_name", found),
    ]
    assert history[-1].reason == "names 550 U.S. 544"


def test_a_reader_writing_over_a_name_keeps_the_one_it_replaced() -> None:
    read = _name(0, 9, "Twombly", defendant="Twombly")
    citation = _citation(read)
    fuller = _name(0, 26, "Bell Atl. Corp. v. Twombly", "Bell Atl. Corp.", "Twombly")
    citation.record_case_name(fuller, by="adjudicate_case_name")
    assert citation.case_name == fuller
    assert [touch.value for touch in citation.field_log.history(CASE_NAME)] == [read, fuller]


def test_a_citation_that_says_a_reader_built_it_says_so_in_its_log() -> None:
    citation = _citation(None, read_by="adjudication")
    assert citation.field_log.history(CASE_NAME)[0].by == "adjudication"


def test_replacing_a_field_keeps_the_history_and_does_not_reopen_it() -> None:
    """`replace` makes the same citation with one field changed, not a new one."""
    citation = _citation(None)
    found = _name(40, 62, "Bell Atl. Corp. v. Twombly", "Bell Atl. Corp.", "Twombly")
    citation.record_case_name(found, by="adjudicate_case_name")
    attributed = replace(citation, root_id="c0")
    assert attributed.root_id == "c0"
    assert attributed.case_name == found
    assert len(attributed.field_log.history(CASE_NAME)) == 2
