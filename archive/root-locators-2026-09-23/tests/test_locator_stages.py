"""Tests for independently checkpointable complete-locator readings."""

from __future__ import annotations

import contextlib
import io
import json

from mellea_lrc.model.citations import DocketCitation, FullCaseCitation
from mellea_lrc.model.document import Document
from mellea_lrc.extraction.locator_stages import (
    DOCKET_RULE_STAGE,
    REPORTER_RULE_STAGE,
    REPORTER_SITE_STAGE,
    find_docket_locators,
    find_full_reporter_locators,
    mark_full_reporter_locator_hunting_skipped,
    resolve_colocations,
    start_locator_document,
)
from mellea_lrc.extraction.stages import resolve_case_names, resolve_courts, resolve_dates
from mellea_lrc.extraction.rules import stable
from mellea_lrc.preprocessing import preprocess


def _round_trip(document):
    return Document.model_validate(json.loads(json.dumps(document.model_dump(mode="json"))))


def _run(text: str):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = start_locator_document(preprocess(text), rules=stable())
        reporters = find_full_reporter_locators(document, rules=stable())
        dockets = find_docket_locators(_round_trip(reporters), rules=stable())
    return reporters, dockets


def test_rule_locator_checkpoints_hold_only_locator_data_and_trace_its_origin() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. Jan. 1, 2024)."
    reporters, dockets = _run(text)

    assert reporters.passes == (REPORTER_RULE_STAGE,)
    assert [type(record.fields) for record in reporters.citations] == [FullCaseCitation]
    reporter = reporters.citations[0]
    assert reporter.fields.case_name is None
    assert reporter.fields.court is None
    assert reporter.fields.date is None
    assert reporter.fields.pin_cite is None
    assert reporter.trace[0].stage == REPORTER_RULE_STAGE
    assert reporter.trace[0].details["locator"]["text"] == "2024 WL 1234567"

    assert dockets.passes == (REPORTER_RULE_STAGE, DOCKET_RULE_STAGE)
    assert {type(record.fields) for record in dockets.citations} == {FullCaseCitation, DocketCitation}
    assert all(record.colocation_id is None for record in dockets.citations)
    assert all(record.root_id is None for record in dockets.citations)
    docket = next(record for record in dockets.citations if isinstance(record.fields, DocketCitation))
    assert docket.fields.court is None
    assert docket.fields.date is None
    assert docket.fields.case_name is None
    assert docket.trace[0].stage == DOCKET_RULE_STAGE
    assert docket.trace[0].details["locator"]["text"] == "No. 1:24-cv-00123"


def test_checkpoints_round_trip_and_explicit_fields_follow_them() -> None:
    text = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. Jan. 1, 2024)."
    _reporters, dockets = _run(text)
    skipped = mark_full_reporter_locator_hunting_skipped(
        _round_trip(dockets), reason="Disabled for this run: low recovery yield relative to model cost."
    )
    restored = _round_trip(skipped)

    assert restored.passes == (REPORTER_RULE_STAGE, DOCKET_RULE_STAGE, REPORTER_SITE_STAGE)
    assert restored.nodes[0].stage == REPORTER_SITE_STAGE
    assert restored.nodes[0].outcome == "not_run"

    grouped = resolve_colocations(restored, rules=stable())
    assert grouped.passes[-1] == "colocation"
    assert all(record.colocation_id is not None for record in grouped.citations)

    settled = resolve_dates(resolve_courts(grouped, rules=stable()), rules=stable())
    reporter = next(record for record in settled.citations if isinstance(record.fields, FullCaseCitation))
    docket = next(record for record in settled.citations if isinstance(record.fields, DocketCitation))
    assert reporter.fields.court == "azd"
    assert reporter.fields.date is not None and reporter.fields.date.year == "2024"
    assert docket.fields.court == "azd"
    assert docket.fields.date is not None and docket.fields.date.year == "2024"


def test_appeals_court_parenthetical_uses_state_from_reporter() -> None:
    text = "Redhair v. Kinerk, 218 Ariz. 293, 297 (Ct. App. 2008)."
    _reporters, dockets = _run(text)
    grouped = resolve_colocations(dockets, rules=stable())
    settled = resolve_courts(grouped, rules=stable())

    assert len(settled.citations) == 1
    assert settled.citations[0].fields.court == "arizctapp"


def test_case_name_reader_fills_an_in_re_name_for_a_docket_root() -> None:
    text = "In re Muscletech Research and Dev. Inc., No. 1:06-bk-01147 (Bankr. S.D.N.Y. Jan. 18, 2006)."
    _reporters, dockets = _run(text)
    grouped = resolve_colocations(dockets, rules=stable())
    named = resolve_case_names(grouped, rules=stable())
    docket = next(record for record in named.citations if isinstance(record.fields, DocketCitation))

    assert docket.fields.case_name is not None
    assert docket.fields.case_name.text == "In re Muscletech Research and Dev. Inc."


def test_case_name_reader_tolerates_detached_punctuation_and_party_ampersands() -> None:
    examples = (
        (
            "Lazy Seven Coal Sales, Inc. v. Stone & Hinds, P.C. , 813 S.W.2d 400 (Tenn. 1991).",
            "Lazy Seven Coal Sales, Inc. v. Stone & Hinds, P.C.",
        ),
        (
            "Anderson v. Libby Lobby, Inc  ., 477 U.S. 242 (1986).",
            "Anderson v. Libby Lobby, Inc  .",
        ),
        (
            "See Hoover v. Langston Equip. Assocs., Inc., 123 F.3d 456 (5th Cir. 1997).",
            "Hoover v. Langston Equip. Assocs., Inc.",
        ),
        (
            "See, Dodona I, LLC v. Goldman, Sachs & Co., 300 F.R.D. 182 (S.D.N.Y. 2014).",
            "Dodona I, LLC v. Goldman, Sachs & Co.",
        ),
        (
            "Id . See Hrobowski v. Worthington Steel Co ., 358 F.3d 473 (7th Cir. 2004).",
            "Hrobowski v. Worthington Steel Co .",
        ),
    )
    for text, expected in examples:
        document = start_locator_document(preprocess(text), rules=stable())
        located = find_full_reporter_locators(document, rules=stable())
        grouped = resolve_colocations(located, rules=stable())
        named = resolve_case_names(grouped, rules=stable())
        assert named.citations[0].fields.case_name.text == expected


def test_case_name_reader_does_not_read_through_an_interleaved_docket_prefix() -> None:
    text = (
        "Robinson\n\nPlaintiff, Case No.                      \n\n"
        "v. Mo. Pac. R.R. Co ., 16 F.3d 1083 (10th Cir. 1994)."
    )
    document = start_locator_document(preprocess(text), rules=stable())
    located = find_full_reporter_locators(document, rules=stable())
    grouped = resolve_colocations(located, rules=stable())
    named = resolve_case_names(grouped, rules=stable())
    assert named.citations[0].fields.case_name is None


def test_case_name_reader_keeps_in_re_when_spacing_is_damaged() -> None:
    text = "In  re Example Holdings, Inc., No. 1:24-cv-00123 (D. Ariz. 2024)."
    _reporters, dockets = _run(text)
    grouped = resolve_colocations(dockets, rules=stable())
    named = resolve_case_names(grouped, rules=stable())
    docket = next(record for record in named.citations if isinstance(record.fields, DocketCitation))
    assert docket.fields.case_name is not None
    assert docket.fields.case_name.text == "In  re Example Holdings, Inc."
