"""Tests for independently checkpointable complete-locator readings."""

from __future__ import annotations

import contextlib
import io
import json

from mellea_lrc.core.citations import DocketCitation, FullCaseCitation
from mellea_lrc.extraction import (
    DOCKET_RULE_STAGE,
    REPORTER_RULE_STAGE,
    REPORTER_SITE_STAGE,
    find_docket_locators,
    find_full_reporter_locators,
    mark_full_reporter_locator_hunting_skipped,
    resolve_case_names,
    resolve_colocations,
    resolve_courts,
    resolve_dates,
    stable,
    start_locator_document,
)
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.serialization import deserialize_document, serialize_document


def _round_trip(document):
    return deserialize_document(json.loads(json.dumps(serialize_document(document))))


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
    assert [type(record.source) for record in reporters.citations] == [FullCaseCitation]
    reporter = reporters.citations[0]
    assert reporter.source.case_name is None
    assert reporter.source.court is None
    assert reporter.source.date is None
    assert reporter.source.pin_cite is None
    assert reporter.trace[0].stage == REPORTER_RULE_STAGE
    assert reporter.trace[0].details["locator"]["text"] == "2024 WL 1234567"

    assert dockets.passes == (REPORTER_RULE_STAGE, DOCKET_RULE_STAGE)
    assert {type(record.source) for record in dockets.citations} == {FullCaseCitation, DocketCitation}
    assert all(record.colocation_id is None for record in dockets.citations)
    assert all(record.root_id is None for record in dockets.citations)
    docket = next(record for record in dockets.citations if isinstance(record.source, DocketCitation))
    assert docket.source.court is None
    assert docket.source.date is None
    assert docket.source.case_name is None
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
    reporter = next(record for record in settled.citations if isinstance(record.source, FullCaseCitation))
    docket = next(record for record in settled.citations if isinstance(record.source, DocketCitation))
    assert reporter.stated.court == "azd"
    assert reporter.stated.date is not None and reporter.stated.date.year == "2024"
    assert docket.stated.court == "azd"
    assert docket.stated.date is not None and docket.stated.date.year == "2024"


def test_case_name_reader_fills_an_in_re_name_for_a_docket_root() -> None:
    text = "In re Muscletech Research and Dev. Inc., No. 1:06-bk-01147 (Bankr. S.D.N.Y. Jan. 18, 2006)."
    _reporters, dockets = _run(text)
    grouped = resolve_colocations(dockets, rules=stable())
    named = resolve_case_names(grouped, rules=stable())
    docket = next(record for record in named.citations if isinstance(record.source, DocketCitation))

    assert docket.stated.case_name is not None
    assert docket.stated.case_name.text == "In re Muscletech Research and Dev. Inc."
