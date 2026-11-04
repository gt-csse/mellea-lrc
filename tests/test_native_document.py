"""Document collections preserve source-grounded citation histories and checkpoints."""

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from mellea_lrc.model import (
    Document,
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
    Rule,
    SourceFormat,
    SourceMetadata,
    Span,
    TableOfAuthoritiesComponent,
)
from mellea_lrc.model.citations import FullDocketCitation, FullReporterCitation, latest
from mellea_lrc.model.citations.fields import CitationDate, NotStatedField
from mellea_lrc.model.execution import CheckpointRun

TEXT = (
    "Préface. Alpha v. Beta, 556 U.S. 662, 678 (2009). "
    "Gamma v. Delta, Case No. 1:24-cv-00123, ECF No. 128 (D. Md. 2024). "
    "A later date was printed as February 30, 2024."
)


def span(quote):
    start = TEXT.index(quote)
    return Span(start, start + len(quote))


def preprocessed():
    return PreprocessedDocument(
        text=TEXT,
        source_metadata=SourceMetadata(path="brief.pdf", format=SourceFormat.PDF),
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.DOCLING,
            backend_version="test-version",
            rules=(Rule.TABLE_OF_AUTHORITIES,),
        ),
        index_spans=(TableOfAuthoritiesComponent(0, 8),),
    )


def empty():
    return Document.from_preprocessed(preprocessed())


def reporter(substage="discover", citation_id="reporter:0"):
    return FullReporterCitation.from_locator(
        citation_id=citation_id,
        substage=substage,
        source=TEXT,
        span=span("556 U.S. 662"),
    )


def docket(substage="discover", citation_id="docket:0"):
    return FullDocketCitation.from_locator(
        citation_id=citation_id,
        substage=substage,
        source=TEXT,
        span=span("Case No. 1:24-cv-00123"),
        number_span=span("1:24-cv-00123"),
    )


def discovered():
    return empty().add_citation(reporter()).complete_substage("discover")


def with_fields():
    document = discovered()
    citation = (
        document.citations[0]
        .record("read_fields")
        .with_case_name(TEXT, span("Beta"))
        .with_date(TEXT, span("2009"))
        .with_pin_cite(TEXT, span("678"))
    )
    return document.replace_citation(citation).complete_substage("read_fields")


def test_from_preprocessed_retains_source_coordinates_metadata_and_components():
    source = preprocessed()
    document = Document.from_preprocessed(source)

    assert document.text == source.text
    assert document.source_metadata == source.source_metadata
    assert document.preprocessing_metadata == source.preprocessing_metadata
    assert document.index_spans == source.index_spans
    assert document.source_path == "brief.pdf"
    assert document.citations == document.runs == document.substage_runs == ()
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_collection_addition_is_immutable_ordered_and_preserves_concrete_citation_types():
    original = empty()
    with_docket = original.add_citation(docket())
    document = with_docket.add_citation(reporter())

    assert original.citations == ()
    assert tuple(item.id for item in with_docket.citations) == ("docket:0",)
    assert tuple(item.id for item in document.citations) == ("reporter:0", "docket:0")
    restored = Document.model_validate_json(document.model_dump_json())
    assert restored == document
    assert isinstance(restored.citations[0], FullReporterCitation)
    assert isinstance(restored.citations[1], FullDocketCitation)


def test_citations_at_the_same_source_position_have_stable_identifier_order():
    document = empty().add_citation(reporter(citation_id="z")).add_citation(reporter(citation_id="a"))
    assert tuple(item.id for item in document.citations) == ("a", "z")
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_duplicate_identifiers_are_rejected_when_adding_or_loading():
    citation = reporter()
    document = empty().add_citation(citation)
    with pytest.raises(ValueError):
        document.add_citation(citation)
    payload = document.model_dump(mode="json")
    payload["citations"].append(payload["citations"][0])
    with pytest.raises(ValidationError):
        Document.model_validate_json(json.dumps(payload))


def test_native_loading_rejects_noncanonical_collection_order():
    document = empty().add_citation(docket()).add_citation(reporter())
    payload = document.model_dump(mode="json")
    payload["citations"].reverse()
    with pytest.raises(ValidationError):
        Document.model_validate_json(json.dumps(payload))


def test_discriminated_collection_rejects_an_unavailable_citation_kind():
    payload = empty().add_citation(reporter()).model_dump(mode="json")
    payload["citations"][0]["kind"] = "supra"
    with pytest.raises(ValidationError):
        Document.model_validate_json(json.dumps(payload))


def test_collection_rejects_a_citation_grounded_in_a_different_document():
    citation = reporter()
    different = preprocessed().model_dump(mode="python")
    different["text"] = TEXT.replace("556 U.S. 662", "999 U.S. 111")
    document = Document.model_validate(different)
    with pytest.raises(ValueError):
        document.add_citation(citation)


def test_native_loading_checks_superseded_quotes_against_the_owning_source():
    document = with_fields()
    corrected = document.citations[0].record("review").with_case_name(TEXT, span("Alpha v. Beta"))
    document = document.replace_citation(corrected).complete_substage("review")
    payload = document.model_dump(mode="json")
    payload["citations"][0]["case_name"][0]["quote"] = "Zeta"
    with pytest.raises(ValidationError):
        Document.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("construction", ("copy", "construct"))
def test_copied_or_constructed_citation_instances_cannot_bypass_local_history_validation(construction):
    citation = reporter()
    if construction == "copy":
        forged = citation.model_copy(update={"id": "foreign"})
    else:
        values = {name: getattr(citation, name) for name in type(citation).model_fields}
        forged = FullReporterCitation.model_construct(**{**values, "id": "foreign"})
    payload = empty().model_dump(mode="python")
    payload["citations"] = (forged,)
    with pytest.raises(ValidationError):
        Document.model_validate(payload)


@pytest.mark.parametrize("operation", ("add", "replace"))
def test_public_collection_updates_revalidate_a_copied_locator_before_using_its_source_site(operation):
    citation = reporter()
    forged = citation.model_copy(update={"locator": ()})
    document = empty() if operation == "add" else empty().add_citation(citation)
    with pytest.raises(ValueError):
        if operation == "add":
            document.add_citation(forged)
        else:
            document.replace_citation(forged)


def test_replacing_a_citation_appends_a_decision_without_mutating_earlier_documents():
    original = discovered()
    changed = original.citations[0].record("read_fields").with_case_name(TEXT, span("Alpha v. Beta"))
    document = original.replace_citation(changed)

    assert original.citations[0].case_name == ()
    assert len(original.citations[0].nodes) == 1
    assert document.citations[0] == changed
    assert len(document.citations[0].nodes) == 2
    assert document.get_substage("discover") == original


def test_replacement_rejects_unknown_identifiers_and_a_changed_concrete_type():
    document = empty().add_citation(reporter())
    with pytest.raises(KeyError):
        document.replace_citation(reporter(citation_id="missing"))
    with pytest.raises(ValueError):
        document.replace_citation(docket(citation_id="reporter:0"))


def test_replacement_cannot_remove_a_recorded_decision_or_field_history():
    document = with_fields()
    with pytest.raises(ValueError):
        document.replace_citation(reporter())
    citation = document.citations[0]
    payload = citation.model_dump(mode="python")
    payload["case_name"] = ()
    shortened = FullReporterCitation.model_validate(payload)
    with pytest.raises(ValueError):
        document.replace_citation(shortened)


def test_replacement_cannot_rewrite_an_earlier_grounded_reading():
    document = with_fields()
    payload = document.citations[0].model_dump(mode="python")
    alternative = reporter().record("read_fields").with_case_name(TEXT, span("Alpha v. Beta"))
    payload["case_name"] = alternative.case_name
    rewritten = FullReporterCitation.model_validate(payload)
    with pytest.raises(ValueError):
        document.replace_citation(rewritten)


def test_new_readings_cannot_be_backfilled_into_a_decision_already_saved_in_the_document():
    document = discovered()
    citation = document.citations[0].record("read_fields").with_case_name(TEXT, span("Beta"))
    document = document.replace_citation(citation)
    with pytest.raises(ValueError):
        document.replace_citation(citation.with_date(TEXT, span("2009")))


@pytest.mark.parametrize("operation", ("add", "replace"))
def test_completed_substages_cannot_acquire_later_citations_or_nodes(operation):
    document = discovered()
    with pytest.raises(ValueError):
        if operation == "add":
            document.add_citation(docket())
        else:
            document.replace_citation(document.citations[0].record("discover"))


def test_only_one_substage_can_have_pending_citation_changes():
    document = empty().add_citation(reporter(substage="reporters"))
    with pytest.raises(ValueError):
        document.add_citation(docket(substage="dockets"))
    with pytest.raises(ValueError):
        document.complete_substage("unrelated")


def test_native_loading_rejects_interleaved_pending_substages():
    payload = empty().model_dump(mode="python")
    payload["citations"] = (reporter(substage="reporters"), docket(substage="dockets"))
    with pytest.raises(ValidationError):
        Document.model_validate(payload)


def test_pending_changes_cannot_be_followed_by_an_earlier_completed_producer_on_loading():
    document = with_fields()
    payload = document.model_dump(mode="json")
    citation = payload["citations"][0]
    citation["nodes"].extend(
        (
            {"id": f"{citation['id']}:node:2", "substage": "pending_review"},
            {"id": f"{citation['id']}:node:3", "substage": "read_fields"},
        )
    )
    with pytest.raises(ValidationError):
        Document.model_validate_json(json.dumps(payload))


def test_native_loading_rejects_citation_nodes_that_move_backward_between_completed_substages():
    payload = with_fields().model_dump(mode="json")
    payload["citations"][0]["nodes"][0]["substage"] = "read_fields"
    payload["citations"][0]["nodes"][1]["substage"] = "discover"
    with pytest.raises(ValidationError):
        Document.model_validate_json(json.dumps(payload))


def test_no_change_completions_are_distinct_replayable_events():
    original = empty()
    first = original.complete_substage("first_noop")
    second = first.complete_substage("second_noop")
    loaded = Document.model_validate_json(second.model_dump_json())

    assert original.runs == ()
    assert first.runs == (CheckpointRun(name="first_noop"),)
    assert second.substage_runs == ("first_noop", "second_noop")
    assert loaded.get_substage("first_noop") == first
    assert loaded.get_substage("second_noop") == second
    assert first != second


def test_completion_events_are_unique_and_unknown_checkpoints_raise():
    document = discovered()
    with pytest.raises(ValueError):
        document.complete_substage("discover")
    payload = document.model_dump(mode="json")
    payload["runs"].append(payload["runs"][0])
    with pytest.raises(ValidationError):
        Document.model_validate_json(json.dumps(payload))
    with pytest.raises(KeyError):
        document.get_substage("not_run")


@pytest.mark.parametrize("name", ("", " ", "WrongCase", "read fields", "a.b.c.d"))
def test_checkpoint_events_require_valid_named_producers(name):
    with pytest.raises(ValidationError):
        CheckpointRun(name=name)


def test_roundtrip_replay_recovers_each_boundary_and_excludes_later_citations_and_pending_readings():
    discovered_document = discovered()
    noop = discovered_document.complete_substage("noop")
    read = noop.replace_citation(
        noop.citations[0]
        .record("read_fields")
        .with_case_name(TEXT, span("Beta"))
        .with_date(TEXT, span("2009"))
        .with_pin_cite(TEXT, span("678"))
    ).complete_substage("read_fields")
    reviewed = read.replace_citation(
        read.citations[0].record("review").with_case_name(TEXT, span("Alpha v. Beta"))
    ).complete_substage("review")
    with_docket = reviewed.add_citation(docket(substage="find_dockets")).complete_substage("find_dockets")
    pending = with_docket.replace_citation(
        with_docket.citations[0]
        .record("pending_review")
        .with_case_name_not_stated()
        .with_pin_cite_not_stated()
    )
    loaded = Document.model_validate_json(pending.model_dump_json())

    for name, expected in (
        ("discover", discovered_document),
        ("noop", noop),
        ("read_fields", read),
        ("review", reviewed),
        ("find_dockets", with_docket),
    ):
        assert loaded.get_substage(name) == expected
    assert len(loaded.get_substage("review").citations) == 1
    assert loaded.get_substage("discover").citations[0].pin_cite is None
    assert loaded.get_substage("read_fields").citations[0].get_pin_cite()[0].first == 678
    assert loaded.get_substage("review").citations[0].get_case_name().as_citation() == "Alpha v. Beta"
    assert isinstance(loaded.citations[0].case_name[-1], NotStatedField)
    assert isinstance(loaded.citations[0].pin_cite[-1], NotStatedField)
    with pytest.raises(KeyError):
        loaded.get_substage("pending_review")


def test_failed_latest_reading_and_explicit_absence_survive_replay_without_reviving_older_values():
    good = with_fields()
    invalid = good.replace_citation(
        good.citations[0].record("bad_date").with_date(TEXT, span("February 30, 2024"))
    ).complete_substage("bad_date")
    cleared = invalid.replace_citation(
        invalid.citations[0].record("clear_date").with_date_not_stated()
    ).complete_substage("clear_date")
    loaded = Document.model_validate_json(cleared.model_dump_json())

    assert latest(loaded.get_substage("read_fields").citations[0].date) == CitationDate(year=2009)
    failed = loaded.get_substage("bad_date").citations[0].date[-1]
    assert failed.quote == "February 30, 2024"
    assert not failed.normalizable
    with pytest.raises(ValueError):
        latest(loaded.get_substage("bad_date").citations[0].date)
    assert latest(loaded.citations[0].date) is None
    assert len(loaded.citations[0].date) == 3


def test_document_and_checkpoint_events_cannot_be_mutated_in_place():
    document = discovered()
    with pytest.raises(ValidationError):
        document.citations = ()
    with pytest.raises(ValidationError):
        document.runs[0].name = "changed"
    with pytest.raises(ValidationError):
        document.citations[0].id = "changed"


def test_document_import_does_not_load_processing_or_provider_modules():
    script = """
import sys
from mellea_lrc.model import Document
from mellea_lrc.model.execution import CheckpointRun

prefixes = (
    "mellea_lrc.extraction",
    "mellea_lrc.validation",
    "mellea_lrc.preprocessing",
    "mellea_lrc.providers",
    "mellea_lrc.llm",
    "mellea_lrc.workflows",
)
unexpected = [name for name in sys.modules if name.startswith(prefixes)]
assert not unexpected, unexpected
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("container", (tuple, list, iter))
def test_every_citation_iterable_revalidates_copied_nested_readings(container):
    citation = reporter()
    invalid_reading = citation.locator[0].model_copy(update={"unchecked_normalized": None})
    forged = citation.model_copy(update={"locator": (invalid_reading,)})
    payload = {**preprocessed().model_dump(mode="python"), "citations": container((forged,))}
    with pytest.raises(ValidationError):
        Document.model_validate(payload)
