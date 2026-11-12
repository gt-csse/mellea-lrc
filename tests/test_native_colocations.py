"""Colocations preserve occurrence histories and bound shared reading context."""

import pytest

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction import (
    find_docket_locators,
    find_full_reporter_locators,
    resolve_colocations,
)
from mellea_lrc.model import Document, PreprocessedDocument, PreprocessingMetadata, SourceMetadata
from mellea_lrc.model.citation_windows import after, before
from mellea_lrc.model.citations.history import RelationshipUpdate, latest
from mellea_lrc.model.execution import CheckpointRun

REPORTERS = "grow_roots.discover_root_locators.find_full_reporter_locators"
DOCKETS = "grow_roots.discover_root_locators.find_docket_locators"
COLOCATIONS = "grow_roots.read_root_fields.resolve_colocations"
MIXED = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. 2024)."


def document(text):
    return Document.from_preprocessed(
        PreprocessedDocument(
            text=text,
            source_metadata=SourceMetadata(path="filing.txt"),
            preprocessing_metadata=PreprocessingMetadata(),
        )
    )


def discovered(text):
    return find_docket_locators(find_full_reporter_locators(document(text)))


def group_quotes(result):
    by_id = {citation.id: citation for citation in result.citations}
    return tuple(
        tuple(by_id[identifier].locator[0].quote for identifier in group.citation_ids)
        for group in result.colocations
    )


def test_mixed_locator_group_preserves_source_and_occurrence_histories():
    original = discovered(MIXED)
    result = resolve_colocations(original)

    assert original.colocations == ()
    assert result.text == original.text
    assert result.source_metadata == original.source_metadata
    assert result.preprocessing_metadata == original.preprocessing_metadata
    assert result.full_locators == result.citations
    assert group_quotes(result) == (("No. 1:24-cv-00123", "2024 WL 1234567"),)
    group = result.colocations[0]
    assert group.id == f"colocation:{result.citations[0].id}"
    assert group.citation_ids == tuple(citation.id for citation in result.citations)
    assert "colocations" not in Document.model_fields
    for old, citation in zip(original.citations, result.citations, strict=True):
        assert citation.nodes[:-1] == old.nodes
        assert citation.locator == old.locator
        assert citation.nodes[-1].substage == COLOCATIONS
        assignment = citation.colocation_id[-1]
        assert isinstance(assignment, RelationshipUpdate)
        assert assignment.node_id == citation.nodes[-1].id
        assert latest(citation.colocation_id) == group.id
        assert citation.case_name == citation.court == citation.date == ()
        assert citation.pin_cite is None
        citation.validate_source(result.text)
    restored = Document.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.get_substage(DOCKETS) == original
    assert restored.get_substage(COLOCATIONS) == result


def test_three_distinct_reporter_editions_share_one_parsing_group():
    result = resolve_colocations(discovered("347 U.S. 483, 74 S. Ct. 686, 98 L. Ed. 873."))
    assert group_quotes(result) == (("347 U.S. 483", "74 S. Ct. 686", "98 L. Ed. 873"),)


def test_repeated_reporter_edition_starts_a_new_group_even_after_another_edition():
    result = resolve_colocations(discovered("347 U.S. 483, 74 S. Ct. 686, 556 U.S. 662."))
    assert group_quotes(result) == (("347 U.S. 483", "74 S. Ct. 686"),)
    assert result.citations[-1].colocation_id == ()


@pytest.mark.parametrize("separator", [" … ", " .. ", " v. ", " vs. ", "\n\n", ". Later "])
def test_separate_citation_markers_prevent_colocation(separator):
    result = resolve_colocations(discovered(f"347 U.S. 483{separator}74 S. Ct. 686."))
    assert len(result.citations) == 2
    assert result.colocations == ()


def test_repeated_edition_occurrences_are_not_merged_into_one_group():
    result = resolve_colocations(discovered("347 U.S. 483, 556 U.S. 662."))
    assert len(result.citations) == 2
    assert result.colocations == ()


def test_gap_limit_counts_letters_and_digits_instead_of_punctuation_and_spacing():
    source = discovered("347 U.S. 483 , \t abcde ; 74 S. Ct. 686.")
    assert len(resolve_colocations(source).colocations) == 1
    assert resolve_colocations(source, ExtractionRules(colocation_max_meaningful_gap=4)).colocations == ()


def test_configured_gap_limit_can_admit_a_more_distant_locator():
    source = discovered("347 U.S. 483 abcdef 74 S. Ct. 686.")
    assert resolve_colocations(source).colocations == ()
    assert len(resolve_colocations(source, ExtractionRules(colocation_max_meaningful_gap=6)).colocations) == 1
    assert stable().colocation_max_meaningful_gap == 5
    with pytest.raises(ValueError):
        ExtractionRules(colocation_max_meaningful_gap=-1)


@pytest.mark.parametrize("text", ["No citations here.", "See 347 U.S. 483."])
def test_no_group_run_still_has_a_replayable_checkpoint(text):
    original = find_full_reporter_locators(document(text))
    result = resolve_colocations(original)
    restored = Document.model_validate_json(result.model_dump_json())

    assert result.colocations == ()
    assert result.citations == original.citations
    assert restored.substage_runs == (REPORTERS, COLOCATIONS)
    assert restored.get_substage(REPORTERS) == original
    assert restored.get_substage(COLOCATIONS) == result


def test_colocation_requires_discovery_and_rejects_reruns_or_later_discovery():
    with pytest.raises(ValueError):
        resolve_colocations(document(MIXED))
    reporter_only = find_full_reporter_locators(document(MIXED))
    grouped = resolve_colocations(reporter_only)
    with pytest.raises(ValueError, match="already completed"):
        resolve_colocations(grouped)
    with pytest.raises(ValueError, match="before resolving colocations"):
        find_docket_locators(grouped)


def test_a_pending_singleton_assignment_cannot_be_committed_as_a_group():
    original = discovered(MIXED)
    first, second = original.citations
    pending = original.replace_citation(first.record("manual_colocation").with_colocation("group"))
    with pytest.raises(ValueError):
        pending.complete_substage("manual_colocation")
    complete = pending.replace_citation(second.record("manual_colocation").with_colocation("group"))
    committed = complete.complete_substage("manual_colocation")
    assert committed.colocations[0].citation_ids == (first.id, second.id)


def test_loading_checks_earlier_group_boundaries_before_later_repairs():
    original = discovered(MIXED)
    first, second = original.citations
    payload = original.model_dump(mode="python")
    payload["citations"] = (
        first.record("manual_colocation").with_colocation("group"),
        second.record("later_repair").with_colocation("group"),
    )
    payload["runs"] = (
        *original.runs,
        CheckpointRun(name="manual_colocation"),
        CheckpointRun(name="later_repair"),
    )
    with pytest.raises(ValueError):
        Document.model_validate(payload)


def test_colocation_assignments_need_an_owned_node_and_are_append_only():
    grouped = resolve_colocations(discovered(MIXED))
    payload = grouped.model_dump(mode="json")
    payload["citations"][0]["colocation_id"][0]["node_id"] = "missing-node"
    with pytest.raises(ValueError):
        Document.model_validate(payload)
    citation = grouped.citations[0]
    changed = citation.model_dump(mode="python")
    changed["colocation_id"][0]["value"] = "different-group"
    with pytest.raises(ValueError, match="append-only"):
        grouped.replace_citation(type(citation).model_validate(changed))


def window_document():
    return resolve_colocations(
        discovered(
            "Older 550 U.S. 544. Context Smith, No. 1:24-cv-00123, "
            "2024 WL 1234567 (D. Ariz. 2024). Next 347 U.S. 483 end."
        )
    )


def test_windows_use_the_group_edges_and_stop_at_neighboring_ungrouped_locators():
    result = window_document()
    previous, first, second, following = result.citations
    start = previous.locator_span.end
    stop = following.locator_span.start

    assert len(result.colocations) == 1
    expected_before = (result.text[start : first.locator_span.start], start)
    expected_after = (result.text[second.locator_span.end : stop], second.locator_span.end)
    assert before(result, first, 1000) == before(result, second, 1000) == expected_before
    assert after(result, first, 1000) == after(result, second, 1000) == expected_after


def test_windows_truncate_at_the_limit_and_allow_an_empty_window():
    result = window_document()
    first, second = result.citations[1:3]
    assert before(result, second, 5) == (
        result.text[first.locator_span.start - 5 : first.locator_span.start],
        first.locator_span.start - 5,
    )
    assert after(result, first, 5) == (
        result.text[second.locator_span.end : second.locator_span.end + 5],
        second.locator_span.end,
    )
    assert before(result, first, 0) == ("", first.locator_span.start)
    assert after(result, second, 0) == ("", second.locator_span.end)


@pytest.mark.parametrize("window", [before, after])
def test_window_rejects_negative_limits_and_foreign_or_stale_citations(window):
    original = discovered(MIXED)
    grouped = resolve_colocations(original)
    with pytest.raises(ValueError):
        window(grouped, grouped.citations[0], -1)
    with pytest.raises(ValueError):
        window(grouped, original.citations[0], 20)
    foreign = find_full_reporter_locators(document("See 347 U.S. 483.")).citations[0]
    with pytest.raises(ValueError):
        window(grouped, foreign, 20)
