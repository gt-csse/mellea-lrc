"""Native root grouping retains every citation occurrence and its history."""

import pytest
from pydantic import ValidationError

from mellea_lrc.extraction import (
    find_docket_locators,
    find_full_reporter_locators,
    form_roots,
    resolve_colocations,
    resolve_courts,
)
from mellea_lrc.extraction.form_roots import SUBSTAGE
from mellea_lrc.extraction.resolve_colocations import SUBSTAGE as COLOCATIONS
from mellea_lrc.model import Document, PreprocessedDocument, PreprocessingMetadata, SourceMetadata
from mellea_lrc.model.citations import latest


def discovered(text: str) -> Document:
    source = PreprocessedDocument(
        text=text,
        source_metadata=SourceMetadata(path="filing.txt"),
        preprocessing_metadata=PreprocessingMetadata(),
    )
    return find_docket_locators(find_full_reporter_locators(Document.from_preprocessed(source)))


def test_repeated_reporter_locators_keep_their_spans_and_share_the_earliest_root():
    grouped = resolve_colocations(discovered("See 347 U.S. 483. Later, 347 U.S. 483. Compare 347 U.S. 485."))
    result = form_roots(resolve_courts(grouped))
    first, repeat, other = result.citations

    assert result.roots == (first, other)
    assert [latest(citation.root_id) for citation in result.citations] == [first.id, first.id, other.id]
    assert first.id != repeat.id
    assert first.locator_span != repeat.locator_span
    for before, after in zip(grouped.citations, result.citations, strict=True):
        assert after.locator == before.locator
        assert after.nodes[: len(before.nodes)] == before.nodes
        assert after.nodes[-1].substage == SUBSTAGE
        assert after.root_id[-1].node_id == after.nodes[-1].id

    restored = Document.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.get_substage(COLOCATIONS) == grouped
    assert restored.get_substage(SUBSTAGE) == result


def test_colocation_and_unresolved_reporter_text_do_not_establish_equivalence():
    grouped = resolve_colocations(discovered("347 U.S. 483, 74 S. Ct. 686. See 1 Wash. 2. Later, 1 Wash. 2."))
    result = form_roots(resolve_courts(grouped))
    first, parallel, unresolved, repeat = result.citations

    assert grouped.colocations[0].citation_ids == (first.id, parallel.id)
    assert not unresolved.locator[-1].normalizable
    assert not repeat.locator[-1].normalizable
    assert result.roots == result.citations


def test_dockets_only_share_a_root_with_the_same_normalized_court_and_number():
    text = (
        "Case No. 1:24-cv-00123 (D. Md.). "
        "Case No. 1:24-cv-00123 (D. Maryland). "
        "Case No. 1:24-cv-00123 (D. Ariz.). "
        "Case No. 1:24-cv-00123. "
        "Case No. 1:24-cv-00123."
    )
    grouped = resolve_colocations(discovered(text))
    with pytest.raises(ValueError, match="Resolve courts"):
        form_roots(grouped)
    prepared = resolve_courts(grouped)
    result = form_roots(prepared)
    first, repeat, other_court, unread, unread_repeat = result.citations

    assert result.roots == (first, other_court, unread, unread_repeat)
    assert latest(repeat.root_id) == first.id
    assert result.get_substage(prepared.substage_runs[-1]) == prepared


def test_root_reassignment_updates_the_current_graph_and_replays_the_previous_graph():
    before = form_roots(
        resolve_courts(resolve_colocations(discovered("347 U.S. 483; 74 S. Ct. 686; 347 U.S. 483.")))
    )
    first, parallel, repeat = before.citations
    after = before.replace_citation(first.record("combine_roots").with_root(parallel.id))
    after = after.replace_citation(repeat.record("combine_roots").with_root(parallel.id))
    after = after.complete_substage("combine_roots")

    restored = Document.model_validate_json(after.model_dump_json())
    assert restored.roots == (parallel,)
    assert restored.get_substage(SUBSTAGE) == before
    assert tuple(update.value for update in restored.citations[-1].root_id) == (first.id, parallel.id)


def test_checkpoint_rejects_indirect_or_missing_roots():
    before = form_roots(
        resolve_courts(resolve_colocations(discovered("347 U.S. 483; 74 S. Ct. 686; 347 U.S. 483.")))
    )
    first, parallel, _ = before.citations
    invalid = before.replace_citation(first.record("combine_roots").with_root(parallel.id))
    with pytest.raises(ValidationError, match="point to a self-root"):
        invalid.complete_substage("combine_roots")
    with pytest.raises(ValidationError, match="unknown root"):
        before.replace_citation(first.record("combine_roots").with_root("missing"))


def test_root_formation_checkpoint_requires_an_assignment_for_every_occurrence():
    ready = resolve_courts(resolve_colocations(discovered("347 U.S. 483; 74 S. Ct. 686.")))
    first = ready.citations[0]
    incomplete = ready.replace_citation(first.record(SUBSTAGE).with_root(first.id))
    with pytest.raises(ValidationError, match="must have a root"):
        incomplete.complete_substage(SUBSTAGE)

    grouped = form_roots(ready)
    first = grouped.citations[0]
    incomplete = grouped.replace_citation(first.record("remove_root").with_root(None))
    with pytest.raises(ValidationError, match="must have a root"):
        incomplete.complete_substage("remove_root")


def test_empty_run_replays_and_stage_guards_are_consistent():
    document = discovered("No citations.")
    with pytest.raises(ValueError, match="Resolve colocations"):
        form_roots(document)
    grouped = resolve_colocations(document)
    result = form_roots(resolve_courts(grouped))

    assert result.citations == result.roots == ()
    assert result.get_substage(COLOCATIONS) == grouped
    assert result.get_substage(SUBSTAGE) == result
    with pytest.raises(ValueError, match="already completed"):
        form_roots(result)
