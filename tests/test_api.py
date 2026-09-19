"""Tests for the outer compositional API."""

import asyncio
from pathlib import Path

from mellea_lrc.api import (
    Document,
    ValidatedDocument,
    find_docket_locators,
    find_full_reporter_locators,
    resolve_colocations,
    run_full_reporter_locator_identity,
    start_full_reporter_locator_identity,
)
from mellea_lrc.extraction import extract_from_plain_text


def test_document_and_validation_checkpoints_own_their_serialization_boundary() -> None:
    document = extract_from_plain_text("A filing without citations.")

    recovered_document = Document.from_serialized(document.serialize())
    checkpoint = start_full_reporter_locator_identity(recovered_document)
    recovered_checkpoint = ValidatedDocument.from_serialized(checkpoint.serialize())
    completed = asyncio.run(run_full_reporter_locator_identity(recovered_checkpoint, client=object()))

    assert recovered_document == document
    assert completed == checkpoint


def test_outer_api_composes_locator_stages_from_a_document_owned_constructor() -> None:
    document = Document.from_source(
        "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567 (D. Ariz. Jan. 1, 2024)."
    )

    document = find_full_reporter_locators(document)
    document = find_docket_locators(document)
    document = resolve_colocations(document)

    assert len(document.citations) == 2
    assert all(citation.colocation_id is not None for citation in document.citations)


def test_document_from_source_preserves_a_path_as_source_provenance(tmp_path: Path) -> None:
    path = tmp_path / "filing.txt"
    path.write_text("A filing without citations.", encoding="utf-8")

    document = Document.from_source(path)

    assert document.source_path == str(path)
