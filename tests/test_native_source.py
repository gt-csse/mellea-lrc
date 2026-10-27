"""Source models serialize directly and preserve original text coordinates."""

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from mellea_lrc.model import (
    PreprocessedDocument,
    PreprocessingBackend,
    PreprocessingMetadata,
    Rule,
    SourceFormat,
    SourceMetadata,
    Span,
    TableOfAuthoritiesComponent,
)


def source_document(*, text="Préface. Alpha v. Beta, 556 U.S. 662.", index_spans=()):
    return PreprocessedDocument(
        text=text,
        source_metadata=SourceMetadata(path="filing.pdf", format=SourceFormat.PDF),
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend.DOCLING,
            backend_version="test-version",
            rules=(Rule.TABLE_OF_AUTHORITIES,),
        ),
        index_spans=index_spans,
    )


def test_native_json_preserves_source_provenance_unicode_and_components():
    original = source_document(index_spans=(TableOfAuthoritiesComponent(0, 8),))
    serialized = original.model_dump_json()
    restored = PreprocessedDocument.model_validate_json(serialized)
    assert restored == original
    assert restored.source_path == "filing.pdf"
    assert restored.source_metadata.format is SourceFormat.PDF
    assert restored.preprocessing_metadata.rules == (Rule.TABLE_OF_AUTHORITIES,)
    assert restored.text == "Préface. Alpha v. Beta, 556 U.S. 662."
    assert restored.text[restored.index_spans[0].start : restored.index_spans[0].end] == "Préface."
    assert json.loads(serialized)["index_spans"] == [{"start": 0, "end": 8, "kind": "table_of_authorities"}]


def test_missing_source_path_is_explicit_and_not_invented():
    original = PreprocessedDocument(
        text="Alpha",
        source_metadata=SourceMetadata(),
        preprocessing_metadata=PreprocessingMetadata(),
    )
    restored = PreprocessedDocument.model_validate_json(original.model_dump_json())
    assert restored.source_path is None
    assert restored.source_metadata.format is SourceFormat.UNKNOWN
    assert restored.preprocessing_metadata.backend is PreprocessingBackend.PLAIN_TEXT
    assert restored.index_spans == ()


@pytest.mark.parametrize("start,end", [(-1, 0), (3, 2)])
def test_invalid_source_spans_raise(start, end):
    with pytest.raises(ValueError, match="Invalid span"):
        Span(start, end)


@pytest.mark.parametrize(
    "components",
    [
        (TableOfAuthoritiesComponent(1, 1),),
        (TableOfAuthoritiesComponent(0, 100),),
        (TableOfAuthoritiesComponent(2, 8), TableOfAuthoritiesComponent(0, 2)),
        (TableOfAuthoritiesComponent(0, 8), TableOfAuthoritiesComponent(7, 9)),
    ],
)
def test_invalid_source_components_raise_on_construction_and_native_loading(components):
    with pytest.raises(ValidationError):
        source_document(index_spans=components)
    payload = source_document().model_dump(mode="json")
    payload["index_spans"] = [
        {"start": item.start, "end": item.end, "kind": item.kind} for item in components
    ]
    with pytest.raises(ValidationError):
        PreprocessedDocument.model_validate_json(json.dumps(payload))


def test_half_open_spans_overlap_only_when_they_share_characters():
    assert Span(1, 4).overlaps(Span(3, 6))
    assert not Span(1, 4).overlaps(Span(4, 6))
    assert not Span(2, 2).overlaps(Span(1, 4))
    touching = source_document(
        index_spans=(TableOfAuthoritiesComponent(0, 8), TableOfAuthoritiesComponent(8, 10))
    )
    assert PreprocessedDocument.model_validate_json(touching.model_dump_json()) == touching


def test_native_source_rejects_empty_text_and_unknown_document_attributes():
    with pytest.raises(ValidationError, match="must not be empty"):
        source_document(text="")
    with pytest.raises(ValidationError, match="Extra inputs"):
        PreprocessedDocument.model_validate({**source_document().model_dump(), "stage_runs": []})


def test_source_and_field_imports_do_not_import_runtime_stages():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from mellea_lrc.model import PreprocessedDocument; from mellea_lrc.model.citations.fields import CaseNameField; assert not any(name.startswith(('mellea_lrc.extraction', 'mellea_lrc.validation', 'mellea_lrc.providers', 'mellea_lrc.llm', 'mellea_lrc.workflows')) for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
