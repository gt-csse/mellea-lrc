# Native source and citation fields

This increment defines source metadata, preprocessed text, exact character spans, and typed case-name, date, and docket readings. These models serialize directly through Pydantic. Existing pipelines remain unchanged.

```python
from mellea_lrc.model import PreprocessedDocument, PreprocessingMetadata, SourceMetadata, Span
from mellea_lrc.model.citations.fields import CaseNameField

source = PreprocessedDocument(
    text="Alpha v. Beta, 556 U.S. 662.",
    source_metadata=SourceMetadata(path="filing.txt"),
    preprocessing_metadata=PreprocessingMetadata(),
)
reading = CaseNameField.from_source(source.text, Span(0, 13), node_id="read-name")
assert reading.get_normalized().plaintiff == "Alpha"
assert CaseNameField.model_validate_json(reading.model_dump_json()) == reading
assert PreprocessedDocument.model_validate_json(source.model_dump_json()) == source
```

Each field holds its exact quote, span, producing `node_id`, and typed normalization. Case names distinguish adversarial, in-re, ex-parte, partial, and unstated forms. Dates preserve their available precision. Docket numbers remain opaque, and docket entries are separate fields.

`normalizable=False` preserves an unsuccessful reading with an explicit error; `get_normalized()` raises rather than returning a silent null. Source quotes are checked against their spans, including through `validate_source()` after loading. Model-supplied case-name normalization still requires an exact source quote.

Citation collections, node ownership and histories, root graphs, checkpoints, stage/workflow definitions, other normalizers, and pipeline integration belong to later PRs. No compatibility adapters are introduced.
