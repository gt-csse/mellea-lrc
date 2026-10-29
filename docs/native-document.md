# Native source and citation fields

The native models retain source metadata, preprocessed text, exact character spans, and typed citation-field readings. Court, full reporter-locator, and pinpoint fields extend the existing case-name, date, and docket fields. These models serialize directly through Pydantic. Existing pipelines remain unchanged.

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

Court fields retain the written label and resolve it to a canonical `courts-db` identity. An inferred court accepts a known court ID without inventing a quote or source span. Unrecognized or ambiguous written labels remain explicit failed readings.

Full reporter locators retain the entire written volume, reporter, and opening-page quote. A shared eyecite-based reader tolerates PDF whitespace and tightly joined reporter components while preserving their original offsets. Successful normalization requires one complete locator and a resolved reporter edition; unresolved spellings are not silently stored as normalized reporter identities.

Pinpoint fields normalize pages, star pages, paragraphs, inclusive ranges, and footnotes into typed targets. Abbreviated ranges are expanded, and ambiguous page/footnote lists remain failed readings. These are interpretations of written citation fields, not judgments about an opinion's substantive support or page correctness.

```python
from mellea_lrc.model.citations.fields import CourtField, FullReporterLocator, PinCiteField, PinCiteKind

court_text = "D. Md."
court = CourtField.from_source(court_text, Span(0, len(court_text)), node_id="read-court")
assert court.get_normalized().id == "mdd"

locator_text = "556 U.S. 662"
locator = FullReporterLocator.from_source(locator_text, Span(0, len(locator_text)), node_id="read-locator")
assert locator.get_normalized().edition == "U.S."
assert FullReporterLocator.model_validate_json(locator.model_dump_json()) == locator

pin_text = "at *10–12"
pin = PinCiteField.from_source(pin_text, Span(0, len(pin_text)), node_id="read-pinpoint")
assert pin.get_normalized()[0].kind is PinCiteKind.STAR
assert (pin.get_normalized()[0].first, pin.get_normalized()[0].last) == (10, 12)
```

Citation collections, node ownership and histories, root graphs, checkpoints, stage/workflow definitions, short-form locator fields, and pipeline integration belong to later PRs. No compatibility adapters are introduced.
