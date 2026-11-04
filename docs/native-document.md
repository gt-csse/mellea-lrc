# Native source, citation fields, and histories

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

## Full citation occurrences and decision histories

`FullReporterCitation` and `FullDocketCitation` combine the field types into immutable occurrences. The initial locator belongs to a creation node. Calling `record()` adds a later decision node; named field methods append readings under that node and return a new citation. One decision can update several fields, with at most one reading per field. Every history follows the owning citation's node order.

```python
from mellea_lrc.model.citations import FullReporterCitation, latest

text = "Alpha v. Beta, 556 U.S. 662 (2009)."
locator_start = text.index("556 U.S. 662")
citation = FullReporterCitation.from_locator(
    citation_id="reporter:1",
    substage="discover_locators",
    source=text,
    span=Span(locator_start, locator_start + len("556 U.S. 662")),
)
initial = citation.record("read_fields").with_case_name(text, Span(9, 13))
corrected = initial.record("review_identity").with_case_name(text, Span(0, 13))

assert initial.get_case_name().as_citation() == "Beta"
assert corrected.get_case_name().as_citation() == "Alpha v. Beta"
assert len(corrected.case_name) == 2
assert corrected.case_name[0] == initial.case_name[0]
assert corrected.case_name[-1].node_id == corrected.nodes[-1].id
assert latest(corrected.locator).edition == "U.S."

loaded = FullReporterCitation.model_validate_json(corrected.model_dump_json())
assert loaded == corrected
loaded.validate_source(text)
```

Quoted readings, inferred courts, and explicit absence decisions remain distinct. `with_case_name_not_stated()`, `with_court_not_stated()`, `with_date_not_stated()`, and `with_pin_cite_not_stated()` append a `NotStatedField` without deleting earlier quotes. Empty histories represent unread fields; an unread pinpoint is `None`. `current_reading()` returns the latest quoted or inferred field, or `None` for an unread or explicitly absent field. `latest()` returns its normalized value and raises when the latest quoted reading failed normalization, rather than falling back to an older successful value.

Docket citations additionally retain entry readings and support rereading the number within the same written locator. Loading checks node ownership, history order, and each locator's fixed source site. `validate_source()` checks every historical field quote against the owning text, including readings superseded by corrections or absence.

## Document collections and replayable checkpoints

`Document.from_preprocessed()` carries the source text and its provenance into an ordered collection of full citations. `add_citation()` and `replace_citation()` return new documents. The collection checks citation IDs, source order, every historical quote, and append-only decision and reading histories. New readings belong to new decision nodes.

`complete_substage()` records an atomic completion event, including a run that made no citation changes. `get_substage()` reconstructs the citation collection and field histories at that boundary. Later corrections remain available in the current document, while earlier checkpoints retain their original readings after a JSON round trip.

```python
from mellea_lrc.model import Document

document = Document.from_preprocessed(
    PreprocessedDocument(
        text=text,
        source_metadata=SourceMetadata(path="filing.txt"),
        preprocessing_metadata=PreprocessingMetadata(),
    )
)
discovered = document.add_citation(citation).complete_substage("discover_locators")
read = discovered.replace_citation(initial).complete_substage("read_fields")
reviewed = read.replace_citation(corrected).complete_substage("review_identity")
finished = reviewed.complete_substage("unchanged_review")

restored = Document.model_validate_json(finished.model_dump_json())
assert restored == finished
assert restored.get_substage("discover_locators") == discovered
assert restored.get_substage("read_fields") == read
assert restored.get_substage("review_identity") == reviewed
assert restored.get_substage("unchanged_review") == finished
assert restored.get_substage("discover_locators").citations[0].pin_cite is None
assert restored.substage_runs == ("discover_locators", "read_fields", "review_identity", "unchanged_review")
```

A document commits one pending substage at a time. Completed substages remain closed, and checkpoint loading checks that citation decisions follow the recorded event order.

Root graphs, grouped stage/workflow definitions, short-form locator fields, and pipeline integration follow in later PRs.
