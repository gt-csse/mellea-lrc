# Extraction

The active extraction pipeline ends at root formation. It accepts a `Document` and returns a `Document`; preprocessing is a separate step. The public functions are in [`mellea_lrc.api`](../src/mellea_lrc/api.py).

```python
import asyncio

from mellea_lrc.api import Document, grow_roots, stable

document = Document.from_source("See 347 U.S. 483 (1954).")
document = asyncio.run(grow_roots(document, rules=stable()))
after_dockets = document.get_stage("docket_locators")
```

`grow_roots` is async. It runs reporter locator discovery, docket locator discovery, optional docket site hunting, docket-entry reading, colocation, case-name/court/date/pin-cite reading, and root formation in that order. Omit `hunt_dockets=True` for the rule-only pass. Each stage can also be called separately through the same API. Colocation sets parsing boundaries; it does not establish case identity.

`hunt_docket_locators(document)` is the independent optional stage between rule discovery and colocation. It proposes labelled opaque identifiers and bounded unlabelled candidates in citation context, then asks a model whether each proposed span is a cited case docket. A positive answer must reproduce both the complete locator and its number, with only whitespace variation permitted. An accepted answer creates a `FullDocketCitation` using exact source spans; the next proposal sees that new locator in its mask. Declines and failed grounding remain in `document.site_reviews`, including model attempts. The stage does not read other fields or create short citations. A supplied async `reviewer` makes it runnable offline; the default uses the configured OpenRouter-compatible model endpoint.

`resolve_docket_entries(document)` then attaches an optional nearby `Doc.`, `Dkt.`, `ECF`, or `D.I.` entry to an already found docket citation. It reads both sides, with a narrower rule after the locator: an immediate comma or a short bracketed reference. Entries never create case-docket roots. Competing associations remain unread for later review. The hunter temporarily masks these entry references while looking for case dockets; it does not alter the source text.

```python
from pathlib import Path

from mellea_lrc.api import find_docket_locators, find_full_reporter_locators, hunt_docket_locators, resolve_docket_entries

document = find_full_reporter_locators(Document.from_source(Path("filing.txt")))
document = find_docket_locators(document)
document = asyncio.run(hunt_docket_locators(document))
document = resolve_docket_entries(document)
```

The default reviewer reads `MELLEA_LRC_LLM_API_BASE`, `MELLEA_LRC_LLM_API_KEY`, and `MELLEA_LRC_LLM_MODEL` from the environment or `.env`. `MELLEA_LRC_LLM_SERVICE_TIER` is optional. A provider error aborts the run so it cannot be mistaken for a declined citation.

`find_short_reporter_citations(document)` is an optional separate stage. It records eyecite `ShortCaseCitation` readings as `ShortReporterCitation` objects with a `short_locator` field. Their page is a pinpoint, not a full reporter citation's first page. This stage does not attach leaves or change `full_locators`, colocations, or roots.

`Document` contains the preprocessed text and provenance, citation histories, durable site reviews, and an ordered `stage_runs` tuple. Each parsed field is an append-only log of domain-specific entries carrying an exact `quote`, its source `span`, normalization status, and a citation-local `node_id`. A successful field contains a typed normalized value. A failed field keeps the quote, span, and error with `normalizable=False`; its `get_normalized()` method raises instead of returning `None`. The field class owns normalization and checks successful values against the quote. An inferred court has no quote or span; root and colocation assignments use separate relationship entries. `get_stage(stage)` reconstructs exactly the document returned by that completed stage, including citations untouched in that stage; it raises `KeyError` if the stage has not run.

`CaseNameField` parses an adversarial name into plaintiff and defendant, or stores the subject of an *In re* or *Ex parte* name. `FullReporterLocator` keeps eyecite's `Reporter` object alongside normalized volume, edition, and page in one locator field. Reporter discovery and normalization use the same whitespace-relaxed eyecite reader on the original document and the exact locator quote respectively, so their matching rules agree without changing source spans. A reporter edition that cannot be resolved from the quote stays unnormalizable. `FullDocketLocator` keeps the written docket number opaque; equivalence is a later identity question. `CourtField`, `DateField`, and `PinCiteField` likewise contain their own typed normalized values and post-validation. Court labels are compared as tokens against `courts-db` citation labels. If that lookup fails, `reporters-db`'s Bluebook-style state abbreviations supply the state part of district and bankruptcy court labels; `courts-db` still supplies the court ID. A fallback is accepted only for a unique court and never overrides a direct database match. A reader leaves a field log empty when it finds no field. If normalization fails, it retains the reading as unnormalizable so later stages can review it. Root formation leaves such locators unmerged until their identity can be resolved. The pin reader handles adjacent pages, star pages, paragraph numbers, and supported footnote forms; unread forms retain their quote for review.

```python
saved = document.model_dump_json()
restored = Document.model_validate_json(saved)
assert restored.get_stage("docket_locators") == after_dockets
```

Identity validation, search, and leaf growth are later layers and are not active in this pipeline.
