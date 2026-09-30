# Extraction

Extraction has separate `grow_roots` and `grow_leaves` workflows. Both accept a `Document` and return a `Document`; preprocessing is a separate step. The public functions are in [`mellea_lrc.api`](../src/mellea_lrc/api.py).

```python
import asyncio

from mellea_lrc.api import Document, grow_roots, stable

document = Document.from_source("See 347 U.S. 483 (1954).")
document = asyncio.run(grow_roots(document, rules=stable()))
after_dockets = document.get_stage("2_docket_locators")
after_entries = document.get_stage("4_docket_entries")
```

`grow_roots` is async. It runs reporter locator discovery, docket locator discovery, optional docket site hunting, docket-entry reading, colocation, case-name/court/date/pin-cite reading, and root formation in that order. Omit `hunt_dockets=True` for the rule-only pass. Each stage can also be called separately through the same API. Colocation sets parsing boundaries; it does not establish case identity.

The locator readers live in their stage modules: `extraction/full_reporter_locator.py`, `extraction/docket_locator.py`, and `extraction/short_reporter_locator.py`. Each module reads source spans and writes its own Document stage. The two reporter stages share the full reporter stage's eyecite tokenizer, so discovery and quote normalization follow the same matching rules without a separate reader layer.

Each extraction entrypoint takes and returns a `Document` and writes a named stage. Docket site hunting is a stage package: its `__init__.py` contains the entrypoint and its adjacent modules contain only its candidate discovery and review logic. Colocation-aware window boundaries live in `model/citation_windows.py` because validation also uses them; extraction-only ordering checks and dated-parenthetical reading live in `extraction/contextual_reading.py`. `config/extraction.py` holds `ExtractionRules` and `stable()`, while `workflows/grow_roots.py` composes the stages. The public `mellea_lrc.api` exposes both levels without making the workflow look like a stage.

`hunt_docket_locators(document)` is the independent optional stage between rule discovery and colocation. It proposes labelled opaque identifiers and bounded unlabelled candidates in citation context, then asks a model whether each proposed span is a cited case docket. The default reviewer uses Mellea's instruct/validate/repair loop, including a schema check and source-grounding check. A positive answer must reproduce both the complete locator and its number, with only whitespace variation permitted. An accepted answer creates a `FullDocketCitation` using exact source spans; the next proposal sees that new locator in its mask. Declines and failed grounding remain in `document.site_reviews`. Each model-backed review contains its entire `ivr` run: answers, validation feedback, and provider requests and responses when the backend exposes them. The stage does not read other fields or create short citations. A supplied async `reviewer` makes it runnable offline.

`resolve_docket_entries(document)` then attaches an optional nearby `Doc.`, `Dkt.`, `ECF`, or `D.I.` entry to an already found docket citation. It reads both sides, with a narrower rule after the locator: an immediate comma or a short bracketed reference. Entries never create case-docket roots. Competing associations remain unread for later review. The hunter temporarily masks these entry references while looking for case dockets; it does not alter the source text.

The evaluator scores each `grow_roots` extraction stage independently using its exact stage checkpoint. See [Grow-roots evaluation](../evaluations/README.md) for the scoring contract and command. Validation stages have separate contracts.

```python
import asyncio
from pathlib import Path

from mellea_lrc.api import (
    Document,
    find_docket_locators,
    find_full_reporter_locators,
    hunt_docket_locators,
    resolve_docket_entries,
)

document = find_full_reporter_locators(Document.from_source(Path("filing.txt")))
document = find_docket_locators(document)
document = asyncio.run(hunt_docket_locators(document))
document = resolve_docket_entries(document)
```

To continue from a saved rule-locator checkpoint, restore its `Document` and call the stage directly. If the saved artifact is from a later stage, `get_stage("2_docket_locators")` recovers the input to hunting. A new full run uses `grow_roots(document, hunt_dockets=True)` instead.

```python
import asyncio
from pathlib import Path

from mellea_lrc.api import Document, hunt_docket_locators

saved = Path("rule-checkpoint.json")
document = Document.model_validate_json(saved.read_text(encoding="utf-8"))
document = asyncio.run(hunt_docket_locators(document.get_stage("2_docket_locators")))
Path("hunted-checkpoint.json").write_text(document.model_dump_json(indent=2), encoding="utf-8")
```

The default reviewer reads `MELLEA_LRC_LLM_API_BASE`, `MELLEA_LRC_LLM_API_KEY`, and `MELLEA_LRC_LLM_MODEL` from the environment or `.env`. `MELLEA_LRC_LLM_SERVICE_TIER` is optional; omitting it uses the normal endpoint. The shared `GroundingEvidence` service supports exact, whitespace-relaxed, and bounded edit-distance comparison; docket admission deliberately uses whitespace relaxation alone, because changing a docket character can identify a different case. A provider error aborts the run so it cannot be mistaken for a declined citation.

`find_short_reporter_citations(document)` is an optional separate stage. It records eyecite `ShortCaseCitation` readings as `ShortReporterCitation` objects with a `short_locator` field. Their page is a pinpoint, not a full reporter citation's first page. This stage does not attach leaves or change `full_locators`, colocations, or roots.

`Document` contains the preprocessed text and provenance, citation histories, durable site reviews, and an ordered `stage_runs` tuple. Each parsed field is an append-only log of domain-specific entries carrying an exact `quote`, its source `span`, normalization status, and a citation-local `node_id`. A successful field contains a typed normalized value. A failed field keeps the quote, span, and error with `normalizable=False`; its `get_normalized()` method raises instead of returning `None`. The field class owns normalization and checks successful values against the quote. An inferred court has no quote or span; root and colocation assignments use separate relationship entries. `get_stage(stage)` reconstructs exactly the document returned by that completed stage, including citations untouched in that stage; it raises `KeyError` if the stage has not run. Calling a completed stage again, including `complete(stage)`, raises rather than silently returning the old document.

`CaseNameField` stores an exact source quote and one of four written forms: adversarial plaintiff and defendant, an *In re* subject, an *Ex parte* subject, or a printed partial fragment. *Matter of* is read as the *In re* form. The partial form does not supply a missing party. The fifth case-name outcome is `not_stated`: when a citation has no case-name reading, `citation.get_case_name()` returns that typed value without inventing a quote or span. A quoted name the parser cannot classify instead makes `get_case_name()` raise, preserving the distinct normalization failure for later review. The official annotation likewise keeps `source.kind: not_stated` with normalization unavailable; it does not claim there is a normalized printed name. `FullReporterLocator` keeps eyecite's `Reporter` object alongside normalized volume, edition, and page in one locator field. Reporter discovery and normalization use the same whitespace-relaxed eyecite reader on the original document and the exact locator quote respectively, so their matching rules agree without changing source spans. A reporter edition that cannot be resolved from the quote stays unnormalizable. `FullDocketLocator` keeps the written docket number opaque; equivalence is a later identity question. `CourtField`, `DateField`, and `PinCiteField` likewise contain their own typed normalized values and post-validation. Court labels are compared as tokens against `courts-db` citation labels. If that lookup fails, `reporters-db`'s Bluebook-style state abbreviations supply the state part of district and bankruptcy court labels; `courts-db` still supplies the court ID. A fallback is accepted only for a unique court and never overrides a direct database match. A reader leaves a field log empty when it finds no field. If normalization fails, it retains the reading as unnormalizable so later stages can review it. Root formation leaves such locators unmerged until their identity can be resolved. The pin reader handles adjacent pages, star pages, paragraph numbers, and supported footnote forms; unread forms retain their quote for review.

```python
saved = document.model_dump_json()
restored = Document.model_validate_json(saved)
assert restored.get_stage("2_docket_locators") == after_dockets
```

Exact reporter-root lookup is the first independent validation stage after root formation; see [Validation](Validation.md).

## Grow leaves

`grow_leaves` needs formed roots. Root validation is optional: a citation whose identity is wrong still needs its short forms attached. The workflow uses the current source-grounded root names and locators. It does not retrieve external evidence or change root identity judgments.

```python
from mellea_lrc.api import grow_leaves

document = await grow_leaves(document)
before_id_review = document.get_stage("36_id_attribution")
```

The individually callable stages are:

| Stage | API | Work |
| --- | --- | --- |
| 28 | `find_short_reporter_citations(document)` | Short reporter sites, using the augmented eyecite tokenizer and shared pinpoint grammar |
| 29 | `find_supra_citations(document)` | Supra sites |
| 30 | `find_id_citations(document)` | Id./Ibid. sites and explicitly joined pinpoints |
| 31 | `find_reference_citations(document)` | Lenient source-name proposals, excluding other citation components |
| 32 | `resolve_leaf_case_names(document)` | Adjacent written names, independently of root attachment |
| 33 | `resolve_leaf_pin_cites(document)` | Exact pinpoint quotes and typed normalization |
| 34 | `attribute_leaves_rule(document)` | Unique preceding reporter/name agreements; ambiguous and bare-name sites route to review |
| 35 | `await review_leaf_attributions(document)` | Citation-use decision and root choice for routed sites |
| 36 | `attribute_id_citations(document)` | Source-order Id. chains, retaining noncase events as barriers |
| 37 | `await review_id_attributions(document)` | Semantic Id. audit for documentary sources missed by the tokenizer |

Use `await grow_leaves(document, review_leaves=False)` for a rule-only pass. Both model stages accept an async `reviewer` for controlled offline tests. Name-only proposals always need semantic review, even when their name has only one root candidate. Full citation repetitions were attached during root formation; grow-leaves evaluation includes them but does not recreate them.

Short reporter volume and edition identify candidate roots; the short page is a pinpoint and is not used as the root's first page. Name-only references can precede a nearby full citation. Id. cannot refer forward: its rule pass follows citation chronology, then its semantic audit distinguishes case-linked docket documents from unrelated briefs, exhibits, statutes, and rules. Each reviewed Id. updates the tree before the next review, and the next prompt receives the recent source sites and their current attachments.

The citation subclasses are `ShortReporterCitation`, `SupraCitation`, `IdCitation`, and `ReferenceCitation`. Their quoted fields, `attributions`, and `reviews` are append-only and point to citation-local nodes. A model returns only `is_citation`, a candidate `root_index` or null, and a reason; the program writes the root assignment. Rejection reattaches the citation to the dummy head. An unsupported semantic choice clears the current attachment while retaining the earlier rule assignment. A failed model call records its trace and preserves the prior rule state.

Each `LeafReview` retains candidate root IDs, the decision or failure, and the complete shared `IvrRun`, including schema repair feedback and provider exchanges. No live model session is needed to restore the document or inspect an earlier stage. Pinpoint **reading** belongs here; validity against the cited opinion remains a separate later workflow.
