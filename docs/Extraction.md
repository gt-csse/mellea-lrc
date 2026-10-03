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

The locator stages live in `extraction/full_reporter_locator.py`, `extraction/docket_locator.py`, and `extraction/short_reporter_locator.py`. Each writes its own Document stage. The full and short reporter stages and their field normalizers share `parsing/reporters.py`, which reads unchanged source text without importing a Document or an extraction stage. The shared pinpoint and supra grammars also live under `parsing/`.

Each extraction entrypoint takes and returns a `Document` and writes a named stage. Docket site hunting is a stage package: its `__init__.py` contains the entrypoint and its adjacent modules contain only its candidate discovery and review logic. Colocation-aware window boundaries live in `model/citation_windows.py` because validation also uses them; extraction-only context, grouping, and ordering helpers live in `extraction/context/`. `config/extraction.py` holds `ExtractionRules` and `stable()`, while `workflows/grow_roots.py` composes the stages. The public `mellea_lrc.api` exposes both levels without making the workflow look like a stage.

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

Each model-backed extraction stage selects its own `MODEL_PROFILE` constant. Its reviewer receives that complete package explicitly; see [model configuration](../README.md#model-configuration). Docket-root reassignment is a stage package with its Document entrypoint in `__init__.py`, source contexts in `context.py`, and model reviewer in `reviewer.py`. The leaf stages share the `leaf_attribution_review/` service. Credentials are resolved from `.env` only when a model call is needed. The shared `GroundingEvidence` service supports exact, whitespace-relaxed, and bounded edit-distance comparison; docket admission deliberately uses whitespace relaxation alone, because changing a docket character can identify a different case. A timeout or exhausted validation loop becomes a retained failed review, not a declined citation. Other provider exceptions propagate and abort the run.

`find_short_reporter_citations(document)` is an optional separate stage. It records eyecite `ShortCaseCitation` readings as `ShortReporterCitation` objects with a `short_locator` field. Their page is a pinpoint, not a full reporter citation's first page. This stage does not attach leaves or change `full_locators`, colocations, or roots.

`Document` contains the preprocessed text and provenance, citation histories, durable site reviews, and an ordered `stage_runs` tuple. Each parsed field is an append-only log of domain-specific entries carrying an exact `quote`, its source `span`, normalization status, and a citation-local `node_id`. A successful field contains a typed normalized value. A failed field keeps the quote, span, and error with `normalizable=False`; its `get_normalized()` method raises instead of returning `None`. The field class owns normalization and checks successful values against the quote. An inferred court has no quote or span; root and colocation assignments use separate relationship entries. `get_stage(stage)` reconstructs exactly the document returned by that completed stage, including citations untouched in that stage; it raises `KeyError` if the stage has not run. Calling a completed stage again, including `complete(stage)`, raises rather than silently returning the old document.

`Citation` owns `pin_cite` for every citation type, including supra. It is `None` when no pinpoint has been read, otherwise a nonempty tuple of `PinCiteField` readings. `citation.record(stage).with_pin_cite(source, span)` appends a grounded reading, and `citation.get_pin_cite()` returns the latest normalized targets. A failed normalization remains a reading whose getter raises; it is not absence. JSON writes an unread field as `null` and rejects an empty pin array. Rewinding before the first reading restores `None`, while later checkpoints retain every reading. Citation subclasses inherit this field and these methods; only the source-reading rules differ by type.

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
from mellea_lrc.api import (
    attribute_reference_citations,
    attribute_short_reporter_citations,
    find_reference_citations,
    find_id_citations,
    attribute_id_citations,
    find_short_reporter_citations,
    resolve_short_reporter_colocations,
    resolve_short_reporter_case_names,
    grow_leaves,
)

document = find_short_reporter_citations(document)
created = document.get_stage("28_short_reporter_citations")
document = resolve_short_reporter_colocations(document)
document = resolve_short_reporter_case_names(document)
document = await attribute_short_reporter_citations(document, review=False)
document = find_reference_citations(document)
references_created = document.get_stage("30_reference_citations")
document = await attribute_reference_citations(document, review=False)
document = find_id_citations(document)
ids_created = document.get_stage("32_id_citations")
document = await attribute_id_citations(document, review=False)
# Or compose the complete workflow from a document with formed roots:
# document = await grow_leaves(document)
```

The individually callable stages are:

| Stage | API | Work |
| --- | --- | --- |
| 28 | `find_short_reporter_citations(document)` | Create short reporter citations with quoted locator and complete pincite; normalize both fields without assigning roots |
| 28.1 | `resolve_short_reporter_colocations(document)` | Group adjacent short reporter sites after pinpoint boundaries are complete |
| 28.2 | `resolve_short_reporter_case_names(document)` | Read the name before a group for every member, or before a singleton |
| 29 | `await attribute_short_reporter_citations(document)` | Attach short reporters using volume/edition/name rules, with optional model review of unresolved attachments |
| 30 | `find_reference_citations(document)` | Create case-name references with their quoted name and explicit adjacent pinpoint; normalize both fields without assigning roots |
| 31 | `await attribute_reference_citations(document)` | Attach references with a unique source-name root candidate, with optional model review of unresolved attachments |
| 32 | `find_id_citations(document)` | Create Id./Ibid. occurrences with their complete citation span and optional normalized pinpoint, without assigning roots |
| 33 | `await attribute_id_citations(document)` | Attribute Id. chains in source order, with optional semantic review of case and noncase antecedents |
| 34 | `find_supra_citations(document)` | Supra sites |
| 35 | `resolve_supra_case_names(document)` | Read each supra citation's written case name |
| 36 | `resolve_supra_pin_cites(document)` | Read each supra citation's optional pinpoint |
| 37 | `attribute_supra_citations_rule(document)` | Attach supra sites and route unresolved supra sites to review |
| 38 | `await review_supra_attributions(document)` | Citation-use decision and root choice for routed supra sites |

Use `await grow_leaves(document, review_leaves=False)` for a rule-only pass. The attribution and review stages accept an async `reviewer` for controlled offline tests. Reference attribution uses current source-grounded names to attach a unique candidate; ambiguous or unmatched references can proceed to semantic review. Full citation repetitions were attached during root formation; grow-leaves evaluation includes them but does not recreate them.

Reference discovery uses current full-citation names, whitespace-relaxed name matching, and a directly adjacent explicit `at`, star-page, or paragraph marker. Bare names and unmarked numbers after names do not create references. The name and pin together must lie outside source-recognized citation components, including citation types created later. Stage 30 records and normalizes both the name and the complete pinpoint on the citation-local creation node. The same grounded name reading supplies `reference_name` and `case_name`; the name alone remains the citation's source site. The dataset retains valid bare-name references as `unit: out_of_scope_citation`, while independently annotated pinpointed references remain `unit: citation`. Stage 30's `case_name_span` and `case_name_normalization` metrics score its name reading, and `pin_cite_span` and `pin_cite_normalization` score its retained pinpoint. Every creation-stage field outcome counts toward its precision, including absent readings, unmatched sites, and failed normalizations. An absent reading is correct only against an explicit annotated not_stated state. Matched annotations missing a required normalization target raise; they are never silently omitted from the denominator. Stage 28 reports `locator_span`, `locator_normalization`, `pin_cite_span`, and `pin_cite_normalization`; stage 28.2 separately reports `case_name_span` and `case_name_normalization`. Stage 28.1 has a checkpoint but no numeric group precision because independent short-group annotations are not yet defined.

Reference discovery makes no LLM calls. Stage 31 attributes already created citations and retains its candidate choices, decisions, and failures; it does not discover reference sites or rewrite their name and pin readings. Use `await attribute_reference_citations(document, review=False)` for unique-candidate rule attachments and unresolved references without model calls. Its optional async `reviewer` supports controlled offline review of unresolved cases. The obsolete LLM bare-name hunter is retained only in the [archived experimental package](../archive/root-locators-2026-09-23/src/mellea_lrc/experimental/leaf_case_name_hunting/__init__.py) and is outside the active workflow.

Short reporter, reference, and Id. creation each precede their own attribution checkpoint. Short reporter creation records the locator and pin on one citation-local creation node. Its separate colocation and case-name stages then run before attribution. The locator quote/span includes the complete pinpoint, but its normalized value contains only volume, Reporter, and edition. `pin_cite` alone owns normalized page/range/list/footnote targets. Attribution changes attachments and records choices or failures without rewriting these source fields. Supra name, pinpoint, and attribution stages act only on supra citations. Source recognition still bounds name reading against intervening citations whose types have not yet been created.

Full and short reporter colocation use the same grouping reader: at most five meaningful characters between adjacent sites by default, excluding whitespace and punctuation, with the same separator and repeated-reporter-edition checks. Recognized intervening nonshort citations also stop a short group. A short reporter with its own immediately preceding written name starts a new occurrence even when that name fits inside the distance allowance. The grouping stage recognizes that boundary through the existing grounded name reader; it does not write name fields. Short groups are exposed separately as `document.short_reporter_colocations`; `document.colocations` still contains full-citation groups. Stage 28.2 uses the window before the first member to quote the same case name for every member. Each member retains its own locator, pinpoint, and eventual root assignment. Grouping neither merges roots nor changes Id. attribution: the default Id. antecedent remains the last preceding resolved citation's root.

Id. has no separate locator or case-name normalization. Stage 32 retains its entire source span, including any explicitly joined pinpoint; the optional `pin_cite` stores normalized targets on the same creation node. A bare Id. has `pin_cite=None`. Its creation scorer reports `citation_span`, `pin_cite_span`, and `pin_cite_normalization`, counting a pinpoint outcome for every created Id., including absence. Stage 33 reports attachment precision independently. `await attribute_id_citations(document, review=False)` uses chronology alone; review mode audits every Id. because eyecite may omit a noncase antecedent. Rule and semantic decisions have separate citation-local nodes within the single attribution checkpoint, and each final attachment is written before the next Id. is processed. An uncreated supra or noncase eyecite event blocks rule attribution; a later supra stage does not retroactively rewrite the Id. checkpoint.

Id. discovery and normalization share a marker grammar built with `fuzzy_literal(..., whitespace=True, newline=True)`. This accepts absent or repeated whitespace at written punctuation and pinpoint joins, retaining exact original offsets. A dotless `Id` or `Ibid` needs an explicit pinpoint join; an ordinary identifier such as `ID` is not a citation. An unlabelled number does not extend a bare Id. site. Whitespace relaxation does not permit changes to the marker's letters. Edit-distance matching remains available in the shared grounding service for copied model evidence, rather than being applied indiscriminately to a two-letter discovery cue.

Supra uses the same whitespace relaxation. The shared `parsing/supra.py` reader supplies both discovery and isolated-quote normalization, preserving a complete written antecedent, optional volume, and adjacent pinpoint. Known names from the Document improve source boundaries but do not establish attribution or suppress a named citation whose root is missing. Bare internal cross-references do not instantiate case citations. Stages 34 through 38 separately create the citation, read its name, read its pinpoint, attribute by rule, and review unresolved attachments.

Short reporter reading discovers all eyecite anchors first, then extends pinpoints back to front, bounded by the next independently recognized citation's start. Citation creation remains in source order. This prevents a following reporter's volume from becoming a comma-separated pinpoint without changing the shared range/list/footnote grammar. Fuzziness alone cannot detect that mistake because the overextended quote can both ground and normalize successfully. A future selective source review could address competing spans or boundaries that the source reader does not recognize. Choosing an antecedent among parallel reporter roots remains a separate attribution question; this boundary pass neither merges roots nor introduces a model review stage.

Short reporter volume and edition identify candidate roots; the short page is a pinpoint and is not used as the root's first page. Pinpointed name references can precede a nearby full citation. Id. cannot refer forward: each rule decision follows citation chronology, then its optional semantic audit distinguishes case-linked docket documents from unrelated briefs, exhibits, statutes, and rules. Each reviewed Id. updates the tree before the next rule decision and review, and the next prompt receives the recent source sites and their current attachments.

There is no legacy `pin_page` loader. Saved leaf Documents with that old normalized field must be regenerated from a native root checkpoint.

The citation subclasses are `ShortReporterCitation`, `SupraCitation`, `IdCitation`, and `ReferenceCitation`. Their quoted fields, `attributions`, and `reviews` are append-only and point to citation-local nodes. Creation leaves root attachment unset until attribution runs. A model returns only `is_citation`, a candidate `root_index` or null, and a reason; the program writes the root assignment. Both rejected and unresolved attribution decisions attach to the same dummy head, using `root_id`; their review logs retain the distinct decisions and reasons. No separate unresolved collection or relationship field exists. Earlier real-root assignments remain in history, and a later supported decision may attach a dummy-head citation to a real root. A failed model call records its trace and preserves the prior rule state, including a dummy-head attachment when the rule could not resolve it.

Each `LeafReview` retains candidate root IDs, the decision or failure, and the complete shared `IvrRun`, including schema repair feedback and provider exchanges. No live model session is needed to restore the document or inspect an earlier stage. Pinpoint **reading** belongs here; validity against the cited opinion remains a separate later workflow.
