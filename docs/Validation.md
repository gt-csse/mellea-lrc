# Root validation

Validation entrypoints take and return a `Document`; each appends only the citation readings and judgments it produces. A stage that needs substantial private logic is a package with its `Document` entrypoint in `__init__.py`, including the reporter and docket model-review stages. Their candidate preparation, prompts, and reviewer implementations stay beside the stage. `reporter_exact/` holds rules shared by reporter lookup stages, while `reporter_review/` holds grounding and field updates shared by their model reviews. Provider clients stay in `courtlistener/` and `govinfo/`; typed lookup responses and judgments stay with the citation model. Bounded citation windows live in `model/citation_windows.py`; deterministic fuzzy matching and quote grounding live in `matching/`. Neither role imports a validation stage.

`reporter_root_lookup(document)` runs as stage `12_reporter_root_lookup` after extraction, which ends with `11_docket_root_equivalence_review` when that optional review is enabled. It requires the `10_roots` checkpoint. It looks up each full reporter root once by normalized volume, reporter edition, and first page. Docket roots and repeated reporter occurrences are untouched. The citation stores one `ReporterExactLookup` object containing the query and complete CourtListener response. Provider failures raise; they are not recorded as lookup misses.

When the response has one cluster, the stage compares the citation's latest case-name, court, and date readings with that cluster. For an adversarial name, the rule comparison requires both parties to appear separately in `caseNameFull`, allowing abbreviations from `reporters-db` without treating different full words as synonyms. A printed partial name cannot pass this complete-name rule; its case-name comparison is unavailable and routes to model review. Courts are compared by recognized court ID when the response has one. CourtListener exact-lookup clusters can omit court metadata: in that case, a court inferred only from the reporter is left unjudged, while an explicitly written court routes to review. A written full date requires the same full date; a written year requires the same year. If either side has no date, the stage makes no date judgment and does not penalize identity for it.

Each available comparison appends a field-specific judgment referencing the absolute index of its citation reading and the index of the cluster in the saved response. It does not copy either value. A unique cluster is marked `CORRECT_IDENTITY` when the case name and all applicable court and date checks match and its listed locator does not conflict. A mismatch or unreadable comparison routes to `14_reporter_root_lookup_unique_llm`; the rule stage does not declare a wrong identity. Multiple clusters route to `13_reporter_root_lookup_ambiguous` without selecting one, and no cluster routes to `reporter_root_search`. The latest identity judgment names the next stage directly. The lookup can retrieve zero, one, or several records; only its one-record branch makes a rule-based identity decision.

```python
from pathlib import Path

from mellea_lrc.api import Document, reporter_root_lookup

checkpoint = Path("reporter-roots.json")
document = Document.model_validate_json(checkpoint.read_text())
document = reporter_root_lookup(document)
checkpoint.write_text(document.model_dump_json())

before_lookup = document.get_stage("10_roots")
after_lookup = document.get_stage("12_reporter_root_lookup")
```

The single lookup object, field judgments, and identity judgment are part of the citation's append-only history. A later checkpoint retains them, while `get_stage("10_roots")` removes them from the recovered earlier view.

## Unique lookup model review

`reporter_root_lookup_unique_llm(document)` processes only roots routed to that stage by the first lookup. It reuses the saved single candidate. One model call rereads the nearby filing text, proposes any case-name, court, or date corrections, and compares all three fields with the candidate. Each field assessment explicitly says whether it proposes a replacement: `propose_replacement=True` requires a nonempty source quote, while `False` requires `quote=None`. A proposed correction is grounded to its actual source span before it becomes a new field reading; the reporter locator is fixed. The stage records the complete model repair trace and all three field judgments on one citation node. Its overall identity verdict is computed from those judgments: a mismatch is wrong identity, an adequately supported agreement is correct identity, and unresolved evidence routes to `reporter_root_search`.

The stages remain separate and return a complete `Document`:

```python
from mellea_lrc.api import reporter_root_lookup_unique_llm

document = await reporter_root_lookup_unique_llm(document)
before_review = document.get_stage("12_reporter_root_lookup")
after_review = document.get_stage("14_reporter_root_lookup_unique_llm")
```

## Docket lookup review

After the reporter stages, `docket_root_lookup(document)` saves the CourtListener search responses at `16_docket_root_lookup`. `docket_root_lookup_review(document)` rereads the citation, selects from the saved shortlist, and records separate docket-number, case-name, court, and date assessments at `17_docket_root_lookup_review`. Each assessment explicitly proposes a replacement quote or keeps the current reading. Proposed quotes must ground to the appropriate filing window before they become append-only field readings; the case-name assessment also supplies its normalized name. If no record was shortlisted, the stage records unavailable field comparisons without a model call. A selected docket card's `dateFiled` is the case filing date. Its date assessment checks only whether the cited decision date could be on or after filing; it does not confirm that an opinion was issued on the cited day. A selected opinion record can instead support a comparison with its own opinion filing date. The review preserves the full model trace and does not yet issue an overall docket-root identity judgment.

## Locator evidence in other document bodies

After root lookup, unresolved roots can be checked for their reporter or docket locator in other opinions and filings. The high-level `validate_roots(document, retrospective_date=cutoff)` workflow runs this locator-first path after stage `19_govinfo_docket_lookup_review`. Retrieval has one stage per source: CourtListener opinion bodies (`20_courtlistener_opinion_locator_body_search`), CourtListener RECAP filing bodies (`21_courtlistener_recap_locator_body_search`), and GovInfo opinion granules (`22_govinfo_opinion_locator_body_search`). These stages search by locator and retain only fetched bodies containing that locator. They save responses, grounded excerpts with surrounding discussion, source identifiers, dates, and failures on the citation. A case-name-only hit cannot establish the locator's identity in this workflow.

`review_locator_body_evidence(document)` is stage `23_locator_body_review`. It selects a grounded occurrence, rereads the source filing, compares the two printed citations field by field, and records how the other document treats that citation. An explicit challenge to a fictitious or incorrect citation needs its own grounded context quote and produces `WRONG_IDENTITY`, even when the printed fields match. A mere mention leaves identity unresolved. An affirmative citation is judged from its field comparisons. These judgments use `basis=third_party`; the citation also retains the selected quote, context, field updates, and complete model trace. Later case-name-led discovery is separate and is not part of this workflow.

The printed-field comparisons are distinct from the identity-field labels in the annotated benchmark: another document can repeat a false citation verbatim. The validation field report therefore keeps the last comparable judgments from stage `19`. It scores stage `23`'s overall identity verdict separately, then reports cumulative identity after that verdict takes precedence over any lookup-derived verdict.

Use the stages individually when developing a provider, or run their composition:

```python
from datetime import date

from mellea_lrc.api import (
    corroborate_root_locator_bodies,
    courtlistener_opinion_locator_body_search,
    courtlistener_recap_locator_body_search,
    govinfo_opinion_locator_body_search,
    review_locator_body_evidence,
)

cutoff = date(2024, 1, 1)  # Optional filing date for retrospective evaluation.
starting_document = document
document = courtlistener_opinion_locator_body_search(document, retrospective_date=cutoff)
document = courtlistener_recap_locator_body_search(document, retrospective_date=cutoff)
document = govinfo_opinion_locator_body_search(document, retrospective_date=cutoff)
document = await review_locator_body_evidence(document)

# Alternatively, start from the same earlier checkpoint:
document = await corroborate_root_locator_bodies(starting_document, retrospective_date=cutoff)
```

With a cutoff, retrieved evidence must have its own reliable issue date on or before that date; undated evidence is excluded. Omitting the cutoff permits later evidence for ordinary research. Every stage returns a serializable `Document`, and `get_stage(...)` recovers its checkpoint.
