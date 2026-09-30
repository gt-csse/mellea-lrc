# Root validation

Validation entrypoints take and return a `Document`; each appends only the citation readings and judgments it produces. A stage that needs substantial private logic is a package with its `Document` entrypoint in `__init__.py`, including the reporter and docket model-review stages. Their candidate preparation, prompts, and reviewer implementations stay beside the stage. `reporter_exact/` holds rules shared by reporter lookup stages, while `reporter_review/` holds grounding and field updates shared by their model reviews. Provider clients and their response models stay in `providers/courtlistener/` and `providers/govinfo/`; saved lookup histories and judgments stay with the citation model. Bounded citation windows live in `model/citation_windows.py`; deterministic fuzzy matching and quote grounding live in `matching/`. Neither role imports a validation stage.

`reporter_root_lookup_cluster_retrieval(document)` runs at stage `12.1_reporter_root_lookup_cluster_retrieval` after extraction. It requires `10_roots` and retrieves each full reporter root once by normalized volume, reporter edition, and first page. The citation stores the query and complete CourtListener response. `reporter_root_lookup_docket_retrieval(document)` runs at stage `12.2_reporter_root_lookup_docket_retrieval`, fetching linked dockets needed for court comparison for both unique and bounded ambiguous results. Provider failures raise; they are not recorded as lookup misses. Neither retrieval stage makes field or identity judgments.

For a unique response, `reporter_root_lookup_unique_rule_judgment(document)` runs at stage `13.1_reporter_root_lookup_unique_rule_judgment` using only saved evidence. It compares the citation's latest case-name, court, and date readings with the cluster. For an adversarial name, the rule comparison requires both parties to appear separately in `caseNameFull`, allowing abbreviations from `reporters-db` without treating different full words as synonyms. A printed partial name cannot pass this complete-name rule and routes to model review. Courts are compared by recognized court ID, including a saved linked docket when the cluster omits its court. A written full date requires the same full date; a written year requires the same year. If either side has no date, the stage makes no date judgment.

Each available comparison appends a field-specific judgment referencing its citation reading and candidate index. A unique cluster is marked `CORRECT_IDENTITY` when the case name and all applicable court and date checks match and its listed locator does not conflict. A mismatch or unreadable comparison routes to `14_reporter_root_lookup_unique_llm_judgment`; the rule stage does not declare a wrong identity. After the same linked-docket retrieval, `13.2_reporter_root_lookup_ambiguous_rule_judgment` makes its judgments for multiple clusters without provider calls. No cluster routes to `reporter_root_search`.

```python
from pathlib import Path

from mellea_lrc.api import Document, reporter_root_lookup_cluster_retrieval, reporter_root_lookup_docket_retrieval, reporter_root_lookup_unique_rule_judgment

checkpoint = Path("reporter-roots.json")
document = Document.model_validate_json(checkpoint.read_text())
document = reporter_root_lookup_cluster_retrieval(document)
lookup_only = document.get_stage("12.1_reporter_root_lookup_cluster_retrieval")
checkpoint.write_text(lookup_only.model_dump_json())
document = reporter_root_lookup_docket_retrieval(document)
document = reporter_root_lookup_unique_rule_judgment(document)

before_lookup = document.get_stage("10_roots")
after_review = document.get_stage("13.1_reporter_root_lookup_unique_rule_judgment")
```

The lookup response, linked docket records, and later judgments are separate append-only nodes. Recovering stage `12.2` retains both provider results and removes the judgments, so rule review can be replayed without another lookup.

## Unique lookup model review

`reporter_root_lookup_unique_llm_judgment(document)` processes only roots routed to that stage by the first lookup. It reuses the saved single candidate. One model call rereads the nearby filing text, proposes any case-name, court, or date corrections, and compares all three fields with the candidate. Each field assessment explicitly says whether it proposes a replacement: `propose_replacement=True` requires a nonempty source quote, while `False` requires `quote=None`. A proposed correction is grounded to its actual source span before it becomes a new field reading; the reporter locator is fixed. The stage records the complete model repair trace and all three field judgments on one citation node. Its overall identity verdict is computed from those judgments: a mismatch is wrong identity, an adequately supported agreement is correct identity, and unresolved evidence routes to `reporter_root_search`.

The stages remain separate and return a complete `Document`:

```python
from mellea_lrc.api import reporter_root_lookup_unique_llm_judgment

document = await reporter_root_lookup_unique_llm_judgment(document)
before_review = document.get_stage("13.2_reporter_root_lookup_ambiguous_rule_judgment")
after_review = document.get_stage("14_reporter_root_lookup_unique_llm_judgment")
```

## Docket lookup review

After the reporter stages, `docket_root_lookup_courtlistener_retrieval(document)` saves the CourtListener search responses at `16_docket_root_lookup_courtlistener_retrieval`. `docket_root_lookup_courtlistener_llm_review(document)` rereads the citation, selects from the saved shortlist, and records separate docket-number, case-name, court, and date assessments at `17_docket_root_lookup_courtlistener_llm_review`. Each assessment explicitly proposes a replacement quote or keeps the current reading. Proposed quotes must ground to the appropriate filing window before they become append-only field readings; the case-name assessment also supplies its normalized name. If no record was shortlisted, the stage records unavailable field comparisons without a model call. A selected docket card's `dateFiled` is the case filing date. Its date assessment checks only whether the cited decision date could be on or after filing; it does not confirm that an opinion was issued on the cited day. A selected opinion record can instead support a comparison with its own opinion filing date. The review preserves the full model trace and does not yet issue an overall docket-root identity judgment.

## Locator evidence in other document bodies

After root lookup, unresolved roots can be checked for their reporter or docket locator in other opinions and filings. The high-level `validate_roots(document, retrospective_date=cutoff)` workflow runs this locator-first path after stage `19_docket_root_lookup_govinfo_llm_review`. Retrieval has one stage per source: CourtListener opinion bodies (`20_locator_body_courtlistener_opinion_retrieval`), CourtListener RECAP filing bodies (`21_locator_body_courtlistener_recap_retrieval`), and GovInfo opinion granules (`22_locator_body_govinfo_opinion_retrieval`). These stages search by locator and retain only fetched bodies containing that locator. They save responses, grounded excerpts with surrounding discussion, source identifiers, dates, and failures on the citation. A case-name-only hit cannot establish the locator's identity in this workflow.

`locator_body_llm_judgment(document)` is stage `23_locator_body_llm_judgment`. It selects a grounded occurrence, rereads the source filing, compares the two printed citations field by field, and records how the other document treats that citation. An explicit challenge to a fictitious or incorrect citation needs its own grounded context quote and produces `WRONG_IDENTITY`, even when the printed fields match. A mere mention leaves identity unresolved. An affirmative citation is judged from its field comparisons. These judgments use `basis=third_party`; the citation also retains the selected quote, context, field updates, and complete model trace. Case-name-led discovery is an optional continuation within `validate_roots`; its stages do not confirm the source locator.

The printed-field comparisons are distinct from the identity-field labels in the annotated benchmark: another document can repeat a false citation verbatim. The validation field report therefore keeps the last comparable judgments from stage `19`. It scores stage `23`'s overall identity verdict separately, then reports cumulative identity after that verdict takes precedence over any lookup-derived verdict.

Use the stages individually when developing a provider, or run their composition:

```python
from datetime import date

from mellea_lrc.api import (
    validate_roots,
    locator_body_courtlistener_opinion_retrieval,
    locator_body_courtlistener_recap_retrieval,
    locator_body_govinfo_opinion_retrieval,
    locator_body_llm_judgment,
)

cutoff = date(2024, 1, 1)  # Optional filing date for retrospective evaluation.
starting_document = document
document = locator_body_courtlistener_opinion_retrieval(document, retrospective_date=cutoff)
document = locator_body_courtlistener_recap_retrieval(document, retrospective_date=cutoff)
document = locator_body_govinfo_opinion_retrieval(document, retrospective_date=cutoff)
document = await locator_body_llm_judgment(document)

# Alternatively, start from the same earlier checkpoint:
document = await validate_roots(starting_document, retrospective_date=cutoff)
```

With a cutoff, retrieved evidence must have its own reliable issue date on or before that date; undated evidence is excluded. Omitting the cutoff permits later evidence for ordinary research. Every stage returns a serializable `Document`, and `get_stage(...)` recovers its checkpoint.

To continue through intended-case discovery, call `validate_roots(document, search_other_fields=True)`. Stages 24 through 27 search other fields and save a possible intended authority; they remain part of the same validation workflow and report. A completed checkpoint skips earlier stages, so a saved stage-23 Document can continue without repeating retrieval.
