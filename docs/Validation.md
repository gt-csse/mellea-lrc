# Validation

`validate_roots` composes `reporter_lookup`, `docket_lookup`, `locator_body_corroboration`, and optional `intended_case_discovery`. `validate_pincite` composes `opinion_preparation`, `citation_preparation`, and `support_review`. Their semantic entrypoints live in `workflows/<workflow>/<stage>.py`; atomic retrieval and judgment writers remain in `validation/`. The [execution catalog](../src/mellea_lrc/model/execution.py) defines membership and order. See [Architecture](Architecture.md) for exact stage and substage checkpoint recovery.

Validation entrypoints take and return a `Document`; each appends only the citation readings and judgments it produces. A substage that needs substantial private logic is a package with its `Document` entrypoint in `__init__.py`, including the reporter and docket model-review substages. Their candidate preparation, prompts, and reviewer implementations stay beside the substage. `reporter_exact/` holds rules shared by reporter lookup substages, while `reporter_review/` holds grounding and field updates shared by their model reviews. `docket_retrieval/` owns shared docket shortlist scoring and failure records. `docket_review/` owns source-window grounding and field updates shared by the CourtListener and GovInfo docket reviews. Body retrieval substages use the shared source services in `body_search/_courtlistener.py` and `body_search/_govinfo.py`. Provider clients and their response models stay in `providers/courtlistener/` and `providers/govinfo/`; saved lookup histories and judgments stay with the citation model. Bounded citation windows live in `model/citation_windows.py`; deterministic fuzzy matching and quote grounding live in `matching/`. Neither role imports a validation substage.

`reporter_root_lookup_cluster_retrieval(document)` runs at substage `validate_roots.reporter_lookup.cluster_retrieval` after extraction. It requires `grow_roots.root_formation.rule` and retrieves each full reporter root once by normalized volume, reporter edition, and first page. The citation stores the query and complete CourtListener response. `reporter_root_lookup_docket_retrieval(document)` runs at substage `validate_roots.reporter_lookup.docket_retrieval`, fetching linked dockets needed for court comparison for both unique and bounded ambiguous results. Provider failures raise; they are not recorded as lookup misses. Neither retrieval substage makes field or identity judgments.

For a unique response, `reporter_root_lookup_unique_rule_judgment(document)` runs at substage `validate_roots.reporter_lookup.unique_rule_judgment` using only saved evidence. It compares the citation's latest case-name, court, and date readings with the cluster. For an adversarial name, the rule comparison requires both parties to appear separately in `caseNameFull`, allowing abbreviations from `reporters-db` without treating different full words as synonyms. A printed partial name cannot pass this complete-name rule and routes to model review. Courts are compared by recognized court ID, including a saved linked docket when the cluster omits its court. A written full date requires the same full date; a written year requires the same year. If either side has no date, the substage makes no date judgment.

Each available comparison appends a field-specific judgment referencing its citation reading and candidate index. A unique cluster is marked `CORRECT_IDENTITY` when the case name and all applicable court and date checks match and its listed locator does not conflict. A mismatch or unreadable comparison routes to `validate_roots.reporter_lookup.unique_llm_judgment`; the rule substage does not declare a wrong identity. After the same linked-docket retrieval, `validate_roots.reporter_lookup.ambiguous_rule_judgment` makes its judgments for multiple clusters without provider calls. A missing result or unnormalizable locator routes to `reporter_root_search`.

```python
from pathlib import Path

from mellea_lrc.api import (
    Document,
    reporter_root_lookup_cluster_retrieval,
    reporter_root_lookup_docket_retrieval,
    reporter_root_lookup_unique_rule_judgment,
)

checkpoint = Path("reporter-roots.json")
document = Document.model_validate_json(checkpoint.read_text())
document = reporter_root_lookup_cluster_retrieval(document)
lookup_only = document.get_substage("validate_roots.reporter_lookup.cluster_retrieval")
checkpoint.write_text(lookup_only.model_dump_json())
document = reporter_root_lookup_docket_retrieval(document)
document = reporter_root_lookup_unique_rule_judgment(document)

before_lookup = document.get_substage("grow_roots.root_formation.rule")
after_review = document.get_substage("validate_roots.reporter_lookup.unique_rule_judgment")
```

The lookup response, linked docket records, and later judgments are separate append-only nodes. Recovering `validate_roots.reporter_lookup.docket_retrieval` retains both provider results and removes the judgments, so rule review can be replayed without another lookup.

## Unique lookup model review

`reporter_root_lookup_unique_llm_judgment(document)` processes only roots routed to that substage by the first lookup. It reuses the saved single candidate. One model call rereads the nearby filing text, proposes any case-name, court, or date corrections, and compares all three fields with the candidate. Each field assessment explicitly says whether it proposes a replacement: `propose_replacement=True` requires a nonempty source quote, while `False` requires `quote=None`. A proposed correction is grounded to its actual source span before it becomes a new field reading; the reporter locator is fixed. The substage records the complete model repair trace and all three field judgments on one citation node. Its overall identity verdict is computed from those judgments: a mismatch is wrong identity, an adequately supported agreement is correct identity, and unresolved evidence routes to `reporter_root_search`.

The substages remain separate and return a complete `Document`:

```python
from mellea_lrc.api import reporter_root_lookup_unique_llm_judgment

document = await reporter_root_lookup_unique_llm_judgment(document)
before_review = document.get_substage("validate_roots.reporter_lookup.unique_rule_judgment")
after_review = document.get_substage("validate_roots.reporter_lookup.unique_llm_judgment")
```

## Docket lookup review

After the reporter substages, `docket_root_lookup_courtlistener_retrieval(document)` saves the CourtListener search responses at `validate_roots.docket_lookup.courtlistener_retrieval`. `docket_root_lookup_courtlistener_llm_review(document)` rereads the citation, selects from the saved shortlist, and records separate docket-number, case-name, court, and date assessments at `validate_roots.docket_lookup.courtlistener_review`. Each assessment explicitly proposes a replacement quote or keeps the current reading. Proposed quotes must ground to the appropriate filing window before they become append-only field readings; the case-name assessment also supplies its normalized name. If no record was shortlisted, the substage records unavailable field comparisons without a model call. A selected docket card's `dateFiled` is the case filing date. Its date assessment checks only whether the cited decision date could be on or after filing; it does not confirm that an opinion was issued on the cited day. A selected opinion record can instead support a comparison with its own opinion filing date. The review preserves the full model trace and does not yet issue an overall docket-root identity judgment.

## Locator evidence in other document bodies

After root lookup, unresolved roots can be checked for their reporter or docket locator in other opinions and filings. The `validate_roots(document, retrospective_date=cutoff)` workflow runs this path after the `docket_lookup` stage. Its final `identity_aggregation` substage combines the saved docket-number, case-name, court, and date judgments without retrieval or a model call; unresolved identities proceed to body corroboration. Retrieval has one substage per source: CourtListener opinion bodies (`validate_roots.locator_body_corroboration.courtlistener_opinion_retrieval`), CourtListener RECAP filing bodies (`validate_roots.locator_body_corroboration.courtlistener_recap_retrieval`), and GovInfo opinion granules (`validate_roots.locator_body_corroboration.govinfo_opinion_retrieval`). These substages search by locator and retain only fetched bodies containing that locator. They save responses, grounded excerpts with surrounding discussion, source identifiers, dates, and failures on the citation. A case-name-only hit cannot establish the locator's identity in this workflow.

`locator_body_llm_judgment(document)` is substage `validate_roots.locator_body_corroboration.llm_judgment`. It selects a grounded occurrence, rereads the source filing, compares the two printed citations field by field, and records how the other document treats that citation. An explicit challenge to a fictitious or incorrect citation needs its own grounded context quote and produces `WRONG_IDENTITY`, even when the printed fields match. A mere mention leaves identity unresolved. An affirmative citation is judged from its field comparisons. These judgments use `basis=third_party`; the citation also retains the selected quote, context, field updates, and complete model trace. Case-name-led discovery is an optional continuation within `validate_roots`; its substages do not confirm the source locator.

The printed-field comparisons are distinct from the identity-field labels in the annotated benchmark: another document can repeat a false citation verbatim. The validation field report therefore keeps the last comparable judgments from `validate_roots.docket_lookup.govinfo_review`. It scores `validate_roots.locator_body_corroboration.llm_judgment`'s overall identity verdict separately, then reports cumulative identity after that verdict takes precedence over any lookup-derived verdict.

Use the substages individually when developing a provider, or run their composition:

```python
from datetime import date

from mellea_lrc.api import corroborate_locator_bodies

cutoff = date(2024, 1, 1)  # Optional filing date for retrospective evaluation.
document = await corroborate_locator_bodies(document, retrospective_date=cutoff)
retrievals_only = document.get_substage("validate_roots.locator_body_corroboration.govinfo_opinion_retrieval")
complete_bodies = document.get_stage("validate_roots.locator_body_corroboration")
```

For the complete root-validation workflow, call `await validate_roots(roots, retrospective_date=cutoff)` from a Document with formed roots. `lookup_reporter_roots`, `lookup_docket_roots`, and `corroborate_locator_bodies` also expose its individual stages.

With a cutoff, retrieved evidence must have its own reliable issue date on or before that date; undated evidence is excluded. Omitting the cutoff permits later evidence for ordinary research. Every substage returns a serializable `Document`, and `get_substage(...)` recovers its checkpoint.

To continue through intended-case discovery, call `await validate_roots(document, search_other_fields=True)` or the independent `await discover_intended_cases(document)` stage. Its three retrieval substages search CourtListener opinions, CourtListener RECAP filings, and GovInfo opinions by other fields; `llm_selection` saves a possible intended authority. These remain part of the same validation workflow and report. A saved completed `locator_body_corroboration` stage can continue without repeating its retrievals.

## Reporter opinion pages

The reporter pinpoint workflow has three stages. Opinion preparation retrieves all cluster writings and indexes explicit pages. Citation preparation resolves each occurrence's pages and writing, reads its filing proposition, and prepares evidence references. Support review checks selected pages, falls back to full opinions when needed, and records a final verdict. Each atomic step has its own checkpoint.

```python
from mellea_lrc.api import prepare_citation_evidence, prepare_root_opinions, review_citation_support

document = prepare_root_opinions(document)
document = await prepare_citation_evidence(document)
document = await review_citation_support(document)
after_page_review = document.get_substage("validate_pincite.support_review.page_review")
complete_support = document.get_stage("validate_pincite.support_review")
```

`await validate_pincite(document)` composes all three stages and resumes a valid partial checkpoint.

The root holds downloaded opinions and the shared page index. Each occurrence
holds references to its locator/pinpoint readings and candidate pages. Selection
uses reporter pagination and citation context, with no automatic majority/dissent
priority. Ranges can cross writings. Null choices stay unresolved. Opinion
retrieval, pagination indexing, page resolution, and writing selection neither
change identity nor decide whether an opinion supports an argument. Proposition
reading and the following evidence and review substages judge the attributed use.

The opinion source binding contains the original cluster metadata. Retrieval,
pagination, and support objects do not depend on identity's unique/ambiguous
lookup structure. Requiring a correct identity before default source retrieval
is a workflow policy. The cited source is distinct from a third-party document
that merely corroborated identity.

A root retains source bodies and page indexes. Each occurrence retains append-only
proposition readings, prepared page references, grounded opinion evidence,
support reviews, and pinpoint judgments. Judgment and review indexes point to
the exact earlier readings; source quotes have spans into the retained filing
or indexed opinion. Every review saves its reason and full IVR repair trace.
Native Pydantic serialization, `get_substage`, and `get_stage` preserve these references.

Page review receives only selected page text. A negative page review cannot
establish that the full opinion lacks support. Full review runs when pages are
missing, ambiguous, uncertain, or fail to settle the attribution. Identical root
source text is sent in a stable system prefix, with the occurrence's proposition
and target in the dynamic instruction. Full reviews are grouped by root for
KV-cache reuse. No retrieval is repeated in these review substages.

The shared source prefix is independent of the occurrence and stays byte
identical across its full-opinion reviews. Prompt refinements belong in the
dynamic instruction unless they describe shared source evidence. Every review
assesses all material assertions and qualifications in the grounded attribution;
full-opinion fallback expands the available evidence while preserving that
attribution and the standard of support. A narrower supported proposition does
not establish a broader claim attributed by the filing.

`CORRECT_PINCITE` follows the dataset's support convention: support on another
page still establishes the attributed use. Each review and final judgment carries
`pagination_available`, `correct_page`, and typed `found_pages` ranges. Availability
describes usable cited-reporter pagination in the source used for that judgment.
Unavailable pagination requires a null page judgment and no recovered pages.
With usable pagination, the page judgment may be true, false, or null; null means
placement remains uncertain. Content support and page placement are independent:
a passage on the correct page may contradict the filing's attributed claim.
A range needs the relevant material somewhere within it, not on every page.
Recovered pages are known locations, not an exhaustive list. Finding a passage
elsewhere alone does not establish a wrong written target.

Mellea validates the output schema and cross-field invariants, then grounds each
quote with shared whitespace relaxation and 98% edit-distance matching. Where
source markers establish pagination, reported pages must locate those quoted
passages. Failed requirements enter the ordinary repair loop and remain in the
saved IVR trace. The `support_review.judgment` substage copies the accepted page assessment and checks retained
quote spans again. Page assessment is evaluated independently of content support;
unknown gold locations are never treated as negative page judgments.
A wrong support judgment needs affirmative contradiction in a full review or
absence established after reading every available subopinion. Missing/empty
source text and model failures yield `UNDETERMINED`, never a negative by themselves.
TOA entries and occurrences without pinpoints receive no support verdict.
Docket pinpoint validation remains outside this reporter workflow.

Preprocessing serializes typed TOA components in `index_spans`, and citation
creation attaches a node-bound tag to occurrences inside them. Root identity
checks and shared root-opinion retrieval still include TOA citations; proposition
and support substages skip the tagged occurrence. Body leaves do not inherit a
root's TOA tag. Older untagged citations need replay through creation to acquire
it. Unrecognized TOA entries may also be read as having no proposition, which
skips support review and final judgment.

Citation signals, quotations, writing parentheticals, and designated footnotes
are interpreted in the filing's context, rather than reducing support to shared
words. The design uses [Cornell's citation guidance](https://www.law.cornell.edu/citation/6-300),
[FRAP 28's description of tables of authorities](https://www.law.cornell.edu/rules/frap/rule_28),
and [CourtListener's cluster/opinion documentation](https://wiki.free.law/c/courtlistener/help/api/rest/v4/case-law).
