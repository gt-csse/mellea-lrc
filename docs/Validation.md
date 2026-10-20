# Validation

`validate_roots_citation_identity` composes `lookup_reporter_roots`, `lookup_docket_roots`, `corroborate_locator_bodies`, and optional `discover_intended_cases`. `validate_all_citation_content` composes `prepare_root_opinions`, `prepare_citation_evidence`, and `review_citation_content`. Each stage's path uses the same name as its API. Their semantic entrypoints live in `workflows/<workflow>/<stage>.py`; atomic retrieval and judgment writers remain in `validation/`. The [execution catalog](../src/mellea_lrc/model/execution.py) defines membership and order. See [Architecture](Architecture.md) for exact stage and substage checkpoint recovery.

Validation entrypoints take and return a `Document`; each appends only the citation readings and judgments it produces. A substage that needs substantial private logic is a package with its `Document` entrypoint in `__init__.py`, including the reporter and docket model-review substages. Their candidate preparation, prompts, and reviewer implementations stay beside the substage. `reporter_exact/` holds rules shared by reporter lookup substages, while `reporter_review/` holds grounding and field updates shared by their model reviews. `docket_retrieval/` owns shared docket shortlist scoring and failure records. `docket_review/` owns source-window grounding and field updates shared by the CourtListener and GovInfo docket reviews. Body retrieval substages use the shared source services in `body_search/_courtlistener.py` and `body_search/_govinfo.py`. Provider clients and their response models stay in `providers/courtlistener/` and `providers/govinfo/`; saved lookup histories and judgments stay with the citation model. Bounded citation windows live in `model/citation_windows.py`; deterministic fuzzy matching and quote grounding live in `matching/`. Neither role imports a validation substage.

`retrieve_reporter_root_clusters(document)` runs at substage `validate_roots_citation_identity.lookup_reporter_roots.retrieve_clusters` after extraction. It requires `grow_roots.form_root_groups.form_roots` and retrieves each full reporter root once by normalized volume, reporter edition, and first page. The citation stores the query and complete CourtListener response. `retrieve_reporter_root_dockets(document)` runs at substage `validate_roots_citation_identity.lookup_reporter_roots.retrieve_dockets`, fetching linked dockets needed for court comparison for both unique and bounded ambiguous results. Provider failures raise; they are not recorded as lookup misses. Neither retrieval substage makes field or identity judgments.

For a unique response, `judge_unique_reporter_roots_rule(document)` runs at substage `validate_roots_citation_identity.lookup_reporter_roots.judge_unique_rule` using only saved evidence. It compares the citation's latest case-name, court, and date readings with the cluster. For an adversarial name, the rule comparison requires both parties to appear separately in `caseNameFull`, allowing abbreviations from `reporters-db` without treating different full words as synonyms. A printed partial name cannot pass this complete-name rule and routes to model review. Courts are compared by recognized court ID, including a saved linked docket when the cluster omits its court. A written full date requires the same full date; a written year requires the same year. If either side has no date, the substage makes no date judgment.

Each available comparison appends a field-specific judgment referencing its citation reading and candidate index. A unique cluster is marked `CORRECT_IDENTITY` when the case name and all applicable court and date checks match and its listed locator does not conflict. A mismatch or unreadable comparison routes to `validate_roots_citation_identity.lookup_reporter_roots.review_unique_llm`; the rule substage does not declare a wrong identity. After the same linked-docket retrieval, `validate_roots_citation_identity.lookup_reporter_roots.judge_ambiguous_rule` makes its judgments for multiple clusters without provider calls. A missing result or unnormalizable locator routes to `search_reporter_roots`.

```python
from pathlib import Path

from mellea_lrc.api import (
    Document,
    retrieve_reporter_root_clusters,
    retrieve_reporter_root_dockets,
    judge_unique_reporter_roots_rule,
)

checkpoint = Path("reporter-roots.json")
document = Document.model_validate_json(checkpoint.read_text())
document = retrieve_reporter_root_clusters(document)
lookup_only = document.get_substage(
    "validate_roots_citation_identity.lookup_reporter_roots.retrieve_clusters"
)
checkpoint.write_text(lookup_only.model_dump_json())
document = retrieve_reporter_root_dockets(document)
document = judge_unique_reporter_roots_rule(document)

before_lookup = document.get_substage("grow_roots.form_root_groups.form_roots")
after_review = document.get_substage(
    "validate_roots_citation_identity.lookup_reporter_roots.judge_unique_rule"
)
```

Lookup responses, linked docket records, and later judgments are stored separately on the citation and linked to citation-local nodes. Recovering `validate_roots_citation_identity.lookup_reporter_roots.retrieve_dockets` retains both provider results and removes the judgments, so rule review can be replayed without another lookup.

## Unique lookup model review

`review_unique_reporter_roots_llm(document)` processes only roots routed to that substage by the unique rule judgment. It reuses the saved single candidate. One model call rereads the nearby filing text, proposes any case-name, court, or date corrections, and compares all three fields with the candidate. Each field assessment has an explicit `action`: `keep`, `replace`, or `not_stated`. Only `replace` requires a nonempty filing quote. `keep` and `not_stated` require `quote=None`; clearing also requires an `unavailable` field judgment and no case-name normalization. A proposed correction is grounded to its actual source span before it becomes a new field reading; the reporter locator is fixed. A `not_stated` correction appends a `NotStatedField` reading at that node; it preserves the prior quote and normalization in history. It means the extraction was unsupported, not that a printed value disagrees with the record. The substage records the complete model repair trace and all three field judgments on one citation node. Its overall identity verdict is computed from those judgments: a mismatch is wrong identity, an adequately supported agreement is correct identity, and unresolved evidence routes to `search_reporter_roots`.

The substages remain separate and return a complete `Document`:

```python
from mellea_lrc.api import review_unique_reporter_roots_llm

document = await review_unique_reporter_roots_llm(document)
before_review = document.get_substage(
    "validate_roots_citation_identity.lookup_reporter_roots.judge_unique_rule"
)
after_review = document.get_substage(
    "validate_roots_citation_identity.lookup_reporter_roots.review_unique_llm"
)
```

## Docket lookup review

After the reporter substages, `retrieve_courtlistener_docket_roots(document)` saves the CourtListener search responses at `validate_roots_citation_identity.lookup_docket_roots.retrieve_courtlistener_dockets`. `review_courtlistener_docket_roots_llm(document)` rereads the citation, selects from the saved shortlist, and records separate docket-number, case-name, court, and date assessments at `validate_roots_citation_identity.lookup_docket_roots.review_courtlistener_dockets_llm`. Each assessment explicitly proposes a replacement quote or keeps the current reading. Proposed quotes must ground to the appropriate filing window before they become append-only field readings; the case-name assessment also supplies its normalized name. If no record was shortlisted, the substage records unavailable field comparisons without a model call. A selected docket card's `dateFiled` is the case filing date. Its date assessment checks only whether the cited decision date could be on or after filing; it does not confirm that an opinion was issued on the cited day. A selected opinion record can instead support a comparison with its own opinion filing date. The review preserves the full model trace and does not yet issue an overall docket-root identity judgment.

## Locator evidence in other document bodies

After root lookup, unresolved roots can be checked for their reporter or docket locator in other opinions and filings. The `validate_roots_citation_identity(document, retrospective_date=cutoff)` workflow runs this path after the `lookup_docket_roots` stage. Its final `judge_identity_from_fields` substage combines the saved docket-number, case-name, court, and date judgments without retrieval or a model call; unresolved identities proceed to body corroboration. Retrieval has one substage per source: CourtListener opinion bodies (`validate_roots_citation_identity.corroborate_locator_bodies.retrieve_courtlistener_opinions`), CourtListener RECAP filing bodies (`validate_roots_citation_identity.corroborate_locator_bodies.retrieve_courtlistener_filings`), and GovInfo opinion granules (`validate_roots_citation_identity.corroborate_locator_bodies.retrieve_govinfo_opinions`). These substages search by locator and retain only fetched bodies containing that locator. They save responses, grounded excerpts with surrounding discussion, source identifiers, dates, and failures on the citation. A case-name-only hit cannot establish the locator's identity in this workflow.

`review_locator_bodies_llm(document)` is substage `validate_roots_citation_identity.corroborate_locator_bodies.review_locator_bodies_llm`. It selects a grounded occurrence, rereads the source filing, compares the two printed citations field by field, and records how the other document treats that citation. An explicit challenge to a fictitious or incorrect citation needs its own grounded context quote and produces `WRONG_IDENTITY`, even when the printed fields match. A mere mention leaves identity unresolved. The model issues the final identity assessment from the printed comparisons and surrounding evidence; it is not computed from field agreement alone. `PARTIALLY_CORROBORATED` distinguishes a corroborated docket case from an unconfirmed cited decision or date. Those roots remain eligible for further discovery. These judgments use `basis=third_party`; the citation also retains the selected quote, context, field updates, and complete model trace. Case-name-led discovery is an optional continuation within `validate_roots_citation_identity`; its substages do not confirm the source locator.

The printed-field comparisons are distinct from the identity-field labels in the annotated benchmark: another document can repeat a false citation verbatim. The validation field report therefore keeps the last comparable judgments from `validate_roots_citation_identity.lookup_docket_roots.review_govinfo_dockets_llm`. It scores `validate_roots_citation_identity.corroborate_locator_bodies.review_locator_bodies_llm`'s overall identity verdict separately, then reports cumulative identity after that verdict takes precedence over any lookup-derived verdict. Strict identity metrics do not count `PARTIALLY_CORROBORATED` as correct identity; the separate `identity_with_partial` metrics include it and appear in parentheses beside the strict metrics.

Use the substages individually when developing a provider, or run their composition:

```python
from datetime import date

from mellea_lrc.api import corroborate_locator_bodies

cutoff = date(2024, 1, 1)  # Optional filing date for retrospective evaluation.
document = await corroborate_locator_bodies(document, retrospective_date=cutoff)
retrievals_only = document.get_substage(
    "validate_roots_citation_identity.corroborate_locator_bodies.retrieve_govinfo_opinions"
)
complete_bodies = document.get_stage("validate_roots_citation_identity.corroborate_locator_bodies")
```

For the complete root-validation workflow, call `await validate_roots_citation_identity(roots, retrospective_date=cutoff)` from a Document with formed roots. `lookup_reporter_roots`, `lookup_docket_roots`, and `corroborate_locator_bodies` also expose its individual stages.

With a cutoff, retrieved evidence must have its own reliable issue date on or before that date; undated evidence is excluded. Omitting the cutoff permits later evidence for ordinary research. Every substage returns a serializable `Document`, and `get_substage(...)` recovers its checkpoint.

To continue through intended-case discovery, call `await validate_roots_citation_identity(document, search_other_fields=True, retrospective_date=cutoff)` or the independent `await discover_intended_cases(document, retrospective_date=cutoff)` stage. Resumed searches must retain the same cutoff as their saved evidence. Its three retrieval substages search CourtListener opinions, CourtListener RECAP filings, and GovInfo opinions by other fields; `review_intended_cases_llm` saves a possible intended authority. These remain part of the same validation workflow and report. A saved completed `corroborate_locator_bodies` stage can continue without repeating its retrievals.

## Reporter opinion pages

`validate_all_citation_content` currently handles reporter citation content through three stages. `prepare_root_opinions` retrieves all cluster writings and indexes explicit pages. `prepare_citation_evidence` resolves each occurrence's pages and writing, reads its filing proposition, and prepares evidence references. `review_citation_content` checks selected pages, falls back to full opinions when needed, and records a final verdict. Each atomic step has its own checkpoint.

```python
from mellea_lrc.api import prepare_citation_evidence, prepare_root_opinions, review_citation_content

document = prepare_root_opinions(document)
document = await prepare_citation_evidence(document)
document = await review_citation_content(document)
after_page_review = document.get_substage(
    "validate_all_citation_content.review_citation_content.review_cited_pages_llm"
)
complete_support = document.get_stage("validate_all_citation_content.review_citation_content")
```

`await validate_all_citation_content(document)` composes all three stages and resumes a valid partial checkpoint.

The root holds downloaded opinions and the shared page index. Each occurrence
holds references to its locator/pinpoint readings and candidate pages. Selection
uses reporter pagination and citation context, with no automatic majority/dissent
priority. Ranges can cross writings. Null choices stay unresolved. Opinion
retrieval, pagination indexing, page resolution, and writing selection neither
change identity nor decide whether an opinion supports an argument. Proposition
reading preserves the filing's attributed use; evidence preparation saves source
references. The support reviews judge whether the opinion supports that use.

The opinion source binding contains the original cluster metadata. Retrieval,
pagination, and support objects do not depend on identity's unique/ambiguous
lookup structure. Requiring a correct identity before default source retrieval
is a workflow policy. The cited source is distinct from a third-party document
that merely corroborated identity.

A root retains source bodies and page indexes. Each occurrence retains append-only
proposition readings, prepared page references, grounded opinion evidence,
support reviews, and pinpoint judgments. Judgment and review indexes point to
the exact earlier readings; source quotes have spans into the retained filing
or indexed opinion. Review records retain reasons and the IVR repair trace when available.
Native Pydantic serialization, `get_substage`, and `get_stage` preserve these references.

Page review receives only selected page text. Review ends early only when an
accepted decision has `result=supported` and `correct_page=true`. Every other
outcome proceeds to full-opinion review, including a contradiction at the written
page, supported content at a different or unknown page, and failed page review.
A page-only negative remains provisional. Missing pages enter the full-opinion
route directly. Both review scopes use the same reviewer, but their named profiles
are independently assigned in `.env`. No retrieval is repeated in these substages.

Full review receives the saved opinion as explicitly labeled page text. Page
references identify the opinion and pagination namespace as well as the printed
label, so a parallel reporter's page number cannot substitute for the cited
reporter's page. Each proposed evidence quote names its source page or pages;
grounding searches only that named source text and retains the resolved opinion
span. Material outside indexed pages remains available as unpaginated source
text so full review can still assess the complete attribution. When usable
pagination is unavailable, the model is asked for content support and evidence
without page-assessment fields. Native review and judgment records still retain
`pagination_available=false`, `correct_page=null`, and `found_pages=[]`.

Identical root source text and page labels are sent in a stable system prefix,
with the occurrence's proposition and target in the dynamic instruction. Full
reviews are grouped by root for KV-cache reuse. Page blocks from every saved
pagination namespace belong to this prefix, keeping it independent of the
reporter used by a particular occurrence.

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
quote within its named source page or pages using shared whitespace relaxation
and 95% edit-distance matching. A response may contain at most three independent
quotes, each requested as one representative sentence or short fragment. They
need not be adjacent. Every quote is checked separately, with its own opinion
and page references. Detailed explanation belongs in `reason`. Before any
grounding, all quotes are checked against a 1,200-character per-item limit;
oversized responses enter the normal repair loop with feedback to focus on one
or two sentences. The hard limit measures characters, avoiding unreliable
sentence splitting around legal abbreviations. It applies to new reviews, not
to loading saved documents. Recovered locations come from those grounded
page references. Quotes may span adjacent pages. Footnote identification remains
a model assessment; grounding verifies the containing page, and a correct-page
claim with a written footnote must name that footnote. Failed requirements
enter the ordinary repair loop and remain in the saved IVR trace. The
response limit and HTTP request timeout come from each substage's profile in `.env`;
8,000 tokens and 300 seconds are configuration examples, not fixed behavior.
The HTTP client enforces connection/read/write/pool timeouts per request, not
a total wall-clock limit on a response or the review. There is no whole-IVR
deadline, and Mellea's separate stream-chunk deadline is disabled. Each repair
gets its own HTTP timeout; validation does not consume a shared time budget.
The maximum attempt count still bounds the repair loop. Actual HTTP timeouts
remain in the durable IVR trace alongside any earlier responses and repairs.
Invalid or incomplete JSON raises `InvalidJsonResponseError` inside the wrapper; Mellea receives its parser
detail and `PLEASE CHECK IF TOKEN LIMIT VIOLATED` repair instruction. Valid JSON
that violates the schema retains field-specific validation feedback. The
wrapper customizes Mellea's native repair hook only for syntactically invalid
or truncated JSON: the next context keeps earlier history and the instruction
that produced the answer, but omits that malformed assistant response. Mellea
still supplies the validation feedback. Valid JSON with schema, grounding, or
semantic failures keeps ordinary multi-turn repair history. Every answer,
including omitted malformed responses, remains in the durable IVR trace.
This policy applies in all three structured-output transport modes. The
`validate_all_citation_content.review_citation_content.judge_citation_content` substage selects the latest successful full review when
present, copies its accepted page assessment, and checks retained quote spans
again. A failed full review can retain earlier affirmative page support, but
cannot turn a provisional page negative into a final wrong-support verdict.
Page assessment is evaluated independently of content support;
unknown gold locations are never treated as negative page judgments.
A wrong support judgment needs affirmative contradiction in a full review or
absence established after reading every available subopinion. Missing/empty
source text and model failures yield `UNDETERMINED`, never a negative by themselves.
TOA entries and occurrences without pinpoints receive no support verdict.
Docket pinpoint validation remains outside this reporter workflow.

Preprocessing serializes typed TOA components in `index_spans`, and citation
creation attaches a node-bound tag to occurrences inside them. Root identity
checks and shared root-opinion retrieval still include TOA citations. Proposition
reading records an empty attribution without a model call; support review and
judgment skip the tagged occurrence. Body leaves do not inherit a
root's TOA tag. Older untagged citations need replay through creation to acquire
it. Unrecognized TOA entries may also be read as having no proposition, which
skips support review and final judgment.

Citation signals, quotations, writing parentheticals, and designated footnotes
are interpreted in the filing's context, rather than reducing support to shared
words. The design uses [Cornell's citation guidance](https://www.law.cornell.edu/citation/6-300),
[FRAP 28's description of tables of authorities](https://www.law.cornell.edu/rules/frap/rule_28),
and [CourtListener's cluster/opinion documentation](https://wiki.free.law/c/courtlistener/help/api/rest/v4/case-law).

## Clearing an extracted field

All source-field correction paths can record explicit absence for case name,
court, date, and pincite where that path reviews the field. Lookup reviews and
leaf correction use `action=not_stated`. Locator-body review returns a complete
filing reread, so a null case name, court, or date explicitly means not stated;
it is not an implicit instruction to keep a prior value. Valid reporter-inferred
courts remain usable evidence even without a written court quote.

A clear appends a `NotStatedField(node_id=...)` to the field history. It contains
no fabricated quote, span, or normalized value. Current getters treat that latest
entry as absence and never fall back to an earlier reading. A later grounded
replacement can append a new value. Native serialization and stage recovery
preserve both the old value and the clear decision.

Required citation locators cannot be cleared. Correction does not regroup roots
or reassign leaves. Citation-content review continues to issue evidence and
judgments rather than rewriting the citation's printed fields.
