# Evaluation runs

The workflows are `grow_roots`, `validate_roots`, `grow_leaves`, and `validate_pincite`.
Each has one scorer and one Markdown report, with its JSON representation
beside it. Reports follow the catalog hierarchy: workflow, stage, then substage.
The semantic grouping introduces no additional metrics or denominators.

## Fresh primary end-to-end run

From the repository root, run all 26 primary filings through the four workflows:

```sh
uv run --no-sync python -m evaluations --end-to-end \
  --annotation-case-cutoffs --courtlistener-pool proxy --workers 3
```

Each filing runs `grow_roots`, `validate_roots`, `grow_leaves`, then
`validate_pincite` in order. Three filings can progress concurrently; substages
within each filing remain serial. Docket hunting, docket-root reassignment, and
leaf review are enabled. Optional intended-case discovery stays disabled.
Every model package and substage assignment comes from `.env`; the proxy uses
its rotating keys without sending a stored personal CourtListener token.

One timestamp directory contains one cumulative Document per filing, atomically
saved after every completed substage and group. On completion, the existing
scorers write all four workflow JSON and Markdown reports into that directory.
`run.json` records the Git commit, model profile descriptors with credential
references, full source and annotation hashes, worker count, and substage
elapsed seconds. Resume and scoring reject changed annotation content; resume
also rejects changed model settings or source content.

```sh
uv run --no-sync python -m evaluations \
  --resume-run evaluations/results/primary/<UTC timestamp>
```

Resume uses the saved mode and worker count, keeps completed checkpoints, and
continues pending substages in the same directory. Transient retrieval failures
leave the run incomplete; resume after the provider recovers. Original-opinion
HTTP or transport failures retain the preceding checkpoint rather than
committing a missing opinion. Model review failures remain explicit typed
outcomes in the saved Documents. End-to-end mode cannot be combined with saved
input modes, docket-lookup reuse, or the reserved-token pool.

## Workflow hierarchy

The immutable catalog in `src/mellea_lrc/model/execution.py` defines 15 stages:

| Workflow | Stages |
| --- | --- |
| `grow_roots` | Locator discovery; field reading; root formation |
| `validate_roots` | Reporter lookup; docket lookup; locator-body corroboration; optional intended-case discovery |
| `grow_leaves` | Short reporter citations; reference citations; Id. citations; supra citations; leaf field correction |
| `validate_pincite` | Opinion preparation; citation preparation; support review |

Each stage groups the granular substages. Substage identifiers use
`workflow.stage.substage`, such as
`validate_roots.reporter_lookup.cluster_retrieval`. The catalog supplies ordering
and optional membership; names carry no global execution number.

`Document.runs` records typed substage and stage completion events.
`Document.substage_runs` lists completed atomic steps, and `Document.stage_runs`
lists completed groups. `get_substage(name)` recovers an exact atomic checkpoint;
`get_stage(name)` recovers the completed group boundary and raises if that group
is unfinished. A group never adds citation decision nodes. Substages retain
execution, durable-save, retry, and citation-node provenance responsibility.

Each scorer operation accepts only `document` and reads its own atomic
checkpoint. Workflow score objects expose flat independent records as
`substages`; serialized reports group them as follows:

```json
{
  "workflow": "grow_leaves",
  "stages": [
    {
      "stage": "grow_leaves.short_reporter_citations",
      "completed": false,
      "substages": [
        {
          "substage": "grow_leaves.short_reporter_citations.discovery",
          "metrics": {}
        }
      ]
    }
  ]
}
```

Workflow summaries remain at the report root. Retrieval, opinion, page, and
judgment detail appears once under its substage. Markdown uses semantic-stage
headings and locally numbered substage headings, with every detail table shown
sequentially by default. Partial leaf and pinpoint reports include completed
substages and mark a group incomplete until its durable stage marker exists.
Aggregation reports a completed group only when every contributing Document has
that marker. Optional substages may be omitted within a completed group.

Aim for three to five meaningful stages per workflow. Granular execution and
review detail belongs in substages; the stage count is a design guideline rather
than a hard cap.

## Reporter-root opinion retrieval

The first `validate_pincite` substage is `validate_pincite.opinion_preparation.retrieval`.
It retrieves every linked opinion of an admitted reporter root's selected
CourtListener cluster through the configured proxy. It preserves complete raw
responses, original HTML/page markers, missing objects, and empty text on the
citation. It does not choose a majority, concurrence, or dissent by list position
or type, and it does not issue a pinpoint-support judgment.

```python
document = Document.model_validate_json(saved_json)
document = reporter_root_opinion_retrieval(document)
```

```sh
.venv/bin/python -m evaluations.run_validate_pincite \
  --input-documents evaluations/results/primary/<UTC timestamp>/documents
.venv/bin/python -m evaluations.score_run \
  evaluations/results/primary/<new UTC timestamp> --workflow validate_pincite
```

The runner saves one cumulative Document per filing after every completed substage.
`--resume-run` reuses completed substages and filings. Inputs already containing
`validate_pincite.opinion_preparation.retrieval` reuse the downloaded responses.
`--stop-after` accepts any catalog substage name and retains that exact boundary.

`score_reporter_root_opinion_retrieval(document)` and
`score_validate_pincite(document)` each accept only a Document. Opinion retrieval reports
one ratio: `reporter_roots_opinion_retrievals / reporter_roots_correct_identity`.
The denominator is reporter roots with correct identity and a selected original
CourtListener cluster; the numerator counts those with at least one retrieved
opinion body. Downloaded-object counts and annotation-target breakdowns are not
additional denominators.

Page indexing (`validate_pincite.opinion_preparation.page_index`) preserves a rendered source and
explicit page spans for every downloaded opinion. It keeps parallel pagination
separate and records uncertain namespaces. Page resolution
(`validate_pincite.citation_preparation.page_resolution`) resolves each root occurrence and its
attached occurrences using their own reporter edition and pinpoint. An
unqualified Id. may inherit its immediate antecedent's pinpoint; other leaves
do not inherit the root's canonical pinpoint.

Opinion selection (`validate_pincite.citation_preparation.opinion_review`) uses IVR only for ambiguous
page candidates. It selects a representative for each requested page, or leaves
it unresolved. Shared opinion text stays in the prompt prefix; occurrence
context and candidate indexes follow it. Decisions, reasons, every IVR repair,
and failures stay in citation-local histories. `get_reporter_page_selection()`
reads the most recent rule/model selection. Proposition and evidence preparation follow within citation preparation, followed by support review. These substages do not change identity,
source fields, or root attachments.

The remaining citation-preparation and support-review substages read each occurrence's proposition, prepare pinpoint evidence,
review selected pages, review the full saved opinion when necessary, and append
the final support/location judgment. Each substage has its own scorer taking only
`document` and a matching renderer. The report renders substages sequentially under their stages.
Stages without independent gold targets report counts rather than invented accuracy.

The report opens with a dataset-only inventory before opinion preparation: native pin cite
occurrences, the subset under gold `CORRECT_IDENTITY` roots, settled findings,
and separate `SKIPPED` counts for `UNSETTLED`, `TOA`, and `IDENTITY_WRONG`.
Settled correct/wrong findings are distributed across reporter roots, their
leaves, docket roots, and their leaves. A leaf uses its root's gold identity.
Pipeline extraction, opinion retrieval, inherited pinpoints without native pin
annotations, and issued judgments do not change this inventory.

The final pinpoint table separates reporter roots, leaves of reporter roots,
docket roots, leaves of docket roots, and their total. Its
fixed gold cohort is every settled native `CORRECT_PINCITE` or `WRONG_PINCITE`
occurrence under a gold `CORRECT_IDENTITY` root. A leaf uses its root's gold
identity. Families with a wrong gold identity are excluded from precision and
recall even if the pipeline admitted them. Gold `SKIPPED` and no-pin occurrences are excluded.
Missing roots, opinions, or judgments never shrink that recall denominator.
Precision counts correct definitive judgments divided by definitive judgments
within that gold cohort;
recall counts correct definitive judgments divided by all settled gold.
`UNDETERMINED` remains a recall miss. Provider/opinion IDs are not accuracy gates.
Docket pinpoint validation is not implemented, so settled docket annotations
contribute to the denominator as missing judgments.
Pinpoint location is distinct from support. Native annotations record
`pagination_available`, an independent nullable Boolean `correct_page`, and
typed `found_pages` intervals. `correct_page` assesses the written target,
including any specified footnote; `found_pages` records identified locations
of the cited material. A correct support label does not imply a correct page,
and a wrong support label does not imply a page error.

Full-opinion review and final judgment also report page precision. Full-opinion review compares the model's
`correct_page` assessment; final judgment compares the preserved assessment after evidence
grounding and page checks. Explicit Boolean assertions are scored against known
native page judgments within the settled gold-correct-identity cohort,
independently of the support verdict. Unknown gold and null predicted judgments
are counted separately outside page precision; they do not alter the fixed
support recall denominator. The workflow summary uses the latest completed
review/judgment boundary, and its page column uses the same rule.

Found-page agreement is reported separately. It compares recovered typed
intervals with known positive native `found_pages` references, including target
kind and footnote. A recovered interval inside a written target explicitly
judged wrong is a known page error. Unlisted or disjoint alternative locations
are not automatically false: the native positive references are not an
exhaustive map of every possible occurrence. Unknown comparisons and missing
predicted locations are reported separately.

The runner processes three independent filings concurrently by default; use
`--workers` to control concurrency. Each filing is saved after completed atomic
substages. Resume restores the saved worker count and checks the source hashes.

All checkpoints are recovered from the final Document with `get_substage`; no
separate substage snapshots are required. Opinion retrieval is a
writing-choice accuracy evaluation, so its report claims no subopinion precision
or recall. Such an evaluation needs independent occurrence-specific targets.

Field-reading precision counts one outcome for every citation the stage reads,
including a missing quote, `not_stated`, an inferred value, or a failed
normalization. Stage eligibility determines that denominator; annotation
availability and nonnull readings never narrow it. Absence receives credit
only against explicit annotated absence. Missing required gold raises.
Field-judgment precision counts every issued judgment about the selected
record, including mismatch and unavailable outcomes. Internal comparisons
that reject other trial candidates are not predictions of citation-field
correctness and cannot be scored against that gold. A stage that issued no
judgment has no judgment prediction to score. Discovery counts
created citations, attribution counts issued attachments, and retrieval counts
queries, including those returning no records.

## Grow leaves from saved roots

The leaf runner starts from a complete run's cumulative saved Documents. It
infers the corpus from the parent `run.json`: `primary`, `hallucination-set-1`,
`hallucination-set-2`, `reliable-high-profile`, or `reliable-low-profile`. Results
are saved under `evaluations/results/<set>/<UTC timestamp>/`; the primary
paths below remain the default examples. It defaults to stage
`validate_roots.locator_body_corroboration.llm_judgment`, before open-ended internet search. It makes no
CourtListener, GovInfo, or web requests.

```sh
.venv/bin/python -m evaluations.run_grow_leaves \
  --input-documents evaluations/results/primary/<UTC timestamp>/documents
.venv/bin/python -m evaluations.score_run \
  evaluations/results/primary/<new UTC timestamp> --workflow grow_leaves
```

`--rule-only` disables semantic review in short reporter, reference, and Id.
attribution and omits the later supra semantic review. `--stop-after SUBSTAGE` ends
at a selected completed checkpoint and is retained when resuming. To rerun
short reporter creation and attribution, followed by reference creation and attribution,
without model calls:

```sh
.venv/bin/python -m evaluations.run_grow_leaves \
  --input-documents evaluations/results/primary/<UTC timestamp>/documents \
  --rule-only --stop-after grow_leaves.reference_citations.attribution
.venv/bin/python -m evaluations.score_run \
  evaluations/results/primary/<new UTC timestamp> --workflow grow_leaves
```

`--input-substage SUBSTAGE` accepts an
earlier completed root or leaf checkpoint; for example, substage `grow_leaves.short_reporter_citations.discovery`
reuses the created short locators and pinpoints, then performs colocation and
case-name reading before attribution. Substage `grow_leaves.short_reporter_citations.colocations`
reuses those groups and begins with case-name reading; stage
`grow_leaves.short_reporter_citations.case_names` begins with short reporter attribution. Stage
`grow_leaves.reference_citations.discovery` reuses each reference's created name and pinpoint and
begins with reference attribution; substage `grow_leaves.reference_citations.attribution` begins with
Id. discovery. Substage `grow_leaves.id_citations.discovery` reuses Id. spans and normalized pinpoints
and begins with Id. attribution at substage `grow_leaves.id_citations.attribution`.
`--resume-run RUN_DIR`
continues an interrupted run in the same directory. Completed stages and
filings are retained, and the saved corpus, parent filing list, input hashes,
and existing checkpoints are checked before continuing. One cumulative Document is saved atomically after each
completed substage in `documents/`; there are no separate copies of each leaf
checkpoint. `get_substage` recovers those checkpoints from the final Document.

`score_grow_leaves(document)` calls the independent Document-only scorer for
each completed leaf substage, including a run that stops before the workflow
ends. It rejects a missing required substage inside that completed prefix;
unrun substages do not appear in the report. The final attribution summary
appears only after Id. attribution has run. Discovery and reading substages report span and
normalization precision; attribution substages report attachment precision.
Short reporter creation at substage `grow_leaves.short_reporter_citations.discovery` reports
`locator_span`, `locator_normalization`, `pin_cite_span`, and
`pin_cite_normalization`. Substage `grow_leaves.short_reporter_citations.colocations` records the
grouping checkpoint without a numeric precision: independent short-group
annotations are not yet defined. Substage `grow_leaves.short_reporter_citations.case_names`
reports `case_name_span` and `case_name_normalization` for the name before each
group or singleton. Reference creation retains its name and pin readings:
`case_name_span` and `case_name_normalization` score the name, while `pin_cite_span` and
`pin_cite_normalization` score the pin. The attribution stage for each citation
kind scores only its attachments. Id. creation reports
`citation_span`, `pin_cite_span`, and `pin_cite_normalization`; it has no
separate locator or case-name normalization. Its next stage attributes each
Id. in source order, optionally reviewing its antecedent before processing the
next Id. Stages `grow_leaves.supra_citations.case_names`, `grow_leaves.supra_citations.pin_cites`,
`grow_leaves.supra_citations.rule_attribution`, and `grow_leaves.supra_citations.llm_attribution` operate only on
supra citations; the IVR review service is shared with the other leaf types.
Creation-substage span and normalization precision count every field outcome
for each created citation, including `not_stated` or null outcomes. Absence
is correct only against an explicit annotated `not_stated` state. Later
reading substages likewise include missing outcomes for every eligible citation:
supra names and pinpoints. Unmatched predictions
and normalization failures remain in the denominator. Independent
annotation targets determine correctness, never another execution of the same
normalizer. A matched annotation missing its required normalization target
raises instead of silently narrowing the denominator. Supra has controlled
tests but no primary gold examples.

The workflow tables report exact source-span and root-attribution precision
and recall by citation kind and overall. A bounded run includes only completed
discovery types, alongside inherited repeated full citations; unrun types are
omitted rather than counted as misses. The full workflow denominator includes all
annotated nonroot occurrences: repeated full reporter/docket citations as
well as short reporter, Id., supra, and name-only references. Attribution
compares source-root locator spans, without requiring an external record ID.
Name-only attribution aligns a unique overlapping annotated mention and
credits it at most once; exact name boundaries are scored separately in the
source-span table. Other leaf kinds use exact source-site alignment.
Bare reference annotations remain in place with the native
`unit: out_of_scope_citation`; the scorer uses only `unit: citation` rows.
A bare reference left as `unit: citation` raises an annotation-scope error
rather than being silently filtered out.
All citations attached to the dummy head, whether unresolved or rejected,
are excluded from workflow source-span and attribution summary predictions.
Their creation-substage decisions and histories remain available, and the
annotated recall denominators remain unchanged. Apparent false positives still need
occurrence review because an unannotated authority mention is scored as
unmatched. JSON and Markdown are generated directly by the same scorer.

## Root extraction and validation

The primary runner writes one cumulative `Document` per filing under
`evaluations/results/primary/<UTC timestamp>/documents/`. Its directory name
records only when the run began; `run.json` records the input and completion
status. The returned root-growth Document is saved before validation begins;
each completed validation substage atomically updates the same filing artifact.
Every saved `Document` contains its typed completion history, so
`get_substage(substage)` recovers an earlier checkpoint without another provider call.

Run both workflows from source:

```sh
.venv/bin/python -m evaluations
```

To resume saved Documents at `grow_roots.root_formation.rule`, then perform the docket-root review
and root validation, use `--from-roots-documents PATH`. This preserves the
earlier extraction history in the new timestamped run.

To repeat the model reviews without repeating extraction or reporter lookup,
use `--from-reporter-review-documents PATH`. The saved Documents must contain
substage `validate_roots.reporter_lookup.ambiguous_rule_judgment`. Add `--reuse-docket-lookups` when
they also contain substage `validate_roots.docket_lookup.courtlistener_retrieval`: the runner reuses those docket
search results, then reruns both reporter reviews and the docket review.

To replay any validation substage after a saved retrieval, use
`--from-checkpoint-documents PATH --checkpoint-substage SUBSTAGE`. For example,
`validate_roots.reporter_lookup.cluster_retrieval` reruns linked-docket retrieval and both rule reviews using
the saved exact response; `validate_roots.reporter_lookup.docket_retrieval`
reruns both rule reviews using the saved linked dockets. The runner updates
each cumulative Document in `documents/` after every completed validation substage,
so `--resume-run` continues after an interrupted review without repeating
retrieval. There are no separate substage snapshots. This behavior also covers
the three retrieval substages within locator-body corroboration; a saved
`validate_roots.locator_body_corroboration.govinfo_opinion_retrieval` checkpoint
can be reviewed again without repeating its three body searches.

To start after docket review, use `--from-docket-review-documents PATH`. The
saved Documents must contain substage `validate_roots.docket_lookup.courtlistener_review`; the runner
then performs GovInfo docket retrieval, review, and identity aggregation,
followed by the locator-first body stages.

To replay just the locator-first body stages from an earlier validation run,
use `--from-validation-documents PATH`. The saved Documents must contain stage
`validate_roots.docket_lookup.govinfo_review`; identity aggregation and
locator-body corroboration run into a new
timestamped directory. `--retrospective-date YYYY-MM-DD` limits body evidence
to documents issued on or before that date. The cutoff is saved in `run.json`
and reused by `--resume-run`.

For the primary corpus, `--annotation-case-cutoffs` reads `filing` and
`case_cutoff` from the first row of each annotation JSONL file. The cutoff is
the earliest sampled filing date for the same case, including when a later
filing appears elsewhere in the corpus. The `filing` entry carries its source
PDF, page, and evidence; `case_cutoff.source_document` identifies the earliest
sampled filing. The runner checks that each cutoff equals that filing's date
before provider calls. This option cannot be combined with
`--retrospective-date`. `run.json` saves the selected dates and annotation
header hashes; resume checks that the headers have not changed.
Use `--from-validation-documents` to replay locator-body corroboration with these cutoffs.

To continue a completed locator-body corroboration run through case-name body discovery
and intended-case review, use its saved `documents/` directory:

```sh
.venv/bin/python -m evaluations \
  --from-locator-review-documents evaluations/results/primary/<locator-body-run>/documents \
  --annotation-case-cutoffs \
  --courtlistener-pool reserved
```

This creates a new timestamped run. It keeps the original locator-body history
and appends the intended-case discovery substages in order. Each filing's cumulative
Document is saved after every substage in `documents/`. If a run
stops, `--resume-run RUN_DIR` verifies the saved source, annotation cutoffs,
and checkpoints, then continues from the last completed substage. A transient
case-name search failure stops before later providers or the model review and
is retried from that provider stage on resume. The new run must use the same
cutoff mode as its locator-body input.

Add `--courtlistener-pool reserved` to use `COURTLISTENER_API_TOKEN_RESERVED`
from `.env` for the CourtListener opinion and RECAP body stages. The proxy URL
and request timeout come from `COURTLISTENER_BASE_URL` and
`COURTLISTENER_TIMEOUT_SECONDS` in the same file. The runner saves only the pool
name in `run.json` and reuses it on resume; the token stays in `.env`. You can
also add the flag to `--resume-run RUN_DIR` for a run created before selecting a
pool.
Without this flag or a saved pool, the proxy selects its usual rotating tokens.
Use `--courtlistener-pool proxy` when resuming a reserved-pool run to return to
those rotating tokens. `run.json` records the switch in `courtlistener_pool_history`.

If a run is interrupted, use `--resume-run RUN_DIR`. It verifies the saved
source and completed Documents, then continues in the same timestamped
directory without repeating completed filings. When a locator-body provider
has a transient failure, it resumes from the checkpoint before that provider
and reruns only that provider and the later body stages.

Evaluation is a separate read-only step. For a completed run:

```sh
.venv/bin/python -m evaluations.score_run evaluations/results/primary/<UTC timestamp>
```

This writes `grow_roots.md`, `grow_roots.json`, `validate_roots.md`, and
`validate_roots.json` beside `run.json`. Use `--workflow grow_roots` or
`--workflow validate_roots` to regenerate just one report. Neither option
executes extraction or validation.

`score_grow_roots(document)` and `score_validate_roots(document)` are the
workflow scorers. Each named substage scorer takes only a `Document`, recovers
its own checkpoint, and reports precision for that stage's decisions. The
grow-roots workflow summary adds precision and recall for final root fields;
when substage `validate_roots.docket_lookup.govinfo_review` is present, it also scores the
case-name, court, and date readings at that checkpoint with the same field
scorer. The original root-field table uses the completed root-formation result, and later
body-review changes do not enter the GovInfo-review comparison. The validate-roots
summary adds precision and recall for case-name, court, and
date judgments. Both use the official annotations identified by the saved
Document's source path. The body review compares two printed citations and
records a separate identity verdict when it selects an independent citation;
those comparisons are not identity-field judgments and have no matching field
gold. The locator-body review substage of `validate_roots.json` and `validate_roots.md`
counts the identity verdicts it issued. Detailed evidence, printed-field
comparisons, reasons, and routes remain in the serialized `Document`. Field
identity judgments remain scored through GovInfo docket review, and the report separately
scores cumulative root identity. An incomplete run cannot be scored.

Locator-body review can record `partially_corroborated` when an independent citation
supports the case but not the particular decision. It can issue `undetermined`
when the selected evidence does not support a firmer opinion. Both remain
visible in its verdict table. The canonical cumulative identity score treats
both as undetermined. Parenthetical precision and recall figures also count
`partially_corroborated` as a positive prediction, scoring it correct only when
the aligned binary gold is `CORRECT_IDENTITY`. A partial verdict on wrong gold
or an unaligned root therefore lowers that broader precision; both scores use
the same annotated-root recall denominator. When no reviewable citation is
selected or the review fails, the program records a next-stage route without
issuing an identity judgment.

The primary runner enables docket-root LLM reassignment after `grow_roots.root_formation.rule` as
substage `grow_roots.root_formation.docket_llm_reassignment`. Its substage score checks assignments for
the docket roots it reviewed. The grow-roots field summary uses the roots
after that review; the `grow_roots.root_formation.rule` substage score remains available separately.

Root validation groups reporter lookup, docket lookup, and locator-body
corroboration into three stages. Intended-case discovery is an optional fourth
stage. Each retrieval substage reports citations with at least one usable saved
record divided by citations actually queried. The Markdown report includes every
completed substage in execution order under its semantic stage. JSON nests those
same retrieval and judgment records under `stages[].substages[]`.

Locator-body review records printed-field comparisons and an identity verdict
when it selects a citation. Those comparisons do not enter field-identity
precision. The summary uses the latest comparable lookup field judgment against
the same annotated-root denominator. It reports cumulative root identity after
locator-body review separately.

Intended-case discovery retrieves case-name evidence from three endpoints, then
counts likely or possible selected candidates, declines, and review failures.
The annotations have no intended-case candidate gold, so these counts have no
precision or recall. Lookup field judgments and locator-body identity retain
their existing evaluation boundaries.
