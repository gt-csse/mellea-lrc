# Evaluation runs

## Grow leaves from saved roots

The leaf runner starts from the primary corpus's cumulative saved Documents.
It defaults to stage `23_locator_body_llm_judgment`, before open-ended internet
search. It makes no CourtListener, GovInfo, or web requests.

```sh
.venv/bin/python -m evaluations.run_grow_leaves \
  --input-documents evaluations/results/primary/<UTC timestamp>/documents
.venv/bin/python -m evaluations.score_run \
  evaluations/results/primary/<new UTC timestamp> --workflow grow_leaves
```

`--rule-only` omits the two semantic reviews. `--input-stage STAGE` accepts an
earlier completed root or leaf checkpoint; for example, stage `30_id_citations`
repeats only reference discovery and the subsequent leaf stages, while stage
`36_id_attribution` repeats only the Id. semantic audit. `--resume-run RUN_DIR`
continues an interrupted run in the same directory. Completed stages and
filings are retained. One cumulative Document is saved atomically after each
completed stage in `documents/`; there are no separate copies of each leaf
checkpoint. `get_stage` recovers those checkpoints from the final Document.

`score_grow_leaves(document)` calls the independent Document-only scorer for
each completed leaf stage. Discovery and reading stages report span and
normalization precision; attribution stages report attachment precision.
Normalization is scored only against independently annotated targets, never
against another execution of the same normalizer. Missing normalization gold
is shown explicitly. Supra has controlled tests but no primary gold examples.

The workflow tables report exact source-span and root-attribution precision
and recall by citation kind and overall. The denominator includes all
annotated nonroot occurrences: repeated full reporter/docket citations as
well as short reporter, Id., supra, and name-only references. Attribution
compares source-root locator spans, without requiring an external record ID.
Name-only attribution aligns a unique overlapping annotated mention and
credits it at most once; exact name boundaries are scored separately in the
source-span table. Other leaf kinds use exact source-site alignment.
Withdrawn proposals remain available in earlier checkpoints and review logs,
but do not count as final active leaves. Apparent false positives still need
occurrence review because an unannotated authority mention is scored as
unmatched. JSON and Markdown are generated directly by the same scorer.

## Root extraction and validation

The primary runner writes one cumulative `Document` per filing under
`evaluations/results/primary/<UTC timestamp>/documents/`. Its directory name
records only when the run began; `run.json` records the input and completion
status. Each final `Document` contains all completed extraction and validation
stage histories, so `get_stage(stage)` recovers an earlier checkpoint without
another provider call.

Run both workflows from source:

```sh
.venv/bin/python -m evaluations
```

To resume saved Documents at `10_roots`, then perform the docket-root review
and root validation, use `--from-roots-documents PATH`. This preserves the
earlier extraction history in the new timestamped run.

To repeat the model reviews without repeating extraction or reporter lookup,
use `--from-reporter-review-documents PATH`. The saved Documents must contain
stage `13.2_reporter_root_lookup_ambiguous_rule_judgment`. Add `--reuse-docket-lookups` when
they also contain stage `16_docket_root_lookup_courtlistener_retrieval`: the runner reuses those docket
search results, then reruns both reporter reviews and the docket review.

To replay any validation stage after a saved retrieval, use
`--from-checkpoint-documents PATH --checkpoint-stage STAGE`. For example,
`12.1_reporter_root_lookup_cluster_retrieval` reruns linked-docket retrieval and both rule reviews using
the saved exact response; `12.2_reporter_root_lookup_docket_retrieval`
reruns both rule reviews using the saved linked dockets. The runner
also saves cumulative Documents after each validation stage in
`checkpoints/stage12.1/`, `checkpoints/stage12.2/`, and the other stage
directories, so `--resume-run` can continue after an interrupted review
without repeating retrieval. The same checkpoint and resume behavior covers
stages `20`–`22` before `23_locator_body_llm_judgment`; a saved stage `22` Document
can be reviewed again without repeating its three body searches.

To start after docket review, use `--from-docket-review-documents PATH`. The
saved Documents must contain stage `17_docket_root_lookup_courtlistener_llm_review`; the runner
then performs GovInfo docket lookup and its review (stages `18` and `19`),
followed by the locator-first body stages.

To replay just the locator-first body stages from an earlier validation run,
use `--from-validation-documents PATH`. The saved Documents must contain stage
`19_docket_root_lookup_govinfo_llm_review`; stages `20` through `23` run into a new
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
Use `--from-validation-documents` to replay stages `20`–`23` with these cutoffs.

To continue a completed stage-`23` primary run through case-name body discovery
and intended-case review, use its saved `documents/` directory:

```sh
.venv/bin/python -m evaluations \
  --from-locator-review-documents evaluations/results/primary/<stage-23-run>/documents \
  --annotation-case-cutoffs \
  --courtlistener-pool reserved
```

This creates a new timestamped run. It keeps the original stage-`23` history
and appends stages `24` through `27` in order. Each filing's cumulative
Document is saved after every stage under `checkpoints/stage24/` through
`checkpoints/stage27/`; the final Document is also in `documents/`. If a run
stops, `--resume-run RUN_DIR` verifies the saved source, annotation cutoffs,
and checkpoints, then continues from the last completed stage. A transient
case-name search failure stops before later providers or the model review and
is retried from that provider stage on resume. The new run must use the same
cutoff mode as its stage-`23` input.

Add `--courtlistener-pool reserved` to use `COURTLISTENER_API_TOKEN_RESERVED`
for the CourtListener opinion and RECAP body stages. The proxy URL still comes
from `COURTLISTENER_BASE_URL`. The runner saves only the pool name in `run.json`
and reuses it on resume; the token stays in the environment. You can also add
the flag to `--resume-run RUN_DIR` for a run created before selecting a pool.
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
workflow scorers. Each named stage scorer takes only a `Document`, recovers
its own checkpoint, and reports precision for that stage's decisions. The
grow-roots workflow summary adds precision and recall for final root fields;
when stage `19_docket_root_lookup_govinfo_llm_review` is present, it also scores the
case-name, court, and date readings at that checkpoint with the same field
scorer. The original root-field table remains fixed at stage `11`, and later
body-review changes do not enter the stage-`19` comparison. The validate-roots
summary adds precision and recall for case-name, court, and
date judgments. Both use the official annotations identified by the saved
Document's source path. The body review compares two printed citations and
records a separate identity verdict when it selects an independent citation;
those comparisons are not identity-field judgments and have no matching field
gold. The stage-`23` section of `validate_roots.json` and `validate_roots.md`
counts the identity verdicts it issued. Detailed evidence, printed-field
comparisons, reasons, and routes remain in the serialized `Document`. Field
identity judgments remain scored through stage `19`, and the report separately
scores cumulative root identity. An incomplete run cannot be scored.

Stage `23` can record `partially_corroborated` when an independent citation
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

The primary runner enables docket-root LLM reassignment after `10_roots` as
stage `11_docket_root_llm_reassignment`. Its stage score checks assignments for
the docket roots it reviewed. The grow-roots field summary uses the roots
after that review; the `10_roots` stage score remains available separately.

Root validation continues with numbered stages `12.1` through `23`: separate
reporter retrieval and review stages, CourtListener docket/opinion search and its
number-based shortlist review, GovInfo docket lookup and review, then three
locator-first body searches and `23_locator_body_llm_judgment`. Retrieval stages
report one metric: citations with at least one usable saved record divided by citations actually queried. The Markdown report lists every
completed stage in execution order, including retrieval stages with their record-return metric; its JSON
`stage_order` records the same sequence separately from scored stages. When it
selects a citation, the locator-body review records printed-field comparisons
and an identity verdict;
these do not enter the field-identity precision table. The workflow summary
uses the latest comparable field judgment for each root against the same
annotated-root denominator.
Saved runs ending at stage `19` retain their stage-specific field report format.

Runs that continue through stages `24`–`27` list those stages after `23` in the
same report. Stages `24`–`26` retrieve case-name evidence and report the same citation-level record-return metric.
Stage `27` counts selected likely or possible intended-case candidates,
declines, and review failures. The annotations have no intended-case candidate
gold, so these counts have no precision or recall. Field judgments remain
scored through `19`, and root identity remains scored after `23`.
