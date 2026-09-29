# Evaluation runs

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
stage `13_reporter_root_lookup_ambiguous`. Add `--reuse-docket-lookups` when
they also contain stage `16_docket_root_lookup`: the runner reuses those docket
search results, then reruns both reporter reviews and the docket review.

To start after docket review, use `--from-docket-review-documents PATH`. The
saved Documents must contain stage `17_docket_root_lookup_review`; the runner
then performs GovInfo docket lookup and its review (stages `18` and `19`),
followed by the locator-first body stages.

To replay just the locator-first body stages from an earlier validation run,
use `--from-validation-documents PATH`. The saved Documents must contain stage
`19_govinfo_docket_lookup_review`; stages `20` through `23` run into a new
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
when stage `19_govinfo_docket_lookup_review` is present, it also scores the
case-name, court, and date readings at that checkpoint with the same field
scorer. The original root-field table remains fixed at stage `11`, and later
body-review changes do not enter the stage-`19` comparison. The validate-roots
summary adds precision and recall for case-name, court, and
date judgments. Both use the official annotations identified by the saved
Document's source path. The body review compares two printed citations and
records a separate identity verdict; those comparisons are not identity-field
judgments and have no matching field gold. The validation report scores the
new stage-`23` identity verdicts and cumulative root identity, while field
identity judgments remain scored through stage `19`. An incomplete run cannot
be scored.

For docket citations, stage `23` can record `case_identity_supported` when an
independent citation supports the case but not the particular decision. That
qualified judgment stays in the saved `Document` and does not count as a
correct full-citation admission in the binary identity score.

The primary runner enables docket-root equivalence review after `10_roots` as
stage `11_docket_root_equivalence_review`. Its stage score checks assignments for
the docket roots it reviewed. The grow-roots field summary uses the roots
after that review; the `10_roots` stage score remains available separately.

Root validation continues with numbered stages `12` through `23`: four reporter
lookup and review stages, CourtListener docket/opinion search and its
number-based shortlist review, GovInfo docket lookup and review, then three
locator-first body searches and `23_locator_body_review`. Retrieval stages
record evidence but make no field judgment. The locator-body review records
printed-field comparisons and an identity verdict, but does not enter the
field-identity precision table. The workflow summary uses the latest comparable
field judgment for each root against the same annotated-root denominator.
Saved runs ending at stage `19` retain their original six-stage report format.
