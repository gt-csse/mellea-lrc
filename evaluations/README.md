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

If a run is interrupted, use `--resume-run RUN_DIR`. It verifies the saved
source and completed Documents, then continues in the same timestamped
directory without repeating completed filings.

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
the validate-roots summary adds precision and recall for case-name, court, and
date judgments. Both use the official annotations identified by the saved
Document's source path. An incomplete run cannot be scored.

The primary runner enables docket-root equivalence review after `10_roots` as
stage `11_docket_root_equivalence_review`. Its stage score checks assignments for
the docket roots it reviewed. The grow-roots field summary uses the roots
after that review; the `10_roots` stage score remains available separately.

Root validation then saves a CourtListener docket/opinion search for each
docket root and reviews the number-based shortlist, followed by the reporter
lookup stages. The docket retrieval stage records evidence but makes no field
judgment, so only its review appears in the validation stage-precision table.
The workflow summary includes both docket and reporter root judgments against
the same annotated-root denominator.
