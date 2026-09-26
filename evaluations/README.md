# Evaluate a saved run

From the repository root, create or resume the reporter lookup run, then evaluate it:

```bash
uv run python -m evaluations.run_reporter_root_lookup \
  --run-dir local/reporter-root-lookup
uv run python -m evaluations.evaluate_run \
  --run-dir local/reporter-root-lookup
```

This scores the `primary` set by default and writes `summary.json`,
`occurrences.json`, and `report.md` to the run's `evaluation/` directory. Use
`--output-dir` to choose another destination, or repeat `--set` to select other
annotated sets explicitly. The command reads completed serialized Documents; it
does not rerun extraction, CourtListener lookups, or model calls.

The evaluator discovers the completed stages in the saved Documents and scores
each stage's new product against the annotations. Its JSON groups summaries and
occurrences by stage. The Markdown report includes extraction span,
normalization, colocation, and root-formation scores, followed by reporter
one-record lookup case-name, court, and date judgment precision and recall when that
stage has run. After a rule-only reporter ambiguity run, it reports only
case-name, court, and date judgment precision for the selected candidate.
Deferred judgments name the next stage directly; the
ambiguity stage processes only citations routed to its stage name. To resume
saved lookups without repeating them, run:

```bash
uv run python -m evaluations.run_reporter_root_lookup_ambiguous \
  --input-run-dir local/reporter-root-lookup \
  --run-dir local/reporter-root-lookup-ambiguous
uv run python -m evaluations.evaluate_run \
  --run-dir local/reporter-root-lookup-ambiguous
```

For unresolved single-candidate lookups, run the model review from the saved
lookup checkpoint, then evaluate its result:

```bash
uv run python -m evaluations.run_reporter_root_lookup_unique_llm \
  --input-run-dir local/reporter-root-lookup \
  --run-dir local/reporter-root-lookup-unique-llm
uv run python -m evaluations.evaluate_run \
  --run-dir local/reporter-root-lookup-unique-llm
```

The unique-review section reports only case-name, court, and date judgment
precision. It scores judgments written by that stage when the annotated root
occurrence and the latest corrected field reading align with an explicit field
label. Its `occurrences.json` entries retain each field's reading, label,
prediction, and scoring outcome.

When docket-locator rules have run, the report also includes the separate
unreviewed site-proposal diagnostic; proposals are not counted as admitted
locators. A missing document, inconsistent stage chain, or completed stage
without an evaluator is an error rather than an omitted score.
