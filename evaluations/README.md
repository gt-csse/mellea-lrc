# Evaluate a saved run

From the repository root, create or resume the exact-lookup run, then evaluate it:

```bash
uv run python -m evaluations.run_reporter_exact_lookup \
  --run-dir local/reporter-exact-rule-only
uv run python -m evaluations.evaluate_run \
  --run-dir local/reporter-exact-rule-only
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
exact-lookup case-name, court, and date judgment precision and recall when that
stage has run. After a rule-only reporter ambiguity run, it reports only
admission precision and case-name, court, and date judgment precision for the
selected candidate. Deferred judgments name the next stage directly; the
ambiguity stage processes only citations routed to its stage name. To resume
saved exact lookups without repeating them, run:

```bash
uv run python -m evaluations.run_reporter_exact_ambiguity \
  --input-run-dir local/reporter-exact-rule-only \
  --run-dir local/reporter-exact-ambiguity-rule-only
uv run python -m evaluations.evaluate_run \
  --run-dir local/reporter-exact-ambiguity-rule-only
```

When docket-locator rules have run, the report also includes the separate
unreviewed site-proposal diagnostic; proposals are not counted as admitted
locators. A missing document, inconsistent stage chain, or completed stage
without an evaluator is an error rather than an omitted score.
