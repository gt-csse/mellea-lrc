# Reporter-root field evaluation

Use a new, empty run directory when repeating an experiment. The runners resume
saved documents if their output directory already contains them.

The run proceeds from the source filings through root extraction, unique
reporter lookup, rule review of ambiguous lookups, and the two separate model
review routes. Combine the model-review branches only after both have finished.
Pass the same `--data-root` and `--sets` to each runner.

```bash
python -m evaluations.run_reporter_root_lookup \
  --data-root /path/to/datasets --sets primary --run-dir local/new-run/lookup
python -m evaluations.run_reporter_root_lookup_ambiguous \
  --data-root /path/to/datasets --sets primary \
  --input-run-dir local/new-run/lookup --run-dir local/new-run/ambiguous
python -m evaluations.run_reporter_root_lookup_unique_llm \
  --data-root /path/to/datasets --sets primary \
  --input-run-dir local/new-run/lookup --run-dir local/new-run/unique-review
python -m evaluations.run_reporter_root_lookup_ambiguous_llm \
  --data-root /path/to/datasets --sets primary \
  --input-run-dir local/new-run/ambiguous --run-dir local/new-run/ambiguous-review
python -m evaluations.combine_reporter_reviews \
  --data-root /path/to/datasets --sets primary \
  --ambiguous-run-dir local/new-run/ambiguous-review \
  --unique-run-dir local/new-run/unique-review --run-dir local/new-run/combined
python -m evaluations.evaluate_run \
  --data-root /path/to/datasets --set primary --run-dir local/new-run/combined
```

The default cumulative report contains only case-name, court, and date
judgment precision and recall at each reporter-validation stage. Each stage
scores judgments it wrote. Precision is the share of its decided judgments
that agree with annotation; recall is the share of its eligible annotated
field judgments it gets right. An abstention counts as a recall miss. The
serialized Documents and occurrence details remain available for diagnosis.
