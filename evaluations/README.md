# Grow-roots evaluation

The active evaluator covers the first two extraction stages, `full_reporter_locators` and `docket_locators`. Each stage uses the same two measures: exact locator-span precision and normalization precision conditional on an exact span and a complete normalized identifier on that same annotation row. These are stage decisions, not root-deduplicated citations. The scorer never follows `root_id` to supply missing normalization gold; a repeated citation may spell its locator differently from its root.

Run the rule stages directly on the official text:

```sh
python -m evaluations --data-root /path/to/mellea-lrc-datasets --output-dir local/grow-roots-evaluation
```

Pass `--set NAME` more than once to include other annotated sets. Pass `--run-dir PATH` to score serialized Documents under `PATH/documents/SET/FILENAME.txt.json` instead of rerunning the two stages. The command writes `summary.json` and `report.md`. It fails on missing stage checkpoints, changed source text, inconsistent annotation quotes, or duplicate gold spans.
