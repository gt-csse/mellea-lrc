# Grow-roots evaluation

Each named stage scorer takes one serialized or in-memory `Document`, recovers its checkpoint with `document.get_stage(stage)`, and scores only decisions made at that stage. Each scorer owns its own comparison and denominator rules; there is no generic field-stage scoring function. Field-reading stages currently report span precision and normalization precision. Colocation and root-formation stages report group and root-assignment precision. A missing checkpoint raises.

`score_grow_roots(document)` calls the numbered stage scorers and reports span and normalization precision and recall for the final roots' reporter locator, docket locator, docket entry, case name, court, date, and pin cite. Each stage has its own `render_*` function. `render_grow_roots(score)` renders all stages in order by default, followed by the root-field summary; pass `include_stages=False` for the summary alone. Span and normalization are checked independently. The gold denominators come from the official annotations. A present field without normalized gold is excluded from normalization scoring; a predicted field absent from the annotation counts against precision. Inferred courts have no span to score, but their normalized court IDs are scored.

The scorer locates the annotation beside the document's official source file and checks its text length, SHA-256 digest, source path, and every quoted span. It does not accept an annotation path or extra stage arguments.

Run the primary set from source:

```sh
.venv/bin/python -m evaluations
```

The runner sets `_SET = "primary"` in `evaluations/__main__.py`. It writes each full `Document` to `evaluations/results/primary/documents/` and the combined `summary.json` and `report.md` beside them. Use `--saved-documents PATH` to score previously saved documents without rerunning extraction. `--data-root` and `--output-dir` change file locations, not the scorer API or selected set.
