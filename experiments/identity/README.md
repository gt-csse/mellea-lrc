# Identity model experiments

Compare named `.env` profiles on saved identity-review inputs. Retrieval is never repeated, and the saved baseline does not make new model calls.

```sh
uv run python -m experiments.identity.compare_models \
  --cohort experiments/identity/results/2026-10-08T01-19-53Z/cohort.json \
  --profiles nrp_glm nrp_qwen
```

The cohort lists `route`, `document`, and `citation_id` for each review. Supported routes are unique reporter lookup, ambiguous reporter lookup, CourtListener docket lookup, GovInfo docket lookup, and locator-body corroboration. Each uses its native pre-review checkpoint, production reviewer, schema, and grounding checks. Annotation labels and baseline decisions enter scoring only.

Each timestamped result directory retains the cohort, complete profile settings without credentials, input contexts, outputs, IVR requests and repairs, JSON summary, and a Markdown comparison. Successful and failed answers also pass through the real production review stage and native Document serialization. Other citations reuse their saved baseline answers; these checkpoints verify individual reviews and do not represent a complete run with the new model.

The selected cohort deliberately includes difficult cases. Its field precision is a diagnostic comparison, not an estimate of corpus recall. A valid null selection counts as an accepted answer but does not issue field comparisons. Body field comparisons are reported as diagnostics; the existing workflow's body stage reports identity verdicts instead.

Resume the same experiment by passing its directory with `--destination`. Completed answers are reused, and changed cohort or profile settings raise. Render or materialize saved answers without model calls:

```sh
uv run python -m experiments.identity.compare_models --render RESULTS_DIRECTORY
uv run python -m experiments.identity.compare_models --materialize RESULTS_DIRECTORY
```
