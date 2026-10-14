# Citation-content model experiments

These probes compare named `.env` profiles on saved opinion-review inputs. They
reuse retrieved sources and keep annotation labels out of model prompts. Profile
names, endpoints, model choices, and limits come from `.env`; see
[runtime configuration](../../README.md#runtime-configuration). Page and
full-opinion review assignments are independent.

## Compare a full-opinion input

Use a native Document with the target citation prepared through
`validate_all_citation_content.prepare_citation_evidence.prepare_evidence`:

```sh
uv run python -m experiments.pinpoint.compare_full_opinion_models \
  --document /path/to/document.json \
  --citation-id CITATION_ID \
  --profiles PROFILE_A PROFILE_B
```

The script uses the production context, reviewer, schema, and grounding checks.
It writes the retained context, per-profile IVR traces, summary JSON, and Markdown
under a timestamped local `comparisons/` directory. Acceptance means schema and
grounding checks passed; it does not establish judgment accuracy.
`render_comparison(path)` regenerates Markdown from those saved outcomes without
model requests.

## Replay saved reviews

The cohort JSON contains a `cases` array. Each case specifies `document`,
`citation_id`, `annotation_id`, and `scope` (`cited_pages` or `full_opinion`). The
Document must already contain that review: the probe retains its baseline and
recovers the native checkpoint immediately before it.

```sh
uv run python -m experiments.pinpoint.compare_saved_reviews \
  --cohort /path/to/cohort.json \
  --profile PROFILE_A

# Score and render saved outcomes without model requests.
uv run python -m experiments.pinpoint.score_saved_reviews \
  --render /path/to/experiment
```

Pass `--destination` to resume the same probe. Completed outcomes are reused;
changed profiles or inputs raise. Traces retain requests, responses, validation
feedback, and repairs. The separate scorer uses annotations to evaluate content
and page assessments. A selected diagnostic cohort is not a corpus evaluation.

## Historical results and production runs

[Earlier experiment notes](../../archive/documentation-2026-10-14/pinpoint-experiments.md)
retain the recorded outcomes and historical model assignments. Those assignments
and old numbered stages do not define the current pipeline. Generated experiment
and evaluation results are local artifacts, not bundled with the repository.

For production execution and workflow reports, use the existing
[citation content runner and scorer](../../evaluations/README.md#reporter-root-opinion-retrieval).
