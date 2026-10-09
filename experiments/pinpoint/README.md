# Pinpoint model experiments

NRP exposes `glm-5`; successful responses identify the provider model as `Inferact/GLM-5.3-NVFP4`. The originally requested `glm-5.3-nvfp4` alias returned HTTP 404. Raw endpoint checks are saved under [endpoint/](endpoint/).

Each probe retains its prompts, schema, provider response when one arrived, validation results, and repair feedback. The early controlled page probes used the provider's default reasoning effort; subsequent low-effort probes record that option explicitly. Fabricated cases check the data contract, and primary-corpus cases use saved source text without annotation labels in the prompt.

## Saved outcomes

| Probe | Accepted content result | Pagination available | Correct page | Found pages | Failure |
| --- | --- | --- | --- | --- | --- |
| [Supported, correct page](smoke/2026-10-02T18-11-22Z/supported-correct-page.json) | supported | true | true | 7 | — |
| [Contradicted, correct page](smoke/2026-10-02T18-11-22Z/contradicted-correct-page.json) | contradicted | true | true | 7 | — |
| [Supported, unpaginated](smoke/2026-10-02T18-34-27Z/supported-unpaginated-low.json) | supported | false | null | [] | — |
| [Primary 006-o32, selected pages](results/2026-10-02T18-18-11Z/006-o32.json) | unavailable | true | null | [] | — |
| [Primary 026-o32, full opinion, bounded repair](results/2026-10-02T18-34-09Z/026-o32.json) | — | — | — | — | found_pages must locate quoted passages in the supplied cited-reporter page markers |
| [Primary 026-o32, full opinion, native async](results/2026-10-02T18-43-41Z/026-o32.json) | — | — | — | — | Model call exceeded the configured 600-second timeout. |
| [Unpaginated, native async](smoke/2026-10-02T18-51-08Z/supported-unpaginated-native-async.json) | — | — | — | — | Model call exceeded the configured 120-second timeout. |
| [Unpaginated, direct async SDK](smoke/2026-10-02T18-55-34Z/direct-async-openai-unpaginated.json) | — | — | — | — | TimeoutError: direct SDK request did not complete within 120 seconds |

The bounded full-opinion review returned two schema-valid answers, both rejected because the reported page did not contain the grounded quote. Its trace records both answers and the native repair message. Source-derived repair feedback now includes the quote's offsets and page location. The subsequent full-opinion request reached the 600-second deadline before any answer arrived, so that feedback improvement has not yet been confirmed in a live repair.

The latest native-async and direct-SDK small probes also returned no response within 120 seconds. No schema error was received. These results do not isolate the timeout to Mellea or establish its upstream cause.

## Full-opinion profile comparison

The stage now chooses one complete named package, including endpoint, model,
temperature, output mode, token budget, timeout, and repair budget. Credentials
are resolved separately and are never saved with the run. Every model-backed
stage uses the same `MODEL_PROFILE` convention; the shared reviewer does not
choose a model internally.

Two complete saved opinion contexts were tested without repeating retrieval or
including annotations in the prompt. Their shared source prefixes contain
20,120 and 99,337 characters. Each comparison uses identical instructions,
strict JSON schema, and grounding checks across models:

- [Omaha full-opinion comparison](comparisons/2026-10-02T19-12-36Z/README.md)
- [Mata full-opinion comparison](comparisons/2026-10-02T19-15-07Z/README.md)

Qwen and Luna produced schema-valid, grounded decisions for both contexts. Each
needed one repair for Omaha and none for Mata. Kimi produced one answer with a
grounding failure, then reached its 300-second whole-call deadline. Its failed
trace retains the completed answer and repair feedback.

These are capability checks, not corpus scores. Omaha's annotation treats the
injunction principle as support for the cited TRO proposition; both models
instead rejected the attribution. Both located their quoted passage on page
239, agreeing with the annotated alternative page. For Mata both found a
misquotation, agreeing with its negative content label. Their page locations
follow the saved CourtListener text's pagination; the annotation uses CAP's
source. These outcomes leave content interpretation and source-page comparison
for the existing evaluation workflow rather than treating grounding acceptance
as accuracy.

Full-opinion review currently selects `NRP_QWEN`, which uses the free NRP
endpoint. Page review selects `NRP_GLM`. `OPENROUTER_LUNA` remains an explicit
alternative. There is no automatic fallback between profiles.

The repeatable comparison script is
[compare_full_opinion_models.py](compare_full_opinion_models.py). It persists
complete contexts, per-profile IVR traces, summary JSON, and rendered Markdown.
`render_comparison(path)` renders the saved outcomes without model requests.

## Resume the corpus

All 26 primary documents are saved cumulatively through stage 44 at [the input checkpoint](../../evaluations/results/primary/2026-10-02T18-20-42Z/). These retain earlier opinion selection and proposition readings. The existing runner can append the redesigned page/full-opinion reviews using each stage's selected profile, without repeating retrieval:

```sh
.venv/bin/python -m evaluations.run_validate_pincite \
  --input-documents evaluations/results/primary/2026-10-02T18-20-42Z/documents
```

The full-corpus run with these per-stage profiles completed through stage 47 for all 26 filings in [the saved run](../../evaluations/results/primary/2026-10-02T20-47-44Z/). Its workflow metrics are recorded by the existing `evaluations.score_run` command.

## Saved review replay (2026-10-08)

[The fourteen-case GLM experiment](results/2026-10-08T01-53-11Z/README.md)
replays seven selected-page and seven full-opinion inputs from their preceding
native checkpoints. It makes no retrieval or paid-model calls. The production
bindings remain GLM for page review, Qwen for full-opinion review, and Luna for
opinion selection and proposition extraction. These bindings come from `.env`.

The cohort, fixed contexts, historical reviews, new IVR traces, failures, native
pending Documents, and scores are retained together. Gold enters scoring only.
The saved and current contexts contain identical source evidence, dynamic
variables, and output schemas; instruction wording has changed. The deliberately
difficult sample measures capability and diagnosis, not corpus performance.

```sh
uv run python -m experiments.pinpoint.compare_saved_reviews \
  --cohort experiments/pinpoint/results/2026-10-08T01-53-11Z/cohort.json \
  --profile nrp_glm

# Render the saved results without model calls.
uv run python -m experiments.pinpoint.score_saved_reviews \
  --render experiments/pinpoint/results/2026-10-08T01-53-11Z
```

Pass `--destination` to the probe to resume an existing experiment. It rejects
changed profiles or inputs and recovers missing native materializations from
saved outcomes rather than requesting the model again.
