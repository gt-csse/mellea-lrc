# Validation evaluation

Scores identity verdicts against the frozen **False Citation Bench —
Validation (CourtListener Heuristics)** set: 423 occurrences, 387 `match` and
36 `mismatch`.

Read [the shared setup](../README.md) first.

## What is scored

Given a citation that has **already been extracted**, does the authority
CourtListener returns for its locator agree with how the filing cites it?

Occurrences CourtListener cannot decide are absent from the benchmark rather
than labelled. Inferring `mismatch` from a failed lookup is exactly the error
this set exists to expose, so it is not built into the labels. Whether an
occurrence is reachable at all is an
[extraction question](../extraction/README.md), scored separately.

This is not a falsity label. A citation can be a clean `match` here and still
misrepresent the holding it is cited for; that judgement lives in the dataset's
`annotations/`. The 36 mismatches are spread over 14 of the 26 filings, so a
system cannot do well by learning that one document is the bad one.

## Write a run artifact

One JSON object per line:

```json
{"id":"cite:006:fullcasecitation-16-f-3d-1083:1557-1569",
 "locator_id":"fullcasecitation-16-f-3d-1083",
 "locator_span":{"start":1557,"end":1569},
 "verdict":"match"}
```

- `id`, `locator_id`, `locator_span` — **copied from the benchmark row**. The
  evaluator checks them and raises if either identity field differs, so a run
  that has drifted onto different text fails loudly instead of scoring
  nonsense. They are copied rather than recomputed so your notion of a
  locator's identity never has to match this project's.
- `verdict` — any string. `match` and `mismatch` are the expected labels;
  anything else becomes its own row in the matrix.

A missing row is recorded as `missing_artifact_record` rather than counting as
either verdict, so a partial run is visible rather than silently penalised.

The benchmark's `matched_text` is stored whitespace-collapsed, not verbatim —
29 of the 423 differ from the raw source slice by internal whitespace. The
evaluator does not compare it; do not use it for an exact string check.

### Producing one with Mellea-LRC

Needs a CourtListener token and an OpenAI-compatible endpoint in `.env`
(`COURTLISTENER_API_TOKEN`, `MELLEA_LRC_LLM_MODEL`, `MELLEA_LRC_LLM_API_BASE`,
`MELLEA_LRC_LLM_API_KEY`). Leave `MELLEA_LRC_LLM_TEMPERATURE` at `0.0`: the
workflow retries under a requirement loop, and a nonzero temperature makes a
score irreproducible run to run.

```bash
uv run --env-file .env python -m evaluations.validation.run_mellea_lrc \
  --documents data/false-citation-bench/documents_txt \
  --output run-validation

uv run python evaluations/validation/export_mellea_lrc_artifact.py \
  --artifact-dir run-validation \
  --output run-artifact.jsonl
```

Keep the serialized run: it is the expensive artifact, and re-scoring it is
free. `export_mellea_lrc_artifact.py` rebuilds each `id` the way the benchmark
did, so rows join without manual mapping. It is Mellea-LRC-only and not part of
the contract.

## Evaluate

```bash
uv run python evaluations/validation/evaluate.py \
  --benchmark data/false-citation-bench/derived/validation-courtlistener-heuristics.jsonl \
  --artifact run-artifact.jsonl \
  --output-dir evaluation-result
```

```text
| Observed verdict | Expected `match` | Expected `mismatch` |
|---|---:|---:|
| `match` | 371 | 3 |
| `mismatch` | 8 | 32 |
| `not_found` | 8 | 1 |

Evaluated labeled records: 423. Non-agreements: 20.
```

Columns are the truth, rows are what the system said. The diagonal is correct;
`(mismatch, match)` is a false alarm, and `(match, mismatch)` is the dangerous
cell — a bad citation waved through.

## Report abstentions as themselves

Rows below `match` and `mismatch` are abstentions: `possible_match`,
`not_found`, `unavailable`. The evaluator keeps each as its own row rather than
folding it into a verdict, because how to treat them is your decision and
should stay visible.

An abstention on a `mismatch` is a citation problem the system did not surface;
an abstention on a `match` is a citation it could not clear. Both are real
outcomes, and recoding them inflates whichever score they land in. Report the
abstention rate alongside accuracy.

`non_agreements.json` gives each disagreement a reason —
`verdict_disagreement`, `non_confident_verdict`, or `missing_artifact_record`.
Check the last one first: a large count usually means the run covered fewer
documents than expected, not that the system abstained.
