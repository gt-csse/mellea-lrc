# Extraction evaluation

Scores extracted citations against the frozen **False Citation Bench —
Extraction** set: 594 identifiers across 26 filings.

Read [the shared setup](../README.md) first, in particular the coordinate
space.

## End to end

```bash
uv run hf auth login

uv run hf download gt-csse/false-citation-bench --repo-type dataset \
  --local-dir data/false-citation-bench

uv run python -m evaluations.extraction.run --arm production \
  --documents data/false-citation-bench/documents_txt --output run-artifact.jsonl

uv run python evaluations/extraction/evaluate.py \
  --benchmark data/false-citation-bench/derived/extraction.jsonl \
  --artifact run-artifact.jsonl --output-dir evaluation-result
```

Expect 563 occurrences, 100.0% precision and 94.8% recall. The sections below
cover what each step does and how to score a system of your own.

## What is scored

One occurrence is one **citation identifier** at one place:

| `kind` | identifier | records |
|---|---|---:|
| `locator` | volume + reporter + page, e.g. `556 U.S. 662` | 583 |
| `docket` | docket number + court, e.g. `No. 1:19-CV-362` (M.D.N.C.) | 11 |

Each is the least that picks out exactly one authority — the
[minimum sufficient case identifier](https://huggingface.co/datasets/gt-csse/false-citation-bench#the-minimum-sufficient-case-identifier).
A prediction is a true positive when it matches an as-yet-unclaimed benchmark
occurrence on all of:

1. the same `document`;
2. the same identifier, compared with punctuation, spacing and case removed —
   `798 F. Supp. 2d 1215` and `798 F.Supp.2d 1215` both reduce to
   `798|fsupp2d|1215`, and `No. 1:19-CV-362` and `1:19-cv-362` both to
   `119cv362`;
3. for a docket, the same court, given either as written (`M.D.N.C.`) or as the
   courts-db id (`ncmd`);
4. spans that **overlap**.

**Why the identifier and not the span.** The identifier is what reaches the
case, so it is what correctness means; a system that reports the right span
having misread the citation has not extracted it. Comparing normalized keeps
the score independent of the source's damage — a filing that writes
`F.Supp.2d` names the same reporter as one that writes `F. Supp. 2d`, and
neither spelling is more correct.

**Why overlap and not exact spans.** Once the identifier is right, where a
citation's edges lie is a matter of convention. The span's remaining job is to
say *which* occurrence is meant, since one authority is often cited many times
in a filing, and overlap is enough for that.

A locator prediction must therefore carry `volume`, `reporter` and `page`, and
a docket prediction its court. Matching is greedy and each benchmark occurrence
is claimed once, so one citation reported twice earns one true positive and one
false positive.

## Get the dataset

```bash
uv sync

uv run hf auth login

uv run hf download gt-csse/false-citation-bench --repo-type dataset \
  --local-dir data/false-citation-bench
```

The dataset repository is private, so the download needs a Hugging Face account
with access to the `gt-csse` organisation.

## The components

Five components combine into the arms below. The names are used consistently in
the code, the output and this document.

| component | what it does |
|---|---|
| **eyecite as published** | eyecite with no help from us — the floor |
| **whitespace repair** | collapses runs of repeated spaces and tabs before parsing, then maps every span back onto the original text |
| **layout-tolerant tokenizer** | rebuilds eyecite's patterns so the separators between volume, reporter and page tolerate any whitespace, line breaks included |
| **site hunting** | masks what was already found, then sweeps the rest for any reporter string the gazetteer knows with digits on both sides |
| **model adjudication** | a model rules on one candidate site at a time, quoting what it sees so the answer can be grounded back into the document |

Why whitespace repair earns its place: eyecite's generated patterns join volume,
reporter and page with a **literal single space**, so one doubled space — which
PDF extraction leaves behind routinely — makes a citation vanish outright rather
than parse imperfectly. Collapsing those runs and remapping the spans costs
nothing and moves no offsets.

Why site hunting comes before the model: the gazetteer holds 4,795 reporter
strings, but after masking only 27 of them still occur anywhere in this corpus,
at 88 positions. The model is asked about those 88 windows, not about 26 whole
documents — and because the hunt knows *which* reporter flagged each site, the
prompt can carry examples for that reporter specifically.

Why the layout-tolerant tokenizer is experimental: relaxing the separator to
`\s*` also matches newlines, which is how a citation split across a page break
is recovered — and also how PDF margin line numbers get read as a page.

Folding newlines into the whitespace repair instead — `\s+` rather than
`[ \t]{2,}` — was measured and rejected. It is the same trade in a different
place: recall rises 94.5% → 94.8%, and precision falls from 100.0% to 99.8% on
the same `214 F.3d\n\n1`.

## The arms

| arm | components | model |
|---|---|:--:|
| `eyecite` | eyecite as published | — |
| `production` | + whitespace repair | — |
| `production+recovery` | + site hunting + model adjudication | yes |
| `layout-tolerant+recovery` | + layout-tolerant tokenizer, + both | yes |

**`production` is what Mellea-LRC ships.** Everything past it is experimental,
and has no domain-object form yet: an `AdjudicatedLocator` is not an
`ExtractedCitation`, so the experimental arms emit public occurrences directly
rather than a serialized `ExtractedDocument`.

## Run an arm

Mellea-LRC's own extraction, over the benchmark corpus:

```bash
uv run python -m evaluations.extraction.run \
  --arm production \
  --documents data/false-citation-bench/documents_txt \
  --output run-artifact.jsonl
```

| arm | components |
|---|---|
| `eyecite` | eyecite as published |
| `production` | eyecite + whitespace repair — what Mellea-LRC ships |

The two model arms need an OpenAI-compatible endpoint. Point `uv` at your
`.env`, as the validation page describes:

```bash
uv run --env-file .env python -m evaluations.extraction.run \
  --arm production+recovery \
  --documents data/false-citation-bench/documents_txt --output run-artifact.jsonl
```

Eyecite writes `Unknown overlap case…` to stderr as it runs. That is its own
diagnostic about overlapping citation tokens, not an error in your run.

To score a system of your own, either register it in `ARMS` or skip the runner
and write the JSONL directly, as below.

## Write a run artifact

One JSON object per line:

```json
{"document":"001__…__partial-motion-to-dismiss.txt", "span":{"start":2163,"end":2175},
 "volume":"556","reporter":"U.S.","page":"662","matched_text":"556 U.S. 662"}

{"document":"010__…__complaint.txt", "span":{"start":21579,"end":21594},
 "matched_text":"No. 1:19-CV-362","court":"M.D.N.C","court_id":"ncmd"}
```

- `document` — the filename exactly as published under `documents_txt/`.
- `span` — half-open offsets into the document **body**.
- a locator needs `volume`, `reporter`, `page`; a docket needs `matched_text`
  and a court.

Any other field is carried into the report untouched, so add whatever helps you
read a result. Nothing in this format is Mellea-LRC-specific.

## Evaluate

```bash
uv run python evaluations/extraction/evaluate.py \
  --benchmark data/false-citation-bench/derived/extraction.jsonl \
  --artifact run-artifact.jsonl \
  --output-dir evaluation-result
```

```text
| Metric | Value |
|---|---:|
| Expected occurrences | 594 |
| Predicted occurrences | 563 |
| True positives | 563 |
| False positives | 0 |
| False negatives | 31 |
| Precision | 100.0% |
| Recall | 94.8% |
| F1 | 97.3% |
```

## Reference results

Measured against this benchmark, on the 26 published documents.

| arm | predicted | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `eyecite` | 526 | 526 | 0 | 68 | 100.0% | 88.6% | 93.9% |
| `production` | 563 | 563 | 0 | 31 | 100.0% | 94.8% | 97.3% |
| `production+recovery` | 594 | 594 | 0 | 0 | **100.0%** | **100.0%** | **100.0%** |
| `layout-tolerant+recovery` | 595 | 594 | 1 | 0 | 99.8% | 100.0% | 99.9% |

**Whitespace repair alone is worth 37 citations.** Nothing clever happens
between the first two rows — repeated spaces are collapsed and the spans are
mapped back. That is the size of the problem a literal single space in a
generated pattern creates.

**Both recovery arms find everything**, and they are not equally good. The
layout-tolerant one also reports `214 F.3d\n\n1` — margin line numbers after a
page break read as a page, where the true citation is 214 F.3d **1058**. Its
relaxation buys nothing the model does not already recover, and pays for it with
a well-formed locator naming the wrong case: the worst failure available here,
because nothing downstream can tell it is wrong.

What it does buy is cost. It resolves 19 more citations by pattern, so 36 fewer
sites reach the model — 63 calls against 99, roughly 5m30s against 8m23s. If
model calls are the constraint the trade may be worth making; if correctness is,
it is not. **Prefer `production+recovery`.**

Of the 594 identifiers `production+recovery` reports, 563 come from the
extractor, 20 locators from the model, and 11 dockets. The 20 recovered locators
grounded exactly in 15 cases, after normalization in 2, and through
whitespace-relaxed matching in 3.

### A caveat on the perfect score

581 of the benchmark's records were themselves established by the
layout-tolerant tokenizer and 2 more by reading the source by hand, so arms
built from the same components share lineage with the labels. The score
measures agreement with a benchmark these tools helped construct — not
performance on unseen filings.

## Read the disagreements

`non_agreements.json` holds every miss and every spurious report in full — the
benchmark's row for a false negative, yours for a false positive:

```json
{
  "reason": "false_negative",
  "occurrence": {
    "document": "001__reaves-law-firm-pllc-v-baker-donelson-…__partial-motion-to-dismiss.txt",
    "kind": "locator",
    "span": {"start": 2977, "end": 2993},
    "matched_text": "937\n\nS.W.2d  796",
    "volume": "937", "reporter": "S.W.2d", "page": "796"
  }
}
```

This is the file worth reading. `matched_text` on a miss shows *why* it was
missed — here a page break falls between the volume and the reporter, so any
pattern joining them with a literal space cannot match. Grouping misses by that
shape says more than the recall figure does.

Two facts about the benchmark that a result should be read against. A system
that does not attempt docket numbers has a floor of 11 false negatives and
cannot exceed 98.1% recall. And three occurrences are deliberately excluded
because the filing states no complete identifier — a page lost to margin
numbering, a volume stranded in another table cell, a volume never written —
so reporting one scores a false positive.
