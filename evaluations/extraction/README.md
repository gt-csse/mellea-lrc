# Extraction evaluation

Scores extracted citations against the frozen **False Citation Bench —
Extraction** set: 594 identifiers across 26 filings.

Read [the shared setup](../README.md) first, in particular the coordinate
space.

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
