# Extraction evaluation

Scores extracted citations against the frozen **False Citation Bench —
Extraction** set: 594 identifiers across 26 filings.

Read [the shared setup](../README.md) first, in particular the coordinate
space.

## End to end

```bash
uv run hf download gt-csse/false-citation-bench --repo-type dataset \
  --local-dir data/false-citation-bench

uv run python -m evaluations.extraction.run --arm bounded \
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

## The citation tree

`evaluate.py` scores a flat list of identifiers: whether a citation was found,
and nothing else. `tree.py` scores against `extraction-v3.0`, which is a tree --
every place a filing cites a case, which place introduced the case, and which
page each one claims -- and it scores three arms against it.

| arm | what it runs |
|---|---|
| `eyecite` | eyecite as published. The floor, and what a result is read up from |
| `augmented` | the same, with this project's rules: the separator relaxation, the docket reader, the pin cite reader, the case name locator |
| `mellea` | the augmented rules, then the model layers. Today that is the case-name layer; more land here as they are built |

```bash
uv run --env-file .env python -m evaluations.extraction.tree \
  --dataset  <store>/extraction-v3.0/documents \
  --documents <store>/corpus/documents_txt
```

`--arms eyecite augmented` runs the two that call no model, and `--detail`
prints every disagreement under the tables.

```text
counts, recall · precision

              eyecite            augmented          mellea
citations     644/721 · 644/664  710/721 · 710/717  720/721 · 720/727
- roots       408/428 · 408/411  426/428 · 426/426  426/428 · 426/426
- short forms 232/293 · 232/253  283/293 · 283/291  293/293 · 293/301
pin cites     351/463 · 351/357  460/463 · 460/462  460/463 · 460/462
attribution   640/644 · 640/657  707/710 · 707/709  717/720 · 717/719

percentages, recall · precision

              eyecite            augmented          mellea
citations     89.3% · 97.0%      98.5% · 99.0%      99.9% · 99.0%
- roots       95.3% · 99.3%      99.5% · 100.0%     99.5% · 100.0%
- short forms 79.2% · 91.7%      96.6% · 97.3%      100.0% · 97.3%
pin cites     75.8% · 98.3%      99.4% · 99.6%      99.4% · 99.6%
attribution   99.4% · 97.4%      99.6% · 99.7%      99.6% · 99.7%

by kind, recall

                        eyecite            augmented          mellea
- DocketCitation        42/42              42/42              42/42
- FullCaseCitation      548/590            589/590            589/590
- IdCitation            22/32              32/32              32/32
- ReferenceCitation     2/16               6/16               16/16
- ShortCaseCitation     30/41              41/41              41/41
```

**What each arm is worth.** The rules are worth 66 citations and 109 pin cites
over eyecite as published, and they cost nothing: precision rises with recall,
because most of what they add is a citation eyecite read at the wrong edges or
did not read at all. The model layer is worth the last 10, which are the bare
names -- a case named with no identifier at all, which no reporter-driven
tokenizer can see because there is nothing there to tokenize. It takes
`ReferenceCitation` recall from 6/16 to 16/16 and leaves every other measure
where it was.

The one citation still missed at `mellea` is the one whose volume the filing
never wrote.

**What costs precision**, and it is the same seven at every arm: spans read as a
case citation that refer to no case at all -- two `Id.` into motions filed in
the same proceeding, four into an exhibit declaration and a statute, and a
section heading eyecite reads as a bare-name reference. Five reach no root, so
nothing downstream would check anything for them. Two do, and those are the
damaging ones: an `Id.` pointing at `Doc. 387` is attributed to
`Perez v. Sunbeam Prods., Inc.` 1,700 characters earlier, and a section heading
reading `Chen Zhi 32` is attributed to `United States v. Chen Zhi` **with `32`
as a pin cite**, which is a page claim the filing never made.

### What is scored, and how

An annotated citation is **found** when the arm produces a citation at exactly
its `locator` span, or at its `cited_as` span where it has no locator, which is
`Id.` and the bare-name references. Nothing partial counts.

The **roots** are counted apart from the **short forms**, because the two cost
different things: a short form missed costs a page claim, a root missed costs
the case. A root counts as found only when the arm reads it *as* a root -- a
root filed under some other case is a root it did not find, which is what
`Rosenblatt v. Baer, 383 U.S. at 85` is, read as a short form of the case
quoting it.

A root is named by its span rather than by a dataset id, so an arm that knows
nothing about `extraction-v3.0` can be scored the same way.

**Recall is out of what the filings state; precision is out of what the arm
reports.** One without the other hides half of a pass: a reader that reports
every span in the document has perfect recall. Both denominators are taken over
the whole run rather than over the part of it the dataset annotates -- a pin
cite stays in the recall denominator when the citation carrying it was missed,
and an attribution counts against precision even when the thing attributed is
not a citation to a case.

**Attribution is scored over the citations an arm found**, because a citation
nobody read was missed rather than misattributed, and charging it here would
count one failure twice.

### Not measured yet

Two whole columns of the ground truth have no reading here.

**Case names.** Every citation now carries `case_name_span`, located rather
than rebuilt, and `extraction-v3.0` annotates 682 of them as spans. Nothing
compares the two. It is the one field where the stages could be read against
each other -- what the rules locate, what a model patches, what a lookup later
confirms -- which is most of the reason to score it at all. The dataset says in
writing that it does not score case names, so that sentence moves first.

**The normalized half of every field.** A citation carries both what the filing
wrote and what it means: `reporter` beside `reporter_as_written`, `court`
beside `court_text`, `pin_cite.normalized` beside the damaged string. Only the
pin cite's normalization is checked, inside the pin cite measure. Whether
`F.Supp.2d` resolved to the right reporter and `Bankr. S.D. Fla.` to the right
courts-db id is not scored anywhere, and those are exactly the halves a lookup
uses.

**A docket citation's court is unchecked, and it is half the identifier.**
`PROTOCOL.md` puts it plainly -- `1:19-cv-362` exists in every district and
names a case in none of them alone -- and `evaluate.py` refuses a docket
prediction whose court does not match. This table matches on span alone, so all
42 docket citations count as found whether or not the court resolved. They all
do today, which is exactly why nothing here would notice if they stopped.

The **date** is a different case and is deliberately absent: it is not part of
any identifier, and the year a filing states against the year on the record is
a misrepresentation question, which belongs to validation. The same goes for
the court in a reporter citation's parenthetical, which no lookup needs.

`BOUNDED` is not an arm. It was the control full relaxation was read against,
and full relaxation is now at least as good on every measure here. It remains
the library's default and an arm of the published flat bench above.

## Get the dataset

```bash
uv sync

uv run hf download gt-csse/false-citation-bench --repo-type dataset \
  --local-dir data/false-citation-bench
```

## The components

Five components combine into the arms below. The names are used consistently in
the code, the output and this document.

| component | what it does |
|---|---|
| **eyecite as published** | eyecite with no help from us — the floor |
| **bounded relaxation** | rebuilds eyecite's patterns so the separators between volume, reporter and page match whatever whitespace is there, stopping short of a blank line between reporter and page |
| **full relaxation** | the same, with that last bound removed: a page may sit past a blank line |
| **site hunting** | masks what was already found, then sweeps the rest for any reporter string the gazetteer knows with digits on both sides |
| **model adjudication** | a model rules on one candidate site at a time, quoting what it sees so the answer can be grounded back into the document |

Both relaxations are the same code path and the same
`mellea_lrc.extraction.Relaxation` parameter, differing in one join. Nothing
rewrites the text at any level, so no span is ever remapped.

Why relaxation earns its place: eyecite's generated patterns join volume,
reporter and page with a **literal single space**, so one doubled space — which
PDF extraction leaves behind routinely — makes a citation vanish outright rather
than parse imperfectly. Matching the damage where it is costs one substitution
per pattern and moves no offsets, and it also reaches the opposite defect,
`846F.2d746`, which repairing the text could not.

Why site hunting comes before the model: the gazetteer holds 4,795 reporter
strings, but after masking only 27 of them still occur anywhere in this corpus,
at 88 positions. The model is asked about those 88 windows, not about 26 whole
documents — and because the hunt knows *which* reporter flagged each site, the
prompt can carry examples for that reporter specifically.

Why full relaxation is not the default: the join it opens is the one between
reporter and page, and the page is what lands past the break. On pleading paper
that is where the margin line numbers are, so `214 F.3d\n\n1` reads as page 1
when the citation is `214 F.3d 1058` — a well-formed locator naming the wrong
case. On this corpus it costs precision 100.0% → 99.8% for one extra citation.

## The arms

| arm | components | model |
|---|---|:--:|
| `eyecite` | eyecite as published | — |
| `bounded` | + bounded separator relaxation | — |
| `full` | full separator relaxation instead | — |
| `bounded+recovery` | + site hunting + model adjudication | yes |
| `full+recovery` | full relaxation, + both | yes |

**Arms are named for the mechanism they run, not for a status.** None of them
is "what ships": the library on `main` is still eyecite plus a whitespace
repair, which this branch removes, and `bounded` is not the intended
destination either — it is the control `full` is read against. Everything from
`bounded+recovery` on is experimental,
and has no domain-object form yet: an `AdjudicatedLocator` is not an
`ExtractedCitation`, so the experimental arms emit public occurrences directly
rather than a serialized `ExtractedDocument`.

## Run an arm

Mellea-LRC's own extraction, over the benchmark corpus:

```bash
uv run python -m evaluations.extraction.run \
  --arm bounded \
  --documents data/false-citation-bench/documents_txt \
  --output run-artifact.jsonl
```

| arm | components |
|---|---|
| `eyecite` | eyecite as published |
| `bounded` | eyecite + bounded separator relaxation |
| `full` | eyecite + full separator relaxation |

The two model arms need an OpenAI-compatible endpoint. Point `uv` at your
`.env`, as the validation page describes:

```bash
uv run --env-file .env python -m evaluations.extraction.run \
  --arm bounded+recovery \
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
| Predicted occurrences | 582 |
| True positives | 582 |
| False positives | 0 |
| False negatives | 12 |
| Precision | 100.0% |
| Recall | 98.0% |
| F1 | 99.0% |
```

## Reference results

Measured against this benchmark, on the 26 published documents.

| arm | predicted | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `eyecite` | 526 | 526 | 0 | 68 | 100.0% | 88.6% | 93.9% |
| `bounded` | 582 | 582 | 0 | 12 | 100.0% | 98.0% | 99.0% |
| `bounded+recovery` | 594 | 594 | 0 | 0 | **100.0%** | **100.0%** | **100.0%** |
| `full+recovery` | 595 | 594 | 1 | 0 | 99.8% | 100.0% | 99.9% |

The two model arms were measured when `bounded` was eyecite plus a
whitespace repair rather than the bounded relaxation, and have not been
re-measured since. Their totals are unlikely to move — both already reach every
identifier — but the split between what the pattern finds and what the model
recovers has, and so has the number of model calls.

**Bounded relaxation alone is worth 56 citations.** No model, no text rewriting:
the patterns simply match the whitespace that is there. That is the size of the
problem a literal single space in a generated pattern creates.

**`bounded` now sits one citation off its own ceiling.** Eleven of its twelve
misses are docket numbers, which eyecite does not attempt at all — the floor
noted below. The twelfth is `455 US. 363`, a reporter missing the period after
`US`, which is a different defect from the one relaxation addresses.

**Both recovery arms find everything**, and they are not equally good. The
full one also reports `214 F.3d\n\n1` — margin line numbers after a
page break read as a page, where the true citation is 214 F.3d **1058**. Full
relaxation buys nothing the model does not already recover, and pays for it with
a well-formed locator naming the wrong case: the worst failure available here,
because nothing downstream can tell it is wrong.

What it does buy is cost, by resolving more citations before the model is asked.
**Prefer `bounded+recovery`.**

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
    "document": "013__gunter-v-contango-ore-inc-et-al__complaint.txt",
    "kind": "locator",
    "span": {"start": 51325, "end": 51336},
    "matched_text": "455 US. 363",
    "volume": "455", "reporter": "U.S.", "page": "363",
    "note": "reporter written 'US.' rather than 'U.S.'; outside the gazetteer, so no tokenizer reaches it"
  }
}
```

This is the file worth reading. `matched_text` on a miss shows *why* it was
missed — here the reporter is written `US.`, missing a period the gazetteer
requires, which no amount of separator relaxation reaches. Grouping misses by
that shape says more than the recall figure does.

Two facts about the benchmark that a result should be read against. A system
that does not attempt docket numbers has a floor of 11 false negatives and
cannot exceed 98.1% recall. And three occurrences are deliberately excluded
because the filing states no complete identifier — a page lost to margin
numbering, a volume stranded in another table cell, a volume never written —
so reporting one scores a false positive.
