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
page each one claims:

```bash
uv run python -m evaluations.extraction.tree \
  --dataset  <store>/extraction-v3.0/documents \
  --documents <store>/corpus/documents_txt \
  --relaxation FULL
```

```text
                              recall         precision
citations            710/721   98.5%   710/717   99.0%
- roots              426/428   99.5%   426/426  100.0%
- short forms        283/293   96.6%   283/291   97.3%
pin cites            460/463   99.4%   460/460  100.0%
attribution          707/710   99.6%   707/707  100.0%

by kind, recall:
- DocketCitation        42/42
- FullCaseCitation      589/590
- IdCitation            32/32
- ReferenceCitation     6/16
- ShortCaseCitation     41/41

roots:
- 006-o39 'Rosenblatt v. Baer, 383 U.S. at 85' read as a short form
- 025-o13 'Watson v. New York  , WL 6200979 (S.D.N.Y.' not read
…
```

**Recall is out of what the filings state; precision is out of what the run
reports.** One without the other hides half of a pass: a reader that reports
every span in the document has perfect recall. Detail lines are indented
with `- `.

A root counts as found only when the run reads it *as* a root. A root filed
under some other case is a root the run did not find, whatever it did with the
span — which is what `006-o39` is, `Rosenblatt v. Baer, 383 U.S. at 85`, read
by extraction as a short form of the case quoting it.

**Attribution is scored over the citations a run found**, not over all of them:
a citation nobody read was not attributed wrongly, it was missed, and charging
it here would count one failure twice.

**Every metric is counted against the ground truth's own denominator.**
`pincite_parsed` is out of the 446 pin cites the filings state, not out of the
citations this run found -- a denominator that shrank with the run would hide
what the run missed. Detail lines are indented with `- `.

| relaxation | citations | roots | short forms | pin cites | attribution |
|---|---|---|---|---|---|
| `NONE` | 630/721 · 630/664 | 408/428 | 222/293 | 351/463 | 627/630 |
| `BOUNDED` | 709/721 · 709/716 | 426/428 | 282/293 | 460/463 | 705/709 |
| `FULL` | **710/721 · 710/717** | **426/428** | **283/293** | **460/463** | **707/710** |
| `FULL` + the case-name layer | **720/721 · 720/727** | 426/428 | **293/293** | 460/463 | **717/720** |

Each cell is recall, and for citations precision beside it.

`root_parsed` is the roots alone, the 428 citations that state an identifier for
the first time. A short form missed costs a page claim; a root missed costs the
case, and the one that is missed is the citation with no volume.

**What costs precision.** Seven spans the run reads as a case citation refer to
no case at all: two `Id.` into motions filed in the same proceeding, four into
an exhibit declaration and a statute, and a section heading eyecite reads as a
bare-name reference. Five of the seven reach no root, so nothing downstream
would check anything for them. Two do, and those are the damaging ones: an
`Id.` pointing at `Doc. 387` is attributed to `Perez v. Sunbeam Prods., Inc.`
1,700 characters earlier, and a section heading reading `Chen Zhi 32` is
attributed to `United States v. Chen Zhi` **with `32` as a pin cite**, which
is a page claim the filing never made.

Eleven of the eleven misses at `FULL` are citations no reader can reach: ten
bare names, which state no identifier at all, and document 025's
`WL 6200979`, which states no volume. The three pin cites are two shapes: two
carry a footnote marker eyecite's pattern does not admit, and the third sits in
a table of authorities where the leader dots follow the page.

## Case names nothing read

`tree.py` says the gap: at `FULL`, eleven citations are missed and ten of
them are a case name with no identifier at all -- a Bluebook Rule 10.9 short
form, which a reporter-driven tokenizer cannot see because there is nothing
there to tokenize. This layer goes after them.

It masks every citation that was read, sweeps the residue for case names, and
asks a reader what each one is. Four answers:

| reading | what it means |
|---|---|
| `names_a_citation` | the name is the case name of a citation already read, not a citation of its own |
| `short_form` | a proper Rule 10.9 reference to a case cited in full elsewhere, and which root it is |
| `uncited_case` | a case the filing leans on that the document never cites |
| `not_a_citation` | a caption, a heading, a party discussed in prose, a roman numeral `v` |

`names_a_citation` asserts no error. A filing writes `In Boeser v. Sharp , the
court recognized …` and then the citation, and both names are right; what the
answer carries is where the name is written, so a consumer can hold the fuller
of the two. That is `case_name_span` on the citation -- a span, not a parse,
because `In re Flint Water Cases` is a whole name and eyecite files it under
`defendant` with the opening words stripped. The parsed party fields are left
exactly as they were read.

The reader never returns an offset. It quotes the name verbatim and picks a
citation or a root **by number** from lists this layer built, so every part of
the answer grounds back into the record deterministically or fails to.

```bash
uv run --env-file .env python -m evaluations.extraction.name_recovery \
  --dataset  <store>/extraction-v3.0/documents \
  --documents <store>/corpus/documents_txt
```

```text
sites                 54
- bare short forms    13
- uncited cases       18
- not annotated       23
declined              0/54
recovered             13/13
root_right            10/10
- 006 'In Boeser v. Sharp' read as names_a_citation, so no root was named
- 006 'United States  v.  Hassan' read as names_a_citation, so no root was named
- 021 "In Loos v. Lowe's" read as names_a_citation, so no root was named
uncited_right         18/18
invented              0/23
false_defect          2/23
- 010 'In re COvIDrelated' read as a case the filing never cites
- 025 'Watson v. New York' read as a case the filing never cites
```

The headline numbers count only the 31 sites the dataset annotates, so here is
what became of all 54:

| the dataset says | the reader said | n |
|---|---|---:|
| bare short form | `short_form`, root named and right | 10 |
| | `names_a_citation`, the name of the citation a sentence away | 3 |
| uncited case | `uncited_case` | 18 |
| not annotated | `names_a_citation` | 17 |
| | `not_a_citation` | 4 |
| | `uncited_case` | 2 |

52 of the 54 are right. The 17 `names_a_citation` answers on unannotated sites
are the finding this dataset cannot score, because it does not annotate case
names: each is the name of a citation already in the record, fuller than the
one the parse reached. Eleven of the seventeen are one document where the
extraction spaces the apostrophe out of a party name and the name search stops
there.

### What is left

Two sites wrong, both reported as a defect the filing does not have: a name the
converter damaged into something that is not a case name, and a citation whose
volume the filing never wrote, which the reader reads as a case cited nowhere.
Neither enters the record as a citation.

**What `uncited_case` turns on.** It is the answer when the filing leans on the
case for something it wants accepted -- a holding, a standard, or that
something happened -- and nothing in the document cites it. It is not the
answer when a case is named for another reason: whose matter it is, what a
heading says, who the parties to this filing are. Narrowing it to *legal*
propositions alone was tried and is wrong -- a filing that offers a case as
evidence that a pattern of conduct exists is still offering it.

A switch remains for the ordering reading, `--unordered`, which lets a short
form stand before the citation it refers to. It reaches nothing more here and
invents two citations, so it is off by default.

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
