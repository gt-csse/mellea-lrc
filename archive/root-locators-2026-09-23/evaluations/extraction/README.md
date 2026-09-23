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
and nothing else. `tree.py` scores against `annotation-v4.0`, which is a tree --
every place a filing cites a case, which place introduced the case, and which
page each one claims.

| arm | what it runs |
|---|---|
| `eyecite` | eyecite as published, with no docket reader, and its own resolution attaching the leaves. The floor, and what a result is read up from |
| `augmented` | the same, with this project's rules: the separator relaxation, the docket reader, the pin cite reader, the case name locator. The leaves are still eyecite's to attach |
| `grown` | the augmented rules, and then the second growth: the leaves attached from `stated` rather than from the parse |
| `grown+identity` | the same, over roots a recorded identity run has settled -- replayed from `--identity`, so scoring it calls nothing |
| `mellea` | the augmented rules, then the model layers, then the leaves. Today that is the pin-cite review |

The name-only `ReferenceCitation` rows are left out of both sides unless
`--score-bare-names` is given; a reference that states a page is always scored.

```bash
uv run --env-file .env python -m evaluations.extraction.tree \
  --dataset  <store>/annotation-v4.0/documents \
  --documents <store>/corpus/documents_txt \
  --arms eyecite augmented grown
```

`--arms eyecite augmented grown` runs the three that call no model, and
`--detail` prints every disagreement under the tables.

```text
counts, recall · precision

              eyecite            augmented          grown
citations     611/711 · 611/620  707/711 · 707/713  707/711 · 707/709
- roots       374/427 · 374/377  426/427 · 426/426  426/427 · 426/426
- short forms 234/284 · 234/243  281/284 · 281/287  281/284 · 281/283
pin cites     350/463 · 350/355  458/463 · 458/460  458/463 · 458/460
docket courts 0/42 · 0/0         42/42 · 42/42      42/42 · 42/42
courts        474/565 · 474/478  547/565 · 547/551  547/565 · 547/551
dates         487/577 · 487/490  563/577 · 563/564  563/577 · 563/564
attribution   218/284 · 218/243  281/284 · 281/287  281/284 · 281/283

percentages, recall · precision

              eyecite            augmented          grown
citations     85.9% · 98.5%      99.4% · 99.2%      99.4% · 99.7%
- roots       87.6% · 99.2%      99.8% · 100.0%     99.8% · 100.0%
- short forms 82.4% · 96.3%      98.9% · 97.9%      98.9% · 99.3%
pin cites     75.6% · 98.6%      98.9% · 99.6%      98.9% · 99.6%
docket courts 0.0% · --          100.0% · 100.0%    100.0% · 100.0%
courts        83.9% · 99.2%      96.8% · 99.3%      96.8% · 99.3%
dates         84.4% · 99.4%      97.6% · 99.8%      97.6% · 99.8%
attribution   76.8% · 89.7%      98.9% · 97.9%      98.9% · 99.3%

by kind, recall

                        eyecite            augmented          grown
- DocketCitation        0/42               42/42              42/42
- FullCaseCitation      548/590            589/590            589/590
- IdCitation            32/32              31/32              31/32
- ReferenceCitation     2/7                5/7                5/7
- ShortCaseCitation     29/40              40/40              40/40
```

**What the rules are worth.** 96 citations and 108 pin cites over eyecite as
published, at no cost to precision on anything but attribution: most of what
they add is a citation eyecite read at the wrong edges or did not read at all.
42 of the 96 are docket citations, which eyecite attempts none of -- its
tokenizer is built from a reporter gazetteer and a docket number names no
reporter.

**`case names` is where the name is written, and it is scored on roots.** What
is compared is the span: the annotation records the characters the filing wrote,
and an arm that reports the same offsets read the same name. It is a question
about the document and nothing else -- what an *archive* calls the case is
`found`, and comparing the two is validation's finding rather than this one.

**`- overlapping` is the same name asked a weaker way**, and the two are read
together. `case names` is the span the filing writes, which is what a reader
wanting the characters needs. `overlapping` asks only whether the name was read
*at this citation* -- any span touching the annotated one counts -- and that is
the question the stages after this one turn on: identity matches a name against
an archive leniently and correctly, so a name whose edges are a word out still
reaches the right case, while a name read somewhere else entirely reaches
nothing.

The gap between the two rows is the population whose edges are off and whose
case is not: 19 of the corpus's 424 roots, 16 of eval-1's 332, 11 of eval-2's
296. What fails even on overlap is five rows on the corpus -- three where no
name was read at all, one read at another citation's name, and one citation the
arm does not read.

A leaf is left out of both. `Iqbal , 556 U.S. at 678` writes a name and it is a
part of one; whether a short form's `Huri` is the right part is the attachment
question, which `attribution` already scores.

**In this recorded identity-only run, the name is final after the identity stage.**
The later optional leaf case-name site hunt was not part of these measurements.
Measured over the corpus, identity moves 423 root
names from 400 right to 403 -- it puts five right and breaks two, and the 18 it
leaves are almost all span defects rather than identity questions: a name cut
short at the front (`Broadway Assoc., LLC v. Layens` for `3694 Broadway Assoc.,
LLC v. Layens`), or one that swallowed the furniture in front of it
(`Cases Page(s) Ahanchian v. Xenon Pictures`). No archive answers those.

**A leaf states no court**, so it is left out of both sides. `Id. at 570` is its
root's citation and its root's court, which is not a second claim to be right or
wrong about, and an arm carrying the root's court forward is neither credited
nor charged for it.

**Where the court is still lost.** 18 of the corpus's 565, and they are five
things rather than one: four rows in a table of authorities where the court sits
outside the span the reader takes; three `A.D.3d` citations whose department is
written (`3d Dep't`); two `D. N. Mar. I.` the court reader does not reach; two
`Ct. App.` after `218 Ariz. 293`, where the parenthetical names a court that is
only unambiguous inside the state the reporter names; and `022-o01`, given `azd`
where the filing writes a Western District of Virginia citation -- a court
bleeding from the citation before it, which is why the annotation reads its
courts from each row's own characters and not from the parse.

**What the second growth is worth.** The same citations, attached better. On
this corpus it removes four false attributions of an `Id.` whose antecedent is
a statute, which no case root can hold, and attribution precision goes from
97.9% to 99.3%. On the two held-out sets it is worth one short form of recall
on `extraction-eval-1` (167/177 to 168/177, precision 100% either way) and
one on `extraction-eval-2` (159/163 to 160/163, precision 100% either way).
It is not a large difference and it
is not meant to be one: the measurement here is against roots nobody has
corrected, where `stated` is still the parse, and what the growth is for is the
names validation settles.

`grown+identity` replays a recorded identity run onto the roots first -- 31 of
this corpus's roots come back with a corrected name, including
`Cnty. of Bernalillo` written out as
`Solis-Marrufo v. Bd. of Comm'rs for Cnty. of Bernalillo`. It moves no leaf.
Every leaf those names would reach is already reached by the volume, the
reporter and the page, which the filing states at the leaf itself; the name is
what decides a `supra` or a bare-name reference, and those are the forms this
corpus writes least.

**Every denominator is what the filings state.** Recall is out of the 711
citations, the 427 roots, the 284 short forms, the 463 pin cites, the 42 docket
courts -- never out of the part of the corpus the arm happened to reach. A
denominator that shrinks with the run hides what the run missed, which is how
`attribution` once read 99.6% while it was scored over the citations an arm
found.

**Attribution is a short form pointing at its root**, so it is counted over the
284 short forms. A root points at itself and there is nothing there to get
wrong.

**Identification is exact; everything else is scored over an overlap.** A
citation is found when the arm produces one at exactly its span, because the
identifier is what a lookup resolves and nothing partial counts. But a run that
reads `673 F.2d at ` where the filing writes `673 F.2d at 57` *has* that
citation, with the wrong edges, and whether it then attributed it to the right
case is a separate question -- so a reported citation is associated with the
annotated one it overlaps most, and attribution, the pin cite and the docket
court are read through that association.

**`courts` and `dates` are the citation's other two claims.** A court is scored
where the annotation states one -- written in the filing, or named by the
reporter the citation is in, `556 U.S. 662 (2009)` being the Supreme Court's
without anyone saying so. What is compared is the courts-db id, because that is
the half a lookup takes. A citation in a reporter several courts publish in with
no court parenthetical states no court at all, and an arm that reports one there
is charged for it: `206 P. 327 (1922)` says nothing about which court, and a
guess is not a reading.

A date is compared as the annotation writes it -- `2009`, `2023-09`,
`2006-07-13` -- which is exactly as much of the date as the filing states. An
arm that reads a year where the filing writes a day has read less than the
filing states, and the strings differ, which is the answer.

**`docket courts` is the one field checked beside the span.** A docket number
names a case in no district on its own -- `1:19-cv-362` exists in every one of
them -- so the court is half of the identifier rather than decoration, which is
what `PROTOCOL.md` says and what `evaluate.py` already requires of a docket
prediction. It is the courts-db id that is compared, because that is the half a
lookup takes; the spelling the filing used is the document's characters, like
`reporter_as_written`. A short form of a docket states the number again and not
the court, so it is scored against its root's.

`eyecite` reads none of the 42, so its recall is 0% and it has no precision to
state: an arm that reports nothing of a kind cannot be right or wrong about it,
and the two sides of a cell say so separately.

**What costs precision** at `grown` is two spans read as a case citation that
refer to no case: an `Id.` into a motion filed in the same proceeding, and a
section heading read as a bare-name reference with `32` as its pin cite, which
is a page claim the filing never made. `augmented` reports those two and four
more, all of them an `Id.` attached across a statute to the case before it.

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
nothing about `annotation-v4.0` can be scored the same way.

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
than rebuilt, and `annotation-v4.0` annotates 682 of them as spans. Nothing
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
rather than a serialized `Document`.

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

## Locator-layer evaluation

`grow_annotated_corpus` reads the corpus and runs `grow_roots` once per
document. It exposes exactly three scores:

| score | prediction population | question |
|---|---|---|
| `reporter_locators` | every read `FullCaseCitation` | Did the reporter locator occur at the annotated span? |
| `docket_locators` | only audit-admitted `DocketCitation` records | Did the docket locator occur at the annotated span? |
| `colocation` | locator groups before the audit | Were the exact parallel-locator members grouped together? |

`eval_locators(corpus)` returns the first two scores; `eval_colocation(corpus)`
returns the third. `locator_layers.py` is the convenience orchestrator that
returns all three. These commands read only the annotated corpus, not the
held-out evaluation sets.

~~~bash
uv run python -m evaluations.extraction.eval_locators \
  --annotations <store>/annotation-v4.0/documents \
  --texts-root <store>

uv run python -m evaluations.extraction.eval_colocation \
  --annotations <store>/annotation-v4.0/documents \
  --texts-root <store>

uv run python -m evaluations.extraction.locator_layers \
  --annotations <store>/annotation-v4.0/documents \
  --texts-root <store>
~~~

An exact locator match is `(document, start, end)`: two appearances of the
same identifier count twice. The reporter and docket metrics do not read
`is_root`, `root_id`, court, date, or case name; root deduplication and context
resolution are separate questions. Short forms belong to the later leaf layer.

Raw docket-shaped candidates remain in the document trace so the audit can be
inspected, but they are absent from every public locator metric. A caption or
page stamp that the audit withdraws is not a docket prediction. `colocation`
scores exact groups of occurrence spans before the audit, because those groups
are input to the audit itself; repeated parallel citations remain separate
groups.
