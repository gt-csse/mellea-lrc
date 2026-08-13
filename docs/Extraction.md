---
tags: [extraction, eyecite, citations, spans]
status: active
---

# Extraction

Extraction is the middle stage. It takes the plain text a document was
preprocessed into, finds every citation in it, and returns each one as a typed
object with an offset back into that text. It decides nothing about whether a
citation is *real* — a fabricated case and a genuine one are extracted alike.
Judging them is [validation](./Validation%20Model%20Development.md)'s job.

This document walks through the whole stage: how to run it, what comes back,
what the engine is, where it fails, and what is being built to reach the
citations it misses.

---

## Running it

Three entrypoints, differing only in where the text comes from:

```python
from pathlib import Path
from mellea_lrc.extraction import extract

document = extract("See Brown v. Board of Education, 347 U.S. 483, 495 (1954).")
document = extract(Path("filing.pdf"))
```

`extract` dispatches on the argument's type — **a `str` is content, a `Path` is
a location** — and hands off to one of the two explicit forms,
`extract_from_plain_text(text)` and `extract_from_raw_document(path)`. Use those
directly when the distinction matters; a filename that arrives as a string would
otherwise be extracted *from*, rather than opened.

There is deliberately no entrypoint taking a `PreprocessedDocument`. Nothing
serializes one, so it cannot cross a process boundary, and a caller holding one
is already inside the library.

Extraction is offline and deterministic. It needs no credentials and makes no
network calls, which is why the `mellea-lrc` command does not expose it on its
own — it is a step of `validate`, not a thing to run.

---

## What comes back

An `ExtractedDocument`, which is the `PreprocessedDocument` it was built from
plus the citations found in it:

| field | what it is |
|---|---|
| `text` | the preprocessed text every offset indexes into |
| `citations` | one `ExtractedCitation` per occurrence, in document order |
| `extraction_metadata` | which backend ran, and its version |

Each `ExtractedCitation` carries:

| field | what it is |
|---|---|
| `citation_id` | stable identifier for this occurrence |
| `span` | the citation's full extent, including party names and parenthetical |
| `locator_span` | just the part that identifies the authority |
| `matched_text` | the source text under `span` |
| `citation` | the typed object — one of the eight kinds below |
| `resolves_to` | for a back-reference, the `citation_id` it points at |

### `span` and `locator_span` are not the same

For `Brown v. Board of Education, 347 U.S. 483, 495 (1954)`:

- `span` covers the whole thing, party names through the year parenthetical;
- `locator_span` covers `347 U.S. 483` alone.

The locator is what reaches the case in a reporter-indexed database, so it is
what lookup and evaluation both key on. The full span is what you highlight in a
document. Reporting the right locator with a slightly different full span is not
an error; reporting the right span with a misread locator is.

---

## The eight citation kinds

Every citation becomes exactly one of these. Fields not present in the source
are `None`; every kind also carries an optional `parenthetical`.

### Full citations

These state an authority completely and stand on their own.

**`FullCaseCitation`** — a reported case. `Bush v. Gore, 531 U.S. 98, 99 (2000)`

| field | example | notes |
|---|---|---|
| `plaintiff` | `Bush` | party before "v." |
| `defendant` | `Gore` | party after "v." |
| `volume` | `531` | |
| `reporter` | `U.S.` | |
| `page` | `98` | first page of the opinion |
| `pin_cite` | `99` | the specific page cited |
| `extra` | `aff'd, 123 F.3d 456` | subsequent history |
| `year` | `2000` | |
| `court` | `scotus` | eyecite's canonical court code |

**`FullLawCitation`** — a statute, regulation, or code section.
`42 U.S.C. § 1983(a)(1)`

`volume` is the title (`42`), `reporter` the code (`U.S.C.`), `page` the section
(`1983`), `pin_cite` the subsection. Also carries `year` and `publisher`.

**`FullJournalCitation`** — a law review or journal article.
`45 Harv. L. Rev. 123, 125 (2000)`

Fields are `volume`, `reporter`, `page`, `pin_cite`, `year`.

### Back-references

Legal writing states a citation in full once, then refers back to it. These
kinds carry a `resolves_to` pointing at the full citation they depend on, and
mean nothing without it.

| kind | form | fields | resolves to |
|---|---|---|---|
| `ShortCaseCitation` | `531 U.S. at 99` | `volume`, `reporter`, `page`, `pin_cite`, `court` | the `FullCaseCitation` for that reporter |
| `SupraCitation` | `Bush, supra, at 99` | `pin_cite` | the full citation named |
| `IdCitation` | `Id. at 100` | `pin_cite` | the immediately preceding citation |
| `ReferenceCitation` | `Bush v. Gore` | `plaintiff`, `defendant` | the `FullCaseCitation` for that case |

eyecite resolves these itself and the links come through pre-populated. They are
worth checking: an `Id.` crossing a paragraph break, or a `supra` whose
antecedent uses an abbreviated party name, are both places resolution goes
wrong.

### `UnknownCitation`

A span that reads as a citation but parses into no kind. It carries no fields —
only the span survives. Treat it as a signal about the text, not as a citation.

---

## The engine

### eyecite

Extraction is built on [eyecite](https://github.com/freelawproject/eyecite), the
Free Law Project's rule-based parser. It is strong out of the box and is both
the baseline and the working engine. The goal has never been to replace it, only
to reach what it leaves behind.

eyecite matches a gazetteer of literal reporter strings — some 4,800 spellings
drawn from `reporters-db` — and then runs the generated regex for whichever
strings appear. This is fast and precise, and it is also the source of both
failure modes below.

### Whitespace repair

The one addition in the shipping pipeline. eyecite's generated patterns join
volume, reporter and page with a **literal single space**, so one doubled space
makes a citation vanish outright rather than parse imperfectly — and doubled
spaces are exactly what PDF extraction of justified text leaves behind.

So the text is collapsed on `[ \t]{2,}` before parsing, and every resulting span
is mapped back onto the original text with eyecite's `SpanUpdater`. Offsets
still index the text as it was; nothing downstream sees the collapsed copy.

It is a small change that is worth **37 citations** on the benchmark — the
distance between 88.6% and 94.8% recall, with no cleverness in between.

Note what it deliberately does not do: it collapses spaces and tabs, not
newlines. Widening it to `\s+` was measured and rejected. See the trade below.

---

## Where extraction still fails

Two distinct modes, worth separating because they call for different fixes.

**Mis-parsed fields on clean text.** The span is found, but a field boundary is
read wrongly. eyecite parses the plaintiff of
`Methodist Hosp. of Sacramento v. Shalala, 38 F.3d 1225` as `Sacramento`. The
text is not damaged; the rule is. Fixes here transfer to any corpus.

**Non-recognition of damaged text.** The citation is never seen at all, because
a line break, a page break, or an OCR artefact falls inside it. `937\n\nS.W.2d
796` matches no pattern that joins volume to reporter with a space. This is an
upstream failure surfacing as an extraction failure, and it is what the
experimental work below targets.

A practical consequence: **keep a citation on one line.** A break between the
reporter and the page means the citation is missed rather than misread, which is
the quieter of the two problems but not the smaller one.

---

## Reaching the rest: experimental work

Both live in `mellea_lrc.experimental` and neither is in the shipping pipeline.

### Layout-tolerant tokenizer

Rebuilds eyecite's patterns so the separators between volume, reporter and page
tolerate any whitespace, newlines included. This recovers citations split across
a page break — and, because `\s*` cannot tell a page break from a margin, also
lets PDF line numbers be read as a page. It is the same trade as widening
whitespace repair, in a different place.

### Grounded adjudication

The sounder approach, and the one to build on. Four steps:

1. **Mask** everything already extracted, leaving only the text that failed.
2. **Hunt** that residue for any gazetteer string with digits close on both
   sides — the volume-and-page shape.
3. **Adjudicate** each candidate site with a model, one at a time, asking only
   whether the characters at *this* position state a complete identifier.
4. **Ground** the answer by locating the model's verbatim quote back in the
   document. A quote that cannot be found is dropped.

The point of hunting is not speed, it is the shape of the question. Without it
the model is asked "what citations are in this document?" — open, generative,
and unconstrained by what the text actually says. With it the question is closed
and checkable: the position is known before the model speaks, so every answer
can be verified against the document and discarded if it does not ground.

The narrowing is steep. Hunting searches roughly 4,795 gazetteer strings; after
masking, **27 of them occur anywhere in the 26-filing corpus, at 88 positions**.
The model sees 88 short windows instead of 26 whole documents. Because the hunt
knows which string flagged each site, the prompt can be specific to that
reporter.

Two things the numbers do not say. The gazetteer is not a list of case
reporters: `U.S.C.` alone accounts for 20 of those 88 sites and is a statute
code, and several others are obscure state reporters whose abbreviations collide
with ordinary words. And the hunt is deliberately over-permissive — a judge that
rejects freely costs far less than a citation never surfaced.

---

## What it scores

Measured against the frozen extraction benchmark: 594 identifiers across 26
filings.

| arm | predicted | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| eyecite as published | 526 | 526 | 0 | 68 | 100.0% | 88.6% | 93.9% |
| **+ whitespace repair** (ships) | 563 | 563 | 0 | 31 | 100.0% | 94.8% | 97.3% |
| + hunting and adjudication | 594 | 594 | 0 | 0 | **100.0%** | **100.0%** | **100.0%** |
| + layout-tolerant tokenizer | 595 | 594 | 1 | 0 | 99.8% | 100.0% | 99.9% |

The last two both find everything, and they are not equally good. The
layout-tolerant arm also reports `214 F.3d 1`, reading margin line numbers after
a page break as a page, where the true citation is `214 F.3d` **1058**. It buys
nothing the model does not already recover and pays with a well-formed locator
naming the wrong case — the worst failure available here, because nothing
downstream can tell it is wrong.

What it does buy is cost: 19 more citations resolved by pattern means 36 fewer
sites reach the model, 63 calls against 99. If model calls are the constraint
that may be worth it. If correctness is, it is not.

**Read the perfect score carefully.** 581 of the benchmark's 594 records were
themselves established by the layout-tolerant tokenizer, so these arms share
lineage with the labels they are scored against. The number measures agreement
with a benchmark these tools helped build, not performance on unseen filings.

The benchmark itself — what an occurrence is, how a prediction is matched
against one, what the arms are, and how to reproduce any row above — is
documented with the code that runs it, in
[the extraction evaluation](../evaluations/extraction/README.md). Its contents
and provenance are described on
[the dataset card](https://huggingface.co/datasets/gt-csse/false-citation-bench).
