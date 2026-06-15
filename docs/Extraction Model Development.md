---
tags: [extraction, annotation, pre-annotation, eyecite, preprocessing]
status: active
created: 2026-05-26
---

# Extraction Model Development

This document covers the development of our citation extraction system. **Extraction, preprocessing, and annotation are treated here as a unified development cycle, because in practice they are inseparable.**

---

## Overview


The extraction model operates on the **2nd layer** of our benchmark: preprocessed, text-based documents derived from raw court filings. Its job is to identify and classify all citation spans in a document, regardless of whether those citations are real or hallucinated.

Our extraction system is built on top of [`eyecite`](https://github.com/freelawproject/eyecite), a rule-based citation parser developed by the Free Law Project. `eyecite` is strong out of the box and serves as both our baseline and our pre-annotation engine. The extraction model's goal is to improve on `eyecite` where it falls short, without replacing what already works well.

---

## Preprocessing

Preprocessing converts raw documents into the text that the extraction model operates on. Cleaner preprocessing directly improves extraction performance — not because the extraction model got better, but because the input got better.

A common failure mode for `eyecite` is ill-formatted text produced by PDF parsing: an extra whitespace from OCR, a broken line from a multi-column layout, or a hyphenated word across a page break can prevent a citation from being recognized at all. These are upstream failures, not extraction failures.

See [Benchmark](./Benchmark.md) for the design decision that the 2nd layer benchmark is always re-derived from the current preprocessing pipeline.

---

## eyecite: Strengths and Failures

`eyecite` is a strong rule-based parser and our natural starting point. Its pre-annotations serve as a performance baseline: anything the extraction model produces should be at least as good as `eyecite` on clean text.

`eyecite` fails in two distinct ways:

### Failure Type 1 — Incorrect parsing of well-formatted spans

`eyecite` correctly identifies a citation span but parses one of its fields incorrectly.

> **Example**: `eyecite` identifies the plaintiff of `Methodist Hosp. of Sacramento v. Shalala, 38 F.3d 1225` as `Sacramento`, where it should be `Methodist Hosp. of Sacramento`.

This type of error is **agnostic of preprocessing quality** — the text is clean, but the rule-based parser misidentifies a field boundary. This is where we can add the most value: a fix here does not rely on the preprocessing assumptions we made for the 2nd layer benchmark, and a successful implementation would apply in broader contexts beyond our current dataset.


### Failure Type 2 — Non-recognition due to ill-formatted text

`eyecite` fails to recognize a span at all because the text around it is malformed (extra whitespace, broken tokens, OCR artifacts). Preprocessing may help reduce these failures, but applying a fix heuristically is risky — a cleaning step that helps some cases may break others. We need to develop a better understanding of this as we annotate more documents, rather than rushing preprocessing fixes.

---

## The Annotate-as-You-Go Workflow
**Instead of producing a static ground-truth benchmark (2nd layer), our development gradually improves the 2nd layer benchmark itself (by preprocessing development) as well as extraction model performance.**
We follow an iterative workflow:

1. **Pre-annotate** new documents with the current extraction model (`eyecite` + augmentations).
2. **Review** pre-annotations in Label Studio. For failures or missed spans, identify whether the root cause is preprocessing (Type 2) or parsing logic (Type 1).
3. **Fix the pipeline** — improve preprocessing or the extraction model based on what you find.
4. **Re-run** pre-annotation on the corrected pipeline.
5. Repeat until only genuine edge cases remain, then commit those as manual ground truth.

We do not commit manual annotations until the pipeline is stable. Any change to the preprocessing pipeline invalidates existing span annotations. 



---

## Canonical Citation Types

We recognize eight citation types. Each has a fixed set of bibliographic fields used for both annotation and serialization to the 1st layer (Existence Benchmark).

For the complete field definitions and annotation instructions for each type, see the annotation reference below.

### The `resolves_to` Relation

Legal writing rarely repeats a full citation every time. After introducing `Bush v. Gore, 531 U.S. 98 (2000)`, a brief will refer back using shorthand: `531 U.S. at 99`, `Bush, supra, at 99`, `Id. at 100`. These are **reference citations** — they carry meaning only because a full citation was established earlier.

**Resolution** connects each reference citation back to its antecedent full citation. In Label Studio, this is a directed `resolves_to` relation drawn from the reference region to the full citation it refers to.

Four types require a `resolves_to` link:

| Type | Typical form | Resolves to |
|---|---|---|
| `ShortCaseCitation` | `531 U.S. at 99` | The `FullCaseCitation` for that reporter |
| `SupraCitation` | `Bush, supra, at 99` | The `FullCaseCitation` (or `FullJournalCitation`) named |
| `IdCitation` | `Id. at 100` | The immediately preceding citation |
| `ReferenceCitation` | `Bush v. Gore` (bare name) | The `FullCaseCitation` for that case |

`eyecite` attempts automatic resolution and pre-populates these links. Verify them — `eyecite` can mis-resolve an `Id.` that crosses a paragraph break, or fail to resolve a `supra` whose antecedent uses an abbreviated party name.

---

## Citation Type Reference

### FullCaseCitation
A complete citation to a reported case.

**Example:** `Bush v. Gore, 531 U.S. 98, 99 (2000)`

| Field | Example | Notes |
|---|---|---|
| plaintiff | `Bush` | Party before "v." |
| defendant | `Gore` | Party after "v." |
| volume | `531` | |
| reporter | `U.S.` | |
| page | `98` | First page of the opinion |
| pin_cite | `99` | Specific page cited; omit if absent |
| extra | `aff'd, 123 F.3d 456` | Subsequent history; omit if absent |
| year | `2000` | From parenthetical |
| court | `scotus` | eyecite canonical court code; omit if absent |

### FullLawCitation
A citation to a statute, regulation, or code section.

**Example:** `42 U.S.C. § 1983(a)(1)`

| Field | Example | Notes |
|---|---|---|
| volume | `42` | Title or volume number |
| reporter | `U.S.C.` | Code abbreviation |
| page | `1983` | Section number |
| pin_cite | `(a)(1)` | Subsection(s); omit if absent |
| year | `2018` | From parenthetical if present; omit if absent |

### FullJournalCitation
A citation to a law review or journal article.

**Example:** `45 Harv. L. Rev. 123, 125 (2000)`

| Field | Example | Notes |
|---|---|---|
| volume | `45` | |
| reporter | `Harv. L. Rev.` | Journal abbreviation |
| page | `123` | First page of the article |
| pin_cite | `125` | Specific page cited; omit if absent |
| year | `2000` | From parenthetical |

### ShortCaseCitation
A subsequent reference using volume + reporter + "at" + pin cite.

**Example:** `Bush, 531 U.S. at 99`

| Field | Example | Notes |
|---|---|---|
| volume | `531` | |
| reporter | `U.S.` | |
| page | `98` | First page (same as full citation) |
| pin_cite | `99` | The "at ___" page |
| court | `scotus` | omit if absent |

**Relation:** `resolves_to` → the `FullCaseCitation` being referenced.

### SupraCitation
A reference using party name + "supra", optionally with a pin cite.

**Example:** `Bush, supra, at 99`

| Field | Example | Notes |
|---|---|---|
| pin_cite | `99` | Omit if absent |

**Relation:** `resolves_to` → the `FullCaseCitation` or `FullJournalCitation` being referenced.

### IdCitation
A reference using "Id." or "Ibid.", optionally with a pin cite.

**Example:** `Id. at 99`

| Field | Example | Notes |
|---|---|---|
| pin_cite | `99` | Omit if absent |

**Relation:** `resolves_to` → the immediately preceding citation.

### ReferenceCitation
A bare party-name reference with no reporter information.

**Example:** `Bush v. Gore` (mid-sentence, no volume/reporter/page)

| Field | Example | Notes |
|---|---|---|
| plaintiff | `Bush` | |
| defendant | `Gore` | |

**Relation:** `resolves_to` → the `FullCaseCitation` being referenced.

### UnknownCitation
A span that looks like a citation but cannot be parsed into any known type. No bibliographic fields — just label the span and use the Notes field to describe the issue.
