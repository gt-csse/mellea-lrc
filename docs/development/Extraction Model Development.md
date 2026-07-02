# Extraction Model Development

Extraction operates on the **2nd layer**: preprocessed text, in which it locates and classifies every citation span (real or hallucinated). It is built on [`eyecite`](https://github.com/freelawproject/eyecite), which is our baseline and pre-annotation engine; the goal is to improve on it where it falls short without replacing what works.

## Two eyecite failure modes

1. **Bad parse of a clean span** (Type 1) — the span is found but a field is wrong (e.g. plaintiff of `Methodist Hosp. of Sacramento v. Shalala` parsed as `Sacramento`). Preprocessing-agnostic; this is where we add the most durable value.
2. **Non-recognition from ill-formatted text** (Type 2) — OCR artifacts, broken lines, hyphenation across page breaks stop a span from being recognized at all. Upstream problem; fixed in preprocessing, cautiously (a clean-up that helps one case can break another).


## Canonical citation types

Eight types; fields below are used for annotation and for serialization to the 1st layer. Reference citations carry meaning only via a `resolves_to` link to the full citation they refer back to (`eyecite` pre-populates these — verify them, especially `Id.` across paragraph breaks and `supra` with abbreviated party names).

| Type | Example | Key fields | `resolves_to` |
|---|---|---|---|
| `FullCaseCitation` | `Bush v. Gore, 531 U.S. 98, 99 (2000)` | plaintiff, defendant, volume, reporter, page, pin_cite, extra, year, court | — |
| `FullLawCitation` | `42 U.S.C. § 1983(a)(1)` | volume, reporter, page (section), pin_cite, year | — |
| `FullJournalCitation` | `45 Harv. L. Rev. 123, 125 (2000)` | volume, reporter, page, pin_cite, year | — |
| `ShortCaseCitation` | `Bush, 531 U.S. at 99` | volume, reporter, page, pin_cite, court | full case for that reporter |
| `SupraCitation` | `Bush, supra, at 99` | pin_cite | the named full case/journal |
| `IdCitation` | `Id. at 99` | pin_cite | immediately preceding citation |
| `ReferenceCitation` | `Bush v. Gore` (bare name) | plaintiff, defendant | full case for that name |
| `UnknownCitation` | — | none (label span + note) | — |

Field notes: `pin_cite`/`extra`/`year`/`court` are omitted when absent. `court`
is eyecite's canonical code. When eyecite recognizes a reporter but omits the
court, assessment may apply the extraction-level fallback documented in
[Reporter-to-Court Inference](../knowledge/Reporter%20Court%20Inference.md).
Inference is allowed only for reporters that cover one court exclusively; it
does not turn missing CourtListener data into a match or mismatch.
