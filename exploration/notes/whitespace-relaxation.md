# One relaxation, three levels, and what the widest one costs

## What was here before

Extraction had three tokenizer behaviours spread across two modules: the
shipping extractor (eyecite plus a whitespace repair), an experimental module
relaxing every join to `\s*`, and a variant of that module bounding the
reporter-to-page join at a blank line. Choosing between them meant choosing an
import, and no two of them shared a code path, so a comparison between any two
compared more than the tokenizer.

They are now one substitution behind one parameter.

## The levels

| level | volume → reporter | reporter → page | punctuation inside the reporter |
|---|---|---|---|
| `NONE` | literal space | literal space | as eyecite generates it |
| `BOUNDED` (default) | any whitespace | any, stopping at a blank line | relaxed |
| `FULL` | any whitespace | any whitespace | relaxed |

The two joins are bounded differently at `BOUNDED`, and that asymmetry is the
level.

Between volume and reporter, a break leaves reporter and page still adjacent on
the far side, so the page captured is the citation's own. Blank lines are
always safe there and are needed: `937\n\nS.W.2d 796` is a real citation split
by a page break.

Between reporter and page, the page number is what lands beyond the break. On
pleading paper that is where the margin line numbers are, so
`214 F.3d\n\n1\n\n2` reads as page 1 when the citation is `214 F.3d 1058` — a
wrong page rather than a missing one, which sends validation to a different
case and returns a confident verdict about it.

The reporter-punctuation relaxation is a separate thing from the joins. eyecite
already allows whitespace *after* a period inside a reporter, so `N.Y. 2d`
matches; it allows none before one and none around an apostrophe, so `N.Y .2d`
and `F. App ' x` match nothing. Both occur in real extraction.

## The whitespace repair is gone

The previous pipeline collapsed runs of repeated inline whitespace before
extraction and remapped every span afterwards. Measured over 103 documents and
2,603 citations, with the joins and the reporter punctuation relaxed:

| | citations |
|---|---|
| found either way | 2,603 |
| found **only** with the collapse | **0** |
| found only without it | 0 |

It recovered nothing the relaxed patterns do not already reach. Removing it
buys three things: the text is never rewritten, so no span is remapped and
`matched_text` is the source text as written; there is one mechanism able to
find a citation rather than two, which matters when the tokenizer is the thing
under test; and relaxing the pattern also reaches `846F.2d746`, the opposite
defect, which no collapse could.

## What BOUNDED is worth

On the published bench, `production` (eyecite + `BOUNDED`, no model):

| | predicted | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|
| eyecite as published | 526 | 0 | 68 | 100.0% | 88.6% |
| previously (+ whitespace repair) | 563 | 0 | 31 | 100.0% | 94.8% |
| now (+ `BOUNDED`) | 582 | 0 | 12 | 100.0% | 98.0% |

Eleven of the twelve remaining misses are docket numbers, which eyecite does
not attempt at all — the benchmark's own floor for any such system is 11 false
negatives and a 98.1% ceiling. The twelfth is `455 US. 363`, a reporter written
without the period after `US`, which is not a whitespace defect.

## What FULL costs

On a mined corpus of real filings, widening the reporter-to-page join changed
the parse in 6 of 103 documents: **two correct recoveries and four errors.**

| document | what changes | |
|---|---|---|
| `69912445_21` | gains `214 F.3d 1058` | correct |
| `71920595_40` | `, 487 U.S.⏎⏎317.` gains `487 U.S. 317` | correct |
| `test_data/3` | `214 F.3d` + gutter 1–28 + `1058` → page `1` | wrong |
| `70607460_15` | `206 P. 327` → `206 P.3 27` | wrong |
| `72050145_17` | `607 F.3d 355` → `130 S.Ct. 607` | wrong |
| `69912445_49` | `Fed. R. Civ. P. 11(b)(2)` → `1 Fed.R. 1` | wrong |

**Two of the four destroy a citation that parsed correctly before**, which is
worse than adding a bad one: a lost citation reports nothing to check, the
failure mode the whole relaxation exists to remove.

Only the first error involves a page margin. The other three occur in text with
no margin in it, so removing page furniture upstream does not make `FULL` safe.

## What would make FULL defensible

One rule: refuse to cross a blank line when what follows already parses as its
own citation. That covers `607 F.3d 355` → `130 S.Ct. 607` directly and
possibly the digit-eating case too. If it holds, the level collapses back into
one and there is no choice left to make.

## Relationship to the upstream PR

freelawproject/eyecite#339 relaxes both joins to `\s*` in eyecite itself —
`FULL`, unconditionally — and separately fixes adjacent citations being skipped
(`347 U.S. 483,349 U.S. 294` loses the second), which this does **not** address
and which is a real bug.

If #339 merges, `FULL` has no reason to exist here. `BOUNDED` still would: the
measurement above says the reporter-to-page join wants a bound that #339 does
not have. The reporter-punctuation relaxation is not in that PR either.

Both are worth contributing to the discussion, because a wrong page is not a
lost citation, and a regression check that counts citations cannot see the
difference — two of the four errors above leave the count unchanged.
