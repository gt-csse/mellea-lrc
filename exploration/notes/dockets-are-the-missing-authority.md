# Dockets are the missing authority kind

Written 1 September 2026, on `bench/locator-recall`, from the 26 documents of
`false-citation-bench` rendered as `locator-only-v2.0`.

Three separate measurements on this branch kept arriving at the same place, and
the place is not the tokenizer. It is that this project extracts reporter
citations and does not extract docket numbers, and a filing cites both.

## 1. What set this off

The citation tree attributes every return visit -- `Id.`, a short form, a party
name -- to the full citation that introduced the authority. It accounts for all
75 secondary citations in the corpus, but only 53 land on an authority:

| | bounded | full |
|---|---:|---:|
| secondary citations in the text | 75 | 75 |
| attributed to an authority | 53 | 54 |
| unattributed | 17 | 17 |
| out of scope, not a case | 5 | 4 |

The 17 do not move with the relaxation level, which was already known: the same
17 are unattributed at `NONE`, `BOUNDED` and `FULL`, citation for citation.
Whitespace was never what stranded them.

## 2. What the 17 are pointing at

Fifteen of them are in one document -- 016, a declaration -- and they chain
back to this:

```
See Indictment, United States v. Chen Zhi,
    No. 1:25-cr-00312-RPK (E.D.N.Y. filed Oct. 8, 2025).

21. The Indictment alleges that ... Id. ¶¶ 30-31.
```

Every `Id.` after it carries a paragraph pin cite into that indictment's
numbered allegations: `¶ 34`, `¶ 36`, `¶¶ 42-44`.

An earlier note guessed these were references into the filing's *own* numbered
allegations. They are not: they are paragraphs of another court document, cited
by docket number.

**The tree is right to leave them, and for a stronger reason than uncertainty.**
There is no case authority for them to attach to. eyecite produces no citation
for a docket number, so the chain has no head, and attaching paragraph 34 of an
indictment to a reporter page would be wrong even if a head could be found. A
pinpoint check on it would be verifying a claim nobody made.

The other two are unrelated: an `Id. ¶¶26-28` following a reference to the
opposing brief, and a short form quoted inside another case's parenthetical.

So against case-law authorities the tree is effectively complete -- 53 of 55 --
and the 70.7% figure is measuring a gap in docket support.

## 3. How much docket citation the corpus holds

The published bench records **11 docket occurrences in 4 documents**, every one
found by the model-assisted docket hunt. The source annotations cannot settle
this: they record 79 *defects*, not an inventory of citations.

Sweeping the text for the federal docket shape and separating a filing's own
number from the ones it cites:

| | documents | occurrences |
|---|---:|---:|
| the bench records a docket citation | **4** of 26 | 11 |
| cite another case's docket | **5** of 26 | **12** |
| hold a docket number at all | 21 of 26 | 47 |

The middle row is the answer to "how many documents contain docket citations":
**5 of 26**.

An earlier draft of this note said 8 of 26 and 15 occurrences, from a sweep
that separated a filing's own docket from the ones it cites by looking for
`Case No:` before the number. Three of those 15 are the filing's own caption,
written in forms that test does not match -- `Civil Action No: 2:25-cv-02623`
in 001, `Civil Action No. 2:25-cv-00804` in 004, `CaseNo. 2:25-cv-10681` in
014. The sweep is a proxy with errors in both directions and is not ground
truth; the extractor built for section 6 finds all 12 real ones and is.

The remaining gap between 5 and 21 is the filing's own case number, in the
caption and in every ECF page stamp. Document 020 alone carries twenty of them,
all identical, all reading `Case 2:25-cv-01295-GMS Document 1 Filed 04/18/25`,
and the corpus carries 35 in total. Nobody should count those as citations.

**Those stamps should not be in the text.** They are page furniture, and their
being there is the failure `preprocess/repeated-furniture` addresses: Docling
labels the running head `page_header` on some pages and `section_header` on
others, and the ones it mislabels survive `export_to_text`. That rule is
written and measured and is deliberately not wired in.

## 4. Why this matters more than it looks

**A docket number is the only identifier some filings give.** A case too recent
or too minor to be in a reporter is cited by docket, and that is exactly the
population where a fabricated citation is hardest to catch -- there is no
reporter page to check it against.

**Eleven docket occurrences were the reason every recall figure needed a
footnote.** They put a floor of 11 false negatives under every extraction arm
and capped recall at 98.1% for a reason unrelated to reading citations, which is
why the locator-only bench drops them. Dropping them makes the extraction
numbers honest; it does not make the dockets go away.

**Fifteen unattributed back-references are waiting on the same thing.** A tree
that resolved dockets as authorities would attribute all fifteen, and the
`Id. ¶ 34` chains would become checkable claims against a real document.

## 5. What already exists

`mellea_lrc.experimental.grounded_adjudication` has both halves:
`suspected_dockets` hunts docket-shaped strings, and `adjudicate_docket` asks a
model to confirm one, resolving the courts written nearby against courts-db and
offering them as a closed set so the model picks or declines rather than
inventing. The 11 records in the published bench came from exactly that path.

What is missing is the domain object. An adjudicated docket is not an
`ExtractedCitation`, so it cannot enter an `ExtractedDocument`, cannot be an
authority in the tree, and cannot be masked as found. That is why the
evaluation arms emit public occurrences directly rather than a serialized
document.

## 6. What was done about it

Built on `extraction/docket-locator`, stacked on this branch.

`src/mellea_lrc/extraction/dockets.py` registers a `TokenExtractor` that emits
an ordinary `CitationToken` with a synthetic edition, so eyecite builds the
citation itself and does the rest of its work unchanged -- case name in front,
pin cite and year behind, resource grouping, `Id.` resolution. The docket is
the locator, which is exactly the shape the rest of the pipeline already knows
how to carry.

A brand-new citation *class* turned out to be impossible without patching
eyecite: `get_citations` dispatches on the exact token type, so a new `Token`
subclass reaches `citation_tokens` and is then silently dropped. That is worth
knowing before anyone tries it again.

| | before | after |
|---|---:|---:|
| secondary citations attributed | 53 of 75 (70.7%) | **69 of 76 (90.8%)** |
| unattributed | 17 | **2** |
| bench docket records found | 0 of 11 | **11 of 11** |
| docket citations read | 0 | 12, in 5 documents |
| extracted from an ECF page stamp | — | **0 of 35** |
| locator score, `full` arm | 583 / 0 / 3 | unchanged, artifact byte-identical |

The two that remain unattributed are the two this note already accounted for:
an `Id. ¶¶26-28` pointing at the opposing brief, and `383 U.S. at 85` quoted
inside another case's citation, whose antecedent is in the quoted case and not
in the filing. Against case-law authorities the tree is now complete.

Proximity to a court proved insufficient as a rule, which section 4 of an
earlier draft implied it might be: document 022 stamps its own number 40
characters after another case's `(N.D. Cal. May 13, 2011)`. What works is
requiring the court in a bracket *after* the number with no paragraph break
between -- the same block-boundary rule the relaxation uses.

## 7. What is still open

1. **Make a docket a citation kind the core can hold.** Until then everything
   else is a workaround. It is a type-level change to `core.citations` and
   `ExtractedCitation`, not a model problem.
2. **Let the tree root an authority on a docket.** Fifteen of the seventeen
   unattributed back-references resolve the moment it can, and the tree's
   attribution rate stops being a measurement of the wrong thing.
3. **Wire in `repeated_furniture`, or decide not to.** Twenty of the 35 docket
   strings in this corpus are ECF stamps that should never have reached the
   text. Any docket work has to deal with them, and the rule that removes them
   is already written on `preprocess/repeated-furniture`.
4. **Re-measure the four unrecorded docket citations.** The hunt found 11 and
   the text holds 15; whether the other four are misses or correct rejections
   has not been read.

## 8. What this says about the relaxation work

Nothing, and that is the point worth recording. The unattributed count is
identical at every relaxation level, the docket census does not vary with it,
and no amount of whitespace tolerance produces a citation eyecite has no type
for. The remaining extraction failures on this corpus divide into three causes
and only one of them is the tokenizer:

- **the converter** -- tables reassembled out of order, ECF stamps left in the
  body, OCR dropping a period
- **the missing kind** -- dockets, which nothing extracts
- **the tokenizer** -- one citation, `214 F.3d 1058`, which `FULL` reads
  correctly once the margin is removed

Relaxation is close to finished here. The other two are not.
