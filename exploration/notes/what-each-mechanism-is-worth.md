# What each mechanism is worth

`evaluations/extraction/matrix.py` turns off one thing at a time and reports the
same columns, so a mechanism's value is the difference between its row and the
one above it.

Two things make it a reliability table rather than a volume one.

**Half the columns count defects.** A configuration that reads more citations
while carrying more wrong years is worse, and the table says so rather than
hiding it behind a recall number. `pin cite lost` is a page eyecite filed under
`extra`; `date taken from another case` is a span reaching over an unrelated
citation for its court and year; `court written, not recorded` is a
parenthetical naming a court that the citation does not carry.

**Recall is scored against ground truth only where it exists.** The 586 locators
of `false-citation-bench-locator-only-v2.0` are annotated and inclusive -- a
locator the filing states counts whether or not any tokenizer reaches it. The 77
mined filings have no annotation, so their columns are counts and defect counts,
never recall.

Site hunting is deliberately absent: it proposes 185 candidates on the mined
corpus to reach at most 2 real citations, and its cost belongs to a reviewer
rather than to a measurement.

## The result

    false-citation-bench, 586 locators annotated

    arm                          found  spurious  dockets  pin  pin lost  date  date wrong  court  court missed
    eyecite as published           542         0        0  303        44   483          18    391            51
    + reporter joins (bounded)     582         0        0  310        68   517          20    421            51
    + reporter joins (full)        583         0        0  310        68   518          20    422            51
    + pin cites                    583         0        0  386         1   518          21    423            50
    + court and date boundary      583         0        0  386         1   511           0    422            50
    + court resolution             583         0        0  386         1   511           0    458            14
    + docket locators              583         0       12  386         1   511           0    458            14

    mined corpus, 77 unseen filings, no ground truth

    arm                          cites  dockets  pin  pin lost  date  date wrong  court  court missed
    eyecite as published          1847        0 1123        74  1712         110   1450           107
    + reporter joins (bounded)    2021        0 1140       158  1861         133   1560           130
    + reporter joins (full)       2024        0 1140       158  1864         134   1562           130
    + pin cites                   2024        0 1318         9  1864         139   1590           104
    + court and date boundary     2024        0 1318         9  1840           0   1580           105
    + court resolution            2024        0 1318         9  1636*         0   1636            49
    + docket locators             2024       36 1318         9  1840           0   1636            49

Read down the defect columns. **Relaxation buys citations and costs defects**:
40 more locators on the bench, and pin cites lost rising 44 to 68 because there
are more citations to lose one from. Every mechanism after it removes a defect
class without giving back a finding -- pin cites lost 68 to 1, dates taken from
another case 21 to 0, courts written but unrecorded 50 to 14.

**The composite is the only row where every defect column is at its minimum**,
and it is the only row where that is true on the unseen corpus as well. That is
the claim the matrix supports, and it is a narrower claim than "most reliable":
it says no mechanism here is paid for by another's regression.

Two rows are worth reading carefully. The `+ pin cites` row raises `date wrong`
by one on the bench and five on the mined corpus -- reading a pin cite extends a
citation's span, so a span that already reached too far now provably does. The
boundary row then takes all of them, which is why the two belong in that order.
And the docket row moves nothing except its own column, because a docket
citation is not a `FullCaseCitation`: its value is 12 authorities on the bench
and 36 on the mined corpus that were previously not citations at all.

## What the matrix cannot answer

Whether a field that *is* recorded is **right**. Every column here is presence,
absence, or a defect a rule can detect. Nothing checks that the year a citation
carries is the year of that case, except where a span crossing proves it cannot
be. Establishing that needs either an annotated field-level ground truth, which
does not exist for either corpus, or a reader sampling the parsed fields against
the text.

That reader is the obvious use for a model here, and it is cheap -- a hundred
sampled citations, one question each. **It is not run: no LLM credentials are
configured in this environment** (`MELLEA_LRC_LLM_MODEL`, `_API_BASE`, `_API_KEY`
are unset and there is no `.env`). The matrix stands on deterministic evidence
alone, and the field-accuracy row is left empty rather than estimated.

## A trap worth recording, because it caught this file

The first run reported zero for `date taken from another case` in every row,
including the rows where the boundary was switched off. The switch was patching
`post_citation.reread_post_citation`, but `stages` imports that name by value,
so the pipeline kept calling the original and the column measured nothing.

The same by-value import trap has now appeared three times in this project:
`POST_FULL_CITATION_REGEX` baked into `helpers`, `tokenizer_for` imported into
the extractor, and here. **Patch a name where it is looked up, not where it is
defined**, and treat a row of zeros in a differential measurement as a bug in
the measurement until proven otherwise.
