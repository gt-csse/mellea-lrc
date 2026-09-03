# A reporter set in capitals is not a reporter

## The gap

eyecite registers one extractor per reporter string, and the reporter extractors
are **case-sensitive**::

    strings: ['F.4th']   case-insensitive: False

So the same citation is found or missed depending on how the converter rendered
its letters:

    Doe v. Roe, 33 F.4th 693 (2d Cir. 2022).    ->  ['33 F.4th 693']
    Doe v. Roe, 33 F.4TH 693 (2D CIR. 2022).    ->  []
    Doe v. Roe, 33 f.4th 693 (2d Cir. 2022).    ->  []
    Doe v. Roe, 833 F.2D 183 (9th Cir. 1987).   ->  []

A reporter that is already all capitals is unaffected, because there is no case
to get wrong -- `DOE V. ROE, 550 U.S. 544 (2007)` extracts normally, and so do
`F.R.D.` and `B.R.` The exposure is exactly the reporters whose abbreviation
contains lower-case letters: the ordinal series `F.2d`, `F.3d`, `F.4th`, `A.2d`,
`S.E.2d`, `N.Y.2d`, and the spelled words in `F. Supp.`, `S. Ct.`, `N.C. App.`

That is most of them. Of the 80 canonical reporters these corpora use, **67
contain lower-case letters**.

## What it costs here

    bench    0 of 26 documents
    mined    2 of 77 documents

and the two are the same table-of-authorities row in two near-identical filings::

    Dalla-Longa v. Magnetar Capital LLC , 33 F.4TH 693 (2D CIR. 2022)

It matters more than two citations suggests, because the row is a full citation
and the argument later writes `Dalla-Longa, 33 F.4th at 695`. Losing the full
form leaves the short form with nothing to resolve to, so one missed reporter
costs an attribution as well as a citation.

## Why it is not fixed

Two occurrences in 103 documents is below the line for changing how the
tokenizer matches, and the obvious fix is not as cheap as it looks.

Setting `re.I` on every string-bearing extractor moves all of them to the
case-insensitive side of `AhocorasickTokenizer`'s prefilter, which leaves the
case-sensitive automaton with no words in it, and pyahocorasick raises::

    AttributeError: Not an Aho-Corasick automaton yet: call add_word ...

So a fix has to handle an empty filter, and would then need measuring for what
case-insensitivity costs in precision -- `id.` and `supra` are matched by
extractors too. None of that is justified by one table row.

**Recorded so it is known rather than fixed.** On a corpus whose filings set
their tables of authorities in capitals -- which is a house style, not an
accident -- this would stop being a footnote.
