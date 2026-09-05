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

## How it is fixed

Not by making the tokenizer case-insensitive. Setting `re.I` on every
string-bearing extractor moves all of them to the case-insensitive side of
`AhocorasickTokenizer`'s prefilter, which leaves the case-sensitive automaton
with no words in it, and pyahocorasick raises:

    AttributeError: Not an Aho-Corasick automaton yet: call add_word ...

and it would still need a precision measurement nobody has, because `id.` and
`supra` are matched by extractors too.

It is fixed one level up, in two steps that cost nothing per document:

1.  **The candidate generator scans a lower-cased copy.** `reporter_sites`
    searches an ASCII-lower-cased image of the masked document against
    lower-cased gazetteer spellings, so `33 F.4TH 693` is a site. The copy is
    length-preserving, so the offsets still index the original.
2.  **A re-read settles it with no model call.** `promotion.reread_site` reads
    the 480 characters around the site with eyecite's *unfiltered* tokenizer and
    every extractor set to `re.I`. There is no prefilter to leave empty, and the
    cost of the slow tokenizer is irrelevant on one window. It returns an
    ordinary `ExtractedCitation` -- court, date, party names and all -- because
    the real pipeline read it.

The precision worry that blocked the tokenizer fix is answered by scope: the
permissive read happens only where a generator has already found a reporter with
numbers around it, and its output is checked before it is kept. It was worth
checking: on the mined corpus the permissive read turned the footnote marker in
`9 Fed. R. Civ. P. 8(a)(2)` into volume 9, reporter `Fed. R.`, page `Civ` -- a
false citation with every field filled in. A page is printed with numbers, so a
recovered case citation whose page holds no digit is refused.

## What it recovers

    bench    6 sites settled with no call    (all `28 U.s.C.`-style statutes)
    mined    7 sites settled with no call

    33 F.4TH 693                  the table-of-authorities row above, twice
    2007 U.S. Dist. Lexis 40037   `Lexis` for `LEXIS`, twice
    416 U.s. 232                  a lower-case S inside `U.S.`
    279 Ga. App. At 807           a short form with a capitalised `At`
    403 F. Supp. At 1104          the same

Folding capitals is not free, and the cost is a rule of its own. `p.` differs
from the Pacific Reporter's `P.` in nothing but case, and one filing citing its
own exhibits by page (`[Doc. 40, p. 8]`) produced thirty sites; the same fold
turns every sentence-initial `Citing` into the gazetteer's lower-case `citing`.
Both are ruled out by one observation -- a reporter abbreviation is capitalised
-- so a site written differently from the spelling it matched must carry a
capital, and so must the spelling. That drops thirty-five sites and loses one:
`29 ny3d 425`, written entirely in lower case, appears once across both corpora.
