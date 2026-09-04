# What a reader says about the recorded fields

The one column `matrix.py` cannot fill. Every metric there is presence, absence,
or a defect a rule can detect from the document's own structure; none of them
checks that a field which *is* recorded matches the text it was read from, and
nothing deterministic can, because neither corpus has field-level annotation.

`evaluations/extraction/field_review.py` asks a reader instead, on a seeded
sample of 40 citations per corpus, with `gpt-5.6-luna`. 80 calls, none unusable.

## The question, and the line it does not cross

**Does the document state this value for this citation?** The instruction says
explicitly not to judge whether the citation is accurate, whether the case
exists, or whether the year is really that decision's year -- a filing that
writes `550 U.S. 544 (2009)` states 2009, and recording 2009 is a correct read.

## The result

                bench                        mined
    field      match  differ  absent    match  differ  absent
    volume        40       0       0       40       0       0
    reporter      40       0       0       40       0       0
    page          40       0       0       40       0       0
    pin_cite      29       0       0       26       0       0
    date          36       0       0       38       0       0
    court         26       0       2       27       0       3
    plaintiff     25       3       0       26       5       0
    defendant     34       1       0       32       4       1

**The locator, the date and the pin cite are exact.** 240 judgements on volume,
reporter and page with no disagreement; 74 on the date and 55 on the pin cite
with none either. Whatever else is wrong with this extractor, it reads the
identifier and the year off the page correctly.

**Party names are where it fails**, at 13 disagreements in about 126 judgements.
Reading them, the reader is right wherever it can be checked:

- `Country Club  Johnston Cnty., Inc.` -- the filing writes "Country Club **of**
  Johnston Cnty., Inc."
- `St. Amant` recorded as the *defendant* of `390 U.S. 727`, which is
  St. Amant v. Thompson: the plaintiff, in the wrong field, with Thompson gone.
- `Med. Progress` for `890 F.3d 828`, truncated from "Ctr. for Med. Progress".
- `9th Cir. 1999) ......... | 2, 9, 11, 12 |` recorded as a defendant, which is a
  table-of-authorities row read as a party name.

That is the 17% incompleteness measured deterministically, seen from the other
side and with the silent swaps included.

## The finding worth acting on: `court` mixes two things

Every one of the six "absent" verdicts is `scotus`, and every one is **correct**.
`(2007)` states no court. `scotus` is inferred from the reporter being `U.S.`,
by eyecite, and it is a sound inference -- but the record does not say it is an
inference, so a consumer cannot tell a court the filing wrote from one the
library concluded.

That matters here more than usual: a court *stated* is evidence about the
document, and a court *inferred* is evidence about the reporter. Extraction is
scored against the document, so the two should not share a field without a mark
saying which. `court_text` already exists on `DocketCitation` for exactly this
reason and `FullCaseCitation` has no equivalent.

## What this does not establish

The reader is one fallible annotator, not ground truth. Its party verdicts were
spot-checked against the text by hand and held, but only five of the thirteen.
A field-level ground truth is still a separate piece of annotation, and this
review is evidence about where to spend it -- on party names, and nowhere else.
