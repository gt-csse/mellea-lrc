# A screening pass over the recorded fields, and what it is not

`evaluations/extraction/field_review.py` shows a model our recorded values and a
window of the document and asks whether the window states each value. Run on a
seeded sample of 40 citations per corpus with `gpt-5.6-luna`: 80 calls, none
unusable.

**It does not measure accuracy, and calling it that was wrong.** To judge whether
`plaintiff="Med. Progress"` matches the window, the model has to work out what
the plaintiff is -- so it is extracting, silently, and reporting whether it
agrees. The number below is agreement between eyecite and a model, and two
extractors agreeing proves neither right.

What it is worth is narrower and still worth having: **a screening pass that
locates candidate defects for a person to confirm**, which is the same
propose-then-review shape the adjudication layer uses. Five of the thirteen
disagreements were then read against the text by hand and were real.

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

The six "absent" verdicts were all `scotus`, and the screening pass was right --
but this one needed no model at all, and checking it deterministically gives the
real size of it:

    bench    458 citations carry a court   376 named in the parenthetical    81 not named
    mined   1636 citations carry a court  1337 named in the parenthetical   299 not named

**About 18% of recorded courts on each corpus are inferred rather than stated.**
`(2007)` names no court; `scotus` comes from the reporter being `U.S.`. It is a
sound inference, but the record does not say it is one, so a consumer cannot
tell a court the filing wrote from one the library concluded.

That matters here more than usual: a court *stated* is evidence about the
document, and a court *inferred* is evidence about the reporter. Extraction is
scored against the document, so the two should not share a field without a mark
saying which. `court_text` already exists on `DocketCitation` for exactly this
reason and `FullCaseCitation` has no equivalent.

## What this does not establish

Not accuracy, for the reason above. Not a party-name error rate either: 13
disagreements in 126 judgements is how often two extractors differ, and the five
verified by hand are a lower bound on how many of those are ours.

A field-level ground truth is still a separate piece of annotation. What this
run is good for is choosing where to spend it -- on party names, and nowhere
else, since 240 judgements on the locator and 129 on the date and pin cite
produced no disagreement at all.

**And the lesson for the next one: look for the deterministic version first.**
The only finding here that survived scrutiny was reachable by a regex over the
parenthetical, on all 2,094 citations rather than 80, with no model and no
sampling error.
