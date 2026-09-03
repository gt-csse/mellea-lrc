# The 9.3% is a scope problem, not an attribution problem

`build_citation_tree` reports 251 unattributed occurrences on the 77 mined
filings against 18 on the 26-document bench -- 9.3% against 2.7%. Read as an
attribution failure rate that says the association problem is three times worse
than the bench suggested. Read against the documents, it says something else.

## What the 251 are

    112   `Id.` with no paragraph or transcript pin cite
     88   `Id.` into a pleading paragraph          (`Id. ¶ 34`)
     27   short forms
     14   supra
      9   `Id.` into a deposition transcript       (`Id. at 23:5-10`)
      1   party-name reference

And what stands before each one:

    103   no citation of any kind within 1,200 characters
     50   no citation anywhere before it in the document
     70   a full case citation nearby
     26   an unparsed span nearby
      2   a statute nearby

**153 of 251 have nothing to attach to.** Not a chain that broke -- no case
citation within reach at all. Reading them shows what they refer to instead:

- `Id. LLMs are now powerful enough...` -- one filing runs 64 `Id.` off a report
  about generative AI. The antecedent is a secondary source, never a case.
- `( Id. at 3-4.)` -- 42 in a sanctions brief, referring to the court's own
  order and to counsel's filings.
- `Id. ¶ 34`, `Id. at 23:5-10` -- pleadings and deposition transcripts.
- `Supra Argument I.b.` -- the brief pointing at its own section.

None of these is a case, so none of them has an authority to belong to. They are
**out of scope**, and the tree files them under `unattributed` for a reason it
states plainly: only positive evidence sends a citation out of scope, and a bare
`Id.` carries none. That rule is right -- not knowing what something refers to is
not evidence that it refers to a statute -- but it means `unattributed` holds two
different things and only one of them is a failure.

## The failure rate, once the two are separated

    candidates for a real attribution failure    70 of 2,702    2.6%

which is the bench's 2.7%, on three times the documents. **Attribution
generalises as well as the rest of the extraction work.** What does not
generalise is the assumption that a filing's `Id.` chains point at case law: on
this corpus the majority point at the record, and three documents hold 150 of
the 251.

## What this changes

The model's job here is not the one previously written down. "Which authority
does this `Id.` mean?" is a smaller problem than it looked. "Is this reference to
a case at all, or to the record?" is the larger one, and it is a different
question -- classification rather than association, and answerable from the
sentence rather than from a closed set of authorities.

Two cheap signals exist and are worth taking before any model call, because they
are positive evidence of exactly the kind the tree asks for: a page-and-line pin
cite (`23:5-10`) is a transcript, and a paragraph pin cite in a document that
recites a pleading is an allegation. Together they account for 97 of the 251.
The paragraph signal alone is not sufficient -- some state courts number opinion
paragraphs -- which is why it needs the document's own subject matter, and that
is the part worth asking a model.

## The 2.6% is a proxy, not a ground truth

Worth saying plainly, because it is the kind of number that gets quoted.

The denominator is right in principle: an `Id.` pointing at a transcript is not
a citation this project checks, so it is out of scope and belongs outside the
denominator rather than inside the numerator. That is the same rule the bench
already applies to record references.

But 2.6% was not measured against ground truth. It counts occurrences with a
full case citation within 1,200 characters, which is a heuristic standing in for
"refers to a case", and it can be wrong both ways: an `Id.` in the deepfake
chain that happens to sit near a case is counted as a candidate failure, and one
whose real antecedent is a case cited further back than 1,200 characters is
excluded from the count entirely.

The rate we care about is

    attributions the annotation disagrees with
    ----------------------------------------------
    occurrences that refer to a case authority

and it can only be had from an annotated corpus.

## What ground truth actually says, where we have it

`false-citation-bench-tree-v2.0` is annotated, and there the denominator is
real: 255 in-scope returns, of which 61 are secondary -- 34 short forms, 15
`Id.`, 7 party-name references and 5 dockets.

    misattributions the annotation moved off the tree's answer    1 of 61

Document 022's `Id. at 1072-73`, filed under Advanced Textile where the tree
gave it Doe v. Commonwealth's Attorney. Seven further returns have no authority
recorded, but that is the bench's limit rather than the tree's: their authority
is identified by docket, and this bench's authority set is reporter-identified
cases.

**For the 77 mined filings there is no such number, and there cannot be one
without annotating them.** If the association rate is going to be claimed
anywhere, a sample of that corpus has to be read the way §5a of the citation-tree
handoff read the bench -- occurrence by occurrence, against the text.

## Bounds, since a point estimate is not available

The rate is

    occurrences the tree got wrong or could not attribute
    -----------------------------------------------------
    occurrences that refer to a case authority

On the mined corpus both parts are uncertain, but both are bounded. What is
counted:

    attributed occurrences                        2,486   (426 of them secondary)
    unattributed                                    252
      of those, carrying record evidence             98   (`¶ 34`, `at 23:5-10`)

**Lower bound.** Every unattributed occurrence is out of scope, and
misattribution among the attributed runs at the bench's rate of 1 in 61:

    7 / 2,486  =  0.3%

**Upper bound.** Every unattributed occurrence is in scope and is a failure:

    259 / 2,738  =  9.5%

**Narrowed upper bound.** A paragraph or page-and-line pin cite is positive
evidence of a record reference, so those 98 come out of the denominator:

    161 / 2,640  =  6.1%

So **0.3% to 6.1%**, and the honest reading is that the band is too wide to
support a claim in either direction. Two things widen it:

- 154 unattributed occurrences whose scope nobody has decided.
- The misattribution term rests on **one** observation. 1 of 61 has a 95% upper
  bound near 8.8%, which alone would push the top of the range to 7.2%.

## What would collapse the band

Reading the 154, not the 2,738. Deciding in-scope or out-of-scope for each is a
sentence-level judgement -- does this `Id.` point at a case or at the record --
and it removes almost all the width, leaving only the misattribution term.
Constraining that needs a sample of attributed returns read the same way; a
hundred would bring it from one observation to an interval worth quoting.

That is the whole annotation cost of turning a 20-fold band into a number, and
it is why the bound is worth stating rather than the proxy.

## The 154, read

Each was read against the text and asked one question: does this refer to a
court case? Recorded in `exploration/generality/scope_annotation.py`.

    out of scope     97
    in scope         49
    uncertain         8

**The out-of-scope 97 are the record, and they concentrate.** Two sanctions
proceedings account for 61 on their own -- `( Id. )` following `(Doc. 17.)`,
meaning counsel's response and its exhibits -- and one First Amendment brief
runs 28 `Id.` off expert declarations and the complaint. The rest are opposition
briefs cited as `Opp. 21`, a book the filing quotes, and `supra` pointing at the
filing's own argument sections.

**The in-scope 49 are real failures.** Short forms whose full citation is
elsewhere in the document (`833 F.2d at 186`, `33 F.4th at 695`), `Id.` chains
discussing a case (`Id. at *15`, `Id., 153 P.3d at 1240`), and one case
identified only by docket, which has no reporter authority to attach to.

**The 8 uncertain are kept separate rather than forced**, and they are one kind
of hard: a case citation quoted from inside another party's brief, where whether
the filing cites it or reproduces it is a judgement rather than a reading.

## The rate

    attributed occurrences                    2,487   (427 secondary)
    unattributed and in scope                    49 to 57
    unattributed and out of scope                97

    lower   (49 + 7 misattributed) / 2,536  =  2.2%
    upper   (57 + 38 misattributed) / 2,544  =  3.7%

**2.2% to 3.7%**, against the band of 0.3% to 6.1% before the reading, and
against the bench's 2.7%. Attribution generalises.

The width that remains is not the 8 uncertain readings -- they move the answer
by 0.3 points. It is the misattribution term, which still rests on the bench's
single observation, and which spans 7 to 38 of the same denominator. Narrowing
*that* is the annotation still worth doing: a hundred attributed returns read
the way these 154 were.
