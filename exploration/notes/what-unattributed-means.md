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
