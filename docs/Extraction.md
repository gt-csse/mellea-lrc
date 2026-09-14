---
tags: [extraction, eyecite, citations, spans]
status: active
---

# Extraction

Extraction is the middle stage. It takes the plain text a document was
preprocessed into, finds every citation in it, and returns each one as a typed
object with an offset back into that text. It decides nothing about whether a
citation is *real* — a fabricated case and a genuine one are extracted alike.
Judging them is [validation](./Validation.md)'s job.

This document walks through the whole stage: the preprocessing that feeds it,
how to run it, what comes back, what the engine is, where it fails, and what is
being built to reach the citations it misses.

---

## Preprocessing: where the text comes from

Extraction never sees a PDF. It reads the plain text a document was turned into
first, and the quality of that conversion sets a ceiling on everything after it.
A citation broken by the converter cannot be found by any parser downstream, so
a large share of what looks like extraction failure originates here.
[Preprocessing](./Preprocessing.md) documents that stage in full; what follows
is what extraction depends on.

`preprocess(path)` picks a backend from the file's suffix:

| suffix | backend | what happens |
|---|---|---|
| `.txt` | `plain_text` | read as-is, no conversion |
| `.pdf` `.docx` `.pptx` `.xlsx` `.html` `.htm` `.md` | `docling` | converted with [Docling](https://github.com/docling-project/docling) |
| anything else | — | `ValueError` |

Docling is an optional dependency: `uv sync --group preprocessing`. Importing
the backend without it raises with that instruction rather than failing
obscurely.

The result is a `PreprocessedDocument`:

| field | what it is |
|---|---|
| `text` | the converted text; every later offset indexes this, never the original file |
| `source_metadata` | original path and `SourceFormat` |
| `preprocessing_metadata` | which backend ran, and its version |

`text` may not be empty — a conversion that produced nothing raises rather than
handing an empty document downstream.

### A text file is its text

A `.txt` file is read whole. Nothing is split off the front, so an offset into
the file and an offset into the document are the same number, and provenance
lives beside the text rather than inside it.
`preprocess_plain_text_from_string` does the same for text already in memory.

### What the converter does to citations

Two artefacts account for most of the damage, and they are not equally
recoverable:

- **Repeated spaces**, left behind when justified text is flattened. The
  shipping pipeline reads straight through them — see below.
- **Line and page breaks falling inside a citation**, from column layouts and
  page boundaries. A line break is read through as well; a *blank* line is not,
  because crossing one cannot be done without also creating false matches. That
  boundary is the subject of the level table below.

---

## Running extraction

Three entrypoints, differing only in where the text comes from:

```python
from pathlib import Path
from mellea_lrc.extraction import extract

document = extract("See Brown v. Board of Education, 347 U.S. 483, 495 (1954).")
document = extract(Path("filing.pdf"))
```

`extract` dispatches on the argument's type — **a `str` is content, a `Path` is
a location** — and hands off to one of the two explicit forms,
`extract_from_plain_text(text)` and `extract_from_raw_document(path)`. Use those
directly when the distinction matters; a filename that arrives as a string would
otherwise be extracted *from*, rather than opened.

All three take a `relaxation` keyword and nothing else that changes what is
found. There is one extractor, not a production one and a relaxed one.

There is deliberately no entrypoint taking a `PreprocessedDocument`. Nothing
serializes one, so it cannot cross a process boundary, and a caller holding one
is already inside the library.

Extraction is offline and deterministic. It needs no credentials and makes no
network calls, which is why the `mellea-lrc` command does not expose it on its
own — it is a step of `validate`, not a thing to run.

---

## What comes back

An `ExtractedDocument`, which is the `PreprocessedDocument` it was built from
plus the citations found in it:

| field | what it is |
|---|---|
| `text` | the preprocessed text every offset indexes into |
| `citations` | one `ExtractedCitation` per occurrence, in document order |
| `extraction_metadata` | which backend ran, and its version |

Each `ExtractedCitation` carries:

| field | what it is |
|---|---|
| `citation_id` | stable identifier for this occurrence |
| `span` | the citation's full extent, including party names and parenthetical |
| `locator_span` | just the part that identifies the authority |
| `matched_text` | the source text under `span` |
| `citation` | the typed object — one of the eight kinds below |
| `resolves_to` | for a back-reference, the `citation_id` it points at |

### The artifact, which is what validation reads

`serialize_extracted_document` writes the whole of an `ExtractedDocument` as
JSON and `deserialize_extracted_document` reads it back. That artifact is the
**input to the identity stage**, not a by-product of an evaluation: extraction
is offline and deterministic, validation is neither, and the boundary between
them is a file.

It is versioned. `schema_version` is **9**, and a reader refuses a payload that
does not say so rather than guessing at a field it does not recognise.

What a citation carries there, beyond its spans and its parse:

| field | what it holds |
|---|---|
| `case_name` | the whole name: `span`, `text` as the page holds it, `plaintiff`, `defendant` |
| `case_name_span` | the same position alone, for a reader that wants only that |
| `field_log` | every touch on a logged field, in order. The last is the value |
| `pin_cite` / `pin_cite_pages` | the written form, and the pages it claims |
| `root_id`, `colocation_id` | which identifier a return inherits, and what was written beside it |

Three things changed at 9, all of them because a reading can now be better than
the parse it started from. `case_name` used to be a bare span. `field_log` did
not exist, so a name a reader repaired had nowhere to go that did not erase what
it replaced. And a pin cite's pages now carry `footnote` where the citation
names one, and `note` where nothing could be read — the kind formerly called
`nonconforming` is `unread`, which says what it means: **this reader** could not
turn the written form into pages, and the filing may be perfectly proper.

### `span` and `locator_span` are not the same

For `Brown v. Board of Education, 347 U.S. 483, 495 (1954)`:

- `span` covers the whole thing, party names through the year parenthetical;
- `locator_span` covers `347 U.S. 483` alone.

The locator is what reaches the case in a reporter-indexed database, so it is
what lookup and evaluation both key on. The full span is what you highlight in a
document. Reporting the right locator with a slightly different full span is not
an error; reporting the right span with a misread locator is.

---

## The eight citation kinds

Every citation becomes exactly one of these. Fields not present in the source
are `None`; every kind also carries an optional `parenthetical`.

### Full citations

These state an authority completely and stand on their own.

**`FullCaseCitation`** — a reported case. `Bush v. Gore, 531 U.S. 98, 99 (2000)`

| field | example | notes |
|---|---|---|
| `plaintiff` | `Bush` | party before "v." |
| `defendant` | `Gore` | party after "v." |
| `volume` | `531` | |
| `reporter` | `U.S.` | |
| `page` | `98` | first page of the opinion |
| `pin_cite` | `99` | the specific page cited |
| `extra` | `aff'd, 123 F.3d 456` | subsequent history |
| `year` | `2000` | |
| `court` | `scotus` | eyecite's canonical court code |

**`FullLawCitation`** — a statute, regulation, or code section.
`42 U.S.C. § 1983(a)(1)`

`volume` is the title (`42`), `reporter` the code (`U.S.C.`), `page` the section
(`1983`), `pin_cite` the subsection. Also carries `year` and `publisher`.

**`FullJournalCitation`** — a law review or journal article.
`45 Harv. L. Rev. 123, 125 (2000)`

Fields are `volume`, `reporter`, `page`, `pin_cite`, `year`.

### Back-references

Legal writing states a citation in full once, then refers back to it. These
kinds carry a `resolves_to` pointing at the full citation they depend on, and
mean nothing without it.

| kind | form | fields | resolves to |
|---|---|---|---|
| `ShortCaseCitation` | `531 U.S. at 99` | `volume`, `reporter`, `page`, `pin_cite`, `court` | the `FullCaseCitation` for that reporter |
| `SupraCitation` | `Bush, supra, at 99` | `pin_cite` | the full citation named |
| `IdCitation` | `Id. at 100` | `pin_cite` | the immediately preceding citation |
| `ReferenceCitation` | `Bush v. Gore` | `plaintiff`, `defendant` | the `FullCaseCitation` for that case |

eyecite resolves these itself and the links come through pre-populated. They are
worth checking: an `Id.` crossing a paragraph break, or a `supra` whose
antecedent uses an abbreviated party name, are both places resolution goes
wrong.

### `UnknownCitation`

A span that reads as a citation but parses into no kind. It carries no fields —
only the span survives. Treat it as a signal about the text, not as a citation.

---

## The engine

### eyecite

Extraction is built on [eyecite](https://github.com/freelawproject/eyecite), the
Free Law Project's rule-based parser. It is strong out of the box and is both
the baseline and the working engine. The goal has never been to replace it, only
to reach what it leaves behind.

eyecite matches a gazetteer of literal reporter strings — some 4,800 spellings
drawn from `reporters-db` — and then runs the generated regex for whichever
strings appear. This is fast and precise, and it is also the source of both
failure modes below.

### Separator relaxation

The one addition in the shipping pipeline, and the only knob extraction has.
eyecite's generated patterns join volume, reporter and page with a **literal
single space**, so one doubled space makes a citation vanish outright rather
than parse imperfectly — and doubled spaces are exactly what PDF extraction of
justified text leaves behind.

Rather than repair the text, the patterns are rebuilt so those joins match
whatever whitespace is actually there. The text is never rewritten and no span
is ever remapped: every offset indexes the document as it was given.

`Relaxation` names three levels, and every entrypoint takes it:

| level | volume → reporter | reporter → page | punctuation inside the reporter |
|---|---|---|---|
| `NONE` | literal space | literal space | as eyecite generates it |
| `BOUNDED` **(default)** | any whitespace | any, stopping at a blank line | relaxed |
| `FULL` | any whitespace | any whitespace | relaxed |

```python
from mellea_lrc.extraction import Relaxation, extract

document = extract(text, relaxation=Relaxation.FULL)
```

The two joins are treated differently at `BOUNDED`, which is the point of the
level. A break between volume and reporter leaves reporter and page still
adjacent on the far side, so the page captured is the citation's own; blank
lines are always safe there. A break between reporter and page puts the page
number beyond the break — which on pleading paper is where the margin line
numbers are, so `214 F.3d\n\n1\n\n2` reads as page 1 when the citation is
`214 F.3d 1058`. That is a wrong page, not a missing one, and it sends
validation to a different case.

`FULL` takes that risk deliberately. Measured over 103 documents and 2,603
citations, widening that one join changed the parse in six of them: two correct
recoveries and four errors, two of which destroyed a citation that had parsed
correctly before. Only one of the four involved a page margin, so removing the
margin upstream would not make `FULL` safe.

Relaxing the pattern also covers the opposite defect, which repairing the text
never could: `846F.2d746`, written without separators at all by OCR or by the
filer's own word processor.

Which level ran is recorded on the result, in
`extraction_metadata.relaxation`. Two levels disagree about whether a given
citation is in the document at all, so a result that does not say which one
produced it cannot be compared with another.

---

## Where extraction still fails

Two distinct modes, worth separating because they call for different fixes.

**Mis-parsed fields on clean text.** The span is found, but a field boundary is
read wrongly. eyecite parses the plaintiff of
`Methodist Hosp. of Sacramento v. Shalala, 38 F.3d 1225` as `Sacramento`. The
text is not damaged; the rule is. Fixes here transfer to any corpus.

**Non-recognition of damaged text.** The citation is never seen at all, because
a page break or an OCR artefact falls inside it in a way no level will cross.
This is an upstream failure surfacing as an extraction failure, and it is what
the experimental work below targets.

A practical consequence: **do not let a blank line fall between a reporter and
its page.** That is the one break the default will not join, and joining it is
not free — see the level table above.

---

## Reaching the rest: experimental work

Both live in `mellea_lrc.experimental` and neither is in the shipping pipeline.

### Grounded adjudication

The sounder approach, and the one to build on. Four steps:

1. **Mask** everything already extracted, leaving only the text that failed.
2. **Hunt** that residue for any gazetteer string with digits close on both
   sides — the volume-and-page shape.
3. **Adjudicate** each candidate site with a model, one at a time, asking only
   whether the characters at *this* position state a complete identifier.
4. **Ground** the answer by locating the model's verbatim quote back in the
   document. A quote that cannot be found is dropped.

The point of hunting is not speed, it is the shape of the question. Without it
the model is asked "what citations are in this document?" — open, generative,
and unconstrained by what the text actually says. With it the question is closed
and checkable: the position is known before the model speaks, so every answer
can be verified against the document and discarded if it does not ground.

The narrowing is steep. Hunting searches roughly 4,795 gazetteer strings; after
masking, **27 of them occur anywhere in the 26-filing corpus, at 88 positions**.
The model sees 88 short windows instead of 26 whole documents. Because the hunt
knows which string flagged each site, the prompt can be specific to that
reporter.

Two things the numbers do not say. The gazetteer is not a list of case
reporters: `U.S.C.` alone accounts for 20 of those 88 sites and is a statute
code, and several others are obscure state reporters whose abbreviations collide
with ordinary words. And the hunt is deliberately over-permissive — a judge that
rejects freely costs far less than a citation never surfaced.

### What a reader is allowed to establish, and what it is not

The honest statement of the gap: a reader could in principle establish anything
the rules missed — a root nothing parsed, a member of a colocation, a short form
that states a page — and the case-name layer establishes two things. It writes a
name onto a citation the rules already read, and it records a bare name as a
reference to a root. Nothing else it answers becomes a row.

That is narrower than it sounds, because of where the residue actually is. Over
the corpus, the rules alone, by kind:

| kind | read |
|---|---|
| `DocketCitation` | 42 / 42 |
| `ShortCaseCitation` | 41 / 41 |
| `IdCitation` | 32 / 32 |
| `FullCaseCitation` | 589 / 590 |
| `ReferenceCitation` | **6 / 19** |

Every kind that states an identifier is read. The one kind that states none —
a case name and nothing else — is six of nineteen, and eyecite reaches those six
only because it had the root's party names to match on. The thirteen it misses
are the ones whose root was read with no name, which is the same cause as the
false defects above. **So the likely finding is a bare name that is a reference
citation, and that is what the layer is built to establish.**

Three things it cannot, each with an instance on the held-out set:

1.  **A root.** `In re Barteca Restaurants, LLC , Ser. Nos. 85202482 & 85202583
    (T.T.A.B. Feb. 1, 2013)` is a decision cited by application serial number
    and extraction reads nothing at it. A case-name site cannot establish it: a
    name states no identifier, and what is missing here is the identifier. That
    is the locator reviewer's shape — `adjudicate_locator` and
    `promote_locator`, which exist and are not in the pipeline.
2.  **A colocation.** Parallel citations are grouped by a rule in
    `structure/colocation.py` and nothing reviews the grouping, so a position
    the rule splits or joins wrongly stays that way.
3.  **A short form that claims a page.** When the layer accepts a `short_form`
    it records a `ReferenceCitation` with no pin cite, because a bare name has
    none. A name whose page the rules failed to read — `Caraway , at 1301` with
    the `at 1301` damaged — comes back as a reference with the page claim
    flattened out of it, and the page claim is the thing validation checks.

### The record: what the rules read, what it is now, and the evidence between

A case name is not read once. The rules read it from the offsets eyecite gives;
a reader asked about a name standing outside every citation may then say the
name belongs to *this* citation and be right where the rules were not. Keeping
only the second answer loses which one a measurement is measuring; keeping only
the first throws the reader's work away.

So the citation the pipeline holds is not the one extraction produced, and
:class:`~mellea_lrc.core.record.CitationRecord` is where that is written down:

```
CitationRecord
├── citation_id                  never changes
├── source:  Citation            frozen. exactly what the rules read
├── stated:  Citation            the same citation as currently read
├── found:   Resolution | None   what an archive holds        (validation's)
├── root_id / authority_id       what extraction read / what a lookup found
└── trace:   tuple[Node, ...]    evidence, and the corrections it justified
```

Four rules, and between them they are the whole design.

**`source` is never touched**, so the diff against `stated` is exactly what was
changed. They are the same type, so it compares field by field.

**A change lives inside the evidence for it.** A `Correction` is a field of the
`Node` that justified it, so a change with no evidence is not something that can
be built rather than something checked for. `record.corrections` reads them back
in order, and the state at any point is a fold over a prefix of the trace --
which is what an evaluation slices on, and what an as-of read is.

**What the filing states and what an archive holds are kept apart.** `stated` is
only ever the filing's reading; an archive's answer goes on `found`. A filing
citing the right case under the wrong year keeps its wrong year on `stated` and
gets the right one on `found`, and the disagreement between them is the finding.

**A node is named for what it read, not for the stage that ran it.** `reads` is
`DOCUMENT` or `RECORD`, and that is what decides where it may write: document
evidence corrects `stated`, record evidence settles `found`. A model re-reading
a name from the filing's own text produces document evidence whether it runs in
the case-name layer or inside identity -- those are the same operation, and only
`stage` differs.

On corpus document 006 the case-name layer corrects four names, and the parties
are where the repair shows: eyecite read `Boeser` / `Sharp ,  No.
CIVA03CV00031WDMMEH`, and the reader writes `Boeser` / `Sharp`; it read no
plaintiff for `Hassan`, and the reader writes `United States` against it. Each
correction carries the reader's own sentence for why, and what it replaced is
still on `source`.

### Where the held-out gap actually is, after the layer

Measured on `evaluation_set_1` with all three arms, so the reminder is a
number rather than an impression. The layer takes the seven bare names both
rule arms miss — `ReferenceCitation` recall 0/7, 0/7, **7/7** — and attributes
every one to the right root, carrying citations from 98.3% to 99.6% and
attribution from 89.2% to 93.0%.

What it does **not** reach, which is what the next move should be chosen
against:

| what is left | count | is it a case-name problem |
|---|---|---|
| pin cites | 16 | no — converter damage |
| attribution | 13 | no — `Id.` the resolver drops or sends to the other half of a parallel pair |
| citations | 2 | no — a decision cited by application serial number, which needs a *root* established |
| roots and short forms | 6 | no — annotation semantics: four short forms that are their own root, and `604 U.S. ___` having no page |

**So case-name hunting is close to done on this set and the remaining gap is
elsewhere.** What would improve the layer further is better case names on the
*roots* — its three false defects are all roots read with no name — and that is
what validation produces. Which is the loop below, and the reason to hand over
rather than keep pushing here.

### Roots first, leaves after validation

**Design, agreed and not yet built.** It replaces the section that used to sit
here, which said the case-name layer should run a second time after validation.
That was the right observation about one layer and the wrong size: the same
thing is true of every citation that hangs off another, and the fix belongs in
the data model rather than in a second pass.

#### The invariant

**A leaf cannot exist without an admitted root.** Not "should not" -- the type
refuses it. `root_id` is not a field a leaf is built with empty and filled in
later; a leaf is built *from* a confirmed root or it is not built.

A **root** is a citation that states a complete identifier: a volume, a reporter
and a first page, or a docket number with its court. A **leaf** is every
citation whose meaning is which root it points at -- a short form, an `Id.`, a
`supra`, a bare-name reference. `556 U.S. at 678` is characters anyone can read;
what it *claims* is page 678 of a case those characters do not name, and that is
not knowable from them.

#### Why the order has to be this way

Everything after the locator is keyed on the case name, and the case name is the
one field extraction is worst at. eyecite's short-form and `supra` resolution
matches an `antecedent_guess` against the parsed party names; the bare-name
sweep searches the document for them. After the initial pass those names are
whatever the parser made of them: `Cnty.` for
`Huri v. Office of the Chief Judge of the Cir. Ct. of Cook Cnty.`, `Inc.` for a
party that was lost, half a name where a spaced apostrophe stopped the search.
Matching against those is matching against a guess, and the evidence says so --
a search for the names eyecite itself refuses proposed thirty sites across 127
filings and not one of them was a citation.

Validation resolves each root against the archives and, where the lookup cannot
reach it, through open search with the root's context. It checks the case name
the filing wrote against the authority and rewrites it where they disagree. So
after validation's identity stage the record holds each root's **real** name,
and every question that was being asked against a guess has something true to
match on.

#### The three stages

    initialization   roots only. Every citation that states a complete
                     identifier, with its locator, its pin cite and its name as
                     the filing wrote it. No leaf of any kind is emitted.

    identity         validation, per root: lookup, then a check on the case
                     name, then open search for what the lookup cannot reach.
                     Each root comes back admitted or not, with a name.

    leaves           back in extraction, over the admitted roots: the short
                     forms, the `Id.` chains, the `supra` forms and the
                     bare-name references, all matched against real names.

#### What is dropped, and what that costs

The initial pass **drops the leaves entirely**. It does not record a span and
leave the kind off; that would be a leaf-shaped hole and the invariant would be
a convention again rather than a property. The leaf pass re-reads the document,
which it has to do anyway to find a bare name, so the position is recovered
rather than carried.

What that costs is that the artifact between the two stages holds fewer
citations than the filing writes, and anything reading it has to know that. It
is worth saying out loud: **an artifact from the initialization is not a reading
of the document's citations.** It is the roots, which is what identity needs and
all that identity needs.

#### How a leaf finds its root

`structure/attachment.root_for` decides it, and it reads `stated` and never
`source`. That is the point of the second growth: `source` is the parse and
never changes, `stated` is the citation after a reader and validation have
corrected it, and matching a leaf against the parse would throw every one of
those corrections away.

A short form is matched by the volume and reporter it states, against the roots
that state the same. One candidate is the answer; several are narrowed by the
name it writes, and then by the page -- of the roots that begin at or before the
page claimed, the last one holds it. `supra` and a bare-name reference are
matched by the name alone, since neither states an identifier. `Id.` takes the
root of the citation before it, refused when the page it claims cannot fall
inside that root.

Only the whole case name is read, with the parsed parties as a fallback when
there is none. eyecite fills `plaintiff` and `defendant` from the words in front
of a citation, and those are often the words of the citation before it: the
`Bell Atl. Corp. v. Twombly , 550 U.S. 544` two sentences after `Ashcroft v.
Iqbal` is parsed with `defendant='Iqbal'`. Reading the parties alongside the
name let `Iqbal , supra` reach Twombly.

Nothing is guessed. A leaf that matches no root, or matches two and cannot be
narrowed, is not grown at all -- `CitationRecord` refuses a leaf without a root,
so an undecided leaf is a leaf that does not exist rather than one pointing at
the wrong case. Over the 26 corpus filings the parse finds 86 leaves and 77 are
placed.

#### Open: a leaf that cannot be grown is still a finding

The initialization drops a leaf it cannot attach, and the leaf pass drops one
whose root the document does not hold. Both are right for the record and wrong
for the ledger: a short form for a case the filing never gives in full is a
**defect**, and one of the ground truth's `nonconforming_citation` classes.
`orphan_short_forms` used to propose exactly those, and there is now nothing in
the document for it to propose.

So the leaf pass has to report what it could not grow, beside what it grew.
Where that report lives -- a field on the document, a candidate kind, a unit of
its own -- is not decided. Until it is, `tests/test_adjudication.py` carries a
strict `xfail` so the day it starts working is not silent.

The nine refused over the corpus say what such a report would have to hold.
Four are an `Id.` whose antecedent is a statute, which no case root can take:
the tree places citations under case roots only, so `Id. § 1231(g)` after
`8 U.S.C. § 1231` is unplaceable by construction and not a defect at all. Two
follow no citation of any kind. The remaining three are name matches that failed
or were ambiguous, and eyecite placed one of them.

#### Where the site hunting goes, and why it is last

A site is a place in the document the rules came up empty, found against a copy
with every extracted citation blanked out. So how much of the document is
masked decides how many sites there are, and a leaf sitting in the residue
looks exactly like an unread citation -- a case named with no full citation
beside it, which is what a site is. Hunting before the leaves are grown pays a
model to rediscover citations the rules already read.

Growing them first barely moves the mask and collapses the sites:

    set        documents   leaves grown   sites, roots only -> after the leaves
    corpus        26            78            83  ->  42
    eval-1        10            77           121  ->   69
    eval-2        10            75            79  ->   19

A third of a percent more of the text is covered, and half to three quarters of
the questions disappear. What is left is one or two sites a document, which is
few enough to put every one of them in front of a model rather than rank them.

**The hunt is triggered by the leaves that could not be grown.** A refused leaf
is a case the filing cites and the document never introduced, named at a known
span, and it has two answers, both of them findings: the hunt reaches the root
the tokenizer missed, and the leaf grows on the next round; or there is no root
to reach, and the leaf is a `nonconforming_citation`. The second is the class
`orphan_short_forms` used to propose and now has nothing to propose, so the
trigger closes that hole rather than only raising recall.

**Roots are hunted in the first pass instead**, before validation, because a
root is what everything else hangs off: a false leaf is one wrong page claim, a
false root is a case that does not exist plus every leaf that then attaches to
it. Hunting them where their output still has to pass identity before anything
is built on it is the only order that is safe. It is off by default on this
corpus, where the rules already read 1,055 of the 1,057 roots the three sets
state -- the two missed are `Watson v. New York , WL 6200979`, whose volume the
filing never wrote, and `Scheuer v. Rhodes, 416 U.s. 232`, whose reporter is
spelled with a lowercase `s`. Both are the shape a reader would catch; two
recovered against the chance of a fabricated root is not a trade worth making
by default.

**A root found after the leaves have grown is safe and incomplete**, which is
not the same as wrong. `CitationRecord` refuses a leaf with no root and the
leaf pass attaches only to roots the document holds, so a late root cannot
silently acquire leaves -- it has none until another growth runs. What is
missing is the artifact saying the tree is not finished, which is the same open
item as the refused leaves above.

#### What the measurement says

Both growths are scored together, against the same ground truth, because the
artifact the two stages hand between them is not a reading of the document's
citations and a score over it would be the wrong question. `tree.py` runs the
whole sequence per arm -- the roots, then the leaves -- and the arms differ in
how a leaf finds its root: `augmented` keeps eyecite's own resolution, `grown`
decides from `stated`, and `grown+identity` does it over roots a recorded
identity run has settled.

Over the 26 corpus filings the second growth removes four false attributions
and loses nothing: an `Id.` whose antecedent is a statute is not the earlier
case's, and attribution precision goes from 97.9% to 99.3%. Over the two
held-out sets it is worth one short form on each of `extraction-eval-1` and
`extraction-eval-2`. The identity names move no leaf on this
corpus: every leaf they would reach is already reached by the volume, the
reporter and the page, which the filing states at the leaf itself. What a name
decides is a `supra` or a bare-name reference, and those are the forms these
filings write least.

The bare names are left out of both sides, which the evaluator does by default
and `--score-bare-names` undoes. The datasets do not change -- they are ground
truth for the whole document, and a bare name is a conforming citation whatever
the pipeline is asking today.

---

---

## How it is measured

Nothing above says how well any of it works, deliberately. What an occurrence
is, how a prediction is matched against one, what each arm scores, and how to
reproduce a number are documented with the code that runs them, in
[the extraction evaluation](../evaluations/extraction/README.md). The benchmark's
contents and provenance are on
[the dataset card](https://huggingface.co/datasets/gt-csse/false-citation-bench).
