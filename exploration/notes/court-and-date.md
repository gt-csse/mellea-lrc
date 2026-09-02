# Court and date: what eyecite does with them, measured

What the library does as published, measured on
`false-citation-bench-locator-only-v2.0` -- 26 documents, 583 full case
citations, extraction at `Relaxation.FULL`. This records the behaviour and
its causes; the fixes are described where they are made.

## 1. The year is sometimes wrong, not missing

518 citations carry a year. **20 of them carry the wrong one** — a year taken
from a different case. Iqbal comes back as 2001, Twombly as 2001 in one place
and 2009 in another, `137 S. Ct. 1285` as 1978.

The cause is two things in eyecite that compound:

- `add_post_citation` calls `match_on_tokens` without `strings_only`, so the
  forward scan stops only at a paragraph break — not at the next citation.
- `POST_FULL_CITATION_REGEX` spells its leftover group `(?P<extra>[^(;]*)`,
  unbounded until the next bracket or semicolon.

So a citation with no parenthetical of its own runs forward and takes the court
and year of a later, unrelated citation. In document 009,
`Koulkina, 2009 WL 2103627, at *3.` stands two sentences before
`Spector v. Torenberg, 852 F. Supp. 201, 205 (S.D.N.Y. 1994)` and comes back
carrying 1994.

## 2. Crossing the next citation is not always wrong

A parallel citation is one decision printed in several reporters, written as a
comma-separated run with a single date parenthetical at the end. The first
member therefore *has* to reach across the others to find its year, and 30
citations here do exactly that, correctly.

What the corpus shows about the shapes:

| between two adjacent full citations | same case | different cases |
|---|---:|---:|
| a comma, or nothing but a pin cite | 14 | — |
| a semicolon somewhere | **0** | 159 |
| prose between them | 2 | 375 |

A semicolon never appears inside a parallel run. It is a reliable **negative**
signal. A comma is not a positive one — plenty of unrelated citations are
comma-adjacent too.

**One year does not prove one case.** The single date parenthetical is the
formatting convention, not evidence about identity: two different cases decided
the same year produce the same shape. The implication runs the other way — two
citations with *different* years cannot be the same decision, so a year refutes
parallelism but never confirms it.

**Subsequent history is the exception that breaks the run.** `2019 WL 1085179,
at *78 (D.N.M. Mar. 7, 2019), aff'd, 950 F.3d 754` is the same *case* and a
different *decision*, comma-joined, and the first member carries its own earlier
date. So "one date at the end of a comma run" does not hold even for same-case
runs: `aff'd`, `rev'd`, `cert. denied` and `vacated` end one run and start
another.

## 3. Why `3d Cir.` resolves to nothing though the Bluebook prescribes it

`get_court_by_paren` strips every non-word character, lowercases, and compares
against one field: `citation_string`. Exact match first, then a `startswith`
fallback.

courts-db carries **one** citation string per court and no aliases, and it is
not internally consistent about ordinals:

    ca2    citation_string '2d Cir.'        the Bluebook form
    ca3    citation_string '3rd Cir.'       not the Bluebook form
    bap2   citation_string '2nd Cir. BAP'

So `3d Cir.` — correct Bluebook — matches nothing, while `3rd Cir.` matches. And
`2nd Cir.` finds no exact match, falls through to `startswith`, hits
`2ndcirbap`, and returns **bap2**: the Bankruptcy Appellate Panel, a different
court, reported with no indication that a guess was made.

The `regex` field on each record is not a way out. It matches long-form names
("Third Circuit Court of Appeals"), not citation abbreviations, and
`courts_db.find_court` returns nothing for any of `3d Cir.`, `2d Cir.` or
`S.D.N.Y.`.

## 4. The New York departments are a data-model gap, not a spelling one

32 of the 51 unrecorded courts are Appellate Division departments — `2d Dept.`,
`1st Dep't`, `3d Dep't`. courts-db models the Appellate Division as a single
court, `nyappdiv`, named "The Four Departments of the Appellate Division". There
is no entry for a department, so there is nothing for the parenthetical to match.

The reporter already carries the information: `155 A.D.3d 781` is an Appellate
Division citation whatever the parenthetical says. Court can be resolved from
the reporter, with the department kept as written.

## 5. Where the 65 missing years come from

| cause | count |
|---|---:|
| Westlaw or LEXIS — the volume **is** the year | 35 |
| whitespace inside the parenthetical (`2024 )`, `Cir.1991`, `,2004`) | 13 |
| the converter lost the opening bracket | 6 |
| the filing states no year | 11 |

Reproduce with `exploration.court_and_date.survey_missing` and
`exploration.court_and_date.survey_wrong`.
