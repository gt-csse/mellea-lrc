# What counts as an authority

Written 2 September 2026, on `bench/locator-recall`. The scope rule had been
implicit, and re-derived decision by decision — what goes in the bench, what
counts as a miss, what the residue hunt should bother offering a model. Stating
it once.

## The rule

**An authority is a court case, however it is identified.**

Not "a case-law citation": that phrasing loses docket numbers, and a case cited
by docket is as real, as verifiable and as forgeable as one cited by reporter.

| | in scope | why |
|---|---|---|
| `550 U.S. 544` | yes | a case, identified by reporter |
| `No. 1:25-cr-00312-RPK (E.D.N.Y.)` | yes | a case, identified by docket |
| `Id. at 570` following either | yes | a claim about that case's page |
| `28 U.S.C. § 636` | no | not a case |
| `Pl. Br. ¶21`, `Doc. 351`, `ECF No. 42` | no | a document in *this* case |
| `Id. ¶¶ 23-24` into the filing's own complaint | no | the same |
| a short form nested inside a quoted case | no | the filing is not making the claim |

The line is **external identifiability**. A citation is in scope when it names
a case to someone who does not have this case's record in front of them. That
is exactly the population a verification tool can check, and exactly the
population where a fabricated citation does damage.

## What that marks out, on this corpus

Everything below is *correct* behaviour, not residual failure. It is written
down so nobody spends time closing gaps that are not gaps.

**The tree's two unattributed citations.** `Id. ¶¶26-28` in document 005 points
at `Pl. Br.` — the opposing brief, named only by abbreviation, eight times,
never by an identifier. `383 U.S. at 85` in document 006 sits inside a quotation
of what *Anaya* cited; the filing never states `383 U.S. 75` and is not asserting
it. Neither has an authority to attach to, and neither ever will.

**Seventeen back-references the extractor never produced.** Document 021 writes
`(Id. ¶¶ 23 -24)`, `(Id. ¶ 32)` and so on, summarising its own complaint's
numbered allegations. They are missed because the parenthesised form is not what
eyecite matches — but they are out of scope regardless, so the miss costs
nothing. `hunt_secondary` now labels them rather than counting them against
recall.

**Eleven docket occurrences dropped from the locator bench.** In scope as
authorities, out of scope for a *locator* bench, which measures whether a
volume-reporter-page identifier was read. They belong in a bench of their own,
not nowhere. `extraction/docket-locator` now extracts all of them.

## What it does not excuse

Two things surfaced by the same hunt are squarely in scope and are real misses.

**`Caraway , at 1301`** in document 006 and **`Rafiyev at 861`** in document
026. Both are cases the same filing cites in full elsewhere, referred to again
by party name with a pin cite — the exact shape eyecite's
`ReferenceCitation` exists for, and the exact shape that, when missed, lets the
next `Id.` attach itself to whatever citation happens to precede it. That is how
the corpus's one misattribution happened; see section 5a of
`citation-tree-handoff.md`.

### Why they fail, which is not what it looked like

Not a name mismatch. `is_valid_name` passes both, the captured party names are
`Caraway` and `Rafiyev` exactly, and both references sit after their full
citation where eyecite looks.

They fail on **whitespace**, and it is the same defect this whole branch exists
to correct:

```
'Caraway ,  at  1301'   doubled spaces   no match
'Rafiyev at  861.'      doubled space    no match
'Bell at 546.'          single spaces    matches
```

eyecite's `reference_pin_cite_re` composes the party name with `PIN_CITE_REGEX`,
which spells its separators as a single optional literal space, `\ ?`. PDF
extraction of justified text leaves doubled spaces, and the reference vanishes —
exactly as a reporter citation used to vanish before the joins were relaxed.

**The relaxation does not reach this.** `mellea_lrc.extraction.relaxation`
rebuilds the *reporter* extractors' regexes; reference extraction happens later,
in `find.py`, against a regex the tokenizer never sees. So the corpus has one
class of citation still failing for the reason every other class has stopped
failing for.

That makes it the highest-yield item outstanding. It is a matching problem with
a correct answer, not a heuristic; it is upstream of the `Id.` misattribution;
and unlike the join relaxation it is a handful of characters in one pattern.
It is also worth reporting upstream, since it is eyecite's own regex and not
anything this project does to it.
