# Rules we cannot prove belong in a candidate layer, not in the tokenizer

A design decision, and a rule of thumb it comes from.

**When a fix looks easy but the data cannot show it generalises, do not harden
it. Emit a candidate and have a reader confirm it.** The thinness of the
evidence is exactly what makes the review affordable: a pattern with two
instances in a hundred documents costs two calls.

The uppercase reporter was the case that made the argument. `33 F.4TH 693` is
not a citation to eyecite because its reporter extractors are case-sensitive
(see [uppercase-reporters.md](uppercase-reporters.md)). Hardening that means
making 67 reporter spellings case-insensitive, which needs an empty-filter fix
in the Aho-Corasick prefilter and a precision measurement nobody has, to buy
**two** citations.

It is now handled without either -- the generator scans a lower-cased copy, and
a re-read of the window with an unfiltered case-insensitive tokenizer settles it
with no model call -- which sharpens rather than weakens the rule of thumb. The
question was never "rule or reader". It was **where** the permissive read is
allowed to happen: on a whole corpus it needs a precision argument, and on one
window a generator has already flagged it needs only a check on the result.

## The layer is composite, and its name no longer fits

`suspected_locators` finds one thing: a reporter string with digits around it
that produced no citation. That is one generator among several, and the others
have nothing to do with hunting sites:

| generator | candidates on the 77 mined filings | what they are |
|---|---:|---|
| reporter site, strict stage | 161 | letterheads, procedural rules, legislative journals; 7 real, settled by a re-read |
| reporter site, fuzzy stage | 3 | 1 real (`Fordham Intl. L.J.`), 2 a packaging label |
| short form with no antecedent | 27 | 20 filings citing a case they never introduced, 2 our own miss |
| stranded locator parts | -- | the table rows the converter reassembled; already built in `build_locator_bench` |

What unites them is not that they are sites. It is that each is **something
citation-shaped, or citation-defect-shaped, that the deterministic pass did not
record** -- a candidate for the record, needing adjudication before it enters.
`recovery`, `candidates` or `unrecorded` all describe that better than `hunting`.

## The broad generator is two stages, and neither is a search for citations

`reporter_sites` answers "is there a reporter here" twice, because the question
has two answers.

**Strict** finds the gazetteer's own spellings, up to capitalisation and one
optically confused character (`S0.` for `So.`, `U.s.C.` for `U.S.C.`). On clean
text this is almost the whole yield.

**Fuzzy** finds a number, a short letter run and a number where the letters
reduce to a reporter once punctuation stops mattering. `556 U,S, 662` is
invisible to the strict stage -- no gazetteer string matches `U,S,` -- and is a
`U.S.` citation to anyone reading it. The number on either side is what makes
the stage affordable: without it the pattern matches most of a page of prose,
and with it the whole mined corpus yields three sites.

Three things keep the fuzzy stage from swamping the queue, and each was measured
after it did:

*   **A reporter is capitalised.** Folding case makes every `at` in `Doc. 174 at
    5` the gazetteer's `At.`, and every `[Doc. 40, p. 8]` the Pacific Reporter.
*   **A short key must be written like an abbreviation.** Below four characters,
    the site must carry a full stop or be all capitals -- which keeps `U.S.`,
    `US`, `U,S,` and `F.2d` and drops the prose.
*   **Approximate matching guesses, so it is allowed only on long keys.** At
    four characters `Case` is within one edit of `Chase`, `Page` of `Paige`, and
    `after` of `A.F.T.R.`, all of which it proposed. At eight characters and a
    0.9 cutoff it proposes nothing on either corpus that punctuation-insensitive
    equality did not already reach, so its value is unproven and it is kept only
    because the cost of keeping it is measured at zero.

## The economics only work if the broad generator is demoted

    all generators                        192 candidates / 77 documents  = 2.5 per document
    without the reporter-site generator    28 / 77                       = 0.4 per document

164 of the 192 come from the one generator whose yield on this corpus is
approximately zero -- though 7 of those are now settled by a re-read before any
call, so the reviewer's queue is 157. The argument that review is cheap because the evidence is
thin holds for the narrow generators and fails for the broad one -- so the layer
becomes worth wiring at the moment the broad generator stops being all of it.

That does not reverse
[site-hunting-stays-unwired.md](site-hunting-stays-unwired.md); it explains it.
The reporter-site generator stays out, or behind a precision filter. The layer
it lives in is what becomes useful.

One measurement supports the filter half of that. Refusing a site whose window
holds a section sign -- a statute, not a case -- was worth 16 candidates on the
bench and **52 on the mined corpus**, 237 down to 185, on documents it was not
written from. Cheap rules do generalise; it is the expensive ones that need a
reader.

## What the record has to carry

If a candidate is adjudicated into the record, the record must say so: which
generator proposed it, and that a reader accepted it. Otherwise a recovered
citation is indistinguishable from a parsed one, and the next person measuring
extraction is measuring the reviewer instead.
