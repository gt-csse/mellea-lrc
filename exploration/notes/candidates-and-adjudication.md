# Rules we cannot prove belong in a candidate layer, not in the tokenizer

A design decision, and a rule of thumb it comes from.

**When a fix looks easy but the data cannot show it generalises, do not harden
it. Emit a candidate and have a reader confirm it.** The thinness of the
evidence is exactly what makes the review affordable: a pattern with two
instances in a hundred documents costs two calls.

The uppercase reporter is the case that makes the argument. `33 F.4TH 693` is
not a citation to eyecite because its reporter extractors are case-sensitive
(see [uppercase-reporters.md](uppercase-reporters.md)). Hardening that means
making 67 reporter spellings case-insensitive, which needs an empty-filter fix
in the Aho-Corasick prefilter and a precision measurement nobody has, to buy
**two** citations. Demoting it means a pattern that produced exactly two
candidates across 103 documents, both real, for a reader to confirm in a second.

## The layer is composite, and its name no longer fits

`suspected_locators` finds one thing: a reporter string with digits around it
that produced no citation. That is one generator among several, and the others
have nothing to do with hunting sites:

| generator | candidates on the 77 mined filings | what they are |
|---|---:|---|
| reporter string with no citation | 185 | letterheads, procedural rules, legislative journals |
| short form with no antecedent | 27 | 20 filings citing a case they never introduced, 2 our own miss |
| uppercase reporter locator | 2 | both real |
| stranded locator parts | -- | the table rows the converter reassembled; already built in `build_locator_bench` |

What unites them is not that they are sites. It is that each is **something
citation-shaped, or citation-defect-shaped, that the deterministic pass did not
record** -- a candidate for the record, needing adjudication before it enters.
`recovery`, `candidates` or `unrecorded` all describe that better than `hunting`.

## The economics only work if the broad generator is demoted

    all four generators        214 candidates / 77 documents   = 2.8 per document
    without the reporter-site generator   29 / 77              = 0.4 per document

185 of the 214 come from the one generator whose yield on this corpus is
approximately zero. The argument that review is cheap because the evidence is
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
