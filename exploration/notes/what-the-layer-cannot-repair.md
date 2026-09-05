# What the adjudication layer cannot repair, probed deliberately

The relaxations widen separators, so they read damage *between* the parts of a
citation. `evaluations/extraction/damage_probe.py` is a small hand-made document
of damage *inside* a part, which no gap-widening can reach, and it asks four
questions of each case in the order the pipeline asks them: do the rules miss
it, does a stage propose it, does a re-read settle it without a call, and does
the reviewer recover it as something eyecite can then parse.

    case                            expected            rules read      stages     reviewer
    periods gone from the reporter  410 US 113          410 US 113      none       []
    commas for periods              556 U,S, 662                        fuzzy (1)  ['556 U.S. 662 -> parsed']
    period gone from the series     654 F3d 404         654 F3d 404     none       []
    letter l for 1 in the page      833 F.2d l83                        strict (1) ['833 F.2d 183 [repaired] -> parsed']
    letter O for 0 in the volume    32O N.C. 1                          strict (1) ['320 N.C. 1 [repaired] -> parsed']
    short form, periods gone        556 US 678          556 US at 678   none       []
    clean, a control                347 U.S. 483        347 U.S. 483    none       []
    doubled spaces, a control       550 U.S. 544        550  U.S.  544  none       []
    a statute                       -- not a citation --                none       []
    a letterhead                    -- not a citation --                strict (1) []
    a procedural rule               -- not a citation --                none       []
    a docket number                 -- not a citation --                none       []

## Missing punctuation is not a gap at all

`410 US 113`, `654 F3d 404` and `556 US at 678` are all read by the rules. Not
by any relaxation of ours -- reporters-db carries punctuation-free spellings as
**variations**, so `US` and `F3d` are reporters eyecite already knows. Three of
the cases written to be hard turned out not to be, which is the probe doing its
job.

## The limit that was structural, and how it was removed

**A reporter the gazetteer cannot see used to produce no candidate.** `556 U,S,
662` is missed by the rules, and for as long as the generator searched only for
gazetteer spellings it proposed nothing either: `U,S,` is not one of them. No
reviewer can be asked about a citation nobody proposed, and no prompt improves
that.

The fuzzy stage is what closed it. It matches a number, a short letter run and a
number, and asks whether the letters reduce to a reporter once punctuation stops
mattering -- `U,S,` reduces to the same characters as `U.S.` and as `US`. The
site then goes to the reviewer like any other, and the reviewer is told about
`U.S.` rather than about `U,S,`, because a description built from the damage
says only that the database has never heard of it.

What comes back is a locator, not a citation, so it is substituted back into the
window and re-read: `promote_locator` puts `556 U.S. 662` where `556 U,S, 662`
is written, hands the window to eyecite, and maps the resulting full span back
through a `SpanUpdater`. The `locator_span` on the record is the **damaged**
span -- the record points at the characters the filing contains, never at a
repair -- and the date and party names come from the real parser.

**Damage inside a number reaches a reviewer, and is recovered.**
`833 F.2d l83` proposes a candidate, and the model answers it correctly --

    {"text":"833 F.2d l83","volume":"833","reporter":"F.2d","page":"183"}

quoting the damage verbatim and reading the letter back as the digit, which is
exactly what the instruction asks for. It is then refused, by `_validate_parts`:
the quote, reduced, must equal the three parts concatenated, and `833f2dl83` is
not `833f2d183`.

That rule is what stops invention -- a model cannot report a part that is not in
the window, because the quote would contradict it. It also forbade repairing a
character, and those were the same rule seen from two sides.

`_validate_parts` now accepts a second reading: the quote and the parts may also
match once the characters a scanner confuses are folded together -- `l` onto
`1`, `O` onto `0`, `S` onto `5` and four more. Folding is applied to both sides,
so it admits a **substitution** and nothing else. `833f2dl83` matches
`833f2d183`; it does not match `833f2d999`, and an inserted word changes the
length, which no folding hides.

    letter l for 1 in the page   833 F.2d l83   ->  ['833 F.2d 183 [repaired]']
    letter O for 0 in the volume 32O N.C. 1     ->  ['320 N.C. 1 [repaired]']

**The rules are untouched.** A citation with a character damaged is still missed
by extraction, still becomes a candidate, and is recovered only where a reviewer
confirms it -- which is the point of putting the repair here rather than in the
tokenizer. It cannot be proved general, so it proposes and is reviewed instead of
being hardened.

`AdjudicatedLocator.repaired` records which citations arrived that way. Reading a
letter back as a digit is a judgement and not a parse: the document does not say
`183` anywhere. A consumer that wants only what the page states can decline
them, and one that does not can tell the two apart.

On the bench nothing changes: 42 sites, 6 settled by a re-read and 36 declined,
nothing recovered and nothing spurious. The confusion set costs no precision
there because no candidate on that corpus is a damaged locator.

## What is recovered is not yet stable

Four runs of the probe, same prompt, same temperature:

    556 U,S, 662    recovered 4 of 4
    833 F.2d l83    recovered 4 of 4
    32O N.C. 1      recovered 2 of 4
    a letterhead    declined  4 of 4

The provider routes across backends, so a run is not reproducible. Punctuation
damage and a damaged page are recovered reliably; a damaged **volume** is not,
and `32O N.C. 1` is the case that fails. The refusals are stable, which is the
half that would be expensive to get wrong.

## The refusals are correct

A statute, a procedural rule and a docket number propose no candidate at all --
the section-sign rule and the digit-window test filter them before any call. The
letterhead proposes one and the reviewer declines it. Nothing was invented from
any of the twelve cases.
