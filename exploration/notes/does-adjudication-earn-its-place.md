# Does the adjudication layer find more, and does it reject the rest?

Run for real on `false-citation-bench-locator-only-v2.0`, the only corpus with
ground truth: 47 reporter sites proposed, every one sent to
`review/locator.py` with `gpt-5.6-luna`.

    declined -- no locator in the window            43
    call failed                                      3
    accepted at least one locator                    1
      already extracted                              1
      recovered                                      0
      spurious                                       0

## It rejects the noise, and that is the result worth having

**43 of 44 successful calls correctly returned nothing, and none of the 47
produced a spurious locator.** The windows are dominated by letterheads,
procedural rules and legislative journals -- `Corrales, NM 87048 (505) 220-5691`,
`FED. R. CIV. P. 26`, `Minn. S.J., 93d Leg.` -- and the reviewer declined all of
them.

That is the property that decides whether this layer can ever be wired in. A
reviewer that accepts a mailing address puts a spurious locator into the record
looking exactly like a parsed one, and no downstream check would catch it. On
this evidence it does not.

The design earns the credit rather than the model: the instruction requires the
locator to be **quoted from the window character for character**, requires all
three parts in one run of text, and forbids inventing a missing one. A quote
that is not found in the document is rejected before it leaves the module. The
model is never asked what a citation is, only whether characters already there
form one.

## It recovers nothing, and the reason is not a failure

Zero of the three locators the bench states and extraction misses were
recovered, and the reviewer was right to decline all three:

    021 [5727:5753]  '1013, 1023 (D. Ariz. 2011)'
    022 [8380:8390]  'WL 9137645'
    025 [3981:3991]  'WL 6200979'

Each is a table-of-authorities row the converter reassembled, with the volume or
the year stranded in a different cell. The instruction says exactly what to do
with those -- *"a volume that appears in a different table row or column ... is
not present"* -- so refusing them is the rule working, not the reviewer missing
them.

**They are outside what a grounded reviewer can report at all.** Recovering them
means assembling a citation from parts in different places, which is the one
thing the grounding rule exists to forbid. If they are ever to be recovered it
will be by the stranded-parts reconstruction already written in
`scripts/build_locator_bench.py`, which works on table structure and not on
meaning.

## Two smaller findings

**The one acceptance was not a mistake.** It landed on a citation extraction had
already found. The site generator masks extracted spans, but the reviewer's
window extends 170 characters past the site into unmasked text, so a neighbouring
citation is visible and quoting it is correct behaviour. Harmless, and worth
knowing before someone reads a duplicate as a bug.

**Three calls in 47 failed at the provider.** The runner treats a failure as a
result rather than an exception, which is how it has to be: a layer whose answers
arrive over a network needs a verdict for "no answer" as much as for yes and no.

## What this settles

The layer is safe and, on this corpus, empty. That is consistent with what the
generators already said -- the site generator proposes 185 candidates on the
mined corpus to reach at most 2 real citations -- and it removes the remaining
doubt, which was whether a reviewer would hallucinate its way through them. It
would not.

So the conclusion in [site-hunting-stays-unwired.md](site-hunting-stays-unwired.md)
stands for the reason given there -- yield, not risk -- and the narrow generators
remain the case for keeping the layer at all.
