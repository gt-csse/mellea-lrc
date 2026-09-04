# Does the adjudication layer find more, and does it reject the rest?

Run for real on `false-citation-bench-locator-only-v2.0`, the only corpus with
ground truth: 47 reporter sites proposed, every one sent to
`review/locator.py` with `gpt-5.6-luna`.

    declined -- no locator in the window            47
    call raised                                      0
    accepted at least one locator                    0
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

## The window now hides what has already been read

The first run accepted one locator, and it was a citation extraction had already
found. Not a hallucination -- the reviewer's window ran 170 characters past the
site into text that was never masked, so a neighbouring citation was simply
there to be quoted.

The window is now built from the masked copy, and the choice of mask matters
more than it looks:

    full-span mask : '                                                    ...'
    locator mask   : 'Norton v. Shelby County,             , 442 (1886). Doe v. Colgate Univ. , 2016 WL1448829, ...'

**Blanking full spans would hide the candidates.** A full span covers text that
is not a citation, and at `Relaxation.NONE` one span reaches across a whole
sentence -- in the example above it swallows the unread `2016 WL1448829` that a
reviewer exists to find. Blanking **locators** is enough for the purpose: a
locator quote needs a volume, a reporter and a page, and with those characters
gone a neighbour cannot be quoted, while everything unread stays visible.

Hit detection keeps the full-span mask, which is a different job -- not flagging
a reporter that sits inside a citation's party name or court parenthetical --
and is safe there because the court-and-date boundary keeps full spans tight at
the relaxed levels.

After the change the duplicate is gone and nothing is accepted at all.

## The "failed calls" were the reviewer refusing, reported as a crash

Two to four calls per run raised `ValidationError: Invalid JSON: EOF ...
input_value=''`. Read as provider trouble, they were nothing of the kind. The
model returns an empty string, a **validation function** then calls `_parse` on
it and raises, and the exception escapes the whole call:

    requirement.py:271  validate       return self.validation_fn(ctx)
    locator.py:242      _validate_shape    for locator in _parse(...).locators:
    locator.py:229      _parse             return _Locators.model_validate_json(...)

So a refusal was arriving as a failure and being counted as one. Unparseable
output is now a **failed requirement** rather than an exception: the repair loop
gets a reason and another turn, and if the budget runs out the caller sees an
empty result, which the existing handler already reads as a decline.

That is the difference between "the layer is unreliable 8% of the time" and "the
layer declined and the plumbing mislabelled it", and only one of them is true.

## A short form arrived as a locator, once

Run to run the same site gave `[]`, then a refusal, then this:

    spurious: '214  F.3d  at 1071'   from 'Advanced Textile , 214 F.3d at 1071 -72'

Which is a short form. `at 1071` is a pin cite, not a first page, so looking it
up would resolve to the wrong case or to nothing -- the exact failure short forms
were already shown to cause.

The requirements check the model's **quote**; what enters the record is the
**grounded span**, and the whitespace-relaxed match can bind wider than the
quote. The same equality is now applied after grounding: the span's characters,
reduced, must equal volume + reporter + page. `214f3dat1071` is not `214f3d1071`,
so a short form cannot arrive as a locator whatever the model answers.

**Precision was not deterministic before this, and the earlier "0 spurious in 47"
was one run.** It is now a property of the code rather than of a sample.

## What this settles

After both fixes, all 47 sites decline, nothing raises and nothing is spurious.
The layer is safe and, on this corpus, empty. That is consistent with what the
generators already said -- the site generator proposes 185 candidates on the
mined corpus to reach at most 2 real citations -- and it removes the remaining
doubt, which was whether a reviewer would hallucinate its way through them. It
would not.

So the conclusion in [site-hunting-stays-unwired.md](site-hunting-stays-unwired.md)
stands for the reason given there -- yield, not risk -- and the narrow generators
remain the case for keeping the layer at all.
