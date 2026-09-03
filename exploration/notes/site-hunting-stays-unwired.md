# Site hunting is not wired into the pipeline, and the numbers say why

`suspected_locators` finds reporter strings that look like locators and produced
no citation. It was built when the tokenizer missed citations often. Relaxation
made that rare, and this records the decision not to wire the hunter in, with
the measurements behind it.

## Detection is not where the problem is

On `false-citation-bench-locator-only-v2.0`, the only corpus with ground truth:

    stated by the filings   586
    extracted               583      99.5%
    spurious                  0

All three misses are table-of-authorities rows the converter reassembled out of
order -- `1013, 1023 (D. Ariz. 2011)` with its volume and reporter on another
row, and two Westlaw citations whose year was left behind the reporter. The
citation is on the page; the row is not.

## What the hunter would buy, and what it would cost

It is not useless. Its window covers **all three** misses -- for each one it
flags a site within 80 characters, and a judge reading that window would be
reading text that contains the missing citation.

What it flags there is the court abbreviation in the parenthetical, not the
locator: `D.`, `Ariz.`, `Cal.`, `N.Y.` So the candidate is wrong and the window
is right, which is enough for adjudication, since a judge reads the window.

The cost is the yield.

    bench    65 sites for 3 real misses          4.6%
    mined   242 sites, 76 of them in a letterhead

On the 77 unseen filings the remainder is docket numbers, procedural rules
(`FED. R. CIV. P. 26`), legislative journals (`Minn. S.J.`) and stop words
matched as reporters (`citing`, `see also`), and no missed case locator is
visible in the sample. Wiring it in as it stands means a model rejecting
mailing addresses roughly a third of the time to reach at most 0.5% of recall.

**So it stays unwired.** If it is ever wired, a precision filter comes first --
a ZIP code or a phone number in the window is not a citation, and that is a rule,
not a judgement.

## Where the problem actually is

Downstream of detection, and still inside extraction rather than validation:

| | bench | mined |
|---|---:|---:|
| occurrences that could not be attributed | 2.7% | **9.3%** |
| case citations with an incomplete name | 18.5% | 17.3% |

Attribution is the larger of the two and three times worse than the bench
suggested, 210 of its 251 failures being bare `Id.` That is the citation tree's
problem, not the validator's: validation can check a claim once it knows which
authority the claim is about.

One qualification on "detection is solved". It was not free -- the docket
extractor added this session is a detection change, and it is what lets document
016's `Id.` chains resolve at all. Detection is finished for reporter locators,
not by assumption.
