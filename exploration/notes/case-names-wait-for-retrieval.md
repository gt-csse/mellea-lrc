# The case-name decision waits for retrieval

Case names are the worst-parsed field this project has: 17% of case citations
carry no complete party pair on either corpus, and some that do are silently
wrong -- `St. Amant v. Thompson` is recorded with `St. Amant` in the *defendant*
field and Thompson gone entirely. Three unrelated causes, none of them a
separator that could be widened, so the instrument is a model rather than a rule
(see [court-and-date.md](court-and-date.md) and
`exploration/court_and_date/probe_case_names.py`).

**It is still not the next thing to build.** The reason is what a model would be
given.

Asked to read a case name out of a filing, a model has the sentence and nothing
else. Asked *after* the locator has been resolved, it has the sentence and the
**name the reporter gives that case**. Those are different problems. The second
is a comparison against an authority; the first is an extraction with no answer
key, and it is the one where a plausible wrong answer is indistinguishable from
a right one.

The same evidence points at it from the other side. The 22 short forms with no
antecedent cannot be looked up, because a short form states a pin cite and not a
first page -- what identifies them is the party name plus the volume and
reporter. A model given `DCD Programs, 833 F.2d at 186` and nothing else must
recall the case. A model given the same text plus the candidates a search
returns for volume 833 of F.2d is choosing from a closed set, which is the shape
this project has already decided it wants everywhere else.

So the order is: resolve what can be resolved, retrieve, and *then* put the
filing's rendering beside the reporter's. Building case-name extraction first
would be building the version of the problem with the least context available.

Nothing about this is a reason to stop recording what the parser produced.
`case_name_as_written` stays on the record, unscored, exactly as the tree bench
already keeps it.
