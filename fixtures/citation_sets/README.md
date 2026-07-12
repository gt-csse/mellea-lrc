# Curated citation sets

These are small, purpose-specific e2e inputs. Unlike the bookmark fixture,
they are not a research notebook and carry no comments. Every named set is a
directory with one source file per citation, so eyecite resolution cannot bind
parties or years across neighboring test cases.

- `exact-lookup-found/` exercises cheap locator lookup across Supreme Court,
  federal appellate, and district reporters. Every citation is expected to
  return at least one exact-lookup record; identity assessment can still reject
  a record (the Lampe collision is deliberately included).
- `not-found-docket-evidence/` exercises the LLM preparation, opinion/RECAP
  probes, year/date tolerance, and bounded docket-document expansion path.
- `date-recovery/` exercises whitespace-damaged dates: eyecite leaves the
  extraction-side date empty and the grounded preparation path may recover it.

Run a set with, for example:

```bash
uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus \
  --file exact-lookup-found --phase retrieval
```

Broad bookmark fixtures live in `fixtures/bookmarked/sets` and have one JSON
and one text projection per set. Keep this directory for independently shaped,
multi-file execution fixtures only.
