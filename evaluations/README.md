# Evaluations

Two pipeline stages are evaluated separately, each against a frozen dataset
published as part of
[gt-csse/false-citation-bench](https://huggingface.co/datasets/gt-csse/false-citation-bench):

| stage | question | dataset | records |
|---|---|---|---:|
| [Extraction](extraction/README.md) | can a citation identifier be found, and read correctly? | `derived/extraction.jsonl` | 594 |
| [Validation](validation/README.md) | does the found citation agree with CourtListener? | `derived/validation-courtlistener-heuristics.jsonl` | 423 |

They are kept apart on purpose. A system that never finds a citation and a
system that finds it and judges it wrongly fail differently, and one combined
score hides which happened.

Each evaluator compares two files and writes its disagreements out:

```
benchmark JSONL (frozen)  +  run JSONL (yours)  →  evaluate.py  →  metrics + non_agreements.json
```

The run artifact is a small, portable JSONL format with no Mellea-LRC types in
it, so any system can be scored.

## Setup

```bash
uv sync

uv run hf auth login

uv run hf download gt-csse/false-citation-bench --repo-type dataset \
  --local-dir data/false-citation-bench
```

The dataset repository is private, so the download needs a Hugging Face account
with access to the `gt-csse` organisation.

## Spans index the document body, not the file

Both benchmarks locate an occurrence by a character span measured from the
first character **after** the `--- Plain text ---` marker. Each file under
`documents_txt/` opens with a provenance header:

```text
Source PDF: …
Backend: docling

--- Plain text ---
IN THE UNITED STATES DISTRICT COURT …
```

Offset 0 is the `I` of `IN THE`. Getting this wrong does not degrade a score,
it destroys one: all 594 extraction spans align against the body and none
against the raw file, so a system that reads the file whole scores zero while
appearing to work.

```python
MARKER = "--- Plain text ---\n"
body = path.read_text(encoding="utf-8").split(MARKER, 1)[1]
```

To confirm the dataset is present and your offsets line up before spending a
run, check that every benchmark span still slices to its own `matched_text`:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

MARKER = "--- Plain text ---\n"
root = Path("data/false-citation-bench")
rows = [json.loads(line) for line in (root / "derived/extraction.jsonl").open()]
bodies, ok = {}, 0
for row in rows:
    name = row["document"]
    if name not in bodies:
        bodies[name] = (root / "documents_txt" / name).read_text(encoding="utf-8").split(MARKER, 1)[1]
    span = row["span"]
    ok += bodies[name][span["start"]:span["end"]] == row["matched_text"]
print(f"{ok}/{len(rows)} spans align")
PY
```

Expect `594/594`.
