**Project:**
[![License](https://img.shields.io/github/license/gt-csse/mellea-lrc?color=dark-green)](https://github.com/gt-csse/mellea-lrc/blob/master/LICENSE)

**Package:**
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mellea_lrc?color=dark-green)](https://pypi.org/project/mellea_lrc/)
[![PyPI - Version](https://img.shields.io/pypi/v/mellea_lrc?color=dark-green)](https://pypi.org/project/mellea_lrc/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/mellea_lrc)](https://pypistats.org/packages/mellea-lrc)

**Development:**
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pytest](https://img.shields.io/badge/pytest-enabled-brightgreen)](https://docs.pytest.org/)
[![CI](https://github.com/gt-csse/mellea-lrc/actions/workflows/CICD.yml/badge.svg)](https://github.com/gt-csse/mellea-lrc/actions/workflows/CICD.yml)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/y/gt-csse/mellea-lrc?color=dark-green)](https://github.com/gt-csse/mellea-lrc/commits/main/)

<!-- Content above this delimiter will be copied to the generated README.md file. DO NOT REMOVE THIS COMMENT, as it will cause regeneration to fail. -->

## Contents
- [Overview](#overview)
- [Evaluations](#evaluations)
- [Installation](#installation)
- [Development](#development)
- [Additional Information](#additional-information)
- [License](#license)

## Overview

`mellea-lrc` checks whether the citations in a legal filing hold up. It reads a
document, finds every citation in it, and asks CourtListener whether the
authority each one names exists and matches how the filing cites it — case name,
court, year, and the proposition a pinpoint is offered for.

Three layers run in order, each consuming what the last produced:

| Layer | Input | Output |
| --- | --- | --- |
| Preprocessing | PDF or DOCX via [Docling](https://github.com/docling-project/docling), or plain text | `PreprocessedDocument` |
| Extraction | preprocessed text | `ExtractedDocument` |
| Validation | extracted citations | `ValidatedDocument` |

Every citation keeps a span into the preprocessed text, and each validation step
is recorded as its own node, so a verdict can be traced back to the characters
that produced it. Extraction is deterministic and offline; only validation needs
a CourtListener key and a model endpoint.

### How to use `mellea-lrc`

Validation reaches CourtListener and a model, so set those up first:

```bash
cp .env.example .env    # then fill in the values below
```

| variable | what it is |
| --- | --- |
| `COURTLISTENER_API_TOKEN` | citation lookup answers `401` without one |
| `MELLEA_LRC_LLM_MODEL` | model id as your endpoint names it |
| `MELLEA_LRC_LLM_API_BASE` | OpenAI-compatible base URL, ending in `/v1` |
| `MELLEA_LRC_LLM_API_KEY` | key for that endpoint |
| `MELLEA_LRC_LLM_TEMPERATURE` | optional; defaults to `0.0`, and leave it there |

Nothing reads `.env` implicitly, so the commands below are run through
`uv run --env-file .env`. Exporting the variables into your shell works too.

Installing the package provides a `mellea-lrc` command. It parses the citations
out of a source and validates each one. `examples/sample-filing.txt` is a
paragraph to try it on — one citation, stating one proposition the cited page
actually supports:

```bash
uv run --env-file .env mellea-lrc validate \
  --from-file examples/sample-filing.txt --output validated.json
```

```
Parsed 1 citations; validating
=== INFO ====== Sampling was successful.
=== INFO ====== Sampling was successful.
Wrote validated.json
```

Without `--from-file` the argument is the text itself — the same paragraph,
pasted straight in:

```bash
uv run --env-file .env mellea-lrc validate "The District defends its assignment plan on a single ground: that the two campuses it operates receive identical funding, identical staffing, and identical facilities, so no student is disadvantaged by being sent to one rather than the other. That defense is foreclosed. Segregating public school children by race denies them equal educational opportunity even when the physical facilities and other tangible factors at each school are equal. See Brown v. Board of Education, 347 U.S. 483, 495 (1954)." --output validated.json
```

`--output` takes the filename; a bare one after the text is rejected, since that
slot is the text itself.

`validated.json` holds one node per check, in the order they ran. On this
paragraph every one of them agrees:

| node | outcome |
| --- | --- |
| `ExactLocatorLookupNode` | `found` |
| `CandidateEvaluationNode` | `ready` |
| `ExactCaseNameCheckNode` | `match` |
| `YearCheckNode` | `match` |
| `DocketCourtRetrievalNode` | `found` |
| `CourtCheckNode` | `match` |
| `LocatorCandidateAssessmentNode` | `match` |
| `ReporterPageRetrievalNode` | `found` |
| `MelleaCitingPropositionExtractionNode` | `identified` |
| `MelleaPinpointCheckNode` | `supports` |
| `LocatorCitationSummaryNode` | `complete` |

The last one is the substantive check — whether the cited page carries the
proposition the citing sentence attributes to it — and it has to quote the page
to say so:

```json
{
  "node_type": "MelleaPinpointCheckNode",
  "outcome": "supports",
  "evidence_quote": "Separate educational facilities are inherently unequal.",
  "evidence_span": { "start": 172, "end": 227 },
  "evidence_match_method": "exact",
  "evidence_match_score": 1.0
}
```

A verdict the model cannot ground in a quote from the page is discarded rather
than reported, which is why `evidence_span` is always an offset into the text
that was actually retrieved.

Two things worth knowing. Keep a citation on one line: a break between the
reporter and the page — `347 U.S.` / `483` — is damage the extractor does not
repair, and the citation is missed rather than misread. And pass the argument
the surrounding sentences, not the citation alone; the proposition being checked
is read out of the text around the citation, so there has to be one there.

### The Python API

Extraction takes a `str` as content or a `Path` as a location:

```python
from pathlib import Path

from mellea_lrc.extraction import extract

document = extract(Path("filing.pdf"))
for citation in document.full_citations:
    print(citation.matched_text, citation.locator_span)
```

Validation is async, and reads `COURTLISTENER_API_TOKEN` and `MELLEA_LRC_LLM_*`
from the environment:

```python
import asyncio

from mellea_lrc.validation import validate_document

validated = asyncio.run(validate_document(document))
for entry in validated.citations:
    print(entry.citation_id, entry.aggregation)
```

## Evaluations

Extraction and validation are evaluated separately against the frozen
[false-citation-bench](https://huggingface.co/datasets/gt-csse/false-citation-bench)
dataset. Each evaluator reads a small JSONL run artifact, so any system can be
scored, not only this one. See [evaluations/](evaluations/README.md).

<!-- Content below this delimiter will be copied to the generated README.md file. DO NOT REMOVE THIS COMMENT, as it will cause regeneration to fail. -->

## Installation

| Installation Method | Command |
| --- | --- |
| Via [uv](https://github.com/astral-sh/uv) | `uv add mellea_lrc` |
| Via [pip](https://pip.pypa.io/en/stable/) | `pip install mellea_lrc` |



## Development
Please visit [Contributing](https://github.com/gt-csse/mellea-lrc/blob/main/CONTRIBUTING.md) and [Development](https://github.com/gt-csse/mellea-lrc/blob/main/DEVELOPMENT.md) for information on contributing to this project.

## Additional Information
Additional information can be found at these locations.

| Title | Document | Description |
| --- | --- | --- |
| Code of Conduct | [CODE_OF_CONDUCT.md](https://github.com/gt-csse/mellea-lrc/blob/main/CODE_OF_CONDUCT.md) | Information about the norms, rules, and responsibilities we adhere to when participating in this open source community. |
| Contributing | [CONTRIBUTING.md](https://github.com/gt-csse/mellea-lrc/blob/main/CONTRIBUTING.md) | Information about contributing to this project. |
| Development | [DEVELOPMENT.md](https://github.com/gt-csse/mellea-lrc/blob/main/DEVELOPMENT.md) | Information about development activities involved in making changes to this project. |
| CourtListener client | [docs/courtlistener-client.md](docs/courtlistener-client.md) | Direct CourtListener citation-lookup usage. |
| Governance | [GOVERNANCE.md](https://github.com/gt-csse/mellea-lrc/blob/main/GOVERNANCE.md) | Information about how this project is governed. |
| Maintainers | [MAINTAINERS.md](https://github.com/gt-csse/mellea-lrc/blob/main/MAINTAINERS.md) | Information about individuals who maintain this project. |
| Security | [SECURITY.md](https://github.com/gt-csse/mellea-lrc/blob/main/SECURITY.md) | Information about how to privately report security issues associated with this project. |

## License
`mellea-lrc` is licensed under the <a href="https://choosealicense.com/licenses/MIT/" target="_blank">MIT</a> license.
