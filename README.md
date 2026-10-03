# mellea-lrc

The package has four workflows: `grow_roots`, `validate_roots`, `grow_leaves`, and `validate_pincite`. Each takes and returns a `Document`, and each consists of independently callable stages. Locator-body review and intended-case discovery belong to root validation. `validate_pincite` retrieves reporter-root opinions, indexes their explicit pagination, and resolves each citing occurrence to its own pages and writing. It reads grounded occurrence-specific propositions, reviews selected pages, falls back to the saved full opinion when needed, and records support and target location separately. Open-web search is not implemented.

The source layers are:

```text
src/mellea_lrc/
  workflows/       grow_roots, validate_roots, grow_leaves, validate_pincite
  extraction/      citation discovery, field reading, and root/leaf assignment stages
  validation/      retrieval and judgment stages, with local review services
  parsing/         source-text grammars shared by discovery and normalization
  providers/       courtlistener/ and govinfo/ clients and response models
  model/           Document, citations, typed fields, and append-only histories
  preprocessing/   source loading and text preparation
  matching/        fuzzy matching and grounded source quotes
  llm/             model configuration and the reusable IVR wrapper
```

Workflows are defined by the user. A group of stages does not introduce another workflow. `evaluations/` produces one Markdown report and its JSON representation per workflow, including that workflow's stage tables.

See [Architecture](docs/Architecture.md) for the boundaries between stages,
their local services, source readers, and durable citation records.

```python
from pathlib import Path
import asyncio

from mellea_lrc.api import Document, grow_roots, stable

document = Document.from_source(Path("filing.pdf"))
document = asyncio.run(grow_roots(document, rules=stable()))
print(document.full_locators, document.roots)
```

`grow_roots` is async so it can optionally review docket sites with the configured model: pass `hunt_dockets=True` to enable that stage. The document retains citation histories, site reviews, and completed stage runs; after hunting, `document.get_stage("3_docket_locator_site_hunting")` restores that stage's result. See [extraction](docs/Extraction.md), [validation](docs/Validation.md), and [preprocessing](docs/Preprocessing.md).

A `Path` reads a file; a `str` is the document text itself. Install the optional Docling backend for PDFs and other formatted files with `uv sync --group preprocessing`. Plain text works with the base package. Run the active tests with `uv run pytest`.

## Model configuration

Every LLM-backed stage selects one complete, immutable profile from
[`llm/profiles.py`](src/mellea_lrc/llm/profiles.py), alongside its stage name:

```python
from mellea_lrc.llm.profiles import NRP_QWEN

STAGE = "46_reporter_citation_full_opinion_review"
MODEL_PROFILE = NRP_QWEN
```

A profile contains the model, endpoint, credential environment-variable name,
temperature, timeout, token budget, repair budget, output transport, reasoning
effort, and service tier. Each reviewer receives that profile explicitly with
`IvrReporterPinpointReviewer.from_profile(MODEL_PROFILE)`. Page review and
full-opinion review can choose different profiles even though they share that
reviewer. Other stages follow the same convention.

Only credentials live in `.env`: `MELLEA_LRC_NRP_API_KEY` and
`MELLEA_LRC_OPENROUTER_API_KEY`. They are resolved lazily; a missing credential
raises. Model settings do not inherit from environment variables or other
profiles. To test a different package, select another named profile or construct
an explicit `LlmProfile` and pass its reviewer to the stage. Profiles have no
credentials in their serialized representation. The IVR trace records the
profile name and endpoint, while credentials stay outside generation options
and saved documents.

Reasoning effort and service tier are explicit profile settings. An omitted
setting lets the provider choose its default. Endpoint support is verified by
experiments; the project never infers it from the model name.

| Output mode | Endpoint request |
| --- | --- |
| `json_schema` | Strict JSON-schema generation through Mellea's `format` parameter. |
| `json_object` | JSON-object mode, with the full output schema in the prompt. |
| `prompt` | Schema in the prompt, with no endpoint `response_format` parameter. |

Every mode keeps the same Pydantic schema validation, domain requirements, and
model repair loop. Endpoint errors do not silently switch modes, and generated
text is not rewritten into valid JSON. The IVR trace retains the chosen mode,
schema, provider responses, and every repair. Its configured model alias may
differ from the actual model name in the provider response.

The profile's `timeout_seconds` also bounds how long the caller awaits the IVR result across
its repair turns.
If a later request stalls, the failed trace keeps completed attempts captured
before native repair, including their provider exchanges and validation feedback,
then selects a final timeout record. That timeout record contains no invented
provider response; earlier responses remain available for diagnosis.
