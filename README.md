# mellea-lrc

Current development targets [`feat/citation-checking`](https://github.com/gt-csse/mellea-lrc/tree/feat/citation-checking).
Use this branch while its changes are integrated into `main`.

## Quick start

Clone the development branch, install PDF support, and create your runtime configuration:

```bash
git clone --branch feat/citation-checking --single-branch https://github.com/gt-csse/mellea-lrc.git
cd mellea-lrc
uv sync --locked --group preprocessing --extra ml-cite
cp .env.example .env
```

Configure the provider endpoints, credentials, and model profiles in `.env`
(see [Runtime configuration](#runtime-configuration)), then run:

```bash
uv run mellea-lrc validate filing.pdf --output document.json
```

The command runs all four workflows, including docket hunting and docket-root
review, and writes a native `Document` JSON. Upload `document.json` to the
[visualizer](https://mellea-lrc-visualizer.woodygoodenough.com/) to inspect the result.
The same file is updated atomically at each completed checkpoint. If validation
fails, the last saved checkpoint remains available. Completed model reviews are
saved immediately in the adjacent `document.json.run/` directory. Keep this
directory with the Document to resume without repeating recorded reviews:

```bash
uv run mellea-lrc resume document.json
```

Resume calls the same workflow APIs, skipping completed substages. It checks the
source, execution settings, code, and dependencies before continuing. Credentials
remain in `.env`; their values are never copied into run metadata. Existing output
is never overwritten by a new `validate` command. The output must be different
from the source document.
Plain-text input is also supported. Run `uv run mellea-lrc validate --help` for usage.

## Rule and ML field extraction

Locator discovery and citation structure remain part of the Document pipeline.
Rule readers use neighboring citations, colocation groups, and known case names.
Optional ML inference predicts four source spans in one substage and saves them
on each citation. The existing case-name, court, date, and pincite substages then
select `RULE` or `ML` independently and normalize the selected spans.

Use the existing `grow_roots` and `grow_leaves` reports to compare policies on the
same corpus. The standalone ML-cite training project evaluates its own datasets;
it does not import a reduced version of this project's rule pipeline.
See [field extraction](docs/Extraction.md#ml-prediction-and-field-sources).

## Python API

The package organizes citation processing as **workflow → stage → substage**. Its four workflows—`grow_roots`, `validate_roots_citation_identity`, `grow_leaves`, and `validate_all_citation_content`—contain 15 semantic stages and 55 atomic substages. Each level takes and returns a `Document`; substages provide the granular execution and recovery boundaries. The immutable [execution catalog](src/mellea_lrc/model/execution.py) defines their membership, order, and optional steps.

`validate_roots_citation_identity` includes locator-body corroboration and optional intended-case discovery. `validate_all_citation_content` currently handles reporter citation content: it retrieves root opinions, indexes pagination, and resolves each occurrence's pages and writing. It reviews selected pages, falls back to saved full opinions when needed, and records support and target location separately. Open-web search is not implemented.

The source layers are:

```text
src/mellea_lrc/
  workflows/       workflow packages and their semantic stage composers
  extraction/      citation discovery, field reading, and root/leaf assignment substages
  validation/      retrieval and judgment substages, with local review services
  parsing/         source-text grammars shared by discovery and normalization
  providers/       courtlistener/ and govinfo/ clients and response models
  model/           Document, citations, typed fields, and append-only histories
  preprocessing/   source loading and text preparation
  matching/        fuzzy matching and grounded source quotes
  llm/             model configuration and the reusable IVR wrapper
  execution.py     durable execution and resume through the existing workflows
  configuration.py shared .env reader for required runtime settings
```

`evaluations/` produces one Markdown report and its JSON representation per workflow, organized by semantic stages with detailed substage tables.

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

`grow_roots` composes `discover_root_locators`, `read_root_fields`, and `form_root_groups`. Pass `hunt_dockets=True` to enable its docket-hunting substage. The document persists citation histories, site reviews, and one typed checkpoint log, `runs`. `get_substage("grow_roots.discover_root_locators.hunt_docket_locators")` restores the atomic result; `get_stage("grow_roots.discover_root_locators")` restores the completed group. Both return exact snapshots and raise if their checkpoint has not completed. See [extraction](docs/Extraction.md), [validation](docs/Validation.md), and [preprocessing](docs/Preprocessing.md).

A `Path` reads a file; a `str` is the document text itself. The base package
includes the Document workflows and their runtime dependencies. The
`preprocessing` dependency group adds Docling for PDFs and other formatted
files. The optional `ml-cite` extra adds the ML model runtime. Repository
installation is shown in the quick start; run the active tests with
`uv run pytest`.

The same durable runner is available from Python:

```python
from mellea_lrc.execution import validate, resume

document = asyncio.run(validate(Path("filing.pdf"), Path("document.json")))
# After an interrupted execution:
document = asyncio.run(resume(Path("document.json")))
```

Document serialization is unchanged. Existing saved Documents still load and
support `get_stage()` and `get_substage()`. The resume command requires its
execution sidecar; it does not guess missing settings for an older artifact.
Completed successful and failed reviews are both retained. A request interrupted
before its response was durably recorded may need to be sent again.

## Runtime configuration

`.env` defines every named model profile, assigns a profile to each LLM-backed
substage, and configures provider endpoints, credentials, and request timeouts.
Start with [`.env.example`](.env.example), then fill in the credential settings.
Run from the repository root to use its `.env`. That file is authoritative:
ambient environment variables do not override its settings or credentials.
Model choices and settings are not defined in stage scripts or Python profile
constants.

Recommended defaults live in `.env.example`; runtime code never loads that
file or supplies fallback values. Copy those settings into `.env` explicitly.

A substage assignment uses its uppercase identifier with dots replaced by
underscores:

```dotenv
MELLEA_LRC_SUBSTAGE_VALIDATE_ALL_CITATION_CONTENT_REVIEW_CITATION_CONTENT_REVIEW_FULL_OPINIONS_LLM_PROFILE=my_full_review
```

Define that complete package using the prefix `MELLEA_LRC_PROFILE_MY_FULL_REVIEW_`:

| Required suffix | Setting |
| --- | --- |
| `MODEL` | Provider model identifier. |
| `API_BASE` | Endpoint base URL. |
| `API_KEY_ENV` | Name of the credential setting in the same `.env` file. |
| `TEMPERATURE` | Generation temperature. |
| `TIMEOUT_SECONDS` | SDK HTTP connection, read, write, and pool timeout for each request. |
| `MAX_TOKENS` | Output token budget. |
| `MAX_ATTEMPTS` | Maximum generation and repair attempts. |
| `OUTPUT_MODE` | `json_schema`, `json_object`, or `prompt`. |

`REASONING_EFFORT` and `SERVICE_TIER` are optional suffixes. Leave either blank
or omit it to let the provider choose its default. Endpoint support is verified
by experiments; the project never infers it from the model name. Required
settings have no fallback to another profile or a code-defined model package.

Stage code calls [`load_profile(SUBSTAGE)`](src/mellea_lrc/llm/profiles.py) only
when it needs a default reviewer:

```python
from mellea_lrc.llm.profiles import load_profile

SUBSTAGE = "validate_all_citation_content.review_citation_content.review_full_opinions_llm"

# Inside the substage, when an eligible citation needs review:
if service is None:
    service = IvrReporterPinpointReviewer.from_profile(load_profile(SUBSTAGE))
```

Page review and full-opinion review can use different `.env` assignments even
though they share a reviewer. No profile or credential is loaded when a stage
module is imported, when no model review is needed, or when a custom reviewer
is supplied. A missing required profile setting or credential raises when the
default reviewer is needed. The file is read again on each default reviewer
creation, so `.env` edits apply to the next binding without restarting Python.
Change `.env` to select another package; custom reviewers remain available for
controlled tests.

Profiles contain the credential variable name, never the credential itself.
The IVR trace records the resolved profile name and endpoint, while credentials
stay outside generation options and saved documents.

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

There is no whole-review deadline. Each request, including a repair, has its
own HTTP timeout; validation does not consume a shared timeout budget. A stream
can take longer overall while continuing to deliver data. A provider timeout,
connection error, or unsuccessful HTTP status produces a failed trace retaining
earlier attempts and their validation feedback. Missing provider responses and
usage counters are not invented.

Saved traces support failure and repair rates, token usage, and cache statistics
in the workflow reports and a run-level resource summary. These include reporting
coverage, the configured and returned model identifiers, and elapsed time for new
reviews. See [LLM telemetry](evaluations/README.md#llm-telemetry) for definitions
and the command for reporting from existing artifacts without model calls.

Provider factories also read `.env` lazily and require these settings:

| Provider | Required settings |
| --- | --- |
| CourtListener | `COURTLISTENER_BASE_URL`, `COURTLISTENER_TIMEOUT_SECONDS` |
| GovInfo | `GOVINFO_BASE_URL`, `GOVINFO_API_KEY`, `GOVINFO_TIMEOUT_SECONDS` |

CourtListener's proxy normally selects its rotating tokens. Set
`COURTLISTENER_API_TOKEN_RESERVED` in `.env` when using the evaluation runner's
reserved pool. Missing required provider settings raise before a request.
Explicit client configurations and custom reviewers remain available for
controlled tests. Parsing rules and ordinary API options keep their documented
defaults; they are separate from deployment configuration.
