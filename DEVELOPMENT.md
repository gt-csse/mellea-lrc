# Local Development

## Enlistment

| Step | Command Line | Additional Information |
| --- | --- | --- |
| 1. Clone the repository locally. | `git clone https://github.com/gt-csse/mellea-lrc` | https://git-scm.com/docs/git-clone |
| 2. Install [uv](https://github.com/astral-sh/uv). | `curl -LsSf https://astral.sh/uv/install.sh \| sh` on macOS and Linux or <br/>`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` on Windows | https://docs.astral.sh |
| 3. Install dependencies. | `uv sync` (Python 3.10; installs the `dev` group) | https://docs.astral.sh/uv/concepts/projects/sync |
| 4. Install [pre-commit](https://pre-commit.com/) hooks | `uv run pre-commit install` | https://pre-commit.com/#1-install-pre-commit |

### Optional dependency groups

| Group | Use when |
| --- | --- |
| `dev` | Default for CI and local lint/test (`uv sync`) |
| `pipeline` | Notebook or corpus runs through preprocess → assess (`uv sync --group pipeline`) |
| `llm` | LLM provider smoke tests only |
| `preprocessing` | Docling conversion only |
| `label-studio` | Label Studio upload scripts |
| `courtlistener` | CourtListener access service locally |
| `modal` | Deploy or serve Modal apps (includes `pipeline`) |

## Development Activities

| Activity | Command Line | Description | Used During Local Development | Invoked by Continuous Integration |
| --- | --- | --- | :-: | :-: |
| Code Formatting | `uv run ruff format` or<br>`uv run ruff format --check` | Format source code using [ruff](https://github.com/astral-sh/ruff) based on settings in `pyproject.toml`. | :white_check_mark: | :white_check_mark: (via [pre-commit](https://pre-commit.com/)) |
| Static Code Analysis | `uv run ruff check` | Validate source code using [ruff](https://github.com/astral-sh/ruff) based on settings in `pyproject.toml`. | :white_check_mark: | :white_check_mark: (via [pre-commit](https://pre-commit.com/)) |
| Run pre-commit scripts | `uv run pre-commit run` | Run [pre-commit](https://pre-commit.com/) scripts based on settings in `.pre-commit-config.yaml`. | :white_check_mark: | :white_check_mark: |
| Automated Testing | `uv run pytest` or<br/>`uv run pytest --no-cov` | Run automated tests using [pytest](https://docs.pytest.org/) and extract code coverage using [coverage](https://coverage.readthedocs.io/) based on settings in `pyproject.toml`. | :white_check_mark: | :white_check_mark: |
| Semantic Version Generation | `uv run python -m AutoGitSemVer.scripts.UpdatePythonVersion ./pyproject.toml ./src` | Generate a new [Semantic Version](https://semver.org/) based on git commits using [AutoGitSemVer](https://github.com/davidbrownell/AutoGitSemVer). Version information is stored in `pyproject.toml`. | | :white_check_mark: |
| Python Package Creation | `uv build` | Create a python package using [uv](https://github.com/astral-sh/uv) based on settings in `pyproject.toml`. Generated packages will be written to `./dist`. | | :white_check_mark: |
| Python Package Publishing | `uv publish` | Publish a python package to [PyPi](https://pypi.org/) using [uv](https://github.com/astral-sh/uv) based on settings in `pyproject.toml`. | | :white_check_mark: |

## Design Documentation

Implementation notes and behavioral contracts live under `docs/development/`:

- [Preprocessing](docs/development/Preprocessing%20Development.md)
- [Extraction](docs/development/Extraction%20Model%20Development.md)
- [Validation](docs/development/validation/index.md) — retrieval and provenance
  only; this phase does not express an opinion
- [Assessment](docs/development/assessment/index.md) — field-level comparisons
  and conclusions
- [Benchmark architecture](docs/development/Benchmark.md)

Validation and assessment each have nested field/path documents. Add new
behavior to the owning directory and link it from that directory's `index.md`.

## Contributing Changes
Pull requests are preferred, since they are specific. For more about how to create a pull request, see https://help.github.com/articles/using-pull-requests/.

We recommend creating different branches for different (logical) changes, and creating a pull request into the `main` branch when you're done. For more information on creating branches, please see https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/.
