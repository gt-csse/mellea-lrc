"""Observed LLM outcomes and resources, independent of workflow accuracy scores."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path

from mellea_lrc.llm.config import PROFILE_OPTION
from mellea_lrc.model.document import Document
from mellea_lrc.model.execution import WORKFLOW_CATALOG, get_workflow
from mellea_lrc.model.ivr import IvrAttempt, IvrFailureKind, IvrRun


def _sum_reported(left: int | float | None, right: int | float | None) -> int | float | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


_TOKEN_COUNTERS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "reasoning_output_tokens",
)


@dataclass(frozen=True)
class UsageTotals:
    """Observed subtotals; an absent provider counter never becomes zero."""

    attempts: int = 0
    usage_reported_attempts: int = 0
    cache_reported_attempts: int = 0
    cache_rate_attempts: int = 0
    input_reported_attempts: int = 0
    output_reported_attempts: int = 0
    total_reported_attempts: int = 0
    cache_write_reported_attempts: int = 0
    reasoning_reported_attempts: int = 0
    cache_hit_attempts: int = 0
    cache_miss_attempts: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    cache_rate_input_tokens: int | None = None
    cache_rate_cached_input_tokens: int | None = None

    @classmethod
    def from_attempt(cls, attempt: IvrAttempt) -> UsageTotals:
        usage = attempt.usage
        if usage is None:
            return cls(attempts=1)
        cache_reported = usage.cached_input_tokens is not None
        cache_rate_reported = cache_reported and usage.input_tokens is not None
        return cls(
            attempts=1,
            usage_reported_attempts=1,
            cache_reported_attempts=int(cache_reported),
            cache_rate_attempts=int(cache_rate_reported),
            input_reported_attempts=int(usage.input_tokens is not None),
            output_reported_attempts=int(usage.output_tokens is not None),
            total_reported_attempts=int(usage.total_tokens is not None),
            cache_write_reported_attempts=int(usage.cache_write_input_tokens is not None),
            reasoning_reported_attempts=int(usage.reasoning_output_tokens is not None),
            cache_hit_attempts=int(cache_reported and usage.cached_input_tokens > 0),
            cache_miss_attempts=int(cache_reported and usage.cached_input_tokens == 0),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            cache_write_input_tokens=usage.cache_write_input_tokens,
            reasoning_output_tokens=usage.reasoning_output_tokens,
            cache_rate_input_tokens=usage.input_tokens if cache_rate_reported else None,
            cache_rate_cached_input_tokens=usage.cached_input_tokens if cache_rate_reported else None,
        )

    def __add__(self, other: UsageTotals) -> UsageTotals:
        return UsageTotals(
            **{
                field.name: _sum_reported(getattr(self, field.name), getattr(other, field.name))
                for field in fields(self)
            }
        )

    @property
    def cache_rate(self) -> float | None:
        if not self.cache_rate_input_tokens or self.cache_rate_cached_input_tokens is None:
            return None
        return self.cache_rate_cached_input_tokens / self.cache_rate_input_tokens

    @property
    def cache_unreported_attempts(self) -> int:
        return self.attempts - self.cache_reported_attempts

    @property
    def cache_hit_rate(self) -> float | None:
        return _rate(self.cache_hit_attempts, self.cache_reported_attempts)

    @property
    def counter_coverage(self) -> dict[str, int]:
        return dict(
            zip(
                _TOKEN_COUNTERS,
                (
                    self.input_reported_attempts,
                    self.output_reported_attempts,
                    self.total_reported_attempts,
                    self.cache_reported_attempts,
                    self.cache_write_reported_attempts,
                    self.reasoning_reported_attempts,
                ),
                strict=True,
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            **{field.name: getattr(self, field.name) for field in fields(self)},
            "cache_rate": self.cache_rate,
            "cache_unreported_attempts": self.cache_unreported_attempts,
            "cache_hit_rate": self.cache_hit_rate,
            "counter_coverage": self.counter_coverage,
        }


@dataclass(frozen=True)
class ReviewTotals:
    """Review outcomes use runs; failure-event incidence uses attempts.

    Failure categories can overlap. A terminal failure without retained
    diagnostics stays unclassified rather than being called a JSON failure.
    """

    runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    first_attempt_successful_runs: int = 0
    repaired_runs: int = 0
    recovered_after_repair_runs: int = 0
    repair_attempts: int = 0
    attempts_with_failure_events: int = 0
    unclassified_failed_runs: int = 0
    elapsed_reported_runs: int = 0
    elapsed_seconds: float | None = None
    attempt_failure_events: dict[str, int] = field(default_factory=dict)
    final_failure_events: dict[str, int] = field(default_factory=dict)
    failed_requirements: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_run(cls, run: IvrRun) -> ReviewTotals:
        attempts: Counter[str] = Counter()
        requirements: Counter[str] = Counter()
        for attempt in run.attempts:
            attempts.update(_failure_events(attempt))
            requirements.update(
                {
                    requirement.description or "Description unavailable"
                    for requirement in attempt.requirements
                    if not requirement.passed
                }
            )
        final_events = () if run.success else _failure_events(run.attempts[run.selected_attempt])
        return cls(
            runs=1,
            successful_runs=int(run.success),
            failed_runs=int(not run.success),
            first_attempt_successful_runs=int(run.first_attempt_success),
            repaired_runs=int(run.repaired),
            recovered_after_repair_runs=int(run.recovered_after_repair),
            repair_attempts=len(run.attempts) - 1,
            attempts_with_failure_events=sum(bool(_failure_events(attempt)) for attempt in run.attempts),
            unclassified_failed_runs=int(not run.success and not final_events),
            elapsed_reported_runs=int(run.elapsed_seconds is not None),
            elapsed_seconds=run.elapsed_seconds,
            attempt_failure_events=dict(attempts),
            final_failure_events=dict(Counter(final_events)),
            failed_requirements=dict(requirements),
        )

    def __add__(self, other: ReviewTotals) -> ReviewTotals:
        counters = {"attempt_failure_events", "final_failure_events", "failed_requirements"}
        return ReviewTotals(
            **{
                item.name: (
                    dict(Counter(getattr(self, item.name)) + Counter(getattr(other, item.name)))
                    if item.name in counters
                    else _sum_reported(getattr(self, item.name), getattr(other, item.name))
                )
                for item in fields(self)
            }
        )

    @property
    def failure_rate(self) -> float | None:
        return _rate(self.failed_runs, self.runs)

    @property
    def first_attempt_success_rate(self) -> float | None:
        return _rate(self.first_attempt_successful_runs, self.runs)

    @property
    def repair_rate(self) -> float | None:
        return _rate(self.repaired_runs, self.runs)

    @property
    def recovery_rate(self) -> float | None:
        return _rate(self.recovered_after_repair_runs, self.repaired_runs)

    def as_dict(self) -> dict[str, object]:
        return {
            **{item.name: getattr(self, item.name) for item in fields(self)},
            "failure_rate": self.failure_rate,
            "first_attempt_success_rate": self.first_attempt_success_rate,
            "repair_rate": self.repair_rate,
            "recovery_rate": self.recovery_rate,
        }


def _failure_events(attempt: IvrAttempt) -> tuple[str, ...]:
    events = [kind.value for kind in attempt.failure_kinds]
    error = attempt.transport_error
    if error is not None and error.status_code == 429:
        events.append("http_429")
    if error is not None and error.status_code is not None and error.status_code >= 500:
        events.append("http_5xx")
    return tuple(events)


@dataclass(frozen=True)
class ReturnedModelUsage:
    """Each attempt belongs to its actual response.model, or to unknown."""

    model: str | None
    totals: UsageTotals

    def as_dict(self) -> dict[str, object]:
        return {"model": self.model, **self.totals.as_dict()}


@dataclass(frozen=True)
class LlmUsageRow:
    substage: str
    model: str | None
    endpoint: str | None
    totals: UsageTotals
    reviews: ReviewTotals = field(default_factory=ReviewTotals)
    returned_models: tuple[ReturnedModelUsage, ...] = ()
    configurations: tuple[dict[str, object], ...] = ()

    @property
    def key(self) -> tuple[str, str | None, str | None]:
        return self.substage, self.model, self.endpoint

    def as_dict(self) -> dict[str, object]:
        return {
            "substage": self.substage,
            "model": self.model,
            "endpoint": self.endpoint,
            **self.totals.as_dict(),
            "reviews": self.reviews.as_dict(),
            "returned_models": [model.as_dict() for model in self.returned_models],
            "configurations": list(self.configurations),
        }

    def __add__(self, other: LlmUsageRow) -> LlmUsageRow:
        if self.key != other.key:
            raise ValueError("LLM usage rows must share a substage, requested model, and endpoint")
        models: dict[str | None, UsageTotals] = {}
        for model in (*self.returned_models, *other.returned_models):
            models[model.model] = models.get(model.model, UsageTotals()) + model.totals
        configurations = list(self.configurations)
        for configuration in other.configurations:
            if configuration not in configurations:
                configurations.append(configuration)
        return LlmUsageRow(
            self.substage,
            self.model,
            self.endpoint,
            self.totals + other.totals,
            self.reviews + other.reviews,
            tuple(
                ReturnedModelUsage(model, totals)
                for model, totals in sorted(models.items(), key=lambda item: item[0] or "")
            ),
            tuple(configurations),
        )


_SUBSTAGE_ORDER = {
    substage.name: index
    for index, substage in enumerate(
        substage for workflow in WORKFLOW_CATALOG for stage in workflow.stages for substage in stage.substages
    )
}


def _row_order(row: LlmUsageRow) -> tuple[int, str, str, str]:
    return (
        _SUBSTAGE_ORDER.get(row.substage, len(_SUBSTAGE_ORDER)),
        row.substage,
        row.model or "",
        row.endpoint or "",
    )


@dataclass(frozen=True)
class LlmUsageReport:
    rows: tuple[LlmUsageRow, ...] = ()

    def __add__(self, other: LlmUsageReport) -> LlmUsageReport:
        grouped: dict[tuple[str, str | None, str | None], LlmUsageRow] = {}
        for row in (*self.rows, *other.rows):
            previous = grouped.get(row.key)
            grouped[row.key] = row if previous is None else previous + row
        return LlmUsageReport(tuple(sorted(grouped.values(), key=_row_order)))

    @property
    def total(self) -> UsageTotals:
        total = UsageTotals()
        for row in self.rows:
            total += row.totals
        return total

    @property
    def review_total(self) -> ReviewTotals:
        total = ReviewTotals()
        for row in self.rows:
            total += row.reviews
        return total

    def for_workflow(self, workflow: str) -> LlmUsageReport:
        get_workflow(workflow)
        return LlmUsageReport(tuple(row for row in self.rows if row.substage.split(".", 1)[0] == workflow))

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "total": {**self.total.as_dict(), "reviews": self.review_total.as_dict()},
            "usage_source": "Retained provider response.usage, including repair and failed attempts.",
        }


def _run_report(substage: str, run: IvrRun) -> LlmUsageReport:
    profile = run.model_options.get(PROFILE_OPTION)
    endpoint = profile.get("endpoint") if isinstance(profile, dict) else None
    if endpoint is not None and not isinstance(endpoint, str):
        raise TypeError("IVR profile endpoint must be a string or None")
    total = UsageTotals()
    models: dict[str | None, UsageTotals] = {}
    for attempt in run.attempts:
        usage = UsageTotals.from_attempt(attempt)
        total += usage
        model = attempt.response.get("model") if isinstance(attempt.response, dict) else None
        if model is not None and not isinstance(model, str):
            raise TypeError("Provider response model must be a string or None")
        models[model] = models.get(model, UsageTotals()) + usage
    row = LlmUsageRow(
        substage,
        run.model,
        endpoint,
        total,
        ReviewTotals.from_run(run),
        tuple(
            ReturnedModelUsage(model, usage)
            for model, usage in sorted(models.items(), key=lambda item: item[0] or "")
        ),
        ({"backend": run.backend, "model_options": run.model_options},),
    )
    return LlmUsageReport((row,))


def summarize_llm_usage(document: Document) -> LlmUsageReport:
    """Count citation-owned exchanges and site reviews once in this Document.

    Include repaired and failed attempts, including withdrawn citations and
    declined sites. Judgment references are not another provider exchange.
    A recovered checkpoint naturally contains only its surviving histories.
    """
    report = LlmUsageReport()
    for citation in document.citations:
        nodes = {node.id: node.substage for node in citation.nodes}
        for field_name in type(citation).model_fields:
            value = getattr(citation, field_name)
            records = value if isinstance(value, tuple) else (value,)
            for record in records:
                run = getattr(record, "ivr", None)
                if run is not None:
                    report += _run_report(nodes[record.node_id], run)
    for review in document.site_reviews:
        if review.ivr is not None:
            report += _run_report(review.substage, review.ivr)
    return report


def _number(value: int | None) -> str:
    return "Unavailable" if value is None else f"{value:,}"


def _percent(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:.1%}"


def _fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({_percent(_rate(numerator, denominator))})"


def _cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def _model_label(report: LlmUsageReport, row: LlmUsageRow) -> str:
    model = row.model or "Unavailable"
    endpoints = {
        other.endpoint for other in report.rows if (other.substage, other.model) == (row.substage, row.model)
    }
    if len(endpoints) > 1:
        model += f" ({row.endpoint or 'endpoint unavailable'})"
    return _cell(model)


def _table_row(substage: str, model: str, total: UsageTotals) -> str:
    rate = "Unavailable" if total.cache_rate is None else f"{total.cache_rate:.1%}"
    return (
        f"| {substage} | {model} | {total.usage_reported_attempts}/{total.attempts} | "
        f"{_number(total.input_tokens)} | {_number(total.output_tokens)} | "
        f"{_number(total.cached_input_tokens)} | {total.cache_reported_attempts}/{total.attempts} | {rate} |"
    )


def render_llm_usage(report: LlmUsageReport) -> str:
    """Render resource observations without changing the accuracy denominators."""
    lines = [
        "## LLM review outcomes",
        "",
        "Counts cover retained IVR histories only, including failed reviews and every repair attempt. "
        "A review is one instruct/validate/repair run, not one HTTP request. Failure, first-pass success, "
        "and repair rates use all reviews; recovery uses reviews requiring repair. "
        "Untraced or uncheckpointed calls cannot be counted.",
        "",
        "| Substage | Requested model | Reviews | Successful | Final failures | First-pass success | "
        "Requiring repair | Recovered after repair | Attempts | Repair attempts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    observations = [(row.substage, _model_label(report, row), row.totals, row.reviews) for row in report.rows]
    observations.append(("**Total**", "", report.total, report.review_total))
    for substage, model, usage, reviews in observations:
        lines.append(
            f"| {substage} | {model} | {reviews.runs} | {reviews.successful_runs} | "
            f"{_fraction(reviews.failed_runs, reviews.runs)} | "
            f"{_fraction(reviews.first_attempt_successful_runs, reviews.runs)} | "
            f"{_fraction(reviews.repaired_runs, reviews.runs)} | "
            f"{_fraction(reviews.recovered_after_repair_runs, reviews.repaired_runs)} | "
            f"{usage.attempts} | {reviews.repair_attempts} |"
        )
    lines += [
        "",
        "## LLM failure events",
        "",
        "Categories overlap: for example, truncated output can also be invalid JSON. "
        "Attempt incidence uses all retained attempts; terminal incidence uses all reviews and only "
        "their selected final attempt. HTTP 429 and 5xx are subsets of HTTP-status failures. "
        "Requirement-validation events include grounding checks; exact failed requirement descriptions "
        "are listed separately. Unknown causes are not inferred from missing output.",
        "",
        "| Substage | Requested model | Event | Attempt incidence | Terminal incidence |",
        "|---|---|---|---:|---:|",
    ]
    kinds = (*[kind.value for kind in IvrFailureKind], "http_429", "http_5xx")
    for substage, model, usage, reviews in observations:
        for kind in kinds:
            attempts = reviews.attempt_failure_events.get(kind, 0)
            final = reviews.final_failure_events.get(kind, 0)
            if substage == "**Total**" or attempts or final:
                lines.append(
                    f"| {substage} | {model} | {kind} | {_fraction(attempts, usage.attempts)} | "
                    f"{_fraction(final, reviews.runs)} |"
                )
        if substage == "**Total**" or reviews.unclassified_failed_runs:
            lines.append(
                f"| {substage} | {model} | Unclassified terminal failure | Unavailable | "
                f"{_fraction(reviews.unclassified_failed_runs, reviews.runs)} |"
            )
    lines += [
        "",
        "| Substage | Requested model | Failed requirement | Attempt incidence |",
        "|---|---|---|---:|",
    ]
    for substage, model, usage, reviews in observations:
        for description, count in sorted(reviews.failed_requirements.items()):
            lines.append(
                f"| {substage} | {model} | {_cell(description)} | {_fraction(count, usage.attempts)} |"
            )
    if not report.review_total.failed_requirements:
        lines.append("| **Total** | | No failed requirements retained | 0 |")
    lines += [
        "",
        "## Token usage",
        "",
        "Tokens are reported subtotals, including repair attempts. Missing counters are unavailable, not zero. "
        "Cached tokens are already included in input tokens; reasoning tokens are already included in output. "
        "Cache rate uses only attempts reporting both input and cached-input counters.",
        "",
        "| Substage | Model | Usage reported | Input tokens | Output tokens | Cached input tokens | "
        "Cache counters | Cache rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.rows:
        lines.append(_table_row(row.substage, _model_label(report, row), row.totals))
    lines.append(_table_row("**Total**", "", report.total))
    lines += [
        "",
        "Usage comes from the retained provider response's usage object. Counter coverage is shown "
        "independently: a subtotal with incomplete coverage is not a complete token total. "
        "Zero is a reported observation; unavailable is not zero.",
        "",
        "| Substage | Requested model | Input coverage | Output coverage | Total tokens (coverage) | "
        "Cache-write input (coverage) | Reasoning output (coverage) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for substage, model, usage, _ in observations:
        lines.append(
            f"| {substage} | {model} | {usage.input_reported_attempts}/{usage.attempts} | "
            f"{usage.output_reported_attempts}/{usage.attempts} | "
            f"{_number(usage.total_tokens)} ({usage.total_reported_attempts}/{usage.attempts}) | "
            f"{_number(usage.cache_write_input_tokens)} ({usage.cache_write_reported_attempts}/{usage.attempts}) | "
            f"{_number(usage.reasoning_output_tokens)} ({usage.reasoning_reported_attempts}/{usage.attempts}) |"
        )
    lines += [
        "",
        "## Prompt-cache observations",
        "",
        "A request hit means the provider reported positive cached-input tokens; a miss means it "
        "explicitly reported zero. Unreported means no cache counter was retained, whether the provider "
        "omitted it or no response was received. Request hit rate uses only reported cache counters. "
        "Token cache rate weights cached tokens against input tokens only on attempts reporting both "
        "counters. Neither rate infers caching from latency, repeated prefixes, or model names.",
        "",
        "| Substage | Requested model | Hit attempts | Miss attempts | Unreported | Request hit rate | "
        "Paired counters | Token cache rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for substage, model, usage, _ in observations:
        lines.append(
            f"| {substage} | {model} | {usage.cache_hit_attempts} | {usage.cache_miss_attempts} | "
            f"{usage.cache_unreported_attempts} | {_percent(usage.cache_hit_rate)} | "
            f"{usage.cache_rate_attempts}/{usage.attempts} | {_percent(usage.cache_rate)} |"
        )
    lines += [
        "",
        "## Models and configuration",
        "",
        "Requested model and returned response.model are distinct observations. Each attempt appears "
        "under its returned model, or Unavailable when the provider did not identify it. Configuration "
        "is the retained profile and call options; old missing settings remain unknown.",
        "",
        "| Substage | Requested model | Endpoint | Returned model | Attempts | Usage reported | "
        "Input tokens | Output tokens | Cached input tokens |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.rows:
        for returned in row.returned_models:
            usage = returned.totals
            lines.append(
                f"| {row.substage} | {_cell(row.model or 'Unavailable')} | {_cell(row.endpoint or 'Unavailable')} | "
                f"{_cell(returned.model or 'Unavailable')} | {usage.attempts} | "
                f"{usage.usage_reported_attempts}/{usage.attempts} | {_number(usage.input_tokens)} | "
                f"{_number(usage.output_tokens)} | {_number(usage.cached_input_tokens)} |"
            )
    lines += [
        "",
        "| Substage | Requested model | Retained configuration |",
        "|---|---|---|",
    ]
    for row in report.rows:
        for configuration in row.configurations:
            lines.append(
                f"| {row.substage} | {_model_label(report, row)} | "
                f"{_cell(json.dumps(configuration, ensure_ascii=False, sort_keys=True))} |"
            )
    lines += [
        "",
        "## Review duration",
        "",
        "Elapsed seconds measure the complete wrapper review, including generation, validation, and "
        "repairs. They are not per-HTTP latency or corpus wall time. Only genuinely measured durations "
        "are summed; overlapping reviews can run concurrently and old missing durations remain unknown.",
        "",
        "| Substage | Requested model | Measured reviews | Observed elapsed seconds |",
        "|---|---|---:|---:|",
    ]
    for substage, model, _, reviews in observations:
        elapsed = "Unavailable" if reviews.elapsed_seconds is None else f"{reviews.elapsed_seconds:,.3f}"
        lines.append(f"| {substage} | {model} | {reviews.elapsed_reported_runs}/{reviews.runs} | {elapsed} |")
    return "\n".join(lines) + "\n"


def write_llm_usage_run(run_dir: Path, report: LlmUsageReport) -> None:
    """Write one run-wide resource report without consulting accuracy golds."""
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    saved = sum((run_dir / "documents" / f"{name}.json").is_file() for name in record["filings"])
    workflows = {workflow.name: report.for_workflow(workflow.name) for workflow in WORKFLOW_CATALOG}
    metadata = {
        "set": record["set"],
        "run": run_dir.name,
        "run_status": record["status"],
        "saved_documents": saved,
        "expected_documents": len(record["filings"]),
        "scope": "Retained IVR histories in saved Documents only; uncheckpointed or untraced calls are unknown.",
    }
    (run_dir / "llm_usage.json").write_text(
        json.dumps(
            {
                **metadata,
                **report.as_dict(),
                "workflows": {name: usage.as_dict() for name, usage in workflows.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# LLM outcomes and resources",
        "",
        f"Set: {record['set']}. Run: {run_dir.name}. Status: {record['status']}. "
        f"Saved Documents: {saved}/{len(record['filings'])}.",
        "",
        metadata["scope"],
        "",
        "## Workflow totals",
        "",
        "These are resource observations, separate from the four workflows' accuracy scores. "
        "All observed repair attempts are included. Failure and repair rates use reviews.",
        "",
        "| Workflow | Reviews | Final failures | Requiring repair | Attempts | Input tokens | "
        "Output tokens | Cached input tokens | Request cache hit rate | Token cache rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, usage in (*workflows.items(), ("**Total**", report)):
        tokens, reviews = usage.total, usage.review_total
        lines.append(
            f"| {name} | {reviews.runs} | {_fraction(reviews.failed_runs, reviews.runs)} | "
            f"{_fraction(reviews.repaired_runs, reviews.runs)} | {tokens.attempts} | "
            f"{_number(tokens.input_tokens)} | {_number(tokens.output_tokens)} | "
            f"{_number(tokens.cached_input_tokens)} | {_percent(tokens.cache_hit_rate)} | {_percent(tokens.cache_rate)} |"
        )
    (run_dir / "llm_usage.md").write_text(
        "\n".join(lines) + "\n\n" + render_llm_usage(report), encoding="utf-8"
    )


def score_llm_usage_run(run_dir: Path) -> LlmUsageReport:
    """Summarize saved traces even when annotations changed or a run is partial."""
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    expected = {f"{name}.json" for name in record["filings"]}
    paths = tuple(sorted((run_dir / "documents").glob("*.json")))
    actual = {path.name for path in paths}
    if not actual.issubset(expected) or (record["status"] == "complete" and actual != expected):
        raise ValueError("Run Documents do not match its document list")
    report = LlmUsageReport()
    for path in paths:
        report += summarize_llm_usage(Document.model_validate_json(path.read_text(encoding="utf-8")))
    write_llm_usage_run(run_dir, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    score_llm_usage_run(run_dir)
    print(f"Wrote {run_dir / 'llm_usage.md'} and {run_dir / 'llm_usage.json'}")


if __name__ == "__main__":
    main()
