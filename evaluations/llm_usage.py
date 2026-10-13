"""Reported provider token usage, independent of workflow accuracy scores."""

from __future__ import annotations

from dataclasses import dataclass, fields

from mellea_lrc.llm.config import PROFILE_OPTION
from mellea_lrc.model.document import Document
from mellea_lrc.model.execution import WORKFLOW_CATALOG, get_workflow
from mellea_lrc.model.ivr import IvrAttempt, IvrRun


def _sum_reported(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


@dataclass(frozen=True)
class UsageTotals:
    """Observed subtotals; an absent provider counter never becomes zero."""

    attempts: int = 0
    usage_reported_attempts: int = 0
    cache_reported_attempts: int = 0
    cache_rate_attempts: int = 0
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

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            **{field.name: getattr(self, field.name) for field in fields(self)},
            "cache_rate": self.cache_rate,
        }


@dataclass(frozen=True)
class LlmUsageRow:
    substage: str
    model: str | None
    endpoint: str | None
    totals: UsageTotals

    @property
    def key(self) -> tuple[str, str | None, str | None]:
        return self.substage, self.model, self.endpoint

    def as_dict(self) -> dict[str, object]:
        return {
            "substage": self.substage,
            "model": self.model,
            "endpoint": self.endpoint,
            **self.totals.as_dict(),
        }


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
            grouped[row.key] = LlmUsageRow(
                row.substage,
                row.model,
                row.endpoint,
                row.totals if previous is None else previous.totals + row.totals,
            )
        return LlmUsageReport(tuple(sorted(grouped.values(), key=_row_order)))

    @property
    def total(self) -> UsageTotals:
        total = UsageTotals()
        for row in self.rows:
            total += row.totals
        return total

    def for_workflow(self, workflow: str) -> LlmUsageReport:
        get_workflow(workflow)
        return LlmUsageReport(tuple(row for row in self.rows if row.substage.split(".", 1)[0] == workflow))

    def as_dict(self) -> dict[str, object]:
        return {"rows": [row.as_dict() for row in self.rows], "total": self.total.as_dict()}


def _run_report(substage: str, run: IvrRun) -> LlmUsageReport:
    profile = run.model_options.get(PROFILE_OPTION)
    endpoint = profile.get("endpoint") if isinstance(profile, dict) else None
    if endpoint is not None and not isinstance(endpoint, str):
        raise TypeError("IVR profile endpoint must be a string or None")
    total = UsageTotals()
    for attempt in run.attempts:
        total += UsageTotals.from_attempt(attempt)
    return LlmUsageReport((LlmUsageRow(substage, run.model, endpoint, total),))


def summarize_llm_usage(document: Document) -> LlmUsageReport:
    """Count citation-owned exchanges and site reviews once in this Document.

    Include repaired and failed attempts, including withdrawn citations and
    declined sites. Judgment references are not another provider exchange.
    A recovered checkpoint naturally contains only its surviving histories.
    """
    report = LlmUsageReport()
    for citation in document.citations:
        nodes = {node.id: node.substage for node in citation.nodes}
        for field in type(citation).model_fields:
            value = getattr(citation, field)
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
        model = row.model or "Unavailable"
        endpoints = {
            other.endpoint
            for other in report.rows
            if (other.substage, other.model) == (row.substage, row.model)
        }
        if len(endpoints) > 1:
            model += f" ({row.endpoint or 'endpoint unavailable'})"
        lines.append(_table_row(row.substage, model, row.totals))
    lines.append(_table_row("**Total**", "", report.total))
    return "\n".join(lines) + "\n"
