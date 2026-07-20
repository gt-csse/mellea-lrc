"""Flat per-citation validation execution."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, TypeAlias

from mellea_lrc.validation.citation_lookup import run_exact_locator_lookup
from mellea_lrc.validation.model import CitationValidation, ValidationNode

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient

ValidationOperation: TypeAlias = Callable[[CitationValidation], ValidationNode]


def run_citation_loop(
    validation: CitationValidation,
    *,
    client: CourtListenerServiceClient,
) -> CitationValidation:
    """Run one citation from exact lookup until no operations remain.

    Follow-up routing stays here with queue ownership. A route may eventually
    return one operation to continue or several operations to fan out.
    """
    first_operation = partial(run_exact_locator_lookup, client=client)
    pending: deque[ValidationOperation] = deque((first_operation,))
    current = validation
    while pending:
        operation = pending.popleft()
        node = operation(current)
        current = current.append(node)
        pending.extend(_next_operations(node))
    return current


def _next_operations(_node: ValidationNode) -> tuple[ValidationOperation, ...]:
    """Return no follow-up operations until another branch is implemented."""
    return ()
