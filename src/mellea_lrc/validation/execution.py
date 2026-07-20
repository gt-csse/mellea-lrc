"""Flat per-citation validation execution."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import TypeAlias

from mellea_lrc.validation.model import CitationValidation, ValidationNode

ValidationOperation: TypeAlias = Callable[[CitationValidation], ValidationNode]
ValidationRouter: TypeAlias = Callable[[ValidationNode], tuple[ValidationOperation, ...]]


def run_citation_loop(
    validation: CitationValidation,
    *,
    initial_operations: tuple[ValidationOperation, ...],
    route: ValidationRouter,
) -> CitationValidation:
    """Execute, append, and route flat validation nodes until the queue is empty.

    A router returns zero operations to stop, one to continue linearly, or
    several to fan out. No operation owns or nests another node.
    """
    pending = deque(initial_operations)
    current = validation
    while pending:
        operation = pending.popleft()
        node = operation(current)
        current = current.append(node)
        pending.extend(route(node))
    return current
