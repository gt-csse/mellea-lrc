"""A public stage receives and returns independent document state."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, TypeVar, cast

from mellea_lrc.model.document import Document

_F = TypeVar("_F", bound=Callable[..., Any])


def isolated_stage(stage: _F) -> _F:
    """Give a stage its own snapshot before any citation record can be changed.

    Citation records are mutable inside a stage so a node can atomically cause
    several operations. The document passed by the caller is a checkpoint,
    however: a later stage must not alter its records or evidence history.
    This decorator belongs at the public composition boundary, not around
    private readers or each individual operation.
    """
    if iscoroutinefunction(stage):

        @wraps(stage)
        async def async_stage(document: Document, *args: Any, **kwargs: Any) -> Document:
            before = document.snapshot()
            result = await stage(before.snapshot(), *args, **kwargs)
            if not isinstance(result, Document):
                msg = f"{stage.__name__} returned {type(result).__name__}, not Document"
                raise TypeError(msg)
            before.assert_cumulative_successor(result)
            return result

        return cast(_F, async_stage)

    @wraps(stage)
    def sync_stage(document: Document, *args: Any, **kwargs: Any) -> Document:
        before = document.snapshot()
        result = stage(before.snapshot(), *args, **kwargs)
        if not isinstance(result, Document):
            msg = f"{stage.__name__} returned {type(result).__name__}, not Document"
            raise TypeError(msg)
        before.assert_cumulative_successor(result)
        return result

    return cast(_F, sync_stage)
